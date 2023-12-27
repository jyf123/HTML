import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb

import networkx as nx
from wl_test import WL_test, WL_simlar
from subiso import get_w
import datetime
from copy import deepcopy


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1, tau: float = 0.5):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        self.tau: float = tau

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        # iso simlar prediction
        self.proj_iso_head = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.embedding_dim * 2, 1), nn.Sigmoid())

        # subiso
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M, f_e = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y, M, f_e

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

        # predict isomorphism head

    def iso_proj(self, x, edge_index, batch, x_aug, edge_index_aug, batch_aug):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        if x_aug is None:
            x_aug = torch.ones(batch.shape[0]).to(device)

        y, M, _ = self.encoder(x, edge_index, batch)
        y_aug, M_aug, _ = self.encoder(x_aug, edge_index_aug, batch_aug)

        y_comb = torch.cat((y, y_aug), dim=1)
        output = self.proj_iso_head(y_comb)
        return output

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def subiso(self, z):
        z = self.projection(z)
        cos_sim = self.sim(z, z)
        cos_sim = torch.unsqueeze(cos_sim, 2)
        cos_sim = self.fc3(cos_sim)
        return torch.squeeze(cos_sim, 2)


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    epochs = 20
    log_interval = 10
    batch_size = 128
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    CELoss = nn.CrossEntropyLoss(reduction="mean")
    MSEloss = nn.MSELoss(reduction='mean')

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        for data in dataloader:

            data, data_aug = data
            data_copy = deepcopy(data)
            optimizer.zero_grad()

            node_num, _ = data.x.size()

            data = data.to(device)
            x, M, f_e = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                            not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)

            '''
            print(data.edge_index)
            print(data.edge_index.size())
            print(data_aug.edge_index)
            print(data_aug.edge_index.size())
            print(data.x.size())
            print(data_aug.x.size())
            print(data.batch.size())
            print(data_aug.batch.size())
            pdb.set_trace()
            '''

            x_aug, M_aug, f_e_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)


            loss = model.loss_cal(x, x_aug)

            log = ""
            # add isomorphism logic
            if args.iso_logic:

                log += "_wlsimlar_" + str(args.a)

                if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                    edge_idx_ori = data_copy.edge_index.cpu().numpy()
                    _, edge_num_ori = edge_idx_ori.shape
                    idx_ori = [n for n in range(node_num)]
                    node_num_ori = len(idx_ori)

                    edge_idx_ori = [[edge_idx_ori[0, n], edge_idx_ori[1, n]] for n in range(edge_num_ori) if
                                    not edge_idx_ori[0, n] == edge_idx_ori[1, n]]
                    data_copy.edge_index = torch.tensor(edge_idx_ori).transpose_(0, 1)
                data_copy = data_copy.to(device)

                # 1.get isomorphism label
                graphs = []
                graphs_aug = []
                graph_iso_label = []
                edge_list = data_copy.edge_index.cpu().numpy().tolist()
                edge_list_aug = data_aug.edge_index.cpu().numpy().tolist()
                for i in range(data_copy.num_graphs):
                    node_list = torch.nonzero(data_copy.batch == i).squeeze(1).cpu().numpy().tolist()
                    node_list_aug = torch.nonzero(data_aug.batch == i).squeeze(1).cpu().numpy().tolist()

                    graph = nx.DiGraph()
                    graph.add_nodes_from(node_list)
                    edges = []
                    for node in node_list:
                        index = torch.nonzero(data_copy.edge_index[0] == node).squeeze(1).cpu().numpy().tolist()
                        edges += [(edge_list[0][i], edge_list[1][i]) for i in index]

                    graph.add_edges_from(edges)
                    for node in node_list:
                        graph.nodes[node]['label'] = str(graph.degree(node))
                    graphs.append(graph)

                    graph_aug = nx.DiGraph()
                    graph_aug.add_nodes_from(node_list_aug)
                    edges_aug = []
                    for node_aug in node_list_aug:
                        index_aug = torch.nonzero(data_aug.edge_index[0] == node_aug).squeeze(1).cpu().numpy().tolist()
                        edges_aug += [(edge_list_aug[0][i], edge_list_aug[1][i]) for i in index_aug]
                    graph_aug.add_edges_from(edges_aug)
                    for node_aug in node_list_aug:
                        graph_aug.nodes[node_aug]['label'] = str(graph_aug.degree(node_aug))
                    graphs_aug.append(graph_aug)


                    node_labels = nx.get_node_attributes(graph, 'label')
                    node_labels_aug = nx.get_node_attributes(graph_aug, 'label')
                    if graph_aug.number_of_nodes() == 0:
                        graph_iso_label.append(0)
                        continue

                    graph_iso_label.append(WL_simlar(node_labels, node_labels_aug, graph, graph_aug))

                graph_iso_label = torch.tensor(graph_iso_label).to(device)
                # 2.make prediction use simclr
                graph_iso_pre = model.iso_proj(data_copy.x, data_copy.edge_index, data_copy.batch, data_aug.x,
                                               data_aug.edge_index, data_aug.batch)
                graph_iso_pre = torch.squeeze(graph_iso_pre, 1)


                # 3.loss
                loss_iso = MSEloss(graph_iso_pre, graph_iso_label)
                loss+= args.a * loss_iso
            if args.subiso_logic:
                log += "_subiso_" + str(args.b)
                weight = get_w(data)
                mse_loss = 0
                for i in range(data.num_graphs):
                    node_list = torch.nonzero(data.batch == i).squeeze(1)
                    node_embedding = f_e[node_list[0]:node_list[-1] + 1]
                    cos_sim = model.subiso(node_embedding)
                    cos_sim = cos_sim / cos_sim.sum(1, keepdim=True)
                    mse_loss += MSEloss(weight[i], cos_sim)
                loss_subiso = mse_loss / data.num_graphs
                loss += args.b * loss_subiso


            loss_all += loss.item()
            loss.backward()
            optimizer.step()


        print('{}: Epoch {}, Loss {}'.format(datetime.datetime.now(), epoch, loss_all/len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)

    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug + log, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')

