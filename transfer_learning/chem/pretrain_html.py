import argparse

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import networkx as nx
from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

from copy import deepcopy
from wl_test import WL_simlar
from subiso import get_w

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        # iso simlar prediction
        self.proj_iso_head = nn.Sequential(nn.Linear(300 * 2, 300 * 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(300 * 2, 1), nn.Sigmoid())

        # subiso
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x,h_1 = self.gnn(x, edge_index, edge_attr)
        x_g = self.pool(x, batch)
        x = self.projection_head(x_g)
        return x,x_g,h_1

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def iso_proj(self, x_g1, x_g2):

        y_comb = torch.cat((x_g1, x_g2), dim=1)
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

def train(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0
    MSEloss = nn.MSELoss(reduction='mean')

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1,x_g1,h1_1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2,x_g2,h2_1 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        # add isomorphism logic
        if args.iso_logic:

            graphs1 = []
            graphs2 = []
            graph_iso_label = []
            edge_list1 = batch1.edge_index.cpu().numpy().tolist()  # all edges in a batch
            edge_list2 = batch2.edge_index.cpu().numpy().tolist()
            for i in range(batch1.num_graphs):
                node_list1 = torch.nonzero(batch1.batch == i).squeeze(1).cpu().numpy().tolist()
                node_list2 = torch.nonzero(batch2.batch == i).squeeze(1).cpu().numpy().tolist()

                graph1 = nx.Graph()
                graph1.add_nodes_from(node_list1)
                edges1 = []
                for node in node_list1:
                    index = torch.nonzero(batch1.edge_index[0] == node).squeeze(1).cpu().numpy().tolist()
                    edges1 += [(edge_list1[0][i], edge_list1[1][i]) for i in index]

                graph1.add_edges_from(edges1)
                for node in node_list1:
                    graph1.nodes[node]['label'] = str(graph1.degree(node))
                graphs1.append(graph1)

                graph2 = nx.Graph()
                graph2.add_nodes_from(node_list2)
                edges2 = []
                for node2 in node_list2:
                    index2 = torch.nonzero(batch2.edge_index[0] == node2).squeeze(1).cpu().numpy().tolist()
                    edges2 += [(edge_list2[0][i], edge_list2[1][i]) for i in index2]
                graph2.add_edges_from(edges2)
                for node2 in node_list2:
                    graph2.nodes[node2]['label'] = str(graph2.degree(node2))
                graphs2.append(graph2)


                node_labels1 = nx.get_node_attributes(graph1, 'label')
                node_labels2 = nx.get_node_attributes(graph2, 'label')
                if graph2.number_of_nodes() == 0:
                    graph_iso_label.append(0)
                    continue

                graph_iso_label.append(WL_simlar(node_labels1, node_labels2, graph1, graph2))

            graph_iso_label = torch.tensor(graph_iso_label).to(device)

            graph_iso_pre = model.iso_proj(x_g1,x_g2)
            graph_iso_pre = torch.squeeze(graph_iso_pre, 1)


            loss_iso = MSEloss(graph_iso_pre, graph_iso_label)
            loss += args.a * loss_iso

        if args.subiso_logic:

            weight = get_w(batch1)

            mse_loss = 0
            for i in range(batch1.num_graphs):

                node_list = torch.nonzero(batch1.batch == i).squeeze(1)
                node_embedding = h1_1[node_list[0]:node_list[-1] + 1]
                cos_sim = model.subiso(node_embedding)
                cos_sim = cos_sim / cos_sim.sum(1, keepdim=True)

                mse_loss += MSEloss(weight[i], cos_sim)
            loss_subiso = mse_loss / batch1.num_graphs

            loss += args.b * loss_subiso
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    parser.add_argument('--iso_logic', action='store_true')
    parser.add_argument('--a', type=int, default=0, help='the weight of iso loss.')
    parser.add_argument('--subiso_logic', action='store_true')
    parser.add_argument('--b', type=int, default=0, help='the weight of subiso loss.')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, dataset, optimizer)

        print(train_acc)
        print(train_loss)

        if epoch % 10 == 0:
            if args.iso_logic and args.subiso_logic:
                torch.save(gnn.state_dict(),"./models_graphcl/graphcl_iso" + str(args.a) + "_subiso" + str(args.b) + "_" + str(epoch) + ".pth")
            elif args.iso_logic:
                torch.save(gnn.state_dict(), "./models_graphcl/graphcl_iso" + str(args.a) + "_" + str(epoch) + ".pth")
            elif args.subiso_logic:
                torch.save(gnn.state_dict(),"./models_graphcl/graphcl_subiso" + str(args.b) + "_" + str(epoch) + ".pth")
            else:
                torch.save(gnn.state_dict(), "./models_graphcl/graphcl_" + str(epoch) + ".pth")
if __name__ == "__main__":
    main()
