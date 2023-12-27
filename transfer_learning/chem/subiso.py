import numpy as np
import datetime
import torch
import networkx as nx

def read_graph_nodes_relations(data):
    graph_ids = data.batch.cpu().numpy().tolist()
    nodes, graphs = {}, {}
    for node_id, graph_id in enumerate(graph_ids):
        if graph_id not in graphs:
            graphs[graph_id] = []
        graphs[graph_id].append(node_id)
        nodes[node_id] = graph_id
    for graph_id in graphs:
        graphs[graph_id] = np.array(graphs[graph_id])
    return nodes, graphs

def read_graph_adj(data, nodes, graphs):
    edges = data.edge_index.cpu().numpy().tolist()
    adj_dict = {}
    for i in range(len(edges[0])):
        node1 = edges[0][i]
        node2 = edges[1][i]
        graph_id = nodes[node1]
        assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
        if graph_id not in adj_dict:
            n = len(graphs[graph_id])
            adj_dict[graph_id] = np.zeros((n, n))
        ind1 = np.where(graphs[graph_id] == node1)[0]
        ind2 = np.where(graphs[graph_id] == node2)[0]
        assert len(ind1) == len(ind2) == 1, (ind1, ind2)
        adj_dict[graph_id][ind1, ind2] = 1

    adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

    return adj_list
def get_w_old(A_array):
    G = nx.from_numpy_matrix(A_array)

    sub_graphs = []
    subgraph_nodes_list = []
    sub_graphs_adj = []
    sub_graph_edges = []
    new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

    for i in np.arange(len(A_array)):
        s_indexes = []
        for j in np.arange(len(A_array)):
            s_indexes.append(i)
            if (A_array[i][j] == 1):
                s_indexes.append(j)
        sub_graphs.append(G.subgraph(s_indexes))

    for i in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[i].nodes))

    for index in np.arange(len(sub_graphs)):
        sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

    for index in np.arange(len(sub_graphs)):
        sub_graph_edges.append(sub_graphs[index].number_of_edges())

    for node in np.arange(len(subgraph_nodes_list)):
        sub_adj = sub_graphs_adj[node]
        for neighbors in np.arange(len(subgraph_nodes_list[node])):
            index = subgraph_nodes_list[node][neighbors]
            count = torch.tensor(0).float()
            if (index == node):
                continue
            else:
                c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                if index in c_neighbors:
                    nodes_list = subgraph_nodes_list[node]
                    sub_graph_index = nodes_list.index(index)
                    c_neighbors_list = list(c_neighbors)
                    for i, item1 in enumerate(nodes_list):
                        if (item1 in c_neighbors):
                            for item2 in c_neighbors_list:
                                j = nodes_list.index(item2)
                                count += sub_adj[i][j]

                new_adj[node][index] = count / 2
                new_adj[node][index] = new_adj[node][index] / (
                        len(c_neighbors) * (len(c_neighbors) - 1))
                new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

    weight = torch.FloatTensor(new_adj)
    weight = weight / weight.sum(1, keepdim=True)
    weight = torch.where(torch.isnan(weight), torch.full_like(weight, 0), weight)
    return weight

def get_w(data):
    nodes, graphs = read_graph_nodes_relations(data)
    adj_lists = read_graph_adj(data, nodes, graphs)
    dataset_length = len(adj_lists)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for itr in np.arange(dataset_length):
        adj_list = adj_lists[itr]
        n = len(adj_list)
        if n<1000:
            adj_list = torch.FloatTensor(adj_list).to(device)
        else:
            adj_list[range(n), range(n)]=0
            weight = get_w_old(adj_list)
            weight = weight.to(device)
            adj_lists[itr] = weight
            continue

        adj_list = adj_list.fill_diagonal_(fill_value=1)
        node_count = torch.matmul(adj_list,adj_list.transpose(0,1))
        node_count = node_count.fill_diagonal_(fill_value=0)

        subgraph_list = adj_list[:,None,:]*adj_list[:,:,None]
        subgraph_list = subgraph_list*adj_list[None,:,:]

        for i in range(len(subgraph_list)):
            subgraph_list[i] = subgraph_list[i].fill_diagonal_(fill_value=0)

        subgraph_list = subgraph_list.reshape(len(subgraph_list),-1)
        subgraph_list = torch.matmul(subgraph_list,subgraph_list.transpose(0,1))
        subgraph_list = subgraph_list.fill_diagonal_(fill_value=0)

        weight = subgraph_list / 2
        weight = weight / (
                node_count * (node_count - 1))
        weight = torch.where(torch.isnan(weight), torch.full_like(weight, 0), weight)
        weight = weight * (node_count ** 2)

        weight = weight / weight.sum(1, keepdim=True)
        weight = torch.where(torch.isnan(weight), torch.full_like(weight, 0), weight)

        adj_lists[itr] = weight
    return adj_lists


