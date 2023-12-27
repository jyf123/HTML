import networkx as nx
import matplotlib.pyplot as plt
from builtins import str


def draw_graphs(G, G1):
    pos = nx.spring_layout(G)
    pos1 = nx.spring_layout(G1)
    plt.subplot(121)
    nx.draw(G, pos, with_labels=True, node_color='pink', node_size=500)
    plt.subplot(122)
    nx.draw(G1, pos1, with_labels=True, node_size=500)
    plt.show()


def multiset_define(G):
    multiset = {}
    lab = nx.get_node_attributes(G, 'label')
    for u in G.nodes():
        list = []
        a = nx.all_neighbors(G, u)
        for v in a:
            list.append(lab[v])
        list.sort()
        list_new = [str(x) for x in list]
        multiset[u] = ''.join(list_new)
    return multiset


def multiset_join(G, dict):
    node_label = nx.get_node_attributes(G, 'label')
    for u in dict.keys():
        dict[u] = node_label[u] + dict[u]
    return dict


def label_compression(dict1, dict2, G, G1):
    node_label = nx.get_node_attributes(G, 'label')
    node_label1 = nx.get_node_attributes(G1, 'label')

    result = map(int, node_label.values())
    result1 = map(int, node_label1.values())

    n = max(max(result), max(result1)) + 1
    strings = []

    for v in dict1.values():
        strings.append(v)
    for u in dict2.values():
        strings.append(u)
    str_remove = []
    for i in strings:
        if i not in str_remove:
            str_remove.append(i)
    str_remove.sort()
    map_dict = {}
    for m in range(len(str_remove)):
        map_dict[str_remove[m]] = str(n)
        n += 1
    return map_dict


def label_update(dic, map_dict):
    for u in dic.keys():
        for v in map_dict.keys():
            if (dic[u] == v):
                dic[u] = map_dict[v]
    return dic


def WL(dict1, dict2, G, G1):

    i = 1
    list1 = dict1
    list2 = dict2
    result = []
    result1 = []

    while i <= 3:
        list1 = multiset_define(G)
        list2 = multiset_define(G1)
        dict1 = multiset_join(G, list1)
        dict2 = multiset_join(G1, list2)
        map_dict = label_compression(dict1, dict2, G, G1)
        list1 = label_update(dict1, map_dict)
        list2 = label_update(dict2, map_dict)
        nx.set_node_attributes(G, list1, 'label')
        nx.set_node_attributes(G1, list2, 'label')
        result = sorted(map(int, list1.values()))
        result1 = sorted(map(int, list2.values()))
        i += 1
    return result, result1

def WL_test(dict1, dict2, G, G1):
    rG,rG1 = WL(dict1,dict2,G,G1)
    return rG==rG1

def WL_simlar(dict1, dict2, G, G1):
    rG, rG1 = WL(dict1, dict2, G, G1)

    j_num = 0

    for j in range(len(rG)):
        if j>0 and rG[j]==rG[j-1]:
            continue
        if rG[j] in rG1:
            j_num+=min(rG.count(rG[j]),rG1.count(rG[j]))
    sim = round(j_num/(len(rG)+len(rG1)-j_num),2)
    return sim

