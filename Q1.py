from util import *
import matplotlib.pyplot as plt
from community import best_partition, modularity
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
import pdb
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import LFR_benchmark_graph


def draw_graph(partition, algorithm=None):
    color_list = []
    sorted_keys = sorted(partition.keys(), key=lambda s: int(s))
    count = 1
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in sorted_keys if partition[nodes] == com]
        color_list.extend([count] * len(list_nodes))
        count += 1
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color=color_list)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.title("community distribution of {} dataset given by {} algorithm".format(args.data, algorithm))
    plt.savefig("figs/{}_{}.png".format(args.data, algorithm))
    plt.clf()

    ordered_pred = [partition[nodes] for nodes in sorted_keys]
    with open("result/{}_{}.txt".format(args.data, algorithm), "w") as f:
        f.write("NMI of {} dataset given by {} algorithm: {:.3f}\n".format(
            args.data, algorithm, normalized_mutual_info_score(labels, ordered_pred)))
        f.write("ARI of {} dataset given by {} algorithm: {:.3f}\n".format(
            args.data, algorithm, adjusted_rand_score(labels, ordered_pred)))
        f.write("Modularity of {} dataset given by {} algorithm: {:.3f}\n".format(
            args.data, algorithm, modularity(partition, graph)))


def draw_original_community(nodes, labels):
    color_list = []

    count = 1
    for com in set(labels):

        list_nodes = [i for i in range(len(nodes)) if labels[i] == com]
        color_list.extend([count] * len(list_nodes))
        count += 1

    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color=color_list)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.title("original community distribution of {} dataset".format(args.data))
    plt.savefig("figs/{}_original.png".format(args.data))
    plt.clf()


def louvain_partition(graph):
    partition = best_partition(graph)
    draw_graph(partition, "Louvain")


def greedy_partition(graph):
    partition = greedy_modularity_communities(graph)
    res = dict()
    for i, part in enumerate(partition):
        for j in part:
            res[j] = i
    draw_graph(res, "greedy_modularity")


def girvan_newman_partition(graph):
    partition = girvan_newman(graph)
    res = dict()
    for i, part in enumerate(partition):
        for j in part:
            res[str(j)] = i
    draw_graph(res, "Girvan Newman")


def load_labels(graph):
    sorted_nodes = sorted(list(graph.nodes), key=lambda s: int(s))
    global labels

    if args.data in q1_data:
        if args.data == 'karate':
            clubs = list(set([graph.nodes[x]['club'] for x in sorted_nodes]))
            labels = [clubs.index(graph.nodes[x]['club']) for x in sorted_nodes]

        else:
            labels = [graph.nodes[x]['value'] for x in sorted_nodes]

    elif args.data == 'LFR':
        communities = {frozenset(graph.nodes[v]['community']) for v in graph}
        res = np.zeros(len(sorted_nodes)).astype(int)
        for i, part in enumerate(communities):
            for j in part:
                res[j] = i
        labels = res.tolist()

    else:
        if args.data == 'citeseer':
            nonzeros = labels.nonzero()
            max_label = np.max(nonzeros[1])
            diff = len(labels) - len(nonzeros[1])
            res = np.full(labels.shape[0], -1)
            res[nonzeros[0]] = nonzeros[1]
            iso_mask = np.where(res == -1)
            res[iso_mask] = np.arange(max_label + 1, max_label + 1 + diff)
            labels = res.tolist()
        else:
            labels = labels.nonzero()[1].tolist()
    draw_original_community(sorted_nodes, labels)
    return labels


if __name__ == '__main__':
    args = get_args()

    if args.data in q1_data:
        if args.data == 'karate':
            graph = nx.karate_club_graph()
        else:
            graph = nx.read_gml("real-classic/{}.gml".format(args.data), label='id').to_undirected()
        pdb.set_trace()
    elif args.data == 'LFR':
        graph = LFR_benchmark_graph(200, 2.5, 1.5, 0.1, min_community=10, min_degree=5, seed=10)
    else:
        graph, labels = load_data_gcn(args.data)

    pos = nx.spring_layout(graph)
    labels = load_labels(graph)

    lou_partition = louvain_partition(graph)
    gre_partition = greedy_partition(graph)

