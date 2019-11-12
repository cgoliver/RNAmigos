'''
Functions to take a nx object and return (nx, dict_tree, dict_rings)
'''

import networkx as nx
import os, sys
import pickle
import time
from collections import defaultdict, Counter
import multiprocessing as mlt
from tqdm import tqdm


def node_tree(G, v, depth=5):
    """
        Return neighbourhood tree rooted at `v` up to depth `depth`.

        Each key is a node, value is a dictionary where
        each key is a neighbor and its value is the edge label on the edge.

        For now just returning unique visits.

        {'root':  {'nei1': 'l1', 'nei_2':'l2'},
         'nei_1': {'nei_3': 'l3'},
         'nei_2:  {'nei_4: 'l4'}'}

         TODO: index tree by level.
    """
    G = G.to_undirected()
    qu = [v]
    tree = {}
    visits = defaultdict(int)
    # nx.draw_networkx(G)
    while qu:
        current = qu.pop()

        tree[current] = {}

        visits[current] = 1

        for nei in G.neighbors(current):
            if visits[nei] == 0:
                qu = [nei] + qu
                tree[current][nei] = G[current][nei]['label']

    # plt.show()
    '''
    Alternate way to include depth
    G = G.to_undirected()
    qu = [(v, 0)]
    tree = {}
    visits = defaultdict(int)
    nx.draw_networkx(G)
    while qu:
        current, depth = qu.pop()
        tree[current] = {}

        visits[current] = 1

        for nei in G.neighbors(current):
            if visits[nei] == 0:
                qu = [(nei, depth + 1)] + qu
                tree[current][nei] = G[current][nei]['label'], depth + 1

    for k, v in tree.items():
        print(k, v)
    print(tree)
    plt.show()
    pass
    '''
    return tree


def node_2_unordered_rings(G, v, depth=5, unique=True):
    """
    Return rings centered at `v` up to depth `depth`.

        Each key is a node, value is a dictionary where
        each key is a neighbor and its value is the edge label on the edge.

        For now just returning unique visits.

        {'root':  {'nei1': 'l1', 'nei_2':'l2'},
         'nei_1': {'nei_3': 'l3'},
         'nei_2:  {'nei_4: 'l4'}'}

    :param G:
    :param v:
    :param depth:
    :param unique:
    :return:
    """
    G = G.to_undirected()
    node_rings = [[v]]
    edge_rings = [[None]]
    visited = set()
    visited.add(v)
    for k in range(depth):
        ring_k = []
        edge_ring_k = []
        # print('##############################################')
        # print('ring K', node_rings[k])
        for node in node_rings[k]:
            children = []
            e_labels = []

            # print('node', node)
            # print(visited)
            for nei in G.neighbors(node):
                if nei not in visited:
                    if unique:
                        visited.add(nei)
                        # print(unique)
                    children.append(nei)
                    e_labels.append(G[node][nei]['label'])
            ring_k.extend(children)
            edge_ring_k.extend(e_labels)
        node_rings.append(ring_k)
        edge_rings.append(edge_ring_k)
    return node_rings, edge_rings


def build_dict_tree(graph, depth=5):
    """
    :param graph: nx
    :return: dict (node_id : tree)
    """
    dict_tree = {}
    for node in graph.nodes():
        dict_tree[node] = node_tree(graph, node, depth=depth)

    return dict_tree


def build_ring_tree_from_graph(graph, depth=5, unique=True):
    """

    :param graph: nx
    :return: dict (node_id : ring)
    """
    dict_ring = {}
    for node in graph.nodes():
        dict_ring[node] = node_2_unordered_rings(graph, node, depth=depth, unique=unique)

    return dict_ring


def build_ring_tree_from_tree(dict_tree):
    """

    :param graph: nx
    :return: dict (node_id : tree)
    """

    return 0


def annotate(graph, depth=5):
    trees = build_dict_tree(graph)
    rings = build_ring_tree_from_graph(graph, depth=depth)
    return trees, rings


def annotate_one(args):
    """
    To be called by map
    :param args: ( g (name of the graph),
    :return:
    """
    g, graph_path, dump_path, fp = args
    try:
        dump_name = os.path.basename(g).split('.')[0] + "_annot.p"
        dump_full = os.path.join(dump_path, dump_name)
        for processed in os.listdir(dump_path):
            if processed.startswith(dump_name):
                return 0, 0
        graph = nx.read_gpickle(os.path.join(graph_path, g))
        # start = time.perf_counter()
        # print(f"{time.perf_counter() - start}")
        annots = tuple(list(annotate(graph)) + [fp])
        pickle.dump((graph, *annots),
                    open(dump_full, 'wb'))
        return 0, g
    except Exception as e:
        print(e)
        return 1, g


def annotate_all(fp_file="../data/all_ligs_maccs.p", dump_path='../data/annotated/sample', graph_path='../data/chunks_nx', parallel=True):
    """
    Routine for all files in a folder
    :param dump_path: 
    :param graph_path: 
    :param parallel:
    :return: 
    """
    graphs = os.listdir(graph_path)
    failed = 0
    fp_dict = pickle.load(open(fp_file,'rb'))
    lig_name = lambda x: x.split(":")[2]
    pool = mlt.Pool()

    if parallel:
        arguments = [(g, graph_path, dump_path, fp_dict[lig_name(g)]) for g in graphs]
        for res in tqdm(pool.imap_unordered(annotate_one, arguments), total=len(graphs)):
            if res[0]:
                failed += 1
                print(f'failed on {res[1]}, this is the {failed}-th one on {len(graphs)}')
        print(f'failed on {(failed)} on {len(graphs)}')
        return failed
    for graph in tqdm(graphs, total=len(graphs)):
        res = annotate_one((graph, graph_path, dump_path, fp_dict[lig_name(graph)]))
        if res[0]:
            failed += 1
            print(f'failed on {graph}, this is the {failed}-th one on {len(graphs)}')
    pass


if __name__ == '__main__':
    annotate_all(dump_path='../data/annotated/pockets_nx', graph_path='../data/pockets_nx')
