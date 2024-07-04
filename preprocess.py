import dgl
import torch
import numpy as np
import community

from utils import *
from augmentation import pseudo_generation, random_walk_generation, random_generation


def load_data(args):
    assert args.dataset in ['dblp', 'amazon', 'epinions', 'lastfm', 'cora']

    if args.dataset != 'lastfm':
        graph = load_pickle('dataset/'+args.dataset+'/graph.pkl')
        ui_graph = None
        split_edge = load_pickle('dataset/'+args.dataset+'/split_edge.pkl')
    else:
        assert args.dataset_type == 'recommendation'
        graph = load_pickle('dataset/'+args.dataset+'/item_graph.pkl')
        ui_graph = load_pickle('dataset/'+args.dataset+'/ui_graph.pkl')
        split_edge = load_pickle('dataset/'+args.dataset+'/split_edge.pkl')

    return graph, ui_graph, split_edge


def louvain_partition(graph):
    G = dgl.to_networkx(graph).to_undirected()

    print(">> Louvain Clustering...")
    partition = community.best_partition(G)
    membership = []
    for node in partition.keys():
        membership.append(partition[node])

    membership = np.array(membership)
    num_cluster = np.max(membership) + 1
    print(">> Louvain Clustering Finished, {:d} communities detected.".format(num_cluster))

    return membership


def metis_partition(graph, num_parts=50):

    print(">> Metis Clustering...")
    
    dgl.distributed.partition.partition_graph(graph, 
                                'dataset', 
                                num_parts,       
                                out_path="metis/output/", reshuffle=False,
                                balance_ntypes=None,
                                balance_edges=True)
                                
    membership = np.load("metis/output/node_map.npy")

    print(">> Metis Clustering Finished, {:d} communities detected.".format(num_parts))

    return membership



def edge_split(graph, split_edge, augmentation, args):
    # 根据参数决定是否加载预先检测的社区划分
    if args.load_partition == 1:  # 加载预先检测的社区划分
        membership = load_pickle('dataset/' + args.dataset + '/louvain_' + args.dataset + '.pkl')
    elif args.load_partition == 2:  # 加载metis社区划分
        membership = load_pickle('dataset/' + args.dataset + '/metis_' + args.dataset + '.pkl')
    elif args.load_partition == -1:  # 使用Louvain方法进行社区划分并保存
        membership = torch.tensor(louvain_partition(graph))
        save_pickle('dataset/' + args.dataset + '/louvain_' + args.dataset + '.pkl', membership)
    else:  # 使用Metis方法进行社区划分并保存
        membership = torch.tensor(metis_partition(graph))
        save_pickle('dataset/' + args.dataset + '/metis_' + args.dataset + '.pkl', membership)

    # 生成增强的监督集
    if augmentation:
        if args.aug_type == 'jaccard':
            graph, split_edge = pseudo_generation(graph, membership, split_edge, add_edge=args.add_edge, aug_size=args.aug_size,true_aug=0)
        elif args.aug_type == 'random':
            graph, split_edge = random_generation(graph, membership, split_edge, add_edge=args.add_edge, aug_size=args.aug_size)
        else:
            graph, split_edge = random_walk_generation(graph, membership, split_edge, aug_size=args.aug_size)

    # 初始化边的类型（0表示跨社区边，1表示社区内边）
    graph.edata['type'] = torch.zeros(graph.num_edges())
    # 根据社区划分设置边的类型
    graph.edata['type'][membership[graph.edges()[0]] == membership[graph.edges()[1]]] = 1

    # 初始化数据字典，用于存储训练集、验证集和测试集的边
    data = {
        'train': [[] for i in range(2)],
        'valid': [[] for i in range(2)],
        'test': [[] for i in range(2)]
    }

    if args.dataset_type == 'social':
        # 分割训练集边
        train_src_dst = split_edge['train']['edge']
        data['train'][0] = train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]]
        data['train'][1] = train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]]
        print('>> Train | intra-edge: {:d}, inter-edge: {:d}'.format(len(data['train'][0]), len(data['train'][1])))

        # 分割验证集边
        valid_src_dst = split_edge['valid']['edge']
        data['valid'][0] = valid_src_dst[membership[valid_src_dst.t()[0]] == membership[valid_src_dst.t()[1]]]
        data['valid'][1] = valid_src_dst[membership[valid_src_dst.t()[0]] != membership[valid_src_dst.t()[1]]]
        print('>> Valid | intra-edge: {:d}, inter-edge: {:d}'.format(len(data['valid'][0]), len(data['valid'][1])))

        # 分割测试集边
        test_src_dst = split_edge['test']['edge']
        data['test'][0] = test_src_dst[membership[test_src_dst.t()[0]] == membership[test_src_dst.t()[1]]]
        data['test'][1] = test_src_dst[membership[test_src_dst.t()[0]] != membership[test_src_dst.t()[1]]]
        print('>> Test  | intra-edge: {:d}, inter-edge: {:d}'.format(len(data['test'][0]), len(data['test'][1])))

        # 将社区划分信息添加到图节点数据中
        graph.ndata['membership'] = membership
        # 初始化训练集边的置信度
        confidence = torch.ones(len(split_edge['train']['edge']))

    elif args.dataset_type == 'recommendation':
        # 加载用户社区划分信息
        user_membership = load_pickle('dataset/' + args.dataset + '/user_membership.pkl')

        # 分割验证集边
        valid_src_dst = split_edge['valid']['edge']
        data['valid'][0] = valid_src_dst[user_membership[valid_src_dst.t()[0]] == membership[valid_src_dst.t()[1] - 1892]]
        data['valid'][1] = valid_src_dst[user_membership[valid_src_dst.t()[0]] != membership[valid_src_dst.t()[1] - 1892]]
        print('>> Valid | intra-edge: {:d}, inter-edge: {:d}'.format(len(data['valid'][0]), len(data['valid'][1])))

        # 分割测试集边
        test_src_dst = split_edge['test']['edge']
        data['test'][0] = test_src_dst[user_membership[test_src_dst.t()[0]] == membership[test_src_dst.t()[1] - 1892]]
        data['test'][1] = test_src_dst[user_membership[test_src_dst.t()[0]] != membership[test_src_dst.t()[1] - 1892]]
        print('>> Test  | intra-edge: {:d}, inter-edge: {:d}'.format(len(data['test'][0]), len(data['test'][1])))

        # 将社区划分信息添加到图节点数据中
        graph.ndata['membership'] = membership
        # 初始化训练集边的置信度
        confidence = torch.ones(len(split_edge['train']['edge']))

    return data, split_edge, graph, membership, confidence
