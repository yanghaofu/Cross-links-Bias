import dgl
import torch
import networkx as nx
import time
from tqdm import tqdm



def random_walk_generation(graph, membership, split_edge, aug_size):
    '''
    某条边多次出现在随机游走中，它的置信度会更高。
    '''
    # 随机选择一半的节点作为种子节点
    seed_nodes = torch.randperm(graph.num_nodes())[:int(0.5*graph.num_nodes())]

    # 设置随机游走的长度
    walk_length = 10
    # 执行随机游走，生成节点序列
    walks = dgl.sampling.random_walk(graph, seed_nodes, length=walk_length)[0].tolist()

    src, dst = list(), list()
    half_win_size = 15

    # 遍历每一个随机游走序列
    for walk in tqdm(walks, desc='Num walks'):
        l = len(walk)
        for i in range(l):
            real_win_size = half_win_size
            left = i - real_win_size
            if left < 0:
                left = 0
            right = i + real_win_size
            if right >= l:
                right = l - 1
            # 遍历窗口内的节点对
            for j in range(left, right + 1):
                if walk[i] == walk[j]:
                    continue
                elif walk[i] < 0 or walk[j] < 0:
                    continue
                else:
                    if membership[walk[i]] == membership[walk[j]]: 
                        continue
                    else:
                        src.append(walk[i])
                        dst.append(walk[j])

    # 创建虚拟图，存储伪边
    virtual_graph = dgl.graph((torch.tensor(src), torch.tensor(dst)))
    print(len(src))
    virtual_graph.edata['confidence'] = torch.ones(virtual_graph.num_edges())

    # 增加每条边的置信度
    virtual_graph.edata['confidence'][virtual_graph.edge_ids(src, dst)] += 1

    # 获取训练集的边
    train_src_dst = split_edge['train']['edge']

    # 计算社区内边和社区间边的数量
    intra_size = len(train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]])
    inter_size = len(train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]])

    # 计算增强边的数量
    augment_size = int((intra_size - inter_size) * aug_size)
    print('>> 生成 {:d} 条伪边。'.format(augment_size))

    # 根据置信度选择增强边
    pseudo_edges = torch.stack(virtual_graph.edges()).t()[virtual_graph.edata['confidence'].topk(augment_size)[1]]

    # 将增强边添加到训练集边集中
    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], pseudo_edges), dim=0)

    return graph, split_edge




def pseudo_generation(graph, membership, split_edge, add_edge, aug_size, true_aug=1):
    # 初始化图边的数据标签为零
    graph.edata['pseudo_tag'] = torch.zeros(graph.num_edges())

    # 将DGL图转换为NetworkX图
    nx_graph = nx.Graph(dgl.to_networkx(graph))

    # 生成2跳邻居图
    k_hop_graph = dgl.khop_graph(graph, 2)
    # 移除自环
    k_hop_graph = dgl.remove_self_loop(k_hop_graph)

    # 选择跨社区的边
    k_hop_edges = k_hop_graph.edges()
    if true_aug:
        print(">> 真正的增强操作。")
        # 选择社区之间的边
        all_pseudo_edges = torch.stack(k_hop_edges)[:, membership[k_hop_edges[0]] != membership[k_hop_edges[1]]]
    else:
        print(">> 虚假的增强操作。")  # 仅用于消融实验
        # 选择前10000000条边
        all_pseudo_edges = torch.stack(k_hop_edges)[:10000000]

    # 获取训练集的边
    train_src_dst = split_edge['train']['edge']

    # 计算社区内边和社区间边的数量
    intra_size = len(train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]])
    inter_size = len(train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]])

    # 计算增强边的数量
    augment_size = int((intra_size - inter_size) * aug_size)
    print('>> 生成 {:d} 条伪边。'.format(augment_size))

    # 固定随机种子
    torch.manual_seed(0)
    # 随机选择增强边
    pseudo_edges = all_pseudo_edges[:, torch.randperm(len(all_pseudo_edges[0]))[:augment_size*2]]

    # 初始化边的强度
    strength = torch.ones(len(pseudo_edges[0]))

    # 计算jaccard系数作为边的强度
    print('>> 开始预处理...')
    start_time = time.time()
    strength = nx.jaccard_coefficient(nx_graph, list(map(lambda x: tuple(x), pseudo_edges.t().tolist())))
    strength = torch.tensor(list(map(lambda x: x[-1], strength)))
    print('>> 预处理完成。耗时 {:.2f} 秒。'.format(time.time() - start_time))
    
    if true_aug:
        # 选择最强的增强边
        top_pseudo_edges = pseudo_edges[:, strength.topk(augment_size, dim=0)[1]]  # 共同邻居
    else:
        # 选择前augment_size条增强边
        top_pseudo_edges = pseudo_edges[:, :augment_size]

    if add_edge:
        # 将伪边添加到原图中
        new_graph = dgl.add_edges(graph, torch.cat((top_pseudo_edges[0], top_pseudo_edges[1]), dim=0), torch.cat((top_pseudo_edges[1], top_pseudo_edges[0]), dim=0), {'pseudo_tag': torch.ones(2*len(top_pseudo_edges[0]))})
    else:
        new_graph = graph

    # 将伪边添加到训练集中
    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], top_pseudo_edges.t()), dim=0)

    return new_graph, split_edge


def random_generation(graph, membership, split_edge, add_edge, aug_size, true_aug=1):
    # 初始化图边的数据标签为零
    graph.edata['pseudo_tag'] = torch.zeros(graph.num_edges())

    # 将DGL图转换为NetworkX图
    nx_graph = nx.Graph(dgl.to_networkx(graph))

    # 生成2跳邻居图
    k_hop_graph = dgl.khop_graph(graph, 2)
    # 移除自环
    k_hop_graph = dgl.remove_self_loop(k_hop_graph)

    # 选择跨社区的边
    k_hop_edges = k_hop_graph.edges()
    if true_aug:
        print(">> 真正的增强操作。")
        # 选择社区之间的边
        all_pseudo_edges = torch.stack(k_hop_edges)[:, membership[k_hop_edges[0]] != membership[k_hop_edges[1]]]
    else:
        print(">> 虚假的增强操作。")  # 仅用于消融实验
        # 选择前10000000条边
        all_pseudo_edges = torch.stack(k_hop_edges)[:10000000]

    # 获取训练集的边
    train_src_dst = split_edge['train']['edge']

    # 计算社区内边和社区间边的数量
    intra_size = len(train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]])
    inter_size = len(train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]])

    # 计算增强边的数量
    augment_size = int((intra_size - inter_size) * aug_size)
    print('>> 生成 {:d} 条伪边。'.format(augment_size))

    # 固定随机种子
    torch.manual_seed(0)
    # 随机选择增强边
    pseudo_edges = all_pseudo_edges[:, torch.randperm(len(all_pseudo_edges[0]))[:augment_size*2]]

    # 初始化边的强度
    strength = torch.ones(len(pseudo_edges[0]))

    # 计算jaccard系数作为边的强度
    print('>> 开始预处理...')
    start_time = time.time()
    strength = nx.jaccard_coefficient(nx_graph, list(map(lambda x: tuple(x), pseudo_edges.t().tolist())))
    strength = torch.tensor(list(map(lambda x: x[-1], strength)))
    print('>> 预处理完成。耗时 {:.2f} 秒。'.format(time.time() - start_time))
    
    # 选择前augment_size条增强边
    top_pseudo_edges = pseudo_edges[:, :augment_size]

    if add_edge:
        # 添加增强边到图中
        new_graph = dgl.add_edges(graph, torch.cat((top_pseudo_edges[0], top_pseudo_edges[1]), dim=0), torch.cat((top_pseudo_edges[1], top_pseudo_edges[0]), dim=0), {'pseudo_tag': torch.ones(2*len(top_pseudo_edges[0]))})
    else:
        # 保持原图不变
        new_graph = graph

    # 将增强边添加到训练集边集中
    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], top_pseudo_edges.t()), dim=0)

    return new_graph, split_edge
