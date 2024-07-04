import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import dgl

from collections import defaultdict
from utils import *
from dgl.nn import SAGEConv, GINConv, GATConv





class merger(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(merger, self).__init__()

        # 初始化多层感知机（MLP）的层
        self.layers = torch.nn.ModuleList()
        # 添加输入层到输出层的线性变换
        self.layers.append(torch.nn.Linear(in_dim, out_dim))
        # 如果需要多个隐藏层，可以取消下面几行的注释并调整参数
        # for _ in range(num_layers - 2):
        #     self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        # self.layers.append(torch.nn.Linear(hidden_dim, out_dim))

        # 设置dropout的概率
        self.dropout = dropout

    def reset_parameters(self):
        # 重置每层的参数
        for lin in self.layers:
            torch.nn.init.xavier_uniform_(lin.weight)  # 使用Xavier初始化方法初始化权重
            torch.nn.init.constant_(lin.bias, 0)       # 将偏置初始化为0

    def forward(self, emb_ori, emb_aug):
        # 将原始嵌入和增强嵌入在维度1上连接
        x = torch.cat((emb_ori, emb_aug), dim=1)
        # 通过MLP层
        for lin in self.layers[:-1]:
            x = lin(x)                  # 线性变换
            x = F.relu(x)               # ReLU激活函数
            x = F.dropout(x, p=self.dropout)  # Dropout正则化
        x = self.layers[-1](x)          # 输出层

        return x



class PPRGo(torch.nn.Module):
    def __init__(self, graph, emb_dim, dataset, topk, root, device):
        super().__init__()

        # 加载邻居和权重数据
        raw_nei = load_pickle(osp.join(root, dataset+"/"+dataset+"_nei.pkl"))
        raw_wei = load_pickle(osp.join(root, dataset+"/"+dataset+"_wei.pkl"))

        # 获取前 topk 个邻居和权重，并转换为张量
        self.nei = torch.LongTensor(raw_nei[:, :topk]).to(device)
        self.wei = torch.FloatTensor(raw_wei[:, :topk]).to(device)

        # 处理权重，确保权重和为1
        _w = torch.ones(self.nei.shape).to(device)
        _w[self.wei == 0] = 0
        self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)

        # 初始化节点嵌入表，嵌入维度为 emb_dim
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim).to(device)  # 使用正态分布初始化
        self.opt_param_list = []
        self.opt_param_list.extend(self.emb_table.parameters())
        self.device = device

    def forward(self):
        # 将邻居和权重移到设备上
        top_embs = self.emb_table.to(self.device)(self.nei.to(self.device))
        top_weights = self.wei.to(self.device)

        # 计算输出嵌入，按权重加权求和
        out_emb = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        
        return out_emb
    
    def parameters(self):
        return self.opt_param_list



class SAGE(torch.nn.Module):
    def __init__(self, graph, emb_dim, num_layers, device):
        super().__init__()
        # 将图数据移动到指定设备上（如GPU）
        self.graph = graph.to(device)
        # 初始化节点嵌入表，嵌入维度为 emb_dim
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim)  # 使用正态分布初始化

        # 初始化 SAGEConv 层
        self.convs = torch.nn.ModuleList()
        # 添加第一层 SAGEConv，使用 'mean' 聚合器和 ReLU 激活函数
        self.convs.append(SAGEConv(emb_dim, emb_dim, aggregator_type='mean', activation=F.relu))
        # 添加中间的 SAGEConv 层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(emb_dim, emb_dim, aggregator_type='mean', activation=F.relu))
        # 添加最后一层 SAGEConv，使用 'mean' 聚合器
        self.convs.append(SAGEConv(emb_dim, emb_dim, aggregator_type='mean'))
        # 将所有层移动到指定设备上
        self.convs = self.convs.to(device)

        # 保存设备信息
        self.device = device

    def forward(self):
        # 获取所有节点的索引
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        # 获取节点的嵌入向量
        x = self.emb_table(all_idx).to(self.device)
        # 依次通过每一层 SAGEConv
        for conv in self.convs[:-1]:
            x = conv(self.graph, x)  # 执行图卷积
            x = F.relu(x)  # 应用 ReLU 激活函数
        # 最后一层 SAGEConv，不适用激活函数
        emb = self.convs[-1](self.graph, x)
        return emb





class GAT(torch.nn.Module):
    def __init__(self, graph, emb_dim, num_layers, device):
        super().__init__()
        self.graph = dgl.add_self_loop(graph)
        self.graph = self.graph.to(device)
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim)  # initialized from N(0, 1)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(emb_dim, emb_dim, num_heads=4))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(emb_dim, emb_dim, num_heads=4))
        self.convs.append(GATConv(emb_dim, emb_dim, num_heads=4))
        self.convs = self.convs.to(device)

        self.device = device

    def forward(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        for conv in self.convs[:-1]:
            x = conv(self.graph, x)
            x = x.mean(dim=-2)
            x = F.relu(x)
        emb = self.convs[-1](self.graph, x)
        emb = emb.mean(dim=-2)
        return emb


class GIN(torch.nn.Module):
    def __init__(self, graph, emb_dim, num_layers, device):
        super().__init__()
        self.graph = graph.to(device)
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim)  # initialized from N(0, 1)

        self.convs = torch.nn.ModuleList()
        lin = torch.nn.Linear(emb_dim, 256)
        self.convs.append(GINConv(lin, 'max'))
        for i in range(num_layers - 2):
            lin = torch.nn.Linear(256, 256)
            self.convs.append(GINConv(i, 'max'))
        lin = torch.nn.Linear(256, emb_dim)
        self.convs.append(GINConv(lin, 'max'))
        self.convs = self.convs.to(device)

        self.device = device

    def forward(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        for conv in self.convs[:-1]:
            x = conv(self.graph, x)
            x = F.relu(x)
        emb = self.convs[-1](self.graph, x)
        return emb
        


class UltraGCN(nn.Module):

    def __init__(self, graph, config, dataset, device):
        super().__init__()
        self.device = device
        self.config = config
        self.graph = graph
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), config['emb_dim'])  # initialized from N(0, 1)

        constrain_mat_file = 'dataset/'+dataset+'/constrain_mat.pkl'
        topk_neighbors_file = 'dataset/'+dataset+'/ii_topk_neighbors.np.pkl'
        topk_similarity_file = 'dataset/'+dataset+'/ii_topk_similarity_scores.np.pkl'

        if self.config['lambda'] > 0:
            constrain_mat = load_pickle(constrain_mat_file)
            self.beta_uD = torch.FloatTensor(constrain_mat['beta_users']).to(self.device)
            self.beta_iD = torch.FloatTensor(constrain_mat['beta_items']).to(self.device)
            
        if self.config['gamma'] > 0:
            self.ii_topk_neighbors = load_pickle(topk_neighbors_file)
            self.ii_topk_similarity_scores = load_pickle(topk_similarity_file)
            
            topk = config['topk']
            self.ii_topk_neighbors = torch.LongTensor(self.ii_topk_neighbors[:, :topk]).to(self.device)
            self.ii_topk_similarity_scores = torch.FloatTensor(self.ii_topk_similarity_scores[:, :topk]).to(self.device)

    def get_embedding(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        return x

    def forward(self, batch_data):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)

        src, pos, neg = batch_data
        
        src_emb = x[src]
        pos_emb = x[pos]
        neg_emb = x[neg]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)

        _pos = pos
        _neg = neg
        
        if self.config['lambda'] > 0:
            beta_pos = self.beta_uD[src] * self.beta_iD[_pos]
            beta_neg = self.beta_uD[src].unsqueeze(1) * self.beta_iD[_neg]
            pos_coe = 1 + self.config['lambda'] * beta_pos 
            neg_coe = 1 + self.config['lambda'] * beta_neg
        else:
            pos_coe = None
            neg_coe = None
        
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, 
            torch.ones(pos_score.shape).to(self.device),
            weight=pos_coe
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, 
            torch.zeros(neg_score.shape).to(self.device),
            weight=neg_coe
        ).mean(dim = -1)
        
        loss_C_O = (pos_loss + self.config['neg_weight'] * neg_loss).sum()
        
        loss = loss_C_O
        
        # loss L_I
        if self.config['gamma'] > 0:
            ii_neighbors = self.ii_topk_neighbors[_pos]
            ii_scores = self.ii_topk_similarity_scores[_pos]

            _ii_neighbors = ii_neighbors
            ii_emb = x[_ii_neighbors]
            
            pos_ii_score = dot_product(src_emb, ii_emb)
            loss_I = -(ii_scores * pos_ii_score.sigmoid().log()).sum()
            
            loss += self.config['gamma'] * loss_I
        
        # L2 regularization loss
        if self.config['l2_reg_weight'] > 0:
            L2_reg_loss = 1/2 * ((src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum())
            if self.config['gamma'] > 0:
                L2_reg_loss += 1/2 * (ii_emb**2).sum()
            
            loss += self.config['l2_reg_weight'] * L2_reg_loss
    
        return loss


class LightGCN(nn.Module):

    def __init__(self, graph, emb_dim, num_layers, use_sparse_emb, load_from_feat, device):
        super().__init__()
        
        # 获取图的边
        E = graph.edges()
        
        # 计算每个节点的度（入度和出度的平均值）
        d1 = (graph.out_degrees(E[0]) + graph.in_degrees(E[0])) / 2.0
        d2 = (graph.out_degrees(E[1]) + graph.in_degrees(E[1])) / 2.0
        
        # 计算边的权重（度的倒数的平方根）
        edge_weights = (1 / (d1 * d2)).sqrt()
        
        # 创建稀疏邻接矩阵索引
        idx = torch.stack(E)
        num_nodes = graph.num_nodes()
        
        # 创建稀疏的邻接矩阵
        self.full_adj = torch.sparse_coo_tensor(
            idx, edge_weights, (num_nodes, num_nodes)
        ).to(device)

        # 根据参数决定是否从节点特征中加载初始嵌入
        if load_from_feat:
            self.emb_table = torch.nn.Embedding.from_pretrained(graph.ndata['feat'], freeze=False, sparse=use_sparse_emb)
        else:
            self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim, sparse=use_sparse_emb)  # 初始化为正态分布 N(0, 1)
        
        # 将嵌入表移动到指定设备
        self.nodes_embs = self.emb_table.to(device)

        # GCN层数
        self.num_gcn_layer = num_layers

        # 优化器参数列表
        self.opt_param_list = []
        self.opt_param_list.extend(self.emb_table.parameters())

    def forward(self):
        # 获取节点嵌入权重
        X = self.emb_table.weight

        # 逐层进行GCN计算
        for _ in range(self.num_gcn_layer):
            X = torch.sparse.mm(self.full_adj, X)
        
        # 返回最后一层的节点嵌入
        return X

    def parameters(self):
        # 返回需要优化的参数列表
        return self.opt_param_list



def load_model(graph, model, dataset, device):
    if model == 'pprgo':
        config_file = 'config/pprgo-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))

        model = PPRGo(graph, emb_dim=config['emb_dim'], dataset=dataset, topk=config['topk'], root='dataset/', device=device)

    elif model == 'lightgcn':
        config_file = 'config/lightgcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        load_from_feat = False
        num_layers = config['num_layers']
        model = LightGCN(graph, config['emb_dim'], num_layers, config['use_sparse_emb'], load_from_feat, device)

    elif model == 'ultragcn':
        config_file = 'config/ultragcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))

        model = UltraGCN(graph, config, dataset, device)

    elif model == 'gin':
        config_file = 'config/gin-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = GIN(graph, config['emb_dim'], config['num_layers'], device)

    elif model == 'gat':
        config_file = 'config/gat-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = GAT(graph, config['emb_dim'], config['num_layers'], device)

    elif model == 'sage':
        config_file = 'config/sage-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        if dataset == 'epinions':
            model = SAGE(graph, config['emb_dim'], config['num_layers'], device)
        else:
            model = SAGE(graph, config['emb_dim'], config['num_layers'], device)
    
    return model.to(device)