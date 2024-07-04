import torch
import argparse
import logging

from copy import deepcopy
from preprocess import *
from model import load_model, merger

import math
import numpy as np
from torch.utils.data import DataLoader
from utils import evaluate
from tqdm import tqdm


def e2e_train(graph, model, dataloader, args, device):
    # 将图数据移动到指定设备上（如GPU）
    graph = graph.to(device)

    # 在图上进行训练
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for data in dataloader: 
        # 获取嵌入
        embedding = model()
        # 正样本边
        pos_edge = data[0].t()
        pos_out = torch.sigmoid((torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)).unsqueeze(-1))
        # 计算正样本损失
        pos_loss = (-torch.log(pos_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

        # 随机采样负样本边
        neg_edge = torch.randint(0, graph.num_nodes(), pos_edge.size(), dtype=torch.long, device=embedding.device)
        neg_out = torch.sigmoid(torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1).unsqueeze(-1))
        # 计算负样本损失
        neg_loss = (-torch.log(1 - neg_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

        # 总损失
        loss = pos_loss + neg_loss

        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

def e2e_test(model, data, split_edge, args, device, ui_graph=None):
    # 设置模型为评估模式
    model.eval()

    # 获取模型的嵌入
    embedding = model().detach()

    # 处理社交网络类型数据集
    if args.dataset_type == 'social':
        # 将负样本边数据移动到指定设备上（如GPU）
        neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
        neg_test_edge = split_edge['test']['edge_neg'].to(device)

        # 初始化结果字典
        results = {}

        for i in range(2):
            # 验证集正样本预测
            pos_preds = []
            for perm in DataLoader(range(data['valid'][i].size(0)), args.batch_size):
                edge = data['valid'][i][perm].t()
                # 计算正样本边的预测值
                pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            # 合并所有批次的预测结果
            pos_valid_pred = torch.cat(pos_preds, dim=0)

            # 验证集负样本预测
            neg_preds = []
            for perm in DataLoader(range(neg_valid_edge.size(0)), args.batch_size):
                edge = neg_valid_edge[perm].t()
                # 计算负样本边的预测值
                neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            # 合并所有批次的预测结果
            neg_valid_pred = torch.cat(neg_preds, dim=0)

            # 计算验证集的 hit@50 和 hit@100 指标
            results['valid_group_' + str(i) + '_hit@50'] = evaluate(pos_valid_pred, neg_valid_pred)[0]
            results['valid_group_' + str(i) + '_hit@100'] = evaluate(pos_valid_pred, neg_valid_pred)[1]

            # 测试集正样本预测
            pos_preds = []
            for perm in DataLoader(range(data['test'][i].size(0)), args.batch_size):
                edge = data['test'][i][perm].t()
                # 计算正样本边的预测值
                pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            # 合并所有批次的预测结果
            pos_test_pred = torch.cat(pos_preds, dim=0)

            # 测试集负样本预测
            neg_preds = []
            for perm in DataLoader(range(neg_test_edge.size(0)), args.batch_size):
                edge = neg_test_edge[perm].t()
                # 计算负样本边的预测值
                neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            # 合并所有批次的预测结果
            neg_test_pred = torch.cat(neg_preds, dim=0)

            # 计算测试集的 hit@50 和 hit@100 指标
            results['test_group_' + str(i) + '_hit@50'] = evaluate(pos_test_pred, neg_test_pred)[0]
            results['test_group_' + str(i) + '_hit@100'] = evaluate(pos_test_pred, neg_test_pred)[1]

        # 验证集整体正样本预测
        pos_preds = []
        for perm in DataLoader(range(split_edge['valid']['edge'].size(0)), args.batch_size):
            edge = split_edge['valid']['edge'][perm].t()
            # 计算正样本边的预测值
            pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        # 合并所有批次的预测结果
        pos_valid_pred = torch.cat(pos_preds, dim=0)

        # 验证集整体负样本预测
        neg_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), args.batch_size):
            edge = neg_valid_edge[perm].t()
            # 计算负样本边的预测值
            neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        # 合并所有批次的预测结果
        neg_valid_pred = torch.cat(neg_preds, dim=0)

        # 计算验证集的整体 hit@50 和 hit@100 指标
        results['valid_overall_hit@50'] = evaluate(pos_valid_pred, neg_valid_pred)[0]
        results['valid_overall_hit@100'] = evaluate(pos_valid_pred, neg_valid_pred)[1]

        # 测试集整体正样本预测
        pos_preds = []
        for perm in DataLoader(range(split_edge['test']['edge'].size(0)), args.batch_size):
            edge = split_edge['test']['edge'][perm].t()
            # 计算正样本边的预测值
            pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        # 合并所有批次的预测结果
        pos_test_pred = torch.cat(pos_preds, dim=0)

        # 测试集整体负样本预测
        neg_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), args.batch_size):
            edge = neg_test_edge[perm].t()
            # 计算负样本边的预测值
            neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        # 合并所有批次的预测结果
        neg_test_pred = torch.cat(neg_preds, dim=0)

        # 计算测试集的整体 hit@50 和 hit@100 指标
        results['test_overall_hit@50'] = evaluate(pos_test_pred, neg_test_pred)[0]
        results['test_overall_hit@100'] = evaluate(pos_test_pred, neg_test_pred)[1]

    else:
        raise NotImplementedError
    
    return results


parser = argparse.ArgumentParser(description='Twins GNN')
parser.add_argument('--dataset', type=str)
parser.add_argument('--dataset_type', type=str, help='social or recommendation')
parser.add_argument('--device', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--ori_lr', type=float)
parser.add_argument('--aug_lr', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--threshold', type=int)
parser.add_argument('--add_edge', type=int)
parser.add_argument('--n_layer', type=int)
parser.add_argument('--load_partition', type=int)
parser.add_argument('--eval_steps', type=int)
parser.add_argument('--aug_size', type=float, help='supervision augmentation size')
parser.add_argument('--aug_type', type=str, help='rw or jaccard')
parser.add_argument('--alpha', type=float)
parser.add_argument('--train_ratio', type=float)
parser.add_argument('--beta', type=int)

args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# 加载数据
graph, ui_graph, split_edge = load_data(args)
split_edge_ori = deepcopy(split_edge)

# 准备数据
data, split_edge_ori, graph, membership, confidence = edge_split(graph, split_edge_ori, 0, args)

# 加载LightGCN
model = load_model(graph, args.model, args.dataset, device)

# 设置日志配置
logging.basicConfig(filename='Log/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 训练和评估过程
best_hit = 0
best_epoch = 0

for epoch in range(1, 1 + args.epochs):
    # 初始化边数据加载器
    dataloader = EdgeDataloader(split_edge_ori['train']['edge'], confidence, args.batch_size, args.train_ratio)
    
    # 训练模型
    e2e_train(graph, model, dataloader, args, device)
    
    # 每隔一定步数进行一次评估
    if epoch % args.eval_steps == 0:
        results = e2e_test(model, data, split_edge, args, device, ui_graph)
        
        # 打印当前轮次的评估结果
        log_message = f">> EPOCH {epoch:03d}, "
        for i in range(2):
            log_message += f"Group {i} hit@50: {results['test_group_' + str(i) + '_hit@50']:.4f}, "
        log_message += f"Overall hit@50: {results['test_overall_hit@50']:.4f}."
        logging.info(log_message)
        print(log_message)

        # 提前收敛判断
        if results['test_overall_hit@50'] > best_hit:
            best_hit = results['test_overall_hit@50']
            best_epoch = epoch
            # 保存最佳模型
            torch.save(model, 'trained_model/lightgcn_best_model.pt')
            logging.info("Saved best model")

        # 判断是否满足收敛条件，提早停止训练
        if epoch > best_epoch + 5 and epoch > args.threshold - 2:
            logging.info(f"Training stopped early at epoch {epoch}")
            break

# 加载最佳模型进行测试
model = torch.load('trained_model/lightgcn_best_model.pt')
results = e2e_test(model, data, split_edge, args, device, ui_graph)

# 记录测试结果
test_message = ">> TEST\n>> "
print(">> TEST\n>> ", end='')
for i in range(2):
    group_message = 'Group {:d} hit@50: {:.4f}, '.format(i, results['test_group_'+str(i)+'_hit@50'])
    test_message += group_message
    print(group_message, end='')
overall_message = 'Overall hit@50: {:.4f}.'.format(results['test_overall_hit@50'])
test_message += overall_message
print(overall_message)

logging.info(test_message)