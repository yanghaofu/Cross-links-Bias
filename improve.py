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
import datetime


def e2e_train(graph, EPOCH, model_ori, model_aug, model_cal, 
              dataloader_ori, dataloader_aug, data_raw, split_edge, 
              membership, args, device, ui_graph):

    # 将图数据移动到指定设备上（如GPU）
    graph = graph.to(device)

    # 在原始图上进行训练
    model_ori.train()
    optimizer_ori = torch.optim.Adam(model_ori.parameters(), lr=args.ori_lr)
    
    for data in dataloader_ori: 
        # 获取嵌入
        embedding = model_ori()
        # 正样本边
        pos_edge = data[0].t()
        pos_out = torch.sigmoid((torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)).unsqueeze(-1))
        # 计算正样本损失
        pos_loss = (-torch.log(pos_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

        # 随机采样负样本边
        neg_edge = torch.randint(0, graph.num_nodes(), pos_edge.size(), dtype=torch.long,
                            device=embedding.device)
        neg_out = torch.sigmoid(torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1).unsqueeze(-1))
        # 计算负样本损失
        neg_loss = (-torch.log(1 - neg_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

        # 总损失
        loss = pos_loss + neg_loss

        # 梯度更新
        optimizer_ori.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ori.parameters(), 1.0)
        optimizer_ori.step()

    # 在增强图上进行训练
    model_aug.train()
    optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=args.aug_lr)

    for data in dataloader_aug: 
        # 获取嵌入
        embedding = model_aug()
        # 正样本边
        pos_edge = data[0].t()
        pos_out = torch.sigmoid((torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)).unsqueeze(-1))
        # 计算正样本损失
        pos_loss = (-torch.log(pos_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

        # 随机采样负样本边
        neg_edge = torch.randint(0, graph.num_nodes(), pos_edge.size(), dtype=torch.long,
                            device=embedding.device)
        neg_out = torch.sigmoid(torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1).unsqueeze(-1))
        # 计算负样本损失
        neg_loss = (-torch.log(1 - neg_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

        # 总损失
        loss = pos_loss + neg_loss

        # 梯度更新
        optimizer_aug.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_aug.parameters(), 1.0)
        optimizer_aug.step()

    # 动态训练策略
    LR = args.alpha * 1 / (1 + math.exp(-EPOCH + args.threshold))
    STEP = int(args.beta * 1 / (1 + math.exp(-EPOCH + args.threshold))) + 1

#     # 消融实验
#     LR = 0.01
#     STEP = 12
    
    optimizer_cal = torch.optim.Adam(model_cal.parameters(), lr=LR) #消融实验需要删掉
    optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=LR)
    optimizer_ori = torch.optim.Adam(model_ori.parameters(), lr=LR)

    best_hit = 0

    # 保存中间模型
    torch.save(model_cal, 'trained_model/e2e_intermediate_cal_'+args.dataset+'_'+args.model+'_model.pt')

    for epoch in tqdm(range(STEP), desc='>> Merge...'):
        model_cal.train()

        for data in dataloader_ori:
            # 获取原始和增强图的嵌入
            emb_ori = model_ori()
            emb_aug = model_aug()

            if args.model == 'pprgo':
                emb_ori = emb_ori.squeeze(1)
                emb_aug = emb_aug.squeeze(1)

            # 结合原始和增强图的嵌入
            embedding = model_cal(emb_ori, emb_aug)

            # 正样本边
            pos_edge = data[0].t()
            pos_out = torch.sigmoid((torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)).unsqueeze(-1))
            # 计算正样本损失
            pos_loss = (-torch.log(pos_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

            # 随机采样负样本边
            neg_edge = torch.randint(0, graph.num_nodes(), pos_edge.size(), dtype=torch.long,
                                device=embedding.device)
            neg_out = torch.sigmoid(torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1).unsqueeze(-1))
            # 计算负样本损失
            neg_loss = (-torch.log(1 - neg_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

            # 对比嵌入
            contrastive_loss = torch.nn.functional.mse_loss(embedding, emb_aug.detach())

            # 总损失
            loss = pos_loss + neg_loss + contrastive_loss

            # 梯度更新
            optimizer_cal.zero_grad() #消融实验去除
            optimizer_aug.zero_grad()
            optimizer_ori.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_cal.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_aug.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_ori.parameters(), 1.0)
            optimizer_cal.step() #消融实验去除
            optimizer_aug.step()
            optimizer_ori.step()

        # 验证模型效果
        results = e2e_test(model_ori, model_aug, model_cal, data_raw, split_edge, args, device, ui_graph=ui_graph)
        if results['valid_overall_hit@50'] > best_hit:
            best_hit = results['valid_overall_hit@50']
            torch.save(model_cal, 'trained_model/e2e_intermediate_cal_'+args.dataset+'_'+args.model+'_model.pt')
            

def e2e_test(model_ori, model_aug, model_cal, data, split_edge, args, device, ui_graph=None):
    model_ori.eval()
    model_aug.eval()
    model_cal.eval()

    if args.model != 'ultragcn':
        emb_ori = model_ori().detach()
        emb_aug = model_aug().detach()

        if args.model == 'pprgo':
            emb_ori = emb_ori.squeeze(1)
            emb_aug = emb_aug.squeeze(1)
            
        embedding = model_cal(emb_ori, emb_aug).detach()
    
    else:
        emb_ori = model_ori.get_embedding()
        emb_aug = model_aug.get_embedding()

        embedding = model_cal(emb_ori, emb_aug).detach()

    if args.dataset_type == 'social':

        neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
        neg_test_edge = split_edge['test']['edge_neg'].to(device)

        results = {}

        for i in range(2):
            
            # valid preds
            pos_preds = []
            for perm in DataLoader(range(data['valid'][i].size(0)), args.batch_size):
                edge = data['valid'][i][perm].t()
                pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            pos_valid_pred = torch.cat(pos_preds, dim=0) 
            neg_preds = []
            for perm in DataLoader(range(neg_valid_edge.size(0)), args.batch_size):
                edge = neg_valid_edge[perm].t()
                neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            neg_valid_pred = torch.cat(neg_preds, dim=0)
            results['valid_group_'+str(i)+'_hit@50'] = evaluate(pos_valid_pred, neg_valid_pred)[0]
            results['valid_group_'+str(i)+'_hit@100'] = evaluate(pos_valid_pred, neg_valid_pred)[1]

            # test preds
            pos_preds = []
            for perm in DataLoader(range(data['test'][i].size(0)), args.batch_size):
                edge = data['test'][i][perm].t()
                pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            pos_test_pred = torch.cat(pos_preds, dim=0) 
            neg_preds = []
            for perm in DataLoader(range(neg_test_edge.size(0)), args.batch_size):
                edge = neg_test_edge[perm].t()
                neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
            neg_test_pred = torch.cat(neg_preds, dim=0)
            results['test_group_'+str(i)+'_hit@50'] = evaluate(pos_test_pred, neg_test_pred)[0]
            results['test_group_'+str(i)+'_hit@100'] = evaluate(pos_test_pred, neg_test_pred)[1]


        # valid preds
        pos_preds = []
        for perm in DataLoader(range(split_edge['valid']['edge'].size(0)), args.batch_size):
            edge = split_edge['valid']['edge'][perm].t()
            pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_preds, dim=0) 
        neg_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), args.batch_size):
            edge = neg_valid_edge[perm].t()
            neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_preds, dim=0)
        results['valid_overall_hit@50'] = evaluate(pos_valid_pred, neg_valid_pred)[0]
        results['valid_overall_hit@100'] = evaluate(pos_valid_pred, neg_valid_pred)[1]

        # test preds
        pos_preds = []
        for perm in DataLoader(range(split_edge['test']['edge'].size(0)), args.batch_size):
            edge = split_edge['test']['edge'][perm].t()
            pos_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_preds, dim=0) 
        neg_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), args.batch_size):
            edge = neg_test_edge[perm].t()
            neg_preds += [torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_preds, dim=0)
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

graph, ui_graph, split_edge = load_data(args)

split_edge_ori = deepcopy(split_edge)
split_edge_aug = deepcopy(split_edge)

# prepare data for twins GNNs
data, split_edge_ori, graph, membership, confidence = edge_split(graph, split_edge_ori, 0, args)
data, split_edge_aug, graph, membership, confidence = edge_split(graph, split_edge_aug, 1, args)

# Twins GNNs
model_ori = load_model(graph, args.model, args.dataset, device)
model_aug = deepcopy(model_ori)

# embedding fusion module
# the default embedding dimension is 64
model_cal = merger(in_dim=128, hidden_dim=64, out_dim=64, num_layers=args.n_layer, dropout=0).to(device)
model_cal.reset_parameters()

best_hit = 0
best_epoch = 0

# 在训练循环中添加日志记录
for epoch in range(1, 1 + args.epochs):
    dataloader_ori = EdgeDataloader(split_edge_ori['train']['edge'], confidence, args.batch_size, args.train_ratio)
    dataloader_aug = EdgeDataloader(split_edge_aug['train']['edge'], confidence, args.batch_size, args.train_ratio)
    if args.model != 'ultragcn':
        e2e_train(graph, epoch, model_ori, model_aug, model_cal, 
                        dataloader_ori, dataloader_aug, data, split_edge, 
                        membership, args, device, ui_graph)
    else:
        e2e_ultra_train(graph, epoch, model_ori, model_aug, model_cal, 
                        dataloader_ori, dataloader_aug, data, split_edge, 
                        membership, args, device, ui_graph)
    if epoch % args.eval_steps == 0:
        model_cal = torch.load('trained_model/e2e_intermediate_cal_'+args.dataset+'_'+args.model+'_model.pt')
        results = e2e_test(model_ori, model_aug, model_cal, data, split_edge, args, device, ui_graph)
        print(">> EPOCH {:03d}, ".format(epoch), end='')
        for i in range(2):
            print('Group {:d} hit@50: {:.4f}, '.format(i, results['test_group_'+str(i)+'_hit@50']), end='')
        print('Overall hit@50: {:.4f}.'.format(results['test_overall_hit@50']))
        
        # 记录到日志文件
        with open('Log/improve.log', 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - EPOCH {epoch}: ")
            f.write(f"Group 0 hit@50: {results['test_group_0_hit@50']:.4f}, ")
            f.write(f"Group 1 hit@50: {results['test_group_1_hit@50']:.4f}, ")
            f.write(f"Overall hit@50: {results['test_overall_hit@50']:.4f}\n")

        # early converge
        if results['test_overall_hit@50'] > best_hit:
            best_hit = results['test_overall_hit@50']
            best_epoch = epoch
            if args.model != 'gin':
                torch.save(model_ori, 'trained_model/e2e_ori_'+args.dataset+'_'+args.model+'_model.pt')
                torch.save(model_aug, 'trained_model/e2e_aug_'+args.dataset+'_'+args.model+'_model.pt')
                torch.save(model_cal, 'trained_model/e2e_cal_'+args.dataset+'_'+args.model+'_model.pt')

        if epoch > best_epoch + 5 and epoch > args.threshold - 2:  # converge condition 
            converged_epoch = epoch
            break

# 最后的测试结果记录到日志
results = e2e_test(model_ori, model_aug, model_cal, data, split_edge, args, device, ui_graph)
print(">> TEST\n>> ", end='')
for i in range(2):
    print('Group {:d} hit@50: {:.4f}, '.format(i, results['test_group_'+str(i)+'_hit@50']), end='')
print('Overall hit@50: {:.4f}, '.format(results['test_overall_hit@50']))

# 记录到日志文件
with open('Log/improve.log', 'a') as f:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"{timestamp} - TEST RESULTS: ")
    f.write(f"Group 0 hit@50: {results['test_group_0_hit@50']:.4f}, ")
    f.write(f"Group 1 hit@50: {results['test_group_1_hit@50']:.4f}, ")
    f.write(f"Overall hit@50: {results['test_overall_hit@50']:.4f}\n")