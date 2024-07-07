import torch
import argparse

from copy import deepcopy
from preprocess import *
from model import load_model, merger
from process import e2e_train, e2e_test, e2e_ultra_train

from ablation import *




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
# 拆分好的数据，包含伪边的边集，增加了边、指定了membership和type
data, split_edge_ori, graph, membership, confidence = edge_split(graph, split_edge_ori, 0, args)
data, split_edge_aug, graph, membership, confidence = edge_split(graph, split_edge_aug, 1, args)

# Twins GNNs
model_ori = load_model(graph, args.model, args.dataset, device)
model_aug = deepcopy(model_ori)

# embedding fusion module
# the default embedding dimension is 64

# model_cal = merger(in_dim=128, hidden_dim=64, out_dim=64, num_layers=args.n_layer, dropout=0).to(device)
# model_cal.reset_parameters()

# 消融-融合
model_cal=AverageMerger()

best_hit = 0
best_epoch = 0

for epoch in range(1, 1 + args.epochs):
    # 初始化原始和增强图的边数据加载器
    dataloader_ori = EdgeDataloader(split_edge_ori['train']['edge'], confidence, args.batch_size, args.train_ratio)
    dataloader_aug = EdgeDataloader(split_edge_aug['train']['edge'], confidence, args.batch_size, args.train_ratio)
    
    # 根据模型类型选择训练方法
    if args.model != 'ultragcn':
        e2e_train(graph, epoch, model_ori, model_aug, model_cal, 
                  dataloader_ori, dataloader_aug, data, split_edge, 
                  membership, args, device, ui_graph)
    else:
        e2e_ultra_train(graph, epoch, model_ori, model_aug, model_cal, 
                        dataloader_ori, dataloader_aug, data, split_edge, 
                        membership, args, device, ui_graph)

    # 每隔一定步数进行一次评估
    if epoch % args.eval_steps == 0:
        # 加载中间模型进行评估
        model_cal = torch.load('trained_model/e2e_intermediate_cal_'+args.dataset+'_'+args.model+'_model.pt')
        results = e2e_test(model_ori, model_aug, model_cal, data, split_edge, args, device, ui_graph)
        
        # 打印当前轮次的评估结果
        print(">> EPOCH {:03d}, ".format(epoch), end='')
        for i in range(2):
            print('Group {:d} hit@50: {:.4f}, '.format(i, results['test_group_'+str(i)+'_hit@50']), end='')
        print('Overall hit@50: {:.4f}.'.format(results['test_overall_hit@50']))

        # 提前收敛判断
        if results['test_overall_hit@50'] > best_hit:
            best_hit = results['test_overall_hit@50']
            best_epoch = epoch
            # 保存最佳模型
            if args.model != 'gin':
                torch.save(model_ori, 'trained_model/e2e_ori_'+args.dataset+'_'+args.model+'_model.pt')
                torch.save(model_aug, 'trained_model/e2e_aug_'+args.dataset+'_'+args.model+'_model.pt')
                torch.save(model_cal, 'trained_model/e2e_cal_'+args.dataset+'_'+args.model+'_model.pt')

        # 判断是否满足收敛条件，提早停止训练
        if epoch > best_epoch + 5 and epoch > args.threshold - 2:  # 收敛条件
            converged_epoch = epoch
            break


if args.model != 'gin':
    model_ori = torch.load('trained_model/e2e_ori_'+args.dataset+'_'+args.model+'_model.pt')
    model_aug = torch.load('trained_model/e2e_aug_'+args.dataset+'_'+args.model+'_model.pt')
    model_cal = torch.load('trained_model/e2e_cal_'+args.dataset+'_'+args.model+'_model.pt')

results = e2e_test(model_ori, model_aug, model_cal, data, split_edge, args, device, ui_graph)
print(">> TEST\n>> ", end='')
for i in range(2):
    print('Group {:d} hit@50: {:.4f}, '.format(i, results['test_group_'+str(i)+'_hit@50']), end='')
print('Overall hit@50: {:.4f}, '.format(results['test_overall_hit@50']))