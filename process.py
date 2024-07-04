import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from utils import evaluate
from tqdm import tqdm



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

            # 总损失
            loss = pos_loss + neg_loss

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
    # 设置模型为评估模式
    model_ori.eval()
    model_aug.eval()
    model_cal.eval()

    # 获取模型的嵌入
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


    # 处理推荐系统类型数据集
    elif args.dataset_type == 'recommendation':
        item_embedding = embedding
        item_sim_adj = item_embedding @ item_embedding.t()  # 计算物品之间的相似度矩阵
        user_item_adj = ui_graph.adj().to_dense()[:1892, 1892:].to(device)  # 获取用户-物品交互矩阵
        score_adj = user_item_adj @ item_sim_adj  # 计算得分矩阵

        # 获取每个物品的排名
        _, temp_result = torch.sort(score_adj, descending=True)
        _, sort_score_adj = torch.sort(temp_result)

        results = {}

        for i in range(2):
            hits50 = []
            hits100 = []
            for perm in DataLoader(range(data['valid'][i].size(0)), args.batch_size):
                edge = data['valid'][i][perm].t()
                sort_result = sort_score_adj[edge[0], edge[1] - 1892]
                if torch.any(sort_result < 50):
                    hits50.append((sort_result < 50).sum().item() / len(perm))
                else:
                    hits50.append(0)
                if torch.any(sort_result < 100):
                    hits100.append((sort_result < 100).sum().item() / len(perm))
                else:
                    hits100.append(0)

            results['valid_group_' + str(i) + '_hit@50'] = np.array(hits50).mean()
            results['valid_group_' + str(i) + '_hit@100'] = np.array(hits100).mean()

            hits50 = []
            hits100 = []
            for perm in DataLoader(range(data['test'][i].size(0)), args.batch_size):
                edge = data['test'][i][perm].t()
                sort_result = sort_score_adj[edge[0], edge[1] - 1892]
                if torch.any(sort_result < 50):
                    hits50.append((sort_result < 50).sum().item() / len(perm))
                else:
                    hits50.append(0)
                if torch.any(sort_result < 100):
                    hits100.append((sort_result < 100).sum().item() / len(perm))
                else:
                    hits100.append(0)

            results['test_group_' + str(i) + '_hit@50'] = np.array(hits50).mean()
            results['test_group_' + str(i) + '_hit@100'] = np.array(hits100).mean()

        # 验证集整体预测
        hits50 = []
        hits100 = []
        for perm in DataLoader(range(split_edge['valid']['edge'].size(0)), args.batch_size):
            edge = split_edge['valid']['edge'][perm].t()
            sort_result = sort_score_adj[edge[0], edge[1] - 1892]
            if torch.any(sort_result < 50):
                hits50.append((sort_result < 50).sum().item() / len(perm))
            else:
                hits50.append(0)
            if torch.any(sort_result < 100):
                hits100.append((sort_result < 100).sum().item() / len(perm))
            else:
                hits100.append(0)

        results['valid_overall_hit@50'] = np.array(hits50).mean()
        results['valid_overall_hit@100'] = np.array(hits100).mean()

        # 测试集整体预测
        hits50 = []
        hits100 = []
        for perm in DataLoader(range(split_edge['test']['edge'].size(0)), args.batch_size):
            edge = split_edge['test']['edge'][perm].t()
            sort_result = sort_score_adj[edge[0], edge[1] - 1892]
            if torch.any(sort_result < 50):
                hits50.append((sort_result < 50).sum().item() / len(perm))
            else:
                hits50.append(0)
            if torch.any(sort_result < 100):
                hits100.append((sort_result < 100).sum().item() / len(perm))
            else:
                hits100.append(0)

        results['test_overall_hit@50'] = np.array(hits50).mean()
        results['test_overall_hit@100'] = np.array(hits100).mean()

    else:
        raise NotImplementedError
    
    return results

    



def e2e_ultra_train(graph, EPOCH, model_ori, model_aug, model_cal, 
              dataloader_ori, dataloader_aug, data_raw, split_edge, 
              membership, args, device, ui_graph):
    assert args.model == 'ultragcn'

    graph = graph.to(device)

    # train on original graph
    model_ori.train()
    optimizer_ori = torch.optim.Adam(model_ori.parameters(), lr=args.ori_lr)

    for data in dataloader_ori:
        pos_edge = data[0].t()
        src = pos_edge[0]
        pos = pos_edge[1]
        neg = torch.randint(0, graph.num_nodes(), (pos_edge[0].size(0), 64), dtype=torch.long,
                            device=args.device)
        batch_data = (src, pos, neg)

        loss = model_ori(batch_data)
        optimizer_ori.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ori.parameters(), 1.0)
        optimizer_ori.step()


    model_aug.train()
    optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=args.aug_lr)

    for data in dataloader_aug:
        pos_edge = data[0].t()
        src = pos_edge[0]
        pos = pos_edge[1]
        neg = torch.randint(0, graph.num_nodes(), (pos_edge[0].size(0), 64), dtype=torch.long,
                            device=args.device)
        batch_data = (src, pos, neg)

        loss = model_aug(batch_data)
        optimizer_aug.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_aug.parameters(), 1.0)
        optimizer_aug.step()


    # dynamic training
    LR = args.alpha * 1 / (1 + math.exp(-EPOCH+args.threshold))
    STEP = int(args.beta * 1 / (1 + math.exp(-EPOCH+args.threshold))) + 1

    optimizer_cal = torch.optim.Adam(model_cal.parameters(), lr=LR) #消融实验
    optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=LR)
    optimizer_ori = torch.optim.Adam(model_ori.parameters(), lr=LR)

    best_hit = 0
    torch.save(model_cal, 'trained_model/e2e_intermediate_cal_'+args.dataset+'_'+args.model+'_model.pt')

    for epoch in tqdm(range(STEP), desc='>> Merge...'):
        model_cal.train()

        for data in dataloader_ori:
            
            emb_ori = model_ori.get_embedding()
            emb_aug = model_aug.get_embedding()
            embedding = model_cal(emb_ori, emb_aug)

            pos_edge = data[0].t()
            pos_out = torch.sigmoid((torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)).unsqueeze(-1))
            pos_loss = (-torch.log(pos_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

            # Just do some trivial random sampling.
            neg_edge = torch.randint(0, graph.num_nodes(), pos_edge.size(), dtype=torch.long,
                                device=embedding.device)

            neg_out = torch.sigmoid(torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1).unsqueeze(-1))
            neg_loss = (-torch.log(1 - neg_out + 1e-15) * data[1].to(device)).sum() / (data[1].sum())

            loss = pos_loss + neg_loss

            optimizer_cal.zero_grad() #消融实验
            optimizer_aug.zero_grad()
            optimizer_ori.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_cal.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_aug.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_ori.parameters(), 1.0)
            optimizer_cal.step() 
            optimizer_aug.step()
            optimizer_ori.step()

        results = e2e_test(model_ori, model_aug, model_cal, data_raw, split_edge, args, device, ui_graph=ui_graph)
        if results['valid_overall_hit@50'] > best_hit:
            best_hit = results['valid_overall_hit@50']
            torch.save(model_cal, 'trained_model/e2e_intermediate_cal_'+args.dataset+'_'+args.model+'_model.pt')