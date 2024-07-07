# Note
This repository includes the implementation for our NeurIPS 2023 paper: ***[Cross-links Matter for Link Prediction: Rethinking the Debiased GNN from a Data Perspective](https://neurips.cc/virtual/2023/poster/70277).***

## Environments

Python 3.7.6

Packages:
```
dgl_cu102==0.9.1.post1
numpy==1.19.2
python_louvain==0.15
networkx==2.5
tqdm==4.62.3
torch==1.12.1+cu102
community==1.0.0b1
dgl==1.1.0
PyYAML==6.0
```
[community]( https://pypi.org/project/community/ ) is an essential package to deploy the Louvain algorithm used in our work.

Run the following code to install all required packages.
```
> pip install -r requirements.txt
```

## Datasets & Processed files

- Due to size limitation, the processed files and datasets are stored in  [google drive](https://drive.google.com/file/d/1uG43ndQih7OlH477pe3pR3OB4W0sxeSP/view?usp=share_link). The datasets include Epinions, DBLP and LastFM. 
- Each dataset directory contains the following processed files: 
    * graph.pkl: DGLGraph object for storing the graph structure.
    * split_edge.pkl: Splitted training samples, validation samples and test samples.
    * louvain_dataset.pkl: Detected community memberships through Louvain algorithm.
    * Other processed files for running PPRGo and UltraGCN, such as constrain_mat.pkl, ii_topk_neighbors.np.pkl.

## Run the codes

All arguments are properly set in advance in the script files for reproducing our results. 

Here we take GraphSAGE and GAT as examples.

```
> bash script/run_graphsage_e2e.sh
> bash script/run_gat_e2e.sh
```

## BibTeX

If you like our work and use the model for your research, please cite our work as follows.

```bibtex
@inproceedings{luo2023cross-links,
author = {Luo, Zihan and Huang, Hong and Lian, Jianxun and Song, Xiran and Xie, Xing and Jin, Hai},
title = {Cross-links Matter for Link Prediction: Rethinking the Debiased GNN from a Data Perspective},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year = {2023},
month = {October},
url = {https://www.microsoft.com/en-us/research/publication/cross-links-matter-for-link-prediction-rethinking-the-debiased-gnn-from-a-data-perspective/},
}
``` 

## 补充
1. lightgcn.py : 通过运行下面的命令即可进行对比实验，训练日志存储在Log/training.log。
```
> bash script/lightgcn.sh
```
2. improve.py ：尝试在融合模块添加对比损失，以保证融合特征视野全面，训练日志存储在Log/improve.log。这个改进在Epinions数据集上取得优化的效果。
```
> bash script/lightgcn.sh
```
3. 记录.md：记录了复现过程以及结果，对于消融实验的方法在文档中也有提及。
