# OFCL_Cluster
Title – the project or dataset of Outlier-Filtered Contrastive Learning for Attributed Graph Clustering.

The codes for paper "Outlier-Filtered Contrastive Learning for Attributed Graph Clustering"
‎Description – This repository provides the implementation of  "Outlier-Filtered Contrastive Learning for Attributed Graph Clustering", a graph representation learning framework that integrates structural outlier detection with graph contrastive learning. The framework first identifies outlier nodes using structural and attribute features and Isolation Forest, then removes detected outliers to construct a refined graph. Contrastive learning is subsequently performed on the refined graph to learn robust node embeddings for node clustering.
‎
Dataset Information - The experiments are conducted on publicly available benchmark citation network datasets:
Datasets Nodes Edges Attributes Clustering
Cora       2708    10556  1433 7
CiteSeer 3327     9104   3703 6
Wiki        2405    17981  4973 17

External Dataset Source
Wiki Dataset:
Mernyei, P., & Cangea, C. (2020). Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks.
Dataset URL:
https://github.com/cquetai/OFCL_Cluster.
Paper:
https://arxiv.org/abs/2007.02901
Code Information - Main components:
Structural feature extraction
Degree
Clustering coefficient
PageRank
Betweenness centrality
Outlier detection
Isolation Forest
Graph refinement
Outlier node removal
Graph contrastive learning
Edge dropout
Feature masking
Node clustering

Evaluate:
ACC
NMI
ARI

Install dependencies
pip install torch
pip install torch_geometric
pip install scikit-learn
pip install networkx
pip install numpy

‎Requirements：
Pytorch2.40, 
PyTorch Geometric >= 2.4,
cuda12.1,
python -V: Python 3.11.5
NumPy >= 1.24
Scikit-learn >= 1.3
NetworkX >= 3.0
pip show torch
Name: torch Version: 2.4.0+cu121
python -c "import torch; print(torch.__version__)"
2.4.0+cu121

Run node clustering
python OFCL20260417.py

Methodology
Extract graph structural features.
Standardize structural descriptors.
Detect outlier nodes using Isolation Forest.
Remove detected outliers and reconstruct the graph.
Generate graph augmentations.
Train graph contrastive learning model.
Obtain node embeddings.
Perform clustering  evaluation.

Citation
If you use this code, please cite:
OFCL: Outlier-Filtered Contrastive Learning for Attributed Graph Clustering Learning. 

Materials and Methods
CPU: Intel Xeon / Intel Core i9 processor
RAM: 64 GB
GPU: NVIDIA RTX 3090 (24 GB) or equivalent
Storage: SSD
Operating System: Ubuntu 22.04 LTS
Python: 3.11.5
PyTorch: 2.4
PyTorch Geometric: 2.4
CUDA: 12.1

Dataset Sources
Dataset repository:
https://github.com/cquetai/OFCL_Cluster.

Data Preprocessing 
Reference to the paper - Outlier-Filtered Contrastive Learning for Attributed Graph Clustering

