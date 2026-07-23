# -*- coding: utf-8 -*-
"""
Structure + Attribute Outlier Detection
IsolationForest + Graph Contrastive Learning

Evaluate:
ACC
NMI
ARI
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import networkx as nx
import random

from torch_geometric.datasets import Planetoid, WikiCS, AttributedGraphDataset
from torch_geometric.utils import to_networkx, dropout_edge
from torch_geometric.nn import GCNConv

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


# -----------------------------
# clustering accuracy
# -----------------------------
def cluster_acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row, col = linear_sum_assignment(w.max() - w)

    return w[row, col].sum() / y_pred.size


# -----------------------------
# graph encoder
# -----------------------------
class Encoder(nn.Module):

    def __init__(self, in_dim, hidden=256, out_dim=128):

        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x


# -----------------------------
# graph contrastive model
# -----------------------------
class GraphCL(nn.Module):

    def __init__(self, in_dim):

        super().__init__()

        self.encoder = Encoder(in_dim)

        self.project = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

    def forward(self, x, edge_index):

        z = self.encoder(x, edge_index)

        h = self.project(z)

        return z, h


# -----------------------------
# InfoNCE loss
# -----------------------------
def contrastive_loss(z1, z2, tau=0.5):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    sim = torch.mm(z1, z2.t())

    sim = torch.exp(sim / tau)

    pos = sim.diag()

    loss = -torch.log(pos / sim.sum(dim=1))

    #return loss.mean()

##################
    sim1 = torch.mm(z2, z1.t())

    sim1 = torch.exp(sim / tau)

    pos1 = sim.diag()

    loss1 = -torch.log(pos / sim.sum(dim=1))
    #return loss1.mean()
#########################
    return (loss.mean()+loss1.mean())/2

# -----------------------------
# feature dropout
# -----------------------------
def drop_feature(x, drop_prob):

    mask = torch.rand_like(x) > drop_prob

    return x * mask


# -----------------------------
# main pipeline
# -----------------------------
def run(dataset="citeseer", remove_outlier=True,contamination=0.05, epochs=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # -------------------------
    # load dataset
    # -------------------------

    if dataset == "WikiCS":
        #dataset = WikiCS(root="WikiCS")
        data=AttributedGraphDataset(root='./data',name='wiki')
    else:
        dataset = Planetoid(root="data", name=dataset)

    data = dataset[0]

    x = data.x
    y = data.y.numpy()
    print("Number of classes: ", np.unique(y).size)
    edge_index = data.edge_index

    num_nodes = data.num_nodes
    print("Number of nodes: ", num_nodes)
    num_edges = data.num_edges
    print("Number of edges: ", num_edges)

    #num_attributes = data.num_attributes
    #print("Number of attributes: ", num_attributes)
    # -------------------------
    # structural features
    # -------------------------

    G = to_networkx(data, to_undirected=True)

    deg = np.array([d for _, d in G.degree()]).reshape(-1,1)

    clustering = np.array([nx.clustering(G,i) for i in G.nodes()]).reshape(-1,1)

    try:
        pr = np.array(list(nx.pagerank(G).values())).reshape(-1, 1)
    except:
        pr = np.zeros_like(deg)
    bet = np.array(list(nx.betweenness_centrality(G).values())).reshape(-1,1)

    features = np.hstack([deg, clustering, bet])

    # -------------------------
    # attribute + structure
    # -------------------------

    attr = x.numpy()

    features = np.hstack([attr, features])

    scaler = StandardScaler()

    features = scaler.fit_transform(features)

    # -------------------------
    # IsolationForest
    # -------------------------
    if remove_outlier:
        iso = IsolationForest(contamination=contamination)
        pred = iso.fit_predict(features)
        print(pred)
        #exit()
        mask = pred == 1
        keep_idx = np.where(mask)[0]
        removed_idx = np.where(~mask)[0]
        print("original nodes:", num_nodes)
        print("remain nodes:", len(keep_idx))
        print(f"Total nodes: {num_nodes}, kept: {len(keep_idx)}, removed (outliers): {len(removed_idx)}")
    else:
        keep_idx = np.arange(num_nodes)
        removed_idx = np.array([], dtype=int)
        print("Skipping outlier removal.")
        print("original nodes:", num_nodes)
        print("remain nodes:", len(keep_idx))
        print(f"Total nodes: {num_nodes}, kept: {len(keep_idx)}, removed (outliers): {len(removed_idx)}")
        
    # -------------------------
    # remove outliers
    # -------------------------

    x = x[keep_idx]
    y = y[keep_idx]

    id_map = {old:i for i,old in enumerate(keep_idx)}

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    mask = np.isin(src,keep_idx) & np.isin(dst,keep_idx)

    src = src[mask]
    dst = dst[mask]

    src = np.array([id_map[i] for i in src])
    dst = np.array([id_map[i] for i in dst])

    edge_index = torch.tensor([src,dst])

    # -------------------------
    # model
    # -------------------------

    model = GraphCL(x.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = x.to(device)
    edge_index = edge_index.to(device)

    # -------------------------
    # training
    # -------------------------

    for epoch in range(epochs):

        model.train()

        x1 = drop_feature(x,0.3)
        x2 = drop_feature(x,0.3)

        e1,_ = dropout_edge(edge_index,0.2)
        e2,_ = dropout_edge(edge_index,0.2)

        z1,h1 = model(x1,e1)
        z2,h2 = model(x2,e2)

        loss = contrastive_loss(h1,h2)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch%50==0:

            print("epoch:",epoch,"loss:",loss.item())

    # -------------------------
    # embeddings
    # -------------------------

    model.eval()

    with torch.no_grad():

        z,_ = model(x,edge_index)

    z = z.cpu().numpy()

    # -------------------------
    # clustering
    # -------------------------

    n_clusters = len(np.unique(y))

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    pred = kmeans.fit_predict(z)

    acc = cluster_acc(y,pred)

    nmi = normalized_mutual_info_score(y,pred)

    ari = adjusted_rand_score(y,pred)

    print("\nResults")

    print("ACC:",acc)

    print("NMI:",nmi)

    print("ARI:",ari)


# -----------------------------
# run
# -----------------------------
if __name__ == "__main__":
    dataset="WikiCS"
    remove_outlier = False #True
    contamination=0.005
    epochs=301
    run(dataset,
        remove_outlier,
        contamination,
        epochs)  #Cora 0.005 WikiCS citeseer 0.05 pubmed

#citeseer remove_outlier = True contamination=0.005  epochs=301
"""
ACC: 0.6610574018126888
NMI: 0.39587721958749194
ARI: 0.4106902814085543
"""
#citeseer remove_outlier = False contamination=0.005  epochs=301
"""
ACC: 0.6366095581605049
NMI: 0.3810122193114006
ARI: 0.38241809194531184
"""

#Cora remove_outlier = True contamination=0.005  epochs=301
"""
ACC: 0.7446176688938382
NMI: 0.5660502948207308
ARI: 0.5345938593884345
"""

#WikiCS remove_outlier = True contamination=0.005
#Number of edges:  431726
#original nodes: 11701
"""
ACC: 0.4019546469678749
NMI: 0.3509158759158808
ARI: 0.2256592214465724
"""
#WikiCS remove_outlier = False contamination=0.005
"""
ACC: 0.39314124433809076
NMI: 0.32733181240059604
ARI: 0.20870499902395084
"""
