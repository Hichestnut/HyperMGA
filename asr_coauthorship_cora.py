import os
import time
import copy
import torch
import numpy as np
import torch.optim as optim
import pprint as pp
import dhg
import random
import utils.hypergraph_utils as hgut
from models.HGNN import HGNN
import torch.nn.functional as F
torch.set_printoptions(threshold=np.inf)
from utils.ig import IGattack
from utils.fga_iga import FGIGattack
import pickle

from dhg.data import Cora
from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.data import CoauthorshipCora
from dhg.data import CocitationCora
from dhg.data import CocitationCiteseer
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

import random
# initialize data
datasets = 'CoauthorshipCora'

if datasets == 'CoauthorshipCora':
    data = CoauthorshipCora()
    edge_list = data["edge_list"]
    edge_num = data["num_edges"]
    node_num = data["num_vertices"]
    ori_fts = data["features"]
    ori_lbls = data["labels"]
    test_mask = data["test_mask"]

    ori_H = np.zeros((node_num,edge_num))
    print(len(edge_list[0]))
    for i in range(len(edge_list)):
        for j in range(len(edge_list[i])):
            edge_list_index = list(edge_list[i])
            num_index = int(edge_list_index[j])
            ori_H[num_index][i] = 1

    ori_Dv = np.sum(ori_H, axis=1)
    H = np.zeros((ori_H.shape[1], ori_H.shape[1]))
    fts = np.zeros((ori_H.shape[1], ori_fts.shape[1]))
    lbls = np.zeros((ori_H.shape[1]))
    H_n = 0
    for i in range(ori_Dv.shape[0]):
        if ori_Dv[i] != 0 and H_n < ori_H.shape[1]:
            H[H_n] = ori_H[i]
            fts[H_n] = ori_fts[i]
            lbls[H_n] = ori_lbls[i]
            H_n = H_n + 1
    print(H_n)
    print(ori_H.shape[1])
    print(H.shape)
    for i in range(ori_Dv.shape[0]):
        if ori_Dv[i] == 0 and H_n < ori_H.shape[1]:
            H[H_n] = ori_H[i]
            fts[H_n] = ori_fts[i]
            lbls[H_n] = ori_lbls[i]
            H_n = H_n + 1
    print(H_n)
    print(H.shape)
    idx_train = torch.arange(0, 500, 1)
    idx_test = torch.arange(500, 1000, 1)
    n_class = int(lbls.max()) + 1
elif datasets == 'CocitationCora':
    data = CocitationCora()
    edge_list = data["edge_list"]
    edge_num = data["num_edges"]
    node_num = data["num_vertices"]
    ori_fts = data["features"]
    #ori_lbls = data["labels"]
    
    fr_labels = open('Cocitation_Cora_labels.pkl', 'rb')
    inf_labels = pickle.load(fr_labels)
    fr_labels.close()
    ori_lbls = inf_labels
    
    #fr_edge_list = open('edge_list.pkl', 'rb')
    #inf_edge_list = pickle.load(fr_edge_list)
    #fr_edge_list.close()
    #edge_list = inf_edge_list
    
    #fr_features = open('features.pkl', 'rb')
    #inf_features = pickle.load(fr_features)
    #fr_features.close()
    #ori_fts = inf_features

    ori_H = np.zeros((node_num,edge_num))
    print(len(edge_list[0]))
    for i in range(len(edge_list)):
        for j in range(len(edge_list[i])):
            edge_list_index = list(edge_list[i])
            num_index = int(edge_list_index[j])
            ori_H[num_index][i] = 1

    ori_Dv = np.sum(ori_H, axis=1)
    H = np.zeros((ori_H.shape[1], ori_H.shape[1]))
    fts = np.zeros((ori_H.shape[1], ori_fts.shape[1]))
    lbls = np.zeros((ori_H.shape[1]))
    H_n = 0
    for i in range(ori_Dv.shape[0]):
        if ori_Dv[i] != 0 and H_n < ori_H.shape[1]:
            H[H_n] = ori_H[i]
            fts[H_n] = ori_fts[i]
            lbls[H_n] = ori_lbls[i]
            H_n = H_n + 1
    print(H_n)
    print(ori_H.shape[1])
    print(H.shape)
    for i in range(ori_Dv.shape[0]):
        if ori_Dv[i] == 0 and H_n < ori_H.shape[1]:
            H[H_n] = ori_H[i]
            fts[H_n] = ori_fts[i]
            lbls[H_n] = ori_lbls[i]
            H_n = H_n + 1
    print(H_n)
    print(H.shape)
    idx_train = torch.arange(0, 500, 1)
    idx_test = torch.arange(500, 1000, 1)
    n_class = int(lbls.max()) + 1
elif datasets == 'CocitationCiteseer':
    data = CocitationCiteseer()
    #edge_list = data["edge_list"]
    edge_num = data["num_edges"]
    node_num = data["num_vertices"]
    #ori_fts = data["features"]
    #ori_lbls = data["labels"]
    #test_mask = data["test_mask"]

    fr_edge_list = open('CocitationCiteseer_edge_list.pkl', 'rb')
    inf_edge_list = pickle.load(fr_edge_list)
    fr_edge_list.close()
    edge_list = inf_edge_list
        
    fr_features = open('CocitationCiteseer_features.pkl', 'rb')
    inf_features = pickle.load(fr_features)
    fr_features.close()
    ori_fts = inf_features
    
    fr_labels = open('CocitationCiteseer_labels.pkl', 'rb')
    inf_labels = pickle.load(fr_labels)
    fr_labels.close()
    ori_lbls = inf_labels

    fr_test_mask = open('test_mask.pkl', 'rb')
    inf_test_mask = pickle.load(fr_test_mask)
    fr_test_mask.close()
    test_mask = inf_test_mask
    
    ori_H = np.zeros((node_num,edge_num))
    print(len(edge_list[0]))
    for i in range(len(edge_list)):
        for j in range(len(edge_list[i])):
            edge_list_index = list(edge_list[i])
            num_index = int(edge_list_index[j])
            ori_H[num_index][i] = 1

    ori_Dv = np.sum(ori_H, axis=1)
    # H fts lbls
    H = np.zeros((ori_H.shape[1], ori_H.shape[1]))
    fts = np.zeros((ori_H.shape[1], ori_fts.shape[1]))
    lbls = np.zeros((ori_H.shape[1]))
    H_n = 0
    for i in range(ori_Dv.shape[0]):
        if ori_Dv[i] != 0 and H_n < ori_H.shape[1]:
            H[H_n] = ori_H[i]            
            #fts[H_n] = ori_fts[i]
            dense_fts = ori_fts[i].toarray()
            fts[H_n] = dense_fts
            lbls[H_n] = ori_lbls[i]
            H_n = H_n + 1
    print(H_n)
    print(ori_H.shape[1])
    print(H.shape)
    for i in range(ori_Dv.shape[0]):
        if ori_Dv[i] == 0 and H_n < ori_H.shape[1]:
            H[H_n] = ori_H[i]
            fts[H_n] = ori_fts_array[i]
            lbls[H_n] = ori_lbls[i]
            H_n = H_n + 1
    print(H_n)
    print(H.shape)
    idx_train = torch.arange(0, 500, 1)
    idx_test = torch.arange(500, 1000, 1)
    n_class = int(lbls.max()) + 1



print("data loaded done")

#construct H and G
# H = hgut.construct_H_with_KNN(fts, K_neigs=[10], split_diff_scale=False, is_probH=False, m_prob=1)
# H = hgut.construct_H_with_epsilonball(fts, 0.7)
# H = hgut.gen_l1_hg(fts, 2, 20)
# np.savetxt('H.txt', H)

G = hgut._generate_G_from_H(H, variable_weight=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fts = torch.Tensor(fts).to(device)
lbls = torch.tensor(lbls).squeeze().long().to(device)
# H = torch.Tensor(H).to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.tensor(idx_train).long().to(device)
idx_test = torch.tensor(idx_test).long().to(device)
print("G done")

path = data.name + '.pt'
target_model = torch.load(path)
target_model = target_model.to(device)

norm_G = hgut._generate_G_from_H(H, variable_weight=False)
norm_G = torch.Tensor(norm_G).to(device)
result = target_model.forward(fts, norm_G)

def cleanFGA(model, fts, G, H, lbls, idx_train, target_node, n_perturbations):
    start = time.perf_counter()
    ori_H = H.copy()
    pseudo_labels = model.forward(fts, G).detach().argmax(1)
    pseudo_labels[idx_train] = lbls[idx_train]

    G.requires_grad = True

    #     G = utils.normalize_adj_tensor(G)
    for i in range(n_perturbations):
        output = model.forward(fts, G)
        loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
        grad = torch.autograd.grad(loss, G)[0]
        
        index = torch.argmax(grad[target_node])
        #         print(index)
        ori_H[target_node][index] = 1 - ori_H[target_node][index]
        
        G = hgut._generate_G_from_H(ori_H, variable_weight=False)
        G = torch.Tensor(G).to(device)
        G.requires_grad = True
        # print(index)
        # print(grad[target_node][index])
        # print(ori_H[target_node][index])
    end = time.perf_counter()
    running_time = end - start
    return ori_H, running_time
  

def cleanMGA(model, fts, G, H, lbls, idx_train, target_node, n_perturbations, decay_factor=0.5):
    start = time.perf_counter()
    ori_H = H.copy()
    pseudo_labels = model.forward(fts, G).detach().argmax(1)
    pseudo_labels[idx_train] = lbls[idx_train]

    G.requires_grad = True

    #     G = utils.normalize_adj_tensor(G)
    output = model.forward(fts, G)
    loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])

    momentum = torch.autograd.grad(loss, G)[0]

    for i in range(n_perturbations):
        index = torch.argmax(torch.abs(momentum[target_node]))
        
        if momentum[target_node][index].item() > 0: added = 1
        else: added = -1
        
        ori_H[target_node][index] += added
        
        ori_G = hgut._generate_G_from_H(ori_H, variable_weight=False)
        ori_G = torch.Tensor(ori_G).to(device)
        ori_G.requires_grad = True
        
        
        output = model.forward(fts, ori_G)

        loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
        grad = torch.autograd.grad(loss, ori_G)[0]
        
        norm_l1 = torch.norm(grad, p=1)
        
        momentum = (momentum * decay_factor) + (grad / norm_l1)
    end = time.perf_counter()
    running_time = end - start
    return ori_H, running_time


def randomAttack(H, target_node, n_perturbations):
    start = time.perf_counter()
    ori_H = H.copy()  
    index = random.sample(range(0, H.shape[0]), n_perturbations)
    # print(index)
    for i in range(n_perturbations):
        ori_H[target_node][index[i]] = 1 - ori_H[target_node][index[i]]
    end = time.perf_counter()
    running_time = end - start
    return ori_H, running_time


Dv = np.sum(H, axis=1)

num = 100
points = set()

while len(points) < num:
    x = random.randint(0, edge_num - 1)
    if Dv[x] > 2:
        points.add(x)
print(points)
print("Points Ready")

# mga 1-10
mga_suc = []
mga_time = []
for n_perturbations in range(2, 11):
    suc_attack = 0
    running_time = 0
    for index, target_node in enumerate(points):
        mgaH, rt = cleanMGA(target_model, fts, G, H, lbls, idx_train, target_node, n_perturbations)
        running_time = running_time + rt
        mgaG = hgut._generate_G_from_H(mgaH, variable_weight=False)
        mgaG = torch.Tensor(mgaG).to(device)
        for i in range(0,10):
          randomresult = target_model(fts, mgaG)
          if (torch.argmax(result[target_node]) != torch.argmax(randomresult[target_node])):
              suc_attack = suc_attack + 1
    mga_suc.append(suc_attack / (num * 10.0))
    mga_time.append(running_time)

print('The ASR of MGA in' + datasets + ':')
print(mga_suc)
print(mga_time)

# fga 1-10
fga_suc = []
fga_time = []
for n_perturbations in range(2, 11):
    suc_attack = 0
    running_time = 0
    for index, target_node in enumerate(points):
        fgaH, rt = cleanFGA(target_model, fts, G, H, lbls, idx_train, target_node, n_perturbations)
        running_time = running_time + rt
        fgaG = hgut._generate_G_from_H(fgaH, variable_weight=False)
        fgaG = torch.Tensor(fgaG).to(device)
        for i in range(0,10):
          randomresult = target_model(fts, fgaG)
          if (torch.argmax(result[target_node]) != torch.argmax(randomresult[target_node])):
              suc_attack = suc_attack + 1
    fga_suc.append(suc_attack / (num * 10.0))
    fga_time.append(running_time)

print('The ASR of FGA in' + datasets + ':')
print(fga_suc)
print(fga_time)

#hyperattack 
iga_suc = []
iga_time = []
for n_perturbations in range(2,11):
    suc_attack = 0
    running_time = 0 
    modify_zero = 0
    suc_zero = 0
    for index, target_node in enumerate(points):
        model = FGIGattack(model=target_model, nnodes=fts.shape[0], attack_structure=True, attack_features=False,
                        device=device)
        model.attack(target_model, fts, H, lbls, idx_train, target_node, n_perturbations=n_perturbations, m=3)
        running_time = running_time + model.running_time
        igH = model.modified_adj
        igG = hgut._generate_G_from_H(igH, variable_weight=False)
        igG = torch.Tensor(igG).to(device)
        for i in range(0,10):
          randomresult = target_model(fts, igG)
          if (torch.argmax(result[target_node]) != torch.argmax(randomresult[target_node])):
              suc_attack = suc_attack + 1
    iga_suc.append(suc_attack / (num * 10.0))
    iga_time.append(running_time)

print('The ASR of FGIG in' + datasets + ':')
print(iga_suc)
print(iga_time)

# randomAttack 1-10
rdatk_suc = []
rdatk_time = []
for n_perturbations in range(2, 11):
    suc_attack = 0
    running_time = 0
    for index, target_node in enumerate(points):
        rdatkH, rt = randomAttack(H, target_node, n_perturbations)
        running_time = running_time + rt
        rdatkG = hgut._generate_G_from_H(rdatkH, variable_weight=False)
        rdatkG = torch.Tensor(rdatkG).to(device)
        for i in range(0,10):
          randomresult = target_model(fts, rdatkG)
          if (torch.argmax(result[target_node]) != torch.argmax(randomresult[target_node])):
              suc_attack = suc_attack + 1
    rdatk_suc.append(suc_attack / (num * 10.0))
    rdatk_time.append(running_time)

print('The ASR of RandomAttack in' + datasets + ':')
print(rdatk_suc)
print(rdatk_time)

# iga 1-10
iga_suc = []
iga_time = []
for n_perturbations in range(2, 11):
    suc_attack = 0
    running_time = 0
    for index, target_node in enumerate(points):
        model = IGattack(model=target_model, nnodes=fts.shape[0], attack_structure=True, attack_features=True,
                        device=device)
        model.attack(target_model, fts, H, lbls, idx_train, target_node, n_perturbations=n_perturbations)
        running_time = running_time + model.running_time       
        igH = model.modified_adj
        igG = hgut._generate_G_from_H(igH, variable_weight=False)
        igG = torch.Tensor(igG).to(device)
        for i in range(0,10):
          randomresult = target_model(fts, igG)
          if (torch.argmax(result[target_node]) != torch.argmax(randomresult[target_node])):
              suc_attack = suc_attack + 1
    iga_suc.append(suc_attack / (num * 10.0))
    iga_time.append(running_time)

print('The ASR of IGA in' + datasets + ':')
print(iga_suc)
print(iga_time)
