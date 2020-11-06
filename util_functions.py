import dgl
import numpy as np
import networkx as nx             #导入networkx包
import matplotlib.pyplot as plt
import torch
import random
import os
import time
import torch.nn.functional as F
import dgl.function as fn
import multiprocessing as mp
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import Counter

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def get_norm(edges):
    indexs_select = torch.zeros_like(edges.data['rel_mod'])
    indexs = (torch.arange(0, edges.data['rel_mod'].shape[0]).long(), edges.data["rel_type"].long().reshape(-1))
    indexs_select.index_put_(indexs, torch.ones(edges.data['rel_type'].shape[0]))
    node_norm = 1.0 / torch.sum(edges.dst['node_norm'] * indexs_select, -1)
    return node_norm

def node_norm_to_edge_norm(g):
    g = g.local_var()
    g.apply_edges(lambda edges : {'norm' : get_norm(edges)})
    return g.edata['norm']

def message_func(edges):
    indexs = (torch.arange(0, edges.data['rel_mod'].shape[0]).long(), edges.data['rel_type'].long().reshape(-1))
    edges.data['rel_mod'].index_put_(indexs, torch.ones(edges.data['rel_type'].shape[0]))
    return {'msg': edges.data['rel_mod']}

def apply_func(nodes):
    node_norm = nodes.data['o']
    return {'node_norm': node_norm}

def build_all_graph(adj_train, num_user, class_values):
    src = np.array(adj_train[0])
    dst = np.array(adj_train[1]) + num_user
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    G = dgl.DGLGraph((u, v))
    G.edata['rel_type'] = torch.tensor(np.concatenate([adj_train[2], adj_train[2]])).long()
    '''
    G.edata['rel_mod'] = torch.zeros(G.edata['rel_type'].shape[0], len(class_values))
    G.update_all(message_func, fn.sum(msg='msg', out='o'), apply_func)
    G.edata['norm'] = node_norm_to_edge_norm(G).reshape(-1,1)
    '''
    return G



class LocalGraphDataset(object):
    def __init__(self, G, links, labels, class_values, num_user, dataset = None, parallel = False, pre_save = False, testing = False):
        self.G = G
        self.links = links
        self.data_len = len(self.links[0])
        self.labels = labels
        self.num_user = num_user
        self.testing = testing
        self.count = 0
        self.class_values = class_values
        self.pre_save = pre_save
        self.all_indexs = torch.arange(self.data_len)
        self.parallel = parallel
        self.dataset = dataset
        if self.pre_save:
            self.g_lists, self.labels = self.load_subgraphs(dataset)

    def len(self):
        return self.data_len

    def get_graph_tool(self, indexs):
        g_list = []
        labels = []
        index_out = []
        for index in indexs:
            g = self.extract_graph(self.G, self.links[0][index], self.links[1][index])
            index_out.append([self.links[0][index], self.links[1][index]])
            label = self.class_values[self.labels[index]]
            g_list.append(g)
            labels.append(label)
        return g_list, torch.FloatTensor(labels)
    
    def get_graph_tool_save(self, indexs):
        g_list = []
        labels = []
        index_out = []
        S = []
        if not self.parallel:
            pbar = tqdm(range(len(indexs)))
            for index in pbar:
                #g = self.extract_graph_new(self.G, self.links[0][index], self.links[1][index])
                #print(g.edges(form='all', order='srcdst')[0].shape)
                
                g = self.extract_graph(self.G, self.links[0][index], self.links[1][index])
                #print(g.edges(form='all', order='srcdst')[0].shape)
                #dd
                index_out.append([self.links[0][index], self.links[1][index]])
                #degree_l.append(g.number_of_edges())
                label = self.class_values[self.labels[index]]
                g_list.append(g)
                labels.append(label)
            return g_list, torch.FloatTensor(labels)
        else:
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(
                self.extract_graph,
                [
                    (self.G, self.links[0][index], self.links[1][index])
                    for index in indexs
                ]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            while results:
                tmp = results.pop()
                g_list.append(*tmp)
            labels = self.class_values[self.labels[indexs]]
            return g_list, torch.FloatTensor(labels) 

    def get_graphs(self, indexs):
        if not self.pre_save:
            g_list, labels = self.get_graph_tool(indexs)
        else:
            g_list, labels = self.g_lists[indexs], self.labels[indexs]
        return dgl.batch(g_list), labels

    def extract_graph_n(self, G, u_id, v_id):
        subg = dgl.sampling.sample_neighbors(G, [u_id, v_id], 1)
        subg.ndata['node_label'] = torch.zeros([subg.num_nodes(), 4])
        pid = subg.ndata[dgl.NID]
        for i in range(pid.shape[0]):
            if pid[i] == u_id:
                e_u = i
                subg.ndata['node_label'][i, 0] = 1
            elif pid[i] == v_id:
                e_v = i
                subg.ndata['node_label'][i, 1] = 1
            elif pid[i] in u:
                subg.ndata['node_label'][i, 2] = 1
            elif pid[i] in v:
                subg.ndata['node_label'][i, 3] = 1
        if subg.has_edges_between(e_u, e_v):
            e_ids = subg.edge_ids([e_u, e_v], [e_v, e_u])
            subg.remove_edges(e_ids)
        return subg

    def save_subgraphs(self):
        if self.testing:
            file_path = "./data/" + self.dataset + "/test.bin"
            if os.path.exists(file_path):
                return 
        else:
            file_path = "./data/" + self.dataset + "/train.bin"
            if os.path.exists(file_path):
                return 
        g_list, labels = self.get_graph_tool_save(self.all_indexs)
        graph_labels = {"glabel": labels}
        save_graphs(file_path, g_list, graph_labels)

    def load_subgraphs(self):
        if self.testing:
            g_list, label_dict = load_graphs("./data/" + self.dataset + "/test/")
        else:
            g_list, label_dict = load_graphs("./data/" + self.dataset + "/train/")
        return g_list, label_dict["glabel"]

    def extract_graph_new(self, G, u_id, v_id):
        v_id += self.num_user
        static_u = torch.zeros(len(self.class_values))
        static_v = torch.zeros(len(self.class_values))
        start0 = time.time()
        u_nodes, v, e_ids_1 = G.in_edges(v_id, "all")
        u, v_nodes, e_ids_2 = G.out_edges(u_id, "all")
        e_ids = []
        nodes = torch.cat([u_nodes, v_nodes])
        for i in range(u_nodes.shape[0]):
            if u_nodes[i] == u_id:
                e_ids.append(e_ids_1[i])
        for i in range(v_nodes.shape[0]):
            if v_nodes[i] == v_id:
                e_ids.append(e_ids_2[i])
        #start1 = time.time()
        #print(start1 - start0)
        subg = dgl.node_subgraph(G, nodes)
        #start2 = time.time()
        #print(start2 - start1)
        subg.ndata['node_label'] = torch.zeros([subg.num_nodes(), 4])
        pid = subg.ndata[dgl.NID]
        #start3 = time.time()
        #print(start3 - start2)
        for i in range(pid.shape[0]):
            if pid[i] == u_id:
                e_u = i
                subg.ndata['node_label'][i, 0] = 1
            elif pid[i] == v_id:
                e_v = i
                subg.ndata['node_label'][i, 1] = 1
            elif pid[i] in u:
                subg.ndata['node_label'][i, 2] = 1
            elif pid[i] in v:
                subg.ndata['node_label'][i, 3] = 1
        subg = dgl.remove_edges(subg, e_ids)
        start6 = time.time()
        print(start6 - start0)
        print()
        return subg

    def extract_graph(self, G, u_id, v_id):
        v_id += self.num_user
        static_u = torch.zeros(len(self.class_values))
        static_v = torch.zeros(len(self.class_values))
        #start0 = time.time()
        u_nodes, v, e_ids = G.out_edges(u_id, "all")
        u, v_nodes, e_ids = G.in_edges(v_id, "all")
        nodes = torch.cat([u, v])
        if self.testing:
            nodes = torch.cat([nodes, torch.tensor([u_id, v_id])])
        #start1 = time.time()
        #print(start1 - start0)
        subg = G.subgraph(nodes)
        #start2 = time.time()
        #print(start2 - start1)
        subg.ndata['node_label'] = torch.zeros([nodes.shape[0], 4])
        pid = subg.ndata[dgl.NID]
        #start3 = time.time()
        #print(start3 - start2)
        for i in range(pid.shape[0]):
            if pid[i] == u_id:
                e_u = i
                subg.ndata['node_label'][i, 0] = 1
            elif pid[i] == v_id:
                e_v = i
                subg.ndata['node_label'][i, 1] = 1
            elif pid[i] in u:
                subg.ndata['node_label'][i, 2] = 1
            elif pid[i] in v:
                subg.ndata['node_label'][i, 3] = 1
        #start4 = time.time()
        #print(start4 - start3)
        if not self.testing:
            e_ids = subg.edge_ids([e_u, e_v], [e_v, e_u])
            #start5 = time.time()
            #print(start5 - start4)
            subg = dgl.remove_edges(subg, e_ids)
        #start6 = time.time()
        #print(start6 - start0)
        #print()
        return subg
  

if __name__=='__main__':
    adj_train = ([0, 1, 1, 1], [3, 0, 1, 2], [7, 1, 1, 9])
    num_user = 2
    class_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    G = build_all_graph(adj_train, num_user, class_values)
    print(G.edata['rel_type'])
    print(G.edges(form='all', order='srcdst'))
    print(G.edata['norm'])
