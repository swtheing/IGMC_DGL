import torch
import numpy as np
import sys, copy, math, time, pdb, warnings, traceback
import pickle
import scipy.io as sio
import scipy.sparse as sp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from train_eval_test import *
from model import *

import datetime, time
import traceback
import warnings
import sys
import dgl
torch.set_printoptions(profile="full")

def gen_data(file, last_num = 10):
    user_dic = {}
    item_dic = {}
    ucount = 0
    icount = 0
    user_list = []
    item_list = []
    user_item_dic = {}
    data = []
    pos = []
    del_id = []
    user_set = []
    f = open(file, "r")
    for line in f.readlines():
        groups = line.strip().split("\t")
        if groups[2] not in user_dic:
            user_dic[groups[2]] = ucount
            user_item_dic[ucount] = []
            ucount += 1
        if groups[1] not in item_dic:
            item_dic[groups[1]] = icount
            icount += 1
        if len(user_item_dic[user_dic[groups[2]]]) == last_num:
            id = user_item_dic[user_dic[groups[2]]].pop(0)
            del_id.append(id)
        
        user_item_dic[user_dic[groups[2]]].append(len(user_list))
        user_list.append(user_dic[groups[2]])
        item_list.append(item_dic[groups[1]])
        data.append(1)
    del_id.sort(reverse = True)
    for i in range(len(del_id)):
        id = del_id[i]
        user = user_list.pop(id)
        item = item_list.pop(id)
        data.pop(id)
    user_count = {}
    return user_list, item_list, data, user_dic, item_dic

def gen_test_data(file, user_dic, item_dic, i_c = 500):
    f = open(file, "r")
    user = []
    item = []
    item_list = []
    for line in f.readlines():
        groups = line.strip().split("\t")
        if groups[2] not in user_dic or groups[2] in user:
            continue
        if groups[1] not in item_dic:
            continue
        user.append(groups[2])
        item.append(groups[1])
        item_ar = np.array(range(len(item_dic)))
        neg_item = list(np.random.choice(item_ar, i_c, replace = False))
        neg_item = [list(item_dic.keys())[i] for i in neg_item]
        if groups[1] not in neg_item:
            neg_item[0] = groups[1]
        item_list.append(neg_item)
    return user, item, item_list

def load_test_file_rank(user, item, data, user_dic, item_dic, u, v, v_list):
    num_train_user = len(user_dic)
    num_train_item = len(item_dic)
    test_u = []
    test_v = []
    test_data = []
    for i in range(len(u)):
        test_u_item = []
        test_v_item = []
        data_item = []
        for it in v_list[i]:
            test_u_item.append(user_dic[u[i]])
            test_v_item.append(item_dic[it])
            data_item.append(0)
        index = v_list[i].index(v[i])
        data_item[index] = 1
        test_u += test_u_item
        test_v += test_v_item
        test_data += data_item
    O_train = sp.csr_matrix((np.array(data), (user, item)), shape=(num_train_user, num_train_item))     
    return O_train, data, user, item, test_data, test_u, test_v

if __name__ == "__main__":
    user, item, data, user_dic, item_dic = gen_data("./raw_data/group/douban_adj")
    print(len(user_dic))
    u, v, v_list = gen_test_data("./raw_data/group/douban_test_l", user_dic, item_dic) 
    print("user test number:" + str(len(u)))
    adj_train, train_data, train_user, train_item, test_data, test_user, test_item = load_test_file_rank(user, item, data, user_dic, item_dic, u, v, v_list)
    train_item = np.array(train_item)
    train_item += len(user_dic)
    adj_trains = [train_user, list(train_item), train_data]
    test_item = np.array(test_item)
    test_item += len(user_dic)
    train_indices = (train_user, list(train_item))
    test_indices = (test_user, test_item)
    class_values = np.array([0, 1])
    G = build_all_graph(adj_trains, class_values)
    starttime_1 = datetime.datetime.now()
    test_graphs = LocalGraphDataset(G, test_indices, test_data, class_values, testing = True)
    starttime_2 = datetime.datetime.now()
    print("test graph construct time:", (starttime_2 - starttime_1).seconds)
    model = IGMC(1*2+2,
             latent_dim = [32, 32, 32, 32],
             num_rels = 2,
             num_bases = 4)
    model_pos = "results/group_testmode/model_checkpoint36.pth"
    model.load_state_dict(torch.load(model_pos, map_location='cpu'))
    starttime_3 = datetime.datetime.now()
    rmse = test_once(test_graphs, model, test_indices, 500, logger=None)
    starttime_4 = datetime.datetime.now()
    print("test graph time:", (starttime_4 - starttime_3).seconds)

