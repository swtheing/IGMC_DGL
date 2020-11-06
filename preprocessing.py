from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import h5py
import pandas as pd
import pdb
import random
from scipy.sparse import linalg
from data_utils import load_data, map_data, download_dataset
from sklearn.metrics import mean_squared_error
from math import sqrt
def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm

def load_own_file(path_file, filter_num = 10):
    user_dic = {}
    item_dic = {}
    user_item_dic = {}
    user_id = 0
    item_id = 0
    user_select_id = 0
    item_select_id = 0
    user_select = {}
    item_select = {}
    index = []
    indptr = []
    data = []
    for line in open(path_file):
        groups = line.strip().split("\t")
        if groups[3] == "NULL" or groups[3] == '':
            continue
        if groups[0] not in user_dic:
            user_dic[groups[0]] = [user_id, 1]
            user_id += 1
        else:
            user_dic[groups[0]][1] += 1
        if groups[1] not in item_dic:
            item_dic[groups[1]] = [item_id, 1]
            item_id += 1
        else:
            item_dic[groups[1]][1] += 1
        if groups[1] not in item_select and item_dic[groups[1]][1] > filter_num:
            item_select[groups[1]] = item_select_id
            item_select_id += 1
        index.append(groups[0])
        indptr.append(groups[1])
        data.append(float(groups[3]))
    for i in range(len(index)):
        user_id = index[i]
        item_id = indptr[i]
        if item_id in item_select and user_id not in user_select:
            user_select[user_id] = user_select_id
            user_select_id += 1
            
    Train_index = []
    Train_indptr = []
    Train_data = []
    Test_index = []
    Test_indptr = []
    Test_data = []
    All_index = []
    All_indptr = []
    All_data = []
    #filter
    for i in range(len(index)):
        user_id = index[i]
        item_id = indptr[i]
        if user_id in user_select and item_id in item_select:
            All_index.append(user_select[user_id])
            All_indptr.append(item_select[item_id])
            All_data.append(data[i])
            if np.random.randint(0,100) < 20:
                Train_index.append(user_select[user_id])
                Train_indptr.append(item_select[item_id])
                Train_data.append(1)
            else:
                Test_index.append(user_select[user_id])
                Test_indptr.append(item_select[item_id])
                Test_data.append(1)
    M = sp.csr_matrix((All_data, (All_index, All_indptr)), shape=(len(user_select), len(item_select)))
    O_train = sp.csr_matrix((Train_data, (Train_index, Train_indptr)), shape=(len(user_select), len(item_select)))
    O_test = sp.csr_matrix((Test_data, (Test_index, Test_indptr)), shape=(len(user_select), len(item_select))) 
    
    return None, M.toarray(), O_train.toarray(), O_test.toarray(), user_select, item_select

def load_own_file2(path_file, filter_num = 0):
    user_dic = {}
    item_dic = {}
    user_item_dic = {}
    user_id = 0
    item_id = 0
    user_select_id = 0
    item_select_id = 0
    user_select = {}
    item_select = {}
    index = []
    indptr = []
    data = []
    data_c = 0
    data_dic = {}
    most = 0
    for line in open(path_file):
        groups = line.strip().split("\t")
        if groups[3] == "NULL" or groups[3] == '':
            continue
        if groups[0] not in user_dic:
            user_dic[groups[0]] = [user_id, 1]
            user_id += 1
        else:
            user_dic[groups[0]][1] += 1
        if groups[1] not in item_dic:
            item_dic[groups[1]] = [item_id, 1]
            item_id += 1
        else:
            item_dic[groups[1]][1] += 1
            if item_dic[groups[1]][1] > most:
                most = item_dic[groups[1]][1]
        if groups[1] not in item_select and item_dic[groups[1]][1] > filter_num:
            item_select[groups[1]] = item_select_id
            item_select_id += 1
        index.append(groups[0])
        indptr.append(groups[1])
        if groups[0] + "\t" + groups[1] in data_dic:
            last_id = data_dic[groups[0] + "\t" + groups[1]]
            data[last_id] = -1.0
        data_dic[groups[0] + "\t" + groups[1]] = data_c
        data.append(float(groups[3]) - 1)
        data_c += 1
    print("most:", most)
    
    for i in range(len(index)):
        user_id = index[i]
        item_id = indptr[i]
        if item_id in item_select and user_id not in user_select:
            user_select[user_id] = user_select_id
            user_select_id += 1
            
    Train_index = []
    Train_indptr = []
    Train_data = []
    Val_index = []
    Val_indptr = []
    Val_data = []
    Test_index = []
    Test_indptr = []
    Test_data = []
    All_index = []
    All_indptr = []
    All_data = []
    #filter
    Train_item = range(0,1000)
    Train_user = range(0,600000)
    Val_item = range(1000,1050)
    Val_user = range(0, 600000)
    Test_item = range(1050, 1100)
    Test_user = range(0, 600000)
    for i in range(len(index)):
        user_id = index[i]
        item_id = indptr[i]
        if user_id in user_select and item_id in item_select:
            '''
            All_index.append(user_select[user_id])
            All_indptr.append(item_select[item_id])
            All_data.append(data[i])
            if user_select[user_id] in Train_user and item_select[item_id] in Train_item:
                Train_index.append(user_select[user_id])
                Train_indptr.append(item_select[item_id])
                Train_data.append(data[i])
            elif user_select[user_id] in Test_user and item_select[item_id] in Test_item:
                Test_index.append(user_select[user_id])
                Test_indptr.append(item_select[item_id])
                Test_data.append(data[i])
            elif user_select[user_id] in Val_user and item_select[item_id] in Val_item:
                Val_index.append(user_select[user_id])
                Val_indptr.append(item_select[item_id])
                Val_data.append(data[i])

            '''
            rdn = np.random.randint(0,100)
            if rdn < 80:
                Train_index.append(user_select[user_id])
                Train_indptr.append(item_select[item_id])
                Train_data.append(data[i])
            elif rdn < 90:
                Test_index.append(user_select[user_id])
                Test_indptr.append(item_select[item_id])
                Test_data.append(data[i])
            else:
                Val_index.append(user_select[user_id])
                Val_indptr.append(item_select[item_id])
                Val_data.append(data[i])
    num_train_item = len(item_select)
    num_train_user = len(user_select)
    #num_train_item = len(Train_item) + len(Test_item) + len(Val_item)
    #num_train_user = len(Train_user) + len(Test_user) + len(Val_user)
    O_train = sp.csr_matrix((np.array(Train_data) + 1, (Train_index, Train_indptr)), shape=(num_train_user, num_train_item))
    return O_train, Train_index, Train_indptr, Train_data, Val_index, Val_indptr, Val_data, Test_index,Test_indptr, Test_data, user_select, item_select

def gen_data(file):
    user_dic = {}
    item_dic = {}
    ucount = 0
    icount = 0
    user_list = []
    item_list = []
    data = []
    pos = []
    f = open(file, "r")
    for line in f.readlines():
        groups = line.strip().split("\t")
        if groups[2] not in user_dic:
            user_dic[groups[2]] = ucount
            ucount += 1
        if groups[1] not in item_dic:
            item_dic[groups[1]] = icount
            icount += 1
        pos.append(str(user_dic[groups[2]]) + '-' + str(item_dic[groups[1]]))
        user_list.append(user_dic[groups[2]])
        item_list.append(item_dic[groups[1]])
        data.append(1)
    #negtive sampling
    d_l = len(data) * 10
    for i in range(d_l):
        user_id = user_list[0]
        item_id = item_list[0]
        neg = str(user_id) + '-' + str(item_id)
        while neg in pos:
            user_id = random.randint(0, ucount-1)
            item_id = random.randint(0, icount-1)
            neg = str(user_id) + '-' + str(item_id)
        pos.append(neg)
        user_list.append(user_id)
        item_list.append(item_id)
        data.append(0)
    return user_list, item_list, data, user_dic, item_dic

def load_group_file_rank(path_file):
    user, item, data, user_dic, item_dic = gen_data(path_file)
    item = np.array(item)
    train_data = []
    train_user = []
    train_item = []
    val_data = []
    val_user = []
    val_item = []
    test_data = []
    test_user = []
    test_item = []
    test_dic = {}
    degree = {}
    for i in range(len(user)):
        rdn = random.randint(0, 100)
        if rdn > 20:
            train_data.append(data[i])
            train_user.append(user[i])
            train_item.append(item[i]) 
        elif rdn > 10:
            test_data.append(data[i])
            test_user.append(user[i])
            test_item.append(item[i])
            if user[i] in test_dic:
                test_dic[user[i]].append(item[i])
            else:
                test_dic[user[i]] = [item[i]]
        else:
            val_data.append(data[i])
            val_user.append(user[i])
            val_item.append(item[i])
    #add test rank
    debug = {}
    for user in set(test_user):
        item_list = set(train_item) - set(test_dic[user])
        debug[user] = []
        sample_item = np.random.choice(list(item_list), 500, replace=False)
        for item in sample_item:
            test_data.append(0)
            test_user.append(user)
            test_item.append(item)
            debug[user].append(item)
    #print(debug)
    num_train_user = len(user_dic)
    num_train_item = len(item_dic)
    O_train = sp.csr_matrix((np.array(train_data), (train_user, train_item)), shape=(num_train_user, num_train_item))
    return O_train, train_user, train_item, train_data, val_user, val_item, val_data, test_user, test_item, test_data, user_dic, item_dic

def load_group_file(path_file):
    user, item, data, user_dic, item_dic = gen_data(path_file)
    item = np.array(item)
    train_data = []
    train_user = []
    train_item = []
    val_data = []
    val_user = []
    val_item = []
    test_data = []
    test_user = []
    test_item = []
    degree = {}
    for i in range(len(user)):
        rdn = random.randint(0, 100)
        if rdn > 20:
            train_data.append(data[i])
            train_user.append(user[i])
            train_item.append(item[i]) 
        elif rdn > 10:
            test_data.append(data[i])
            test_user.append(user[i])
            test_item.append(item[i])
        else:
            val_data.append(data[i])
            val_user.append(user[i])
            val_item.append(item[i])
    num_train_user = len(user_dic)
    num_train_item = len(item_dic)
    O_train = sp.csr_matrix((np.array(train_data), (train_user, train_item)), shape=(num_train_user, num_train_item))
    return O_train, train_user, train_item, train_data, val_user, val_item, val_data, test_user, test_item, test_data, user_dic, item_dic

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csr_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False, verbose=True, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
    load_data function.
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix.
    """

    if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path, 'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

        if verbose:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(dataset, seed=seed,
                                                                                            verbose=verbose)

        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    neutral_rating = -1

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    if dataset == 'ml_100k':
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    else:
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))

    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])

    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:int(num_train*ratio)]
    val_idx = idx_nonzero[num_train:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    train_pairs_idx = pairs_nonzero[0:int(num_train*ratio)]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values

def load_data_monti_filter(dataset, testing=False, rating_map=None, post_rating_map=None, own = False):
    """
    Loads data from Monti et al. paper.
    if rating_map is given, apply this map to the original rating matrix
    if post_rating_map is given, apply this map to the processed rating_mx_train without affecting the labels
    """

    if not own:
        path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'
        M = load_matlab_file(path_dataset, 'M')
        if rating_map is not None:
            M[np.where(M)] = [rating_map[x] for x in M[np.where(M)]]
        print(M.shape)
        Otraining = load_matlab_file(path_dataset, 'Otraining')
        # filter of dataset
        user_median = np.median(np.sum(Otraining, axis = 0))
        item_median = np.median(np.sum(Otraining, axis = 1))
        user_id = np.arange(Otraining.shape[0]) + 1
        item_id = np.arange(Otraining.shape[1]) + 1
        keep_user = np.where(np.sum(Otraining, axis = 0) < user_median * 1 + 1, user_id, np.zeros_like(user_id))
        keep_item = np.where(np.sum(Otraining, axis = 1) < item_median * 1 + 1, item_id, np.zeros_like(item_id))
        keep_user = keep_user[ keep_user != 0]
        keep_item = keep_item[keep_item != 0]
        '''
        Otraining = Otraining[:, keep_user - 1]
        Otraining = Otraining[keep_item - 1, :]
        user_median = np.min(np.sum(Otraining, axis = 0))
        item_median = np.min(np.sum(Otraining, axis = 1))
        '''
        print(user_median * 3)
        print(item_median * 3)
        Otest = load_matlab_file(path_dataset, 'Otest')
        '''
        Otest = Otest[:, keep_user - 1]
        Otest = Otest[keep_item - 1, :]
        #M = M[:, keep_user - 1]
        #M = M[keep_item - 1, :]
        Otest[:, keep_user - 1] = 0
        Otest[keep_item - 1, :] = 0
        num_users = Otraining.shape[0]
        num_items = Otraining.shape[1]
        '''
    else:
        path_dataset = 'raw_data/' + dataset + '/douban_test'
    print(path_dataset)
    if dataset == 'flixster':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        Wcol = load_matlab_file(path_dataset, 'W_movies')
        u_features = Wrow
        v_features = Wcol
    elif dataset == 'douban':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        #u_features = Wrow
        #v_features = np.eye(num_items)
    elif dataset == 'yahoo_music':
        Wcol = load_matlab_file(path_dataset, 'W_tracks')
        u_features = np.eye(num_users)
        v_features = Wcol
    elif dataset == 'own' or dataset == 'all':
        u_features = None
        v_features = None
        rating_train, Train_index, Train_indptr, Train_data, Val_index, Val_indptr, Val_data, Test_index,Test_indptr, Test_data, user_dic, item_dic = load_own_file2(path_dataset)
        Train_data = np.array(Train_data, dtype = np.int32)
        Train_index = np.array(Train_index)
        Train_indptr = np.array(Train_indptr)
        Val_data = np.array(Val_data, dtype = np.int32)
        Val_index = np.array(Val_index)
        Val_indptr = np.array(Val_indptr)
        Test_data = np.array(Test_data, dtype = np.int32)
        Test_index = np.array(Test_index)
        Test_indptr = np.array(Test_indptr)
        class_values = np.array([1, 2, 3, 4, 5])
        print('number of users = ', len(user_dic))
        print('number of item = ', len(item_dic))
        print("train_labels:")
        print(Train_data)
        print("u_train_idx")
        print(Train_index)
        print("v_train_idx")
        print(Train_indptr)
        print("test_labels")
        print(Test_data)
        print("u_test_idx")
        print(Test_index)
        print("v_test_idx")
        print(Test_indptr)
        print("class_values")
        print(class_values)
        return u_features, v_features, rating_train, Train_data, Train_index, Train_indptr, \
            Val_data, Val_index, Val_indptr, Test_data, Test_index, Test_indptr, class_values

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]
    ratings = M[np.where(M)]
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))
    u_features = np.array(range(num_users)).reshape((-1, 1))
    v_features = np.array(range(num_users, num_users + num_items)).reshape(-1, 1)
    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    print(num_users)
    print(num_items)
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    '''Note here rating matrix elements' values + 1 !!!'''
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))
    if u_features is not None:
        print("user Features:")
        print(u_features)
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        print("Item Features")
        print(v_features)
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))
    print("train_labels:")
    print(train_labels)
    print("u_train_idx")
    print(u_train_idx)
    print("v_train_idx")
    print(v_train_idx)
    print("test_labels")
    print(test_labels)
    print("u_test_idx")
    print(u_test_idx)
    print("v_test_idx")
    print(v_test_idx)
    print("class_values")
    print(class_values)

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, num_users

def load_data_monti(dataset, testing=False, rating_map=None, post_rating_map=None, own = False):
    """
    Loads data from Monti et al. paper.
    if rating_map is given, apply this map to the original rating matrix
    if post_rating_map is given, apply this map to the processed rating_mx_train without affecting the labels
    """

    if not own:
        path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'
        M = load_matlab_file(path_dataset, 'M')
        if rating_map is not None:
            M[np.where(M)] = [rating_map[x] for x in M[np.where(M)]]
        print(M.shape)
        Otraining = load_matlab_file(path_dataset, 'Otraining')
        Otest = load_matlab_file(path_dataset, 'Otest')
        num_users = M.shape[0]
        num_items = M.shape[1]
    else:
        path_dataset = 'raw_data/' + dataset + '/douban_train'
    print(path_dataset)
    if dataset == 'flixster':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        Wcol = load_matlab_file(path_dataset, 'W_movies')
        u_features = Wrow
        v_features = Wcol
    elif dataset == 'douban':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        u_features = Wrow
        v_features = np.eye(num_items)
    elif dataset == 'yahoo_music':
        Wcol = load_matlab_file(path_dataset, 'W_tracks')
        u_features = np.eye(num_users)
        v_features = Wcol
    elif dataset == 'own' or dataset == 'all':
        u_features = None
        v_features = None
        rating_train, Train_index, Train_indptr, Train_data, Val_index, Val_indptr, Val_data, Test_index,Test_indptr, Test_data, user_dic, item_dic = load_own_file2(path_dataset)
        Train_indptr = list(np.array(Train_indptr) + len(user_dic))
        Val_indptr = list(np.array(Val_indptr) + len(user_dic))
        Test_indptr = list(np.array(Test_indptr) + len(user_dic))
        class_values = np.array([1, 2, 3, 4, 5])
        print('number of users = ', len(user_dic))
        print('number of item = ', len(item_dic))
        print("train_labels:")
        print(Train_data)
        print("u_train_idx")
        print(Train_index)
        print("v_train_idx")
        print(Train_indptr)
        print("test_labels")
        print(Test_data)
        print("u_test_idx")
        print(Test_index)
        print("v_test_idx")
        print(Test_indptr)
        print("class_values")
        print(class_values)
        return u_features, v_features, rating_train, Train_data, Train_index, Train_indptr, \
            Val_data, Val_index, Val_indptr, Test_data, Test_index, Test_indptr, class_values
    elif dataset == 'group':
        rating_train, Train_index, Train_indptr, Train_data, Val_index, Val_indptr, Val_data, Test_index,Test_indptr, Test_data, user_dic, item_dic = load_group_file_rank(path_dataset)
        u_features = range(len(user_dic))
        v_features = range(len(user_dic), len(item_dic)+len(user_dic))
        Train_indptr = list(np.array(Train_indptr) + len(user_dic))
        Val_indptr = list(np.array(Val_indptr) + len(user_dic))
        Test_indptr = list(np.array(Test_indptr) + len(user_dic))
        class_values = np.array([0, 1])
        print('number of users = ', len(user_dic))
        print('number of item = ', len(item_dic))
        return u_features, v_features, rating_train, Train_data, Train_index, Train_indptr, \
            Val_data, Val_index, Val_indptr, Test_data, Test_index, Test_indptr, class_values

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]
    print("u_nodes:")
    print(u_nodes_ratings)
    print("v_nodes:")
    print(v_nodes_ratings)
    ratings = M[np.where(M)]
    ''' 
    #Test SVD
    U, s, Vh = linalg.svds(Otraining)
    s_diag_matrix = np.diag(s)
    svd_prediction = np.dot(np.dot(U,s_diag_matrix),Vh)
    prediction_flatten = np.reshape(svd_prediction[Otest.nonzero()], (1,-1))
    test_data_matrix_flatten = Otest[Otest.nonzero()]
    rmse = sqrt(mean_squared_error(prediction_flatten,test_data_matrix_flatten))
    print("SVD rmse:", rmse)
    '''

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    '''Note here rating matrix elements' values + 1 !!!'''
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))


    if u_features is not None:
        print("user Features:")
        print(u_features)
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        print("Item Features")
        print(v_features)
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))
    print("train_labels:")
    print(train_labels)
    print("u_train_idx")
    print(u_train_idx)
    print("v_train_idx")
    print(v_train_idx)
    print("test_labels")
    print(test_labels)
    print("u_test_idx")
    print(u_test_idx)
    print("v_test_idx")
    print(v_test_idx)
    print("class_values")
    print(class_values)
    print(num_users)
    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, num_users


def load_official_trainvaltest_split(dataset, testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """

    sep = '\t'

    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'raw_data/' + fname

    download_dataset(fname, files, data_dir)

    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    filename_train = 'raw_data/' + dataset + '/u1.base'
    filename_test = 'raw_data/' + dataset + '/u1.test'

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    if ratio < 1.0:
        data_array_train = data_array_train[data_array_train[:, -1].argsort()[:int(ratio*len(data_array_train))]]

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train+num_val]
    idx_nonzero_test = idx_nonzero[num_train+num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
    pairs_nonzero_test = pairs_nonzero[num_train+num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])
    
    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if dataset =='ml_100k':

        # movie features (genres)
        sep = r'|'
        movie_file = 'raw_data/' + dataset + '/u.item'
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # user features

        sep = r'|'
        users_file = 'raw_data/' + dataset + '/u.user'
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        age = users_df['age'].values
        age_max = age.max()

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

    elif dataset == 'ml_1m':

        # load movie features
        movies_file = 'raw_data/' + dataset + '/movies.dat'

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # load user features
        users_file = 'raw_data/' + dataset + '/users.dat'
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.
    else:
        raise ValueError('Invalid dataset option %s' % dataset)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, num_users
