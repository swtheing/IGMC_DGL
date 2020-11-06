import torch
import numpy as np
import sys, copy, math, time, pdb, warnings, traceback
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from preprocessing import *
from train_eval import *
from model import *

import traceback
import warnings
import sys

parser = argparse.ArgumentParser(description='Inductive Graph-based Matrix Completion')
parser.add_argument('--testing', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
parser.add_argument('--no-train', action='store_true', default=False,
                    help='if set, skip the training and directly perform the \
                    transfer/ensemble/visualization')
parser.add_argument('--debug', action='store_true', default=False,
                    help='turn on debugging mode which uses a small number of data')
parser.add_argument('--data-name', default='ml_100k', help='dataset name')
parser.add_argument('--own', default=False, help='own dataset')
parser.add_argument('--data-appendix', default='',
                    help='what to append to save-names when saving datasets')
parser.add_argument('--save-appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--max-train-num', type=int, default=None,
                    help='set maximum number of train data to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                    help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                    valid only for ml_1m and ml_10m')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--dynamic-dataset', action='store_true', default=False,
                    help='if True, extract enclosing subgraphs on the fly instead of \
                    storing in disk; works for large datasets that cannot fit into memory')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--save-interval', type=int, default=10,
                    help='save model states every # epochs ')
# subgraph extraction settings
parser.add_argument('--hop', default=1, metavar='S',
                    help='enclosing subgraph hop number')
parser.add_argument('--sample-ratio', type=float, default=1.0,
                    help='if < 1, subsample nodes per hop according to the ratio')
parser.add_argument('--max-nodes-per-hop', default=10000,
                    help='if > 0, upper bound the # nodes per hop by another subsampling')
parser.add_argument('--use-features', action='store_true', default=False,
                    help='whether to use node features (side information)')
# edge dropout settings
parser.add_argument('--adj-dropout', type=float, default=0.2,
                    help='if not 0, random drops edges from adjacency matrix with this prob')
parser.add_argument('--force-undirected', action='store_true', default=False,
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
# optimization settings
parser.add_argument('--continue-from', type=int, default=None,
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-decay-step-size', type=int, default=50,
                    help='decay lr by factor A every B steps')
parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                    help='decay lr by factor A every B steps')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='batch size during training')
parser.add_argument('--ARR', type=float, default=0.001,
                    help='The adjacenct rating regularizer. If not 0, regularize the \
                    differences between graph convolution parameters W associated with\
                    adjacent ratings')
# transfer learning, ensemble, and visualization settings
parser.add_argument('--transfer', default='',
                    help='if not empty, load the pretrained models in this path')
parser.add_argument('--num-relations', type=int, default=5,
                    help='if transfer, specify num_relations in the transferred model')
parser.add_argument('--multiply-by', type=int, default=1,
                    help='if transfer, specify how many times to multiply the predictions by')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='if True, load a pretrained model and do visualization exps')
parser.add_argument('--ensemble', action='store_true', default=False,
                    help='if True, load a series of model checkpoints and ensemble the results')
parser.add_argument('--standard-rating', action='store_true', default=False,
                    help='if True, maps all ratings to standard 1, 2, 3, 4, 5 before training')
# sparsity experiment settings
parser.add_argument('--ratio', type=float, default=1.0,
                    help="For ml datasets, if ratio < 1, downsample training data to the\
                    target ratio")
args = parser.parse_args()
torch.manual_seed(args.seed)
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
args.res_dir = os.path.join(args.file_dir, 'results/{}{}_{}'.format(args.data_name, args.save_appendix, val_test_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
rating_map = None
post_rating_map = None
args.data_name = 'flixster'
if args.data_name in ['flixster', 'douban', 'yahoo_music']:
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values, num_user
    ) = load_data_monti(args.data_name, args.testing, rating_map, post_rating_map)
elif args.data_name == 'ml_100k':
    print("Using official MovieLens split u1.base/u1.test with 20% validation...")
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values, num_user
    ) = load_official_trainvaltest_split(
        args.data_name, args.testing, rating_map, post_rating_map, args.ratio
    )
else:
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values, num_user
    ) = create_trainvaltest_split(
        args.data_name, 1234, args.testing, datasplit_path, True, True, rating_map, 
        post_rating_map, args.ratio
    )

adj_trains = [train_u_indices, train_v_indices, train_labels]
train_indices = (train_u_indices, train_v_indices)
test_indices = (test_u_indices, test_v_indices)

G = build_all_graph(adj_trains, num_user, class_values)
train_save_graph = LocalGraphDataset(G, train_indices, train_labels, class_values, num_user, pre_save = False, testing = False, dataset = args.data_name, parallel = False)
train_save_graph.save_subgraphs()
test_save_graph = LocalGraphDataset(G, test_indices, test_labels, class_values, num_user, pre_save = False, testing = True, dataset = args.data_name, parallel = False)
test_save_graph.save_subgraphs()
train_graphs = LocalGraphDataset(G, train_indices, train_labels, class_values, num_user, testing = False, dataset = args.data_name)
test_graphs = LocalGraphDataset(G, test_indices, test_labels, class_values, num_user, testing = True, dataset = args.data_name)

num_rels = len(class_values)
model = IGMC(args.hop*2+2,
             latent_dim = [32, 32, 32, 32],
             num_rels = num_rels,
             num_bases = 4)

def logger(info, model, optimizer, best_mrr = 0.0):
    epoch, train_loss, test_rmse, test_mrr = info['epoch'], info['train_loss'], info['test_rmse'], info['test_mrr']
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse
            ))
    if type(epoch) == int and best_mrr < test_mrr:
        best_mrr = test_mrr
        print('Saving model states...')
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        print(model.lin2.weight.shape)
        if model is not None:
            torch.save(model.state_dict(), model_name)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_name)
    return best_mrr

train_multiple_epochs(train_graphs,
                      test_graphs,
                      model,
                      args.epochs,
                      args.batch_size,
                      args.lr,
                      lr_decay_factor=args.lr_decay_factor,
                      lr_decay_step_size=args.lr_decay_step_size,
                      weight_decay=0,
                      ARR = args.ARR,
                      logger = logger,
                      continue_from=args.continue_from,
                      res_dir=args.res_dir)


