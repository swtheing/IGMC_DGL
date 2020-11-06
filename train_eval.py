import time
import os
import math
import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train(model, optimizer, loader, device, regression = True, ARR = 0.0):
    model.train()
    total_loss = 0
    t_start = time.perf_counter()
    for graphs, labels in loader:
        t_end = time.perf_counter()
        optimizer.zero_grad()
        #print(graphs.edges(form='all', order='srcdst'))
        graphs = graphs.to(device)
        labels = labels.to(device)
        out = model(graphs)
        if regression:
            loss = F.mse_loss(out, labels.view(-1))
        else:
            loss = F.nll_loss(F.log_softmax(out, dim = -1), labels.view(-1).long())
        if ARR != 0.0:
            for gconv in model.convs:
                w = torch.matmul(
                    gconv.w_comp,
                    gconv.weight.view(gconv.num_bases, -1)
                ).view(gconv.num_rels, gconv.in_feat, gconv.out_feat)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(out)
        optimizer.step()
        torch.cuda.empty_cache()
        t_start = time.perf_counter()
    return total_loss / len(loader.dataset)

def num_graphs(out):
    return out.shape[0]

def take_first(elem):
    return elem[0]    

def eval_loss(model, loader, device, regression=False, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for graphs, labels in pbar:
        graphs = graphs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graphs)
        if regression:
            loss += F.mse_loss(out, labels.view(-1), reduction='sum').item()
        else:
            loss += F.nll_loss(out, labels.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    return loss / len(loader.dataset)

def eval_rmse(model, loader, device, show_progress=False):
    mse_loss = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse

def eval_loss_dic(model, loader, device, regression=True, show_progress=False):
    model.eval()
    loss_dic = {}
    count_dic = {}
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for graphs, labels in pbar:
        graphs = graphs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graphs)
        if regression:
            loss = F.mse_loss(out, labels.view(-1), reduction='none')
        else:
            loss = F.nll_loss(out, labels.view(-1), reduction='none')
        y = degree.view(-1, 4).cpu().numpy().tolist()
        y2 = labels.view(-1).cpu().numpy().tolist()
        O = out.view(-1).cpu().numpy().tolist()
        l = loss.cpu().numpy().tolist()
        for i in range(len(y)):
            key = (y[i][0], y[i][1], y[i][2], y[i][3], y2[i], O[i])
            if key not in loss_dic:
                loss_dic[key] = l[i]
                count_dic[key] = 1
            else:
                loss_dic[key] += l[i]
                count_dic[key] += 1
        torch.cuda.empty_cache()
    for item in loss_dic.keys():
        loss_dic[item] /= count_dic[item]
    return loss_dic

def eval_rmse_dic(model, loader, device, show_progress=False):
    mse_loss_dic = eval_loss_dic(model, loader, device, True, show_progress)
    for item in mse_loss_dic.keys():
        mse_loss_dic[item] = math.sqrt(mse_loss_dic[item])
    mse_loss_dic = sorted(mse_loss_dic.items(), key= lambda d:d[1], reverse=False)
    return mse_loss_dic[-99:]

def eval_mrr(model, loader, device, test_pairs, regression=True, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    ans = {}
    y = []
    hit_5 = 0
    hit_1 = 0
    hit_10 = 0
    count = 0
    debug = {}
    for i in range(len(test_pairs[0])):
        if test_pairs[0][i] not in debug:
            debug[test_pairs[0][i]] = [test_pairs[1][i]]
        else:
            debug[test_pairs[0][i]].append(test_pairs[1][i])
    for key in debug.keys():
        print(len(debug[key]))
    for graphs, labels in pbar:
        graphs = graphs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graphs).cpu().numpy()
            if not regression:
                out = out[:,1]
            y = labels.cpu().numpy()
        for i in range(out.shape[0]):
            if test_pairs[0][count] in ans:
                ans[test_pairs[0][count]].append((out[i], y[i], test_pairs[1][count]))
            else:
                ans[test_pairs[0][count]] = [(out[i], y[i], test_pairs[1][count])]
            count += 1
        torch.cuda.empty_cache()
    c = []
    for key in ans.keys():
        ans[key].sort(key = take_first, reverse = True)
        print(str(key) + "_" + str(ans[key][0][2]))
        for i in range(len(ans[key])):
            if ans[key][i][1] == 1.0:
                c.append(1.0 / (i+1))
                if i < 1:
                    hit_1 += 1.0
                elif i < 5:
                    hit_5 += 1.0
                elif i < 10:
                    hit_10 += 1.0
    print("hit_1: " + str(hit_1 / len(c)))
    print("hit_5: " + str(hit_5 / len(c)))
    print("hit_10: " + str(hit_10 / len(c)))
    return sum(c) / len(c)

def train_multiple_epochs(train_dataset,
                          test_dataset,
                          model,
                          epochs,
                          batch_size,
                          lr,
                          lr_decay_factor,
                          lr_decay_step_size,
                          weight_decay,
                          ARR = 0.0,
                          logger=None,
                          continue_from=None,
                          res_dir=None,
                          regression=True
                          ):
    seeds = torch.arange(train_dataset.len())
    train_loader = DataLoader(seeds, batch_size, collate_fn=train_dataset.get_graphs, shuffle=True, num_workers = mp.cpu_count())
    test_seeds = torch.arange(test_dataset.len())
    test_loader = DataLoader(test_seeds, batch_size,  collate_fn=test_dataset.get_graphs, shuffle=False, num_workers=mp.cpu_count())
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 1
    rmses = []
    mrrs = []
    if continue_from is not None:
        model.load_state_dict(torch.load(os.path.join(res_dir, 'model_checkpoint{}.pth'.format(continue_from))))
        optimizer.load_state_dict(torch.load(os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(continue_from))))
        start_epoch = continue_from + 1
        epochs -= continue_from
    t_start = time.perf_counter()
    pbar = tqdm(range(start_epoch, epochs + start_epoch))
    best_mrr = 0.0
    print("start train:")
    test_pairs = test_dataset.links
    for epoch in pbar:
        t_start = time.perf_counter()
        train_loss = train(model, optimizer, train_loader, device, ARR = ARR)
        t_end = time.perf_counter()
        if regression:
            rmses.append(eval_rmse(model, test_loader, device))
        else:
            rmses.append(-1.0)
        #print(eval_rmse_dic(model, test_loader, device))
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
            'test_mrr': -1.0,
        }
        pbar.set_description('Epoch {}, train loss {:.6f}, test rmse {:.6f}, test mrrs {:.6f}'.format(*eval_info.values()))
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        duration = t_end - t_start
        print("The time in the %d epoch is %f" % (epoch, duration))
        if logger is not None:
            best_mrr = logger(eval_info, model, optimizer, best_mrr)
   
def test_once(test_dataset,
              model,
              test_pairs,
              batch_size,
              logger=None,
              ensemble=False,
              checkpoints=None,
              regression = True):

    test_seeds = torch.arange(test_dataset.len())
    test_loader = DataLoader(test_seeds, batch_size,  collate_fn=test_dataset.get_graphs, shuffle=False, num_workers=mp.cpu_count())
    model.to(device)
    t_start = time.perf_counter()
    if ensemble and checkpoints:
        rmse = eval_rmse_ensemble(model, checkpoints, test_loader, device, show_progress=True)
        mrr = -1.0
    else:
        if regression:
            rmse = -1.0
        else:
            rmse = -1.0
        mrr = eval_mrr(model, test_loader, device, test_pairs, regression=regression)
    t_end = time.perf_counter()
    duration = t_end - t_start
    print('Test Once RMSE: {:.6f}, Duration: {:.6f}'.format(rmse, duration))
    print(mrr)
    print('Test Once mrr: {:.6f}'.format(mrr))
    epoch_info = 'test_once' if not ensemble else 'ensemble'
    eval_info = {
        'test_rmse': rmse,
        'test_mrr': mrr,
    }
    if logger is not None:
        logger(eval_info, None, None)
    return mrr
