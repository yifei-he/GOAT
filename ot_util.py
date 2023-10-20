import ot
import torch
from util import *
import numpy as np
import time
import torch.nn as nn
        

def get_transported_labels(plan, ys, logit=False):
    # plan /= np.sum(plan, 0, keepdims=True)
    ysTemp = ot.utils.label_normalization(np.copy(ys))
    classes = np.unique(ysTemp)
    n = len(classes)
    D1 = np.zeros((n, len(ysTemp)))

    # perform label propagation
    transp = plan

    # set nans to 0
    transp[~ np.isfinite(transp)] = 0

    for c in classes:
        D1[int(c), ysTemp == c] = 1

    # compute propagated labels
    transp_ys = np.dot(D1, transp).T

    if logit:
        return transp_ys 
    
    transp_ys = np.argmax(transp_ys, axis=1)

    return transp_ys


def get_conf_idx(logits, confidence_q=0.2):
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    labels = np.argmax(logits, axis=1)
    
    return labels, indices


def get_OT_plan(X_S, X_T, solver='sinkhorn', weights_S=None, weights_T=None, Y_S=None, numItermax=1e7,
                entropy_coef=1, entry_cutoff=0):

    # X_S, X_T = X_S[:50000], X_T[:50000]
    X_S, X_T = X_S, X_T
    n, m = len(X_S), len(X_T)
    a = np.ones(n) / n if weights_S is None else weights_S
    b = np.ones(m) / m if weights_T is None else weights_T
    print(f'{n} source data, {m} target data. ')
    dist_mat = ot.dist(X_S, X_T).detach().numpy()
    t = time.time()
    if solver == 'emd':
        plan = ot.emd(a, b, dist_mat, numItermax=int(numItermax))
    elif solver == 'sinkhorn':
        plan = ot.sinkhorn(a, b, dist_mat, reg=entropy_coef, numItermax=int(numItermax), stopThr=10e-7)
    elif solver == 'lpl1':
        plan = ot.sinkhorn_lpl1_mm(a, b, Y_S, dist_mat, reg=entropy_coef, numItermax=int(numItermax), stopInnerThr=10e-9)

    if entry_cutoff > 0:
        avg_val = 1 / (n * m)
        print(f'Zero out entries with value < {entry_cutoff}*{avg_val}')
        plan[plan < avg_val * entry_cutoff] = 0

    elapsed = round(time.time() - t, 2)
    print(f"Time for OT calculation: {elapsed}s")
    # plan /= np.sum(plan, 0, keepdims=True)
    # plan[~ np.isfinite(plan)] = 0
    plan = plan * n 

    return plan


def pushforward(X_S, X_T, plan, t):
    print(f'Pushforward to t={t}')
    assert 0 <= t <= 1
    nonzero_indices = np.argwhere(plan > 0)
    weights = plan[plan > 0]
    assert len(nonzero_indices) == len(weights)
    x_t= (1-t)*X_S[nonzero_indices[:,0]] + t*X_T[nonzero_indices[:,1]]
    
    return x_t, weights


def generate_domains(n_inter, dataset_s, dataset_t, plan=None, entry_cutoff=0, conf=0):
    print("------------Generate Intermediate domains----------")
    all_domains = []
    
    xs, xt = dataset_s.data, dataset_t.data
    ys = dataset_s.targets

    if plan is None:
        if len(xs.shape) > 2:
            xs_flat, xt_flat = nn.Flatten()(xs), nn.Flatten()(xt)
            plan = get_OT_plan(xs_flat, xt_flat, solver='emd', entry_cutoff=entry_cutoff)
        else:
            plan = get_OT_plan(xs, xt, solver='emd', entry_cutoff=entry_cutoff)

    logits_t = get_transported_labels(plan, ys, logit=True)
    yt_hat, conf_idx = get_conf_idx(logits_t, confidence_q=conf)
    xt = xt[conf_idx]
    plan = plan[:, conf_idx]
    yt_hat = yt_hat[conf_idx]

    print(f"Remaining data after confidence filter: {len(conf_idx)}")

    for i in range(1, n_inter+1):
        x, weights = pushforward(xs, xt, plan, i / (n_inter+1))
        if isinstance(x, np.ndarray):
            all_domains.append(DomainDataset(torch.from_numpy(x).float(), weights))
        else:
            all_domains.append(DomainDataset(x, weights))
    all_domains.append(dataset_t)

    print(f"Total data for each intermediate domain: {len(x)}")

    return all_domains


def ot_ablation(size, mode):
    ns, nt = size, size
    plan = np.zeros((ns, nt))
    ran = np.arange(ns*nt)
    np.random.shuffle(ran)
    idx = ran[:size]

    for i in idx:
        row = i // nt
        col = i-i//nt * nt
        if mode == "random":
            plan[row, col] = np.random.uniform()
        elif mode == "uniform":
            plan[row, col] = 1
    
    plan /= np.sum(plan, 1, keepdims=True)
    plan[~ np.isfinite(plan)] = 0

    return plan


