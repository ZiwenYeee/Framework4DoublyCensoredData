import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import os
import pandas as pd
# import torchvision 
import torch
import h5py

pred_res = np.load('simulation_0521.npy', allow_pickle = True)[()]

from ortools.sat.python import cp_model


from ortools.sat.python import cp_model

def GetDecision_addV2(num_val, quantiles, price, c1, c2, real_label = None, alpha = 0.1):
    model = cp_model.CpModel()
    sku = {}
    sku_value = {}

    sku_higher = {}
    sku_lossL = {}
    sku_lossR = {}
    
    num_sku = len(num_val)
    if real_label is None:
        real_label = (num_val[:, 49:50]).astype(int)
    for sku_no in range(num_sku):
        sku_value[(sku_no)] = model.NewIntVar(0, 10000, f'y_{sku_no}')
        
        sku_lossL[(sku_no)] = model.NewIntVar(0, 10000, f'll_{sku_no}')
        sku_lossR[(sku_no)] = model.NewIntVar(0, 10000, f'lr_{sku_no}')

        sku_higher[(sku_no)] = model.NewBoolVar(f'h_{sku_no}')
        
        for i in range(1, 100):
            # 创建站点分位数决策
            sku[(sku_no, f"{quantiles[i - 1]}")] = model.NewBoolVar(f'p_{sku_no},{quantiles[i - 1]}')
    for sku_no in range(num_sku):
        model.Add(sum(sku[(sku_no, qp)] for qp in quantiles) == 1)
        for i in range(1, 100):
            model.Add(sku_value[(sku_no)] == int(num_val[sku_no, i - 1])).OnlyEnforceIf(
                sku[(sku_no, quantiles[i - 1])])
            
            loss = 5 * (np.sum(num_val[sku_no,i-1:] * 0.01) - num_val[sku_no, i - 1] * (1 - float(quantiles[i-1].split("_")[1]) * 0.01))
            loss = int(loss)
            model.Add(sku_lossL[(sku_no)] == loss).OnlyEnforceIf(
                sku[(sku_no, quantiles[i - 1])])

        
            # loss = int(np.sum(num_val[k,i-1:] * 0.01) - num_val[k, i - 1] * (1 - float(quantiles[i-1].split("_")[1]) * 0.01))
            loss = 5 * ( float(quantiles[i-1].split("_")[1]) * 0.01 * num_val[sku_no, i - 1] - np.sum(num_val[sku_no, :i] * 0.01) )
            loss = int(loss)
            model.Add(sku_lossR[(sku_no)] == loss).OnlyEnforceIf(
                sku[(sku_no, quantiles[i - 1])])

        model.Add( int(real_label[(sku_no)]) > sku_value[(sku_no)] ).OnlyEnforceIf(sku_higher[(sku_no)])
        model.Add( int(real_label[(sku_no)]) <= sku_value[(sku_no)] ).OnlyEnforceIf(sku_higher[(sku_no)].Not())

    # # for qp in qp_lst[poi]:
    #     self.model.Add(self.y[(poi, order)] == int(60 * order_qp_dict[order][qp])).OnlyEnforceIf(self.p[(poi, qp)])

    model.Add(sum(sku_value[(sku_no)] for sku_no in range(num_sku)) <= int((1 + alpha) * sum(real_label) ))
    # model.Add(sum(sku_higher[(sku_no)] for sku_no in range(num_sku)) <= int(0.05 * num_sku))

    
    model.Maximize(sum(int(sum(num_val[(sku_no)]) * 0.01) * price[sku_no] * 5 - (c1[sku_no] * sku_lossL[(sku_no)] + c2[sku_no] * sku_lossR[(sku_no) ])  for sku_no in range(num_sku)))

    
    solver = cp_model.CpSolver()
    # solver.parameters.num_search_workers = 80  # 设置使用的线程数
    
    # solver.parameters.enumerate_all_solutions = True
    # solution_printer = VarArrayAndObjectiveSolutionPrinter(sku)
    status = solver.Solve(model)
    return solver.ObjectiveValue(), [solver.Value(sku_value[i]) for i in range(num_sku)]


sku_selected = 200
i = 500
np.random.seed(10)
c1 = np.random.uniform(0.1, 1, size = (sku_selected,)) 
c2 = np.random.uniform(0.1, 1, size = (sku_selected,))
price = np.random.uniform(0.1, 10, size = (sku_selected,))



quantiles = [f'q_{i}' for i in range(1, 100)]

full_res = {}
for name in [ 'portnoy', 'doubly', 'excl_censored', 'surv_crps', 'lognorm']:
    print(name)
    tmp_list = []
    num_val = pred_res[name]['quantiles'][i: i + sku_selected]

    real_label = pred_res[name]['real_label'][i: i + sku_selected]

    for alpha in [-0.1, -.05, 0, 0.05, 0.1, 0.15, 0.2]:
        opt_loss, opt_value = GetDecision_addV2(num_val, quantiles, price, c1 * price, c2 * price, alpha = alpha, real_label = real_label)
        overstock_ratio = np.sum(np.maximum(opt_value - real_label.round(), 0))/np.sum(real_label.round())
        oos_ratio = np.sum(np.where(opt_value - real_label.round() < 0, 1, 0))/sku_selected
        real_opt = np.where(opt_value < real_label.round(), opt_value, real_label.round())
    
        # # tot_profit = np.sum(np.maximum(opt_value - real_label.round(), 0) *  c1 * price + np.maximum(real_label.round() - opt_value, 0) *  c2 * price)
        # tot_profit = 0
        # tot_profit = np.sum(real_opt * price) - tot_profit

        tot_profit = np.sum(np.maximum(opt_value - real_label.round(), 0) *  c2 * price + np.maximum(real_label.round() - opt_value, 0) *  c1 * price)
        tot_profit = np.sum(real_label * price) - tot_profit
        
        tmp_dict = {"oos_ratio": oos_ratio, "overstock_ratio": overstock_ratio,
                    "tot_profit": tot_profit, "alpha": alpha,
                    'name':name
                   }
        tmp_list.append(tmp_dict)
    full_res[name] = tmp_list