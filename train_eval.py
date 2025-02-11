import logging
import inspect
import os
import tempfile
import time
import uuid
from typing import Any, List, Optional, Union, Callable
import numpy as np
# import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

logging.basicConfig(level = 'INFO', # DEBUG
        format = "%(asctime)s %(levelname)s:%(lineno)d] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S")


def train(
          train_loader, 
          valid_loader,
          model,
          test_loader = None,
          criteria = "Loss",
          opt_name = 'Adam',
          learning_rate = 3e-4,
          clip_gradient = None,
          eval_step = 2000,
          epoch_num = 1,
          device = 'cpu'
        ):
    
    global_step = 0.
    logging.info("Start Training.")
    epoch_info = {
        }
    valid_trace = []
    optimizer = getattr(torch.optim, opt_name)(model.parameters(), 
                                                       lr=learning_rate,
                                               # weight_decay=1e-4
                                              )
    # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    patience = 0.
    tic = time.time()
    
    for epoch_no in range(epoch_num):
        epoch_loss = 0.
        for batch_no, feat_dic in enumerate(train_loader):
            # for idx, val in enumerate(feat_dic):
            #     feat_dic[idx] = val.to(device)
            optimizer.zero_grad()

            loss = model.training_step(feat_dic)
            loss.backward()
            if clip_gradient is not None:
                nn.utils.clip_grad_norm_(model.parameters(), 
                                         clip_gradient)
            optimizer.step()
            

            epoch_loss += loss.item()

            if global_step % eval_step == 0:
                # logging.info(f"Valid, Epoch {epoch_no}, global steps:{global_step}")
                # logging.info(f"Epoch {epoch_no}|Batch {batch_no}|Loss: {loss:.2f}|Best scrps:{epoch_info['scrps']:.2f}")
                full_arr, metrics_info = evaluate(valid_loader, 
                                                    model, 
                                                    device,
                                                   )
                if 'valid' not in epoch_info:
                    epoch_info['valid'] = metrics_info
                    import copy
                    best_model = copy.deepcopy(model)

                    epoch_info['best_global_step'] = int(global_step)
                    epoch_info['epoch'] = epoch_no
                    epoch_info['max_epoch'] = epoch_num
                    
                    best_info = metrics_info
                else: 
                    if metrics_info[criteria] < epoch_info['valid'][criteria]:
                        # logging.info("Saving best model.")
                        epoch_info['valid'] = metrics_info
                        # os.makedirs(f'/opt/meituan/dolphinfs_yeziwen/tmp/{f_name}', exist_ok = True)
                    # torch.save(model.cpu().state_dict(), epoch_info["params_path"])
                        import copy
                        best_model = copy.deepcopy(model)

                        epoch_info['best_global_step'] = int(global_step)
                        epoch_info['epoch'] = epoch_no
                        epoch_info['max_epoch'] = epoch_num
                        patience = 0
                        best_info = metrics_info
                        
                    else:
                        patience += 1
                # logging.info("Validation evaluation ends.")


            global_step += 1
        epoch_loss = epoch_loss/batch_no
        logging.info(f"Epoch {epoch_no}, Loss:{epoch_loss:.4f}")
        

        
    toc = time.time()
    logging.info("Load the best model.")
#     try:
#         model.load_state_dict(torch.load(epoch_info["params_path"]))
#     except:
#         import copy
#         model = copy.deepcopy(best_model):
    import copy
    model = copy.deepcopy(best_model)
    
    
    epoch_info['tot_time'] = np.round(toc - tic)
    epoch_info['train_loss'] = epoch_loss
    epoch_info['valid_trace'] = valid_trace
    logging.info(f"Model training is over.Training time:{toc - tic:.2f}s")
    return model, epoch_info

 




def metrics_calculation(labels, preds, quantiles = [0.01 * i for i in range(1, 100)]):
    labels = labels.reshape(-1, 1)
    y_diff = labels - preds
    v_a = np.maximum(y_diff, np.zeros(y_diff.shape))
    v_b = np.maximum(-y_diff, np.zeros(y_diff.shape))
    qs = np.array(quantiles)
    scrps = np.matmul(v_a, qs) + np.matmul(v_b, 1 - qs)
    scrps = 2 * np.sum(scrps)/np.sum(labels)
    
    crps = np.mean(np.matmul(v_a, qs) + np.matmul(v_b, 1 - qs)) * 0.01
    
    ece = np.mean(np.abs(np.mean(preds > labels, axis = 0) * 100 - np.arange(1, 100)))
    return scrps, crps, ece


def calculate_right_calibration(y_preds, y_trues, taus, cen_flag):
    val_dif = y_preds - y_trues.reshape(-1, 1)
    q_idx = np.argmin(np.abs(val_dif),axis=1)


    closest_q = []
    for i in range(y_trues.shape[0]):
        closest_q.append(taus[q_idx[i]])
    closest_q=np.array(closest_q)


    dcal_data = []

    for i in range(1, len(taus)):
        a = taus[i - 1]
        b = taus[i]
        lt_b = y_preds[cen_flag == 0, i] > y_trues[cen_flag == 0]
        lt_b_c = y_preds[cen_flag == 1, i] > y_trues[cen_flag == 1]

        gt_a = y_preds[cen_flag == 0, i - 1] <= y_trues[cen_flag == 0]
        gt_a_c = y_preds[cen_flag == 1, i - 1] <= y_trues[cen_flag == 1]
        lt_a_c = y_preds[cen_flag == 1, i - 1] > y_trues[cen_flag == 1]

        fall_w = lt_b * gt_a
        fall_w_c = lt_b_c * gt_a_c
        cen_p1 = fall_w_c * (b - closest_q[cen_flag == 1])/(1 - closest_q[cen_flag == 1])
        cen_p2 = lt_a_c * (b - a)/(1 - closest_q[cen_flag == 1])
        total_points = fall_w.sum() + cen_p1.sum() + cen_p2.sum()
        prop_captured = total_points/y_trues.shape[0]
        dcal_data.append([np.round(b-a, 2),prop_captured])
    dcal_data = np.array(dcal_data)
    return dcal_data

def calculate_right_ece(y_preds, y_trues, taus, cen_flag):
    val_dif = y_preds - y_trues.reshape(-1, 1)
    q_idx = np.argmin(np.abs(val_dif),axis=1)


    closest_q = []
    for i in range(y_trues.shape[0]):
        closest_q.append(taus[q_idx[i]])
    closest_q=np.array(closest_q)
    dece_data = []
    for i in range(len(taus)):
        a = 0.
        b = taus[i]
        lt_b = y_preds[cen_flag == 0, i] > y_trues[cen_flag == 0]
        lt_b_c = y_preds[cen_flag == 1, i] > y_trues[cen_flag == 1]
        gt_a = 0 <= y_trues[cen_flag == 0]
        gt_a_c = 0 <= y_trues[cen_flag == 1]
        lt_a_c = 0 > y_trues[cen_flag == 1]
        fall_w = lt_b * gt_a
        fall_w_c = lt_b_c * gt_a_c
        cen_p1 = fall_w_c * (b - closest_q[cen_flag == 1])/(1 - closest_q[cen_flag == 1])
        cen_p2 = lt_a_c * (b - a)/(1 - closest_q[cen_flag == 1])
        total_points = fall_w.sum() + cen_p1.sum() + cen_p2.sum()
        prop_captured = total_points/y_trues.shape[0]
        dece_data.append([b-a,prop_captured])
    dece_data = np.array(dece_data)
    return dece_data


def calculate_doubly_ece(y_preds, y_trues, taus, cen_flag):
    val_dif = y_preds - y_trues.reshape(-1, 1)
    q_idx = np.argmin(np.abs(val_dif),axis=1)
    closest_q = []
    for i in range(y_trues.shape[0]):
        closest_q.append(taus[q_idx[i]])
    closest_q=np.array(closest_q)
    
    dece_data = []
    for i in range(len(taus)):
        a = 0.
        b = taus[i]
        lt_b = y_preds[cen_flag == 0, i] > y_trues[cen_flag == 0]
        lt_b_right = y_preds[cen_flag == 1, i] > y_trues[cen_flag == 1]
        lt_b_left = y_preds[cen_flag == 2, i] > y_trues[cen_flag == 2]
        gt_b_left = y_preds[cen_flag == 2, i] < y_trues[cen_flag == 2]

        gt_a = 0 <= y_trues[cen_flag == 0]
        gt_a_right = 0 <= y_trues[cen_flag == 1]
        lt_a_right = 0 > y_trues[cen_flag == 1]
        gt_a_left = 0 <= y_trues[cen_flag == 2]
        lt_a_left = 0 > y_trues[cen_flag == 2]

        fall_w = lt_b * gt_a
        fall_w_right = lt_b_right * gt_a_right
        fall_w_left = lt_b_left * gt_a_left

        cen_right_p1 = fall_w_right * (b - closest_q[cen_flag == 1])/(1 - closest_q[cen_flag == 1])
        cen_left_p1 = fall_w_left * (closest_q[cen_flag == 2] - a)/closest_q[cen_flag == 2]
        cen_right_p2 = lt_a_right * (b - a)/(1 - closest_q[cen_flag == 1])
        cen_left_p2 = gt_b_left * (b - a)/closest_q[cen_flag == 2]
        total_points = fall_w.sum() + cen_right_p1.sum() + cen_right_p2.sum() + cen_left_p1.sum() + cen_left_p2.sum()
        prop_captured = total_points/y_trues.shape[0]
        dece_data.append([b-a,prop_captured])
    dece_data = np.array(dece_data)
    return dece_data


def calculate_doubly_calibration(y_preds, y_trues, taus, cen_flag):
    val_dif = y_preds - y_trues.reshape(-1, 1)
    q_idx = np.argmin(np.abs(val_dif),axis=1)
    closest_q = []
    for i in range(y_trues.shape[0]):
        closest_q.append(taus[q_idx[i]])
    closest_q=np.array(closest_q)
    
    dcal_data = []
    for i in range(1, len(taus)):
        a = taus[i - 1]
        b = taus[i]
        lt_b = y_preds[cen_flag == 0, i] > y_trues[cen_flag == 0]
        lt_b_right = y_preds[cen_flag == 1, i] > y_trues[cen_flag == 1]
        gt_b_right = y_preds[cen_flag == 1, i] < y_trues[cen_flag == 1]
        lt_b_left = y_preds[cen_flag == 2, i] > y_trues[cen_flag == 2]
        gt_b_left = y_preds[cen_flag == 2, i] < y_trues[cen_flag == 2]

        gt_a = y_preds[cen_flag == 0, i - 1] <= y_trues[cen_flag == 0]
        gt_a_right = y_preds[cen_flag == 1, i - 1] <= y_trues[cen_flag == 1]
        lt_a_right = y_preds[cen_flag == 1, i - 1] > y_trues[cen_flag == 1]
        gt_a_left = y_preds[cen_flag == 2, i - 1] <= y_trues[cen_flag == 2]
        lt_a_left = y_preds[cen_flag == 2, i - 1] > y_trues[cen_flag == 2]

        fall_w = lt_b * gt_a
        fall_w_right = lt_b_right * gt_a_right
        fall_w_left = lt_b_left * gt_a_left

        cen_right_p1 = fall_w_right * (b - closest_q[cen_flag == 1])/(1 - closest_q[cen_flag == 1])
        cen_left_p1 = fall_w_left * (closest_q[cen_flag == 2] - a)/closest_q[cen_flag == 2]
        cen_right_p2 = lt_a_right * (b - a)/(1 - closest_q[cen_flag == 1])
        cen_left_p2 = gt_b_left * (b - a)/closest_q[cen_flag == 2]
        total_points = fall_w.sum() + cen_right_p1.sum() + cen_right_p2.sum() + cen_left_p1.sum() + cen_left_p2.sum()
        prop_captured = total_points/y_trues.shape[0]
        dcal_data.append([b-a,prop_captured])
    dcal_data = np.array(dcal_data)
    return dcal_data

def get_full_calibration_metrics(eval_tuple):
    taus = eval_tuple['taus']
    cen_flag = eval_tuple['censored']
    cen_flag = cen_flag.astype(np.float32)

    y_preds = eval_tuple['quantiles']
    y_trues = eval_tuple['label']
    
    # cen_flag = (cen_flag == 1).astype(float)
    
    cal_data = []
    for i in range(len(taus)):
        cal_prob = taus[i]
        # cal_large = np.mean(y_preds[cen_flag == 0, i] > y_trues[cen_flag == 0])
        
        
        cal_large = np.mean(y_preds[:, i] > y_trues)
        cal_data.append([cal_prob, cal_large])

    cal_data = np.array(cal_data)
    cal_ece = np.mean(np.abs(cal_data[:, 0] - cal_data[:, 1]))
    
    
    tmp_dcal = []
    tmp_dcal.append([cal_data[0,0], cal_data[0, 1]])
    for i in range(len(cal_data) - 1):
        target = cal_data[i+1,0] - cal_data[i,0]
        captured = cal_data[i+1,1] - cal_data[i,1]
        tmp_dcal.append([target, captured])
    # tmp_dcal.append([1 - cal_data[-1, 0], 1 - cal_data[-1, 1]])
    tmp_dcal = np.array(tmp_dcal)
    
    DCal_score_woc = np.sum(np.square(tmp_dcal[:,0]-tmp_dcal[:,1])) * 100
    
    cum_dcal_woc = np.cumsum(tmp_dcal, axis = 0)
    DCal_ECE_woc = np.mean(np.abs(cum_dcal_woc[:,0] - cum_dcal_woc[:, 1])) * 100
    
    ## right metrics
    dcal_data = calculate_right_calibration(y_preds[cen_flag != 2], y_trues[cen_flag != 2], taus, cen_flag[cen_flag != 2])
    dece_data = calculate_right_ece(y_preds[cen_flag != 2], y_trues[cen_flag != 2], taus, cen_flag[cen_flag != 2])
    
    d2cal_data = calculate_doubly_calibration(y_preds, y_trues, taus, cen_flag)
    d2ece_data = calculate_doubly_ece(y_preds, y_trues, taus, cen_flag)
    
    
    DCal_score_wc = np.sum(np.square(dcal_data[:,0]-dcal_data[:,1])) * 100
    DCal_ECE_wc = np.mean(np.abs(dece_data[:,0] - dece_data[:, 1])) * 100
    
    doubly_cal = np.sum(np.square(d2cal_data[:,0]-d2cal_data[:,1])) * 100
    doubly_ece = np.mean(np.abs(d2ece_data[:,0] - d2ece_data[:, 1])) * 100
    
    cal_metrics = {
        "org_data_dcal": DCal_score_woc,
        "org_data_dece": DCal_ECE_woc,
        "right_cen_DCal": DCal_score_wc,
        "right_cen_DECE": DCal_ECE_wc,
        "doubly_DCal": doubly_cal,
        "doubly_DECE": doubly_ece
    }
    
    return cal_metrics

def evaluate(loader, model, device = 'cpu'):
    full_arr = {
                "quantiles": [],
                "label": [],
                "censored": [],
                "real_label": []
               }

    quantiles = [0.01 * i for i in range(1, 100)]
    epoch_loss = 0.
    for batch_no, feat_dic in enumerate(loader):
        with torch.no_grad():
            q_input = torch.tensor([quantiles]).to(device)
            loss = model.training_step(feat_dic)
            
            preds, real_labels, labels, cen_flag = model.eval_step(feat_dic)
            full_arr['quantiles'].append(preds.detach().cpu().numpy())
            full_arr['label'].append(labels.detach().cpu().numpy())
            full_arr['censored'].append(cen_flag.detach().cpu().numpy())
            
            full_arr['real_label'].append(real_labels.detach().cpu().numpy())
            
            epoch_loss += loss
    epoch_loss = epoch_loss / batch_no
            
    for key, val in full_arr.items():
        full_arr[key] = np.vstack(full_arr[key]).squeeze()
    full_arr['taus'] = quantiles
    
    real_label = full_arr['real_label']
    label_all = full_arr['label']
    q_all = full_arr['quantiles']
    full_scrps, full_crps, full_ece = metrics_calculation(label_all, q_all)
    cen_ind = full_arr['censored'] == 0
    uncen_scrps, uncen_crps, uncen_ece = metrics_calculation(label_all[cen_ind], q_all[cen_ind, :])
    
    

    sharp_q90_q10 = np.mean(q_all[:,89] - q_all[:, 9])


    cal_metrics = get_full_calibration_metrics(full_arr)

    cal_metrics.update({"sharpness": sharp_q90_q10})
    cal_metrics.update({"uncen_crps": uncen_crps, "uncen_ece": uncen_ece})
    from lifelines.utils import concordance_index
    flag = full_arr['censored']
    drop_con = flag == 2
    c_index = concordance_index(full_arr['label'][~drop_con], full_arr['quantiles'][:, 49][~drop_con], 
                      1 - flag[~drop_con])
    cal_metrics['c_index'] = c_index
    
    
    real_scrps, real_crps, real_ece = metrics_calculation(real_label, q_all)
    cen_ind = full_arr['censored'] == 0
    uncen_scrps, uncen_crps, uncen_ece = metrics_calculation(real_label[cen_ind], q_all[cen_ind, :])
    
    cal_metrics.update({"real_crps": real_crps, "real_ece": real_ece})
    cal_metrics.update({"real_un_crps": uncen_crps, "real_un_ece": uncen_ece})
    
    cal_metrics['Loss'] = float(epoch_loss.detach().cpu().numpy())
    
    import copy
    full_arr2 = copy.deepcopy(full_arr)
    full_arr2['label'] = real_label
    cal_metrics2 = get_full_calibration_metrics(full_arr2)
    
    cal_metrics['real_full_DCal'] = cal_metrics2['org_data_dcal']
    
    cal_metrics.update({"name": name})
    return full_arr, cal_metrics


