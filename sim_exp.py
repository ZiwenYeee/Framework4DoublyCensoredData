import os

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import optim, nn, utils, Tensor
from loss_function import *
from model import *
from train_eval import *

def generate_data(n_data, ind = True):
    np.random.seed(4)
    x = np.random.uniform(0, 3, size = (n_data, 1))
    x2 = np.sin(np.random.uniform(0, 5, size = (n_data, 1)))
    if ind:
        L = np.random.uniform(0, 10, size = (n_data, 1)) + 7
        R = L + np.random.uniform(4.5 - 2, 11.5 - 1, size = (n_data, 1))
    else:
        L = np.random.exponential(4 * x + x2 + 3)
        R = L + np.random.uniform(7, 12, size=(n_data, 1))
    beta_0 = 1
    beta_1 = 5
    beta_2 = 2
    eps = (np.random.normal(0, 1, size=(n_data, 1)))
    y = beta_0 + beta_1 * x + beta_2 * x2 + eps + 7
    
    # y = np.exp(y/3)
    # L = np.exp(L/3)
    # R = np.exp(R/3)
    
    obs_y = np.max([L, np.min([y, R], axis = 0)], axis = 0)
    cen_ind = np.where(obs_y == R, 1, np.where(obs_y == L, 2, 0))
    
    x = np.hstack([x, x2])
    return x, y, obs_y, cen_ind


x_train, ry_train, y_train, d_train = generate_data(200000)
x_valid, ry_valid, y_valid, d_valid = generate_data(10000)
x_test, ry_test, y_test, d_test = generate_data(10000)

x_train = torch.Tensor(x_train).float() # transform to torch tensor
y_train = torch.Tensor(y_train).float() # transform to torch tensor
ry_train = torch.Tensor(ry_train).float() # transform to torch tensor
d_train = torch.Tensor(d_train).long() # transform to torch tensor
# tensor_y = torch.Tensor(my_y)

train_data = TensorDataset(x_train, ry_train, y_train, d_train) # create your datset
train_loader = DataLoader(train_data, batch_size = 128, shuffle = True) # create your dataloader



x_valid = torch.Tensor(x_valid).float() # transform to torch tensor
y_valid = torch.Tensor(y_valid).float() # transform to torch tensor
ry_valid = torch.Tensor(ry_valid).float() # transform to torch tensor
d_valid = torch.Tensor(d_valid).long() # transform to torch tensor
valid_data = TensorDataset(x_valid, ry_valid, y_valid, d_valid) # create your datset
valid_loader = DataLoader(valid_data, batch_size = 1024, shuffle = False) # create your dataloader



x_test = torch.Tensor(x_test).float() # transform to torch tensor
y_test = torch.Tensor(y_test).float() # transform to torch tensor
ry_test = torch.Tensor(ry_test).float() # transform to torch tensor
d_test = torch.Tensor(d_test).long() # transform to torch tensor
test_data = TensorDataset(x_test, ry_test, y_test, d_test) # create your datset
test_loader = DataLoader(test_data, batch_size = 1024, shuffle = False) # create your dataloader

# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch import Trainer

final_dict = []
torch.manual_seed(100)

device = 'cuda:0'
# , 'lognorm'
for name in [ 'portnoy', 'doubly', 'excl_censored', 'surv_crps', 'lognorm']:
    logging.info(f"Model: {name}")
    if name == 'lognorm':
        model = LogNormModel(device)
        model.to(device)
    else:
        model = MlpModel(device, name)
        model.to(device)
    max_epoch = 20
    model, epoch_info = train(train_loader, valid_loader, model,
              learning_rate = 1e-4,
              device = device, epoch_num = max_epoch, eval_step = 100)
    # trainer = Trainer(max_epochs=max_epoch, log_every_n_steps = 10,
    #                       callbacks=[checkpoint_callback],
    #                  )
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    full_arr, eval_tuple = evaluate(test_loader, model)
    final_dict.append(eval_tuple)
    # break