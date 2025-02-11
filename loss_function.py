# import lightning as L

from torch import nn
import torch
def quantile_loss(labels, preds, quantiles, reduce = False):
    y_diff = labels - preds
    v_a = torch.max(y_diff, torch.zeros(y_diff.shape).to(y_diff.device))
    v_b = torch.max(-y_diff, torch.zeros(y_diff.shape).to(y_diff.device))

    qs = torch.tensor(quantiles).to(y_diff.device)
    
    loss = v_a * qs + v_b * (1 - qs)
    if reduce:
        loss = torch.mean(loss, axis = 1)
    # loss = torch.matmul(v_a, qs) + torch.matmul(v_b, 1 - qs)
    return loss

def surv_right_loss(labels, 
                    preds, 
                    q_argmin,
                    cen_flag,
                    quantiles = [0.01 * i for i in range(1, 100)],
                ):
    y_max = (preds + 1e-3).detach()
    normal_flag = (cen_flag == 0).float()
    right_flag = (cen_flag == 1).float()
    
    taus = torch.tensor(quantiles).to(labels.device)
    taus = torch.broadcast_to(taus, (labels.shape[0], taus.shape[1]))
    
    tau_lt = (taus < q_argmin).float()

    weights = tau_lt
    
    ql_label = quantile_loss(labels, preds, quantiles)
    loss_normal = quantile_loss(labels, preds, quantiles = quantiles, reduce = True)
    loss_censored = torch.mean(ql_label * weights, axis = 1)

    loss = normal_flag.squeeze() * loss_normal + right_flag.squeeze() * loss_censored
    return loss

def portnoy_loss(labels, 
                 preds, 
                 q_argmin, 
                 cen_flag,
                 quantiles):
    
    y_max = (preds + 1e-3).detach()
    right_flag = (cen_flag == 1).float()
    normal_flag = (cen_flag == 0).float()
    
    taus = torch.tensor(quantiles).to(labels.device)
    taus = torch.broadcast_to(taus, (labels.shape[0], taus.shape[1]))
    
    
    tau_lt = (taus < q_argmin).float()
    tau_gt = (taus >= q_argmin) * (taus - q_argmin)/(1 - q_argmin)
    weights = tau_lt + tau_gt
    # weights = tau_gt
    
    ql_label = quantile_loss(labels, preds, quantiles)
    ql_star = quantile_loss(y_max, preds, quantiles)
    loss_normal = quantile_loss(labels, preds, quantiles = quantiles, reduce = True)
    loss_censored = torch.mean(ql_label * weights + (1 - weights) * (preds - y_max) * (1 - quantiles - 1.), axis = 1)

    loss = normal_flag.squeeze() * loss_normal + right_flag.squeeze() * loss_censored
    return loss


def calculate_right_loss(labels, 
                         preds,  
                         q_argmin,
                         quantiles
                        ):
    y_max = (preds + 1e-6).detach()
    taus = torch.tensor(quantiles).to(labels.device)
    taus = torch.broadcast_to(taus, (labels.shape[0], taus.shape[1]))
    # right censored
    right_tau_lt = (taus < q_argmin).float()
    right_tau_gt = (taus >= q_argmin) * (taus - q_argmin)/(1 - q_argmin)
    right_weights = right_tau_lt + right_tau_gt
    # right_weights = right_tau_gt
    ql_label = quantile_loss(labels, preds, quantiles)
    loss_right_censored = torch.mean(ql_label * right_weights + (1 - right_weights) * (preds - y_max) * (1. - quantiles - 1.), axis = 1)
    return loss_right_censored

def calculate_left_loss(labels,
                        preds,
                        q_argmin,
                        quantiles
                       ):
    
    y_max = (preds - 1e-6).detach()
    
    taus = torch.tensor(quantiles).to(labels.device)
    taus = torch.broadcast_to(taus, (labels.shape[0], taus.shape[1]))

    left_tau_gt = (taus > q_argmin).float()
    left_tau_lt = (taus <= q_argmin) * (q_argmin - taus)/(q_argmin)
    left_weights = left_tau_gt + left_tau_lt
     
    # left_weights = left_tau_lt
    ql_label = quantile_loss(labels, preds, quantiles)
    loss_left_censored = torch.mean(ql_label * left_weights + (1 - left_weights) * (preds - y_max) * (1. - quantiles), axis = 1)
    return loss_left_censored

def doubly_censored_loss(labels, 
                         preds, 
                         q_argmin, 
                         cen_flag, 
                         quantiles = [0.01 * i for i in range(1, 100)]
                        ):
    normal_flag = (cen_flag == 0).float().squeeze()
    right_flag = (cen_flag == 1).float().squeeze()
    left_flag = (cen_flag == 2).float().squeeze()
    
    loss_right_censored = calculate_right_loss(labels, preds, q_argmin, quantiles)
    loss_left_censored = calculate_left_loss(labels, preds, q_argmin, quantiles)
    loss_normal = quantile_loss(labels, preds, quantiles = quantiles, reduce = True)
    final_loss = normal_flag * loss_normal + right_flag * loss_right_censored + left_flag * loss_left_censored
    return final_loss


    
class DynamicRangeExpPWL(nn.Module):
    def __init__(self, num_pieces = 100, start_point = 1, **kwargs):
        
        super().__init__(**kwargs)
        
        
        self.num_pieces = num_pieces
        self.gamma_map = nn.LazyLinear(1)
        self.fixed_knots = torch.arange(start_point, 100 - start_point + 1, 100/num_pieces)/100
        
        self.knots_map = nn.LazyLinear(len(self.fixed_knots))
        self.softplus = torch.nn.Softplus()
        
    def forward(self, x, input_q):
        eps = 1e-4
        q_out = torch.cumsum(F.softplus(self.knots_map(x)), axis = 1)
        
        fixed_knots = self.fixed_knots.to(q_out.device)
        
        beta_right = torch.log((1 - fixed_knots[-2] + eps)/(1 - fixed_knots[-1] + eps) + eps)
        beta_right = beta_right/(q_out[:, -2:-1] - q_out[:, -1:])
        
        beta_left = torch.log((fixed_knots[2] + eps)/(fixed_knots[1] + eps) + eps)
        beta_left = beta_left/(q_out[:, 1:2] - q_out[:, :1])
        
        return self.get_quantiles(q_out, beta_right, beta_left, input_q)
    
    def cdf(self, x, labels):
        eps = 1e-4
        q_out = torch.cumsum(F.softplus(self.knots_map(x)), axis = 1)
        
        fixed_knots = self.fixed_knots.to(q_out.device)
        
        ## modified right_a
        beta_right = torch.log((1 - fixed_knots[-2] + eps)/(1 - fixed_knots[-1] + eps) + eps)
        beta_right = beta_right/(q_out[:, -2:-1] - q_out[:, -1:])
        
        beta_left = torch.log((fixed_knots[2] + eps)/(fixed_knots[1] + eps) + eps)
        beta_left = beta_left/(q_out[:, 1:2] - q_out[:, :1])

        return self.get_cdf(q_out, beta_right, beta_left, labels).squeeze()
        
    
    def get_quantiles(self, q_out, beta_right, beta_left, input_q):
        
        fixed_knots = self.fixed_knots.to(q_out.device)

        right_a = 1 / beta_right
        right_b = - right_a * torch.log(1 - fixed_knots[-1]) + q_out[:, -1:]
        right_cal = torch.log(1 - input_q) * right_a + right_b


        left_a = 1 / beta_left
        left_b = - left_a * torch.log(fixed_knots[0]) + q_out[:, :1]
        left_cal = torch.log(input_q) * left_a + left_b

        left_flag = input_q < fixed_knots[0]
        right_flag = input_q > fixed_knots[-1]
        center_flag = torch.logical_and(~left_flag, ~right_flag)

        q_right_idx = torch.searchsorted(fixed_knots, input_q, right=True)
        q_right_idx = torch.broadcast_to(q_right_idx, (q_out.shape[0], q_right_idx.shape[1]))
        q_right_idx = torch.where(q_right_idx >= len(fixed_knots), 
                                  len(fixed_knots) - 1, 
                                  q_right_idx)
        q_left_idx = torch.where(q_right_idx - 1 >= 0, q_right_idx - 1, 0)


        y_left = torch.gather(q_out, 1, q_left_idx)
        y_right = torch.gather(q_out, 1, q_right_idx)

        q_left = fixed_knots[q_left_idx]
        q_right = fixed_knots[q_right_idx]
        ratio = (input_q - q_left)/(q_right - q_left + 1e-3)
        center_cal = torch.lerp(y_left, y_right, ratio.float())
        output = center_cal * center_flag + left_flag * left_cal + right_flag * right_cal
        return output
    
    def get_cdf(self, q_out, beta_right, beta_left, labels):
        
        tol = 1e-4
        fixed_knots = self.fixed_knots.to(q_out.device)
        
        
        ## modified right_a
        right_a = 1 / beta_right
        right_b = -right_a * torch.log(1 - fixed_knots[-1]) + q_out[:, -1:]


        left_a = 1 / beta_left
        left_b = -left_a * torch.log(fixed_knots[0]) + q_out[:, :1]

        
        idx = torch.searchsorted(q_out, labels, right=True)
        right_idx = torch.where(idx >= len(fixed_knots), len(fixed_knots) - 1, idx)
        left_idx =torch.where(right_idx - 1 >= 0, right_idx - 1, 0)
        left_q = fixed_knots[left_idx]
        right_q = fixed_knots[right_idx]

        left_val = torch.gather(q_out, 1, left_idx)
        right_val = torch.gather(q_out, 1, right_idx)
        ratio = (labels - left_val)/(right_val - left_val + 1e-3)


        find_q = torch.lerp(left_q, right_q, ratio.float())

        right_flag = idx == len(fixed_knots)
        left_flag = idx == 0
        center_flag = torch.logical_and(~right_flag, ~left_flag)
        log_right_alpha = torch.minimum((labels - right_b)/right_a, 
                                        torch.log(1 - fixed_knots[-1]))
        
        right_alpha = 1 - torch.exp(log_right_alpha)

        log_left_alpha = torch.minimum((labels - left_b)/left_a, 
                                       torch.log(fixed_knots[0]))
        left_alpha = torch.exp(log_left_alpha)

        out_q = center_flag * find_q + right_flag * right_alpha + left_flag * left_alpha
        out_q = torch.clip(out_q, 0 + tol, 1 - tol)
        
        return out_q
    

# define the LightningModule
class MlpModel(nn.Module):
    def __init__(self, device, type_ = 'excl_censor'):
        super().__init__()
        
        self.device = device
        
        self.rep_layer = torch.nn.LazyLinear(512)
        self.map_layer = torch.nn.LazyLinear(99)
        self.softplus = torch.nn.Softplus()

#         self.output_layer = DynamicRangeExpPWL(num_pieces = num_pieces, 
#                                     start_point = start_point
#                                    )
        
        self.quantiles = [0.01 * i for i in range(1, 100)]
        
        self.type_ = type_

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        x, ry, y, cen_flag = batch
        
        x = x.to(self.device)
        ry = ry.to(self.device)
        y = y.to(self.device)
        cen_flag = cen_flag.to(self.device)
        
        
        
        
        q_preds = self.rep_layer(x)
        q_preds = self.map_layer(q_preds)
        q_preds = self.softplus(q_preds)
        q_preds = torch.cumsum(q_preds, axis = 1)
        
        loss = self.loss(y, q_preds, cen_flag, self.quantiles, self.type)
        loss = torch.mean(loss)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        return loss

        
    
    def loss(self, labels, q_preds, cen_flag, quantiles, type_ = 'excl_censor'):
        type_ = self.type_
        if type_ == 'excl_censored':
            loss = quantile_loss(labels, q_preds, quantiles)
            loss = loss * (cen_flag == 0)

            flag = cen_flag == 0
            loss = torch.masked_select(loss, flag)
            
        elif type_ == 'portnoy':
            q_input = torch.tensor([quantiles]).to(labels.device)
            q_argmin = self.get_near_q(q_preds, labels, q_input)
            loss = portnoy_loss(labels, q_preds, q_argmin, cen_flag, quantiles = q_input)
            
            flag = cen_flag != 2
            loss = torch.masked_select(loss, flag)
            
        elif type_ == 'doubly':
            q_input = torch.tensor([quantiles]).to(labels.device)
            q_argmin = self.get_near_q(q_preds, labels, q_input)
            loss = doubly_censored_loss(labels, q_preds, q_argmin, cen_flag, quantiles = q_input)
            
        elif type_ == 'surv_crps':
            q_input = torch.tensor([quantiles]).to(labels.device)
            q_argmin = self.get_near_q(q_preds, labels, q_input)
            loss = surv_right_loss(labels, q_preds, q_argmin, cen_flag, quantiles = q_input)
            
            flag = cen_flag != 2
            loss = torch.masked_select(loss, flag)
            # loss = 
        return loss

    def eval_step(self, batch):
        x, ry, y, cen_flag = batch
        
        x = x.to(self.device)
        ry = ry.to(self.device)
        y = y.to(self.device)
        cen_flag = cen_flag.to(self.device)
        
        
        q_preds = self.rep_layer(x)
        q_preds = self.map_layer(q_preds)
        q_preds = self.softplus(q_preds)
        q_preds = torch.cumsum(q_preds, axis = 1)
        return q_preds, ry, y, cen_flag

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def get_near_q(self, preds, labels, quantiles):
        preds_detach = preds.detach()
        abs_diff = torch.abs(labels - preds_detach)
        taus = torch.tensor(quantiles).to(labels.device)
        taus = torch.broadcast_to(taus, abs_diff.shape)
        q_near = abs_diff == torch.min(abs_diff, dim = 1).values.view(-1, 1)
        q_argmin = torch.max(taus * q_near, axis = 1).values.reshape(-1,1)
        return q_argmin.detach()