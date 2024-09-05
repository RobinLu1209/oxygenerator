import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import argparse
from tqdm import tqdm
import time
import math
import re
import os
from datetime import datetime
from utils import *
from models import *
import torch.nn.functional as F
import pdb
import gzip
import logging


# 重定向print到logging
def print_and_log(message):
    print(message)
    logging.info(message)

def get_one_data(pwd, year, area_id):
    file_path = pwd + "year_" + str(year) + "_area_" + str(area_id) +'.gz'
    with gzip.open(file_path, 'rb') as f:
        data = torch.load(f, map_location=torch.device('cpu'))

    numpy_array = data['label'].numpy()
    valid_indices = np.where(~np.isnan(numpy_array))[0]
    pho_indices = torch.nonzero(~torch.isnan(data['features'][:, 7]), as_tuple=False).view(-1)
    nit_indices = torch.nonzero(~torch.isnan(data['features'][:, 8]), as_tuple=False).view(-1)
    return data, valid_indices.tolist(), pho_indices, nit_indices

def select_sample_idx(length, k, select_idx):
    set_random_seed(1024)
    size = int(length/k)
    sample_idx_all = [i for i in range(length)]
    random.shuffle(sample_idx_all)
    test_start = (select_idx - 1) * size
    test_end = select_idx * size
    test_idx = sample_idx_all[test_start:test_end]
    train_val_idx = sample_idx_all[:test_start] + sample_idx_all[test_end:]
    split_point = len(train_val_idx) // 5
    train_idx = train_val_idx[:split_point * 4]
    val_idx = train_val_idx[split_point * 4:]
    return train_idx, val_idx, test_idx

def data_split(data, device):
    x_sample, temporal_do = data['features'].to(device), data['time_series'].to(device)
    x_sample = torch.where(torch.isnan(x_sample), torch.tensor(0.0), x_sample).float()
    temporal_do = torch.where(torch.isnan(temporal_do), torch.tensor(0.0), temporal_do).float()
    temporal_do = torch.cat([temporal_do[:, :5, :], torch.zeros(temporal_do.shape[0], 1, 1).to(device), temporal_do[:, 5:, :]], dim=1)
    space_edge_index, space_edge_feature = data['space_edge_index'], data['space_edge_feature']
    inf_edge_index, inf_edge_feature = data['inf_edge_index'], data['inf_edge_feature'] 
    edge_index = torch.cat((space_edge_index, inf_edge_index), 1).to(device).long()
    edge_attr = torch.cat((space_edge_feature, inf_edge_feature), 0).to(device)
    edge_attr = torch.where(torch.isnan(edge_attr), torch.tensor(0.0), edge_attr).float()
    label = data['label'].to(device).float()
    return x_sample, temporal_do, edge_index, edge_attr, label

def batch_generator(data, batch_size):
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size  # 计算总批次数
    random.shuffle(data)
        
    for i in range(0, num_batches * batch_size, batch_size):
        batch = data[i:i + batch_size]
        yield batch


def main():
    model_pkl_name = f"model_pkl/STModel_{args.tips}_{args.k}.pkl"
    model_name = f"STModel_{args.tips}_{args.k}.pkl"
    # 配置logging
    logging.basicConfig(filename='logging/'+ model_name +'_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    datetime_now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print_and_log(datetime_now)
    begin_time = time.time()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print_and_log("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print_and_log("INFO: CPU")

    # 数据文件夹
    pwd = "./data/"
    file_list = os.listdir(pwd)
    pattern = re.compile(r'year_(\d+)_area_(\d+)')
    data_year_list = [int(re.search(pattern, file).group(1)) for file in file_list if re.search(pattern, file)]
    data_area_list = [int(re.search(pattern, file).group(2)) for file in file_list if re.search(pattern, file)]

    # 数据集划分
    train_idx, val_idx, test_idx = select_sample_idx(length=len(data_year_list), k=4, select_idx=args.k) # train:val:test = 7:1:2
    area_num = len(np.unique(data_area_list))
    if args.partition_by_physics:
        area_num = None

    model = STModel(args.input_dim, args.hidden_dim, args.edge_dim, args.layer_num, area_num=area_num, time_series_dim=args.time_series_dim, only_local_feature=args.only_local_feature).to(args.device)
    model_pkl_name = f"model_pkl/STModel_{args.tips}_{args.k}.pkl"
    for name, param in model.named_parameters():
        print_and_log('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.dynamic_lr:
        print_and_log('Dynamic learning rate')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    criterion = torch.nn.MSELoss(reduction='mean')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, model_name=model_pkl_name, save_model=args.save_model)
    total_epochs = int(args.train_epoches)

    broken_data_files = []
    for epoch_i in range(args.epoch):
        if epoch_i == 0:
            training_epoch = {key: 5 for key in range(32)}
        model.train()
        start_time = time.time()
        train_loss = 0 
        train_idx_i = random.choice(train_idx)
        try:
            data, valid_indices, pho_indices, nit_indices = get_one_data(pwd, data_year_list[train_idx_i], data_area_list[train_idx_i])
        except RuntimeError:
            file_path = pwd + "year_" + str(data_year_list[train_idx_i]) + "_area_" + str(data_area_list[train_idx_i])
            broken_data_files.append(file_path)
            continue
        while valid_indices == []:
            train_idx_i = random.choice(train_idx)
            try:
                data, valid_indices, pho_indices, nit_indices = get_one_data(pwd, data_year_list[train_idx_i], data_area_list[train_idx_i])
            except RuntimeError:
                file_path = pwd + "year_" + str(data_year_list[train_idx_i]) + "_area_" + str(data_area_list[train_idx_i])
                broken_data_files.append(file_path)
                continue
        area_id = data_area_list[train_idx_i]
        x_sample, temporal_do, edge_index, edge_attr, label = data_split(data, args.device)
        x_sample.requires_grad_()
        train_epoch = training_epoch[area_id]
        train_nit_loss = 0
        train_pho_loss = 0
        pho_grad = torch.tensor([0.0]).to(args.device)
        nit_grad = torch.tensor([0.0]).to(args.device)
        for _ in range(train_epoch):
            random.shuffle(valid_indices)
            batch = torch.tensor(valid_indices[: args.batch_size]).to('cuda')
            oxygen_pred = model(x_sample, temporal_do, edge_index, edge_attr, area_id)
            gradients = torch.autograd.grad(outputs=oxygen_pred, inputs=x_sample, grad_outputs=torch.ones_like(oxygen_pred), create_graph=True)
            pho_indices = torch.nonzero(x_sample[batch, 7]).view(-1)
            nit_indices = torch.nonzero(x_sample[batch, 8]).view(-1)
            pho_loss = gradients[0][batch[pho_indices],7].var() if len(pho_indices) > 1 else 0
            nit_loss = gradients[0][batch[nit_indices],8].var() if len(nit_indices) > 1 else 0
            if len(pho_indices) > 1:
                pho_grad = torch.hstack((pho_grad,gradients[0][batch[pho_indices],7]))
            if len(nit_indices) > 1:
                nit_grad = torch.hstack((nit_grad,gradients[0][batch[nit_indices],8]))                
            loss = criterion(oxygen_pred.squeeze(-1)[batch], (label[batch]))
            loss = loss + args.grad_lambda*pho_loss + args.grad_lambda*nit_loss
            if torch.isnan(loss):
                pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.dynamic_lr:
                scheduler.step()
            train_loss += loss / train_epoch
            train_nit_loss += nit_loss / train_epoch
            train_pho_loss += pho_loss / train_epoch
        if len(pho_grad)!=1:    
            pho_grad = pho_grad[1:].mean()
        else:
            pho_grad = 9999.999
            
        if len(nit_grad)!=1:
            nit_grad = nit_grad[1:].mean()
        else:
            nit_grad = 9999.999
        end_time = time.time()
        print_and_log(f"Epoch ({epoch_i}/{args.epoch})NITLoss is {train_nit_loss:.2f}.PHOLoss is {train_pho_loss:.2f}.PHOgrad is {pho_grad:.2f}.NITgrad is {nit_grad:.2f}.Loss is {train_loss:.2f}")

        del data
        if epoch_i % args.val_epoch == 0:
            model.eval()
            val_loss = 0
            area_losses = {}
            area_samples = {}
            with torch.no_grad():
                for id, val_idx_i in enumerate(tqdm(val_idx)):
                    try:
                        data, valid_indices, pho_indices, nit_indices = get_one_data(pwd, data_year_list[val_idx_i], data_area_list[val_idx_i])
                    except RuntimeError:
                        file_path = pwd + "year_" + str(data_year_list[val_idx_i]) + "_area_" + str(data_area_list[val_idx_i])
                        broken_data_files.append(file_path)
                    if valid_indices == []:
                        continue
                    area_id = data_area_list[val_idx_i]
                    if area_id not in area_losses:
                        area_losses[area_id] = 0
                        area_samples[area_id] = 0
                    x_sample, temporal_do, edge_index, edge_attr, label = data_split(data, args.device)
                    oxygen_pred = model(x_sample, temporal_do, edge_index, edge_attr, area_id)
                    # oxygen_pred = model(x_sample)
                    loss = criterion(oxygen_pred[valid_indices].squeeze(-1), label[valid_indices])
                    area_losses[area_id] += loss
                    area_samples[area_id] += 1
                    val_loss += loss
                early_stopping(val_loss, model)
                average_loss = {area: area_losses[area] / area_samples[area] for area in area_losses}
                loss_values = torch.stack(list(average_loss.values()))
                normalized_values = (loss_values - loss_values.min()) / (loss_values.max() - loss_values.min())
                softmax_probs = F.softmax(normalized_values, dim=0)
                training_epochs = (softmax_probs * total_epochs).int()
                training_epoch = {}
                for key, epochs in zip(average_loss.keys(), training_epochs):
                    training_epoch[key] = epochs.item()
                if early_stopping.early_stop:
                    break
            print_and_log(f"Epoch ({epoch_i}/{args.epoch}). Val Loss is {val_loss}")
            if args.dynamic_lr:
                print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    print_and_log(f"INFO: Train finish @{epoch_i}")  

    # test
    print_and_log("INFO: Start testing...")
    mae_list, mape_list, mse_list, rmse_list, r2_list, std_list = [], [], [], [], [], []
    model.load_state_dict(torch.load(model_pkl_name))  
    model.eval()
    count = 0 
    with torch.no_grad():
        for id, test_idx_i in enumerate(tqdm(test_idx)):
            try:
                data, valid_indices, pho_indices, nit_indices = get_one_data(pwd, data_year_list[test_idx_i], data_area_list[test_idx_i])

            except RuntimeError:
                file_path = pwd + "year_" + str(data_year_list[test_idx_i]) + "_area_" + str(data_area_list[test_idx_i])
                broken_data_files.append(file_path)

            if valid_indices == []:
                continue
            count +=1 
            area_id = data_area_list[test_idx_i]
            x_sample, temporal_do, edge_index, edge_attr, label = data_split(data, 'cuda')
            oxygen_pred = model(x_sample, temporal_do, edge_index, edge_attr, area_id)
            if count ==1:
                oxygen_preds = oxygen_pred[valid_indices]
                labels = label[valid_indices]
            else:
                oxygen_preds = torch.vstack((oxygen_preds, oxygen_pred[valid_indices]))
                labels = torch.hstack((labels, label[valid_indices]))
    print_and_log(set(broken_data_files))
    metric_result_dict = calculate_metrics(oxygen_preds, labels, metric_list=['MAE', 'MSE', 'MAPE', 'RMSE', 'R2', 'STD'])
    finish_time = time.time()
    print_and_log("-------------------------------")
    print_and_log(metric_result_dict)
    print_and_log("-------------------------------")
    print_and_log(f"Use Time:{finish_time - begin_time}")

    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_dim', type=int, default=15)
    argparser.add_argument('--hidden_dim', type=int, default=32)
    argparser.add_argument('--edge_dim', type=int, default=5)
    argparser.add_argument('--layer_num', type=int, default=2)
    argparser.add_argument('--output_dim', type=int, default=1)
    argparser.add_argument('--time_series_dim', type=int, default=1)
    argparser.add_argument('--epoch', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--grad_lambda', type=float, default=1)
    argparser.add_argument('--patience', type=int, default=70)
    argparser.add_argument('--train_epoches', type=int, default=500)
    argparser.add_argument('--val_epoch', type=int, default=100)
    argparser.add_argument('--k', type=int, default=1)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--save_model', type=bool, default=True)
    argparser.add_argument('--only_local_feature', action='store_true',default=False)
    argparser.add_argument('--partition_by_physics', action='store_false',default=True)
    argparser.add_argument('--cmip6', action='store_true',default=False)
    argparser.add_argument('--tips', type=str, default='None', help='message for different model settings')
    argparser.add_argument('--dynamic_lr', action='store_true',default=False)
    args = argparser.parse_args()
    main() 