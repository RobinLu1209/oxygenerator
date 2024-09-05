import torch
import torch_geometric
import torch.nn as nn

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv, SAGEConv
from torch_geometric.utils import scatter, mask_feature
import torch.nn.functional as F
import pdb
class MLP(nn.Module):   
    def __init__(self, input_dim, hidden_dim, layer_num=2, activation='relu'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.activation = activation
        self.build()

    def build(self):
        mlp_layer = [
            nn.Linear(self.input_dim, self.hidden_dim)
        ]
        for _ in range(self.layer_num-1):
            mlp_layer.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.mlp_model = nn.ModuleList(mlp_layer)
    
    def forward(self, input):
        for i in range(self.layer_num):
            output = self.mlp_model[i](input)
            if self.activation == 'relu':
                input = torch.relu(output)
            else:
                input = output
        return output


class STModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, layer_num, time_series_dim, area_num, activation='relu', only_local_feature=False):
        super(STModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.layer_num = layer_num
        self.area_num = area_num
        self.activation = activation
        self.only_local_feature = only_local_feature
        self.time_series_dim = time_series_dim
        self.build()
    
    def build(self):
        self.mlp = MLP(self.input_dim, self.hidden_dim, layer_num=2)
        self.temporal_model = nn.LSTM(input_size=self.time_series_dim, hidden_size=16, num_layers = 2, bidirectional = True, batch_first=True)
        gnn_list = [
                OxygenGraphConv(self.hidden_dim + 32, self.hidden_dim, self.edge_dim, self.area_num)
            ]
        if self.layer_num > 1:
            for _ in range(self.layer_num - 1):
                gnn_list.append(OxygenGraphConv(self.hidden_dim, self.hidden_dim, self.edge_dim, self.area_num))
        self.gnn_layer = nn.ModuleList(gnn_list)
        # self.bb_linear = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, 1)
        self.edge_recons_linear = nn.Linear(2*self.hidden_dim, self.edge_dim)

    def forward(self, x_sample, temporal_do, edge_index, edge_attr, area_id, edge_recontruction=False):
        sample_feature = self.mlp(x_sample)
        gnn_output = sample_feature
        temporal_feature, (_, _) = self.temporal_model(temporal_do)
        gnn_input = torch.cat((sample_feature, temporal_feature[:,5,:]), 1)
        
        for i in range(self.layer_num):
            gnn_output = self.gnn_layer[i](gnn_input, edge_index, edge_attr, area_id, x_sample)
            if self.activation == 'relu':
                gnn_output = torch.relu(gnn_output)
            gnn_input = gnn_output
            
        oxygen_pred = self.output_linear(gnn_output)  
        return oxygen_pred
    



class OxygenGraphConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_dim, area_num=None):
        super().__init__(aggr='add')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.area_num = area_num
        self.build()

    def build(self):
        self.feature_transform = Linear(self.input_dim, self.output_dim, bias=False)
        self.edge_transform = Linear(self.edge_dim, 1)

        if self.area_num is not None:
            self.alpha_list = []
            self.beta_list = []
            for _ in range(self.area_num):
                alpha_k = Parameter(torch.eye(self.input_dim)).cuda()
                beta_k = Parameter(torch.zeros(1, self.output_dim)).cuda()
                self.alpha_list.append(alpha_k)
                self.beta_list.append(beta_k)
        else:
            self.alpha_transform =  nn.Linear(7, self.input_dim * self.input_dim)
            nn.init.zeros_(self.alpha_transform.weight)
            nn.init.ones_(self.alpha_transform.bias)
            self.beta_transform = nn.Linear(7, self.output_dim)
            # nn.init.zeros_(self.beta_transform.weight)
            # nn.init.zeros_(self.beta_transform.bias)

    def partition_by_physics(self, meta, input):
        alpha = self.alpha_transform(meta).view(-1, self.input_dim, self.input_dim)#size:N,Din,Din
        beta = self.beta_transform(meta)#size:N,Dout
        result = torch.bmm(alpha, input.unsqueeze(2)).squeeze()#size:N,Din
        return self.feature_transform(result) + beta #size:N,Dout
    
    def forward(self, x, edge_index, edge_attr, area_id=None, x_sample = None): #size: Size = None
        # x = self.feature_transform(x)
        if self.area_num is not None:
            # x_map = map(lambda id,x:self.feature_transform(torch.mm(x.unsqueeze(0),self.alpha_list[int(id)]))+self.beta_list[int(id)], area_id, x)
            # x_list = list(x_map)
            # x = torch.stack(x_list).squeeze(1)
            x = self.feature_transform(torch.mm(x, self.alpha_list[int(area_id)]))+self.beta_list[int(area_id)]
        else:
            meta = torch.cat((x_sample[:, 1:5], x_sample[:, -3:]), dim=1)
            x = self.partition_by_physics(meta, x)
        # print(f"After stack, x shape is {x.shape}")
        # edge_attr = torch.relu(self.edge_transform(edge_attr))
        edge_attr = self.edge_transform(edge_attr)
        edge_attr = torch.exp(edge_attr)
        edge_attr_sum = scatter(edge_attr, edge_index[0], reduce='sum')
        edge_attr = edge_attr / edge_attr_sum[edge_index[0]]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # out = self.propagate(edge_index, x=x, edge_attr=None)
        # print(f"After Message propagation, out shape is {out.shape}")
        out = out + x
        return out
    
    def message(self, x_j, edge_attr):
        msg = edge_attr.view(-1, 1) * x_j
        # msg = x_j
        return msg



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dim)
        # self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        # self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index , edge_attr):
        # 第一层 GCN'
        x = self.linear(x)
        x = self.conv1(x, edge_index)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x1 = self.conv2(x, edge_index)
        x1 = self.batch_norm_2(x1)
        x1 = F.relu(x1)
        x = x + x1
        output = self.fc(x)
        return output


class MLP_GNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, layer_name='GCN'):
        super(MLP_GNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp = MLP(26, self.hidden_dim, layer_num=2)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.layer_name = layer_name
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        print(layer_name)
        if layer_name == 'GCN':
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)        
        elif layer_name == 'GraphSAGE':
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif layer_name == 'GAT':       
            self.conv1 = GATConv(hidden_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
            self.edge_transform = Linear(5, 1)

    def forward(self, x, edge_index , edge_attr):
        gnn_input = self.mlp(x)
        if self.layer_name == 'GAT':
            edge_attr = self.edge_transform(edge_attr)
            edge_attr = torch.exp(edge_attr)
            edge_attr_sum = scatter(edge_attr, edge_index[0], reduce='sum')
            edge_attr = edge_attr / edge_attr_sum[edge_index[0]]
            gnn_output = self.conv1(gnn_input, edge_index, edge_attr)
            gnn_output = self.batch_norm_1(gnn_output)
            gnn_output = F.relu(gnn_output)
            gnn_input = gnn_output + gnn_input
            gnn_output = self.conv2(gnn_input, edge_index, edge_attr)
            gnn_output = self.batch_norm_2(gnn_output)
            output = self.output_layer(gnn_output)
        else:
            gnn_output = self.conv1(gnn_input, edge_index)
            gnn_output = self.batch_norm_1(gnn_output)
            gnn_output = F.relu(gnn_output)
            gnn_input = gnn_output + gnn_input
            gnn_output = self.conv2(gnn_input, edge_index)
            gnn_output = self.batch_norm_2(gnn_output)
            output = self.output_layer(gnn_output)
        return output

class MLP_Transformer(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        self.hidden_dim = hidden_dim
        super(MLP_Transformer, self).__init__()
        self.mlp = MLP(15, self.hidden_dim, layer_num=2)
        self.relu = nn.ReLU()
        self.linear_layer_0 = nn.Linear(1, hidden_dim)
        self.attention_layer_1 = nn.MultiheadAttention(self.hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm_att_1 = nn.BatchNorm1d(self.hidden_dim)
        self.linear_layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_layer_2 = nn.MultiheadAttention(self.hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm_att_2 = nn.BatchNorm1d(self.hidden_dim)
        self.linear_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, x_sample, temporal_do):  
        sample_feature = self.mlp(x_sample)
        att_input = self.linear_layer_0(temporal_do)
        att_output, _ = self.attention_layer_1(att_input, att_input, att_input)
        linear_input = att_output + att_input
        linear_output = self.linear_layer_1(linear_input)
        linear_output = torch.relu(linear_output)
        att_input = linear_output + linear_input
        att_output, _ = self.attention_layer_2(att_input, att_input, att_input)
        linear_input = att_output + att_input
        linear_output = self.linear_layer_2(linear_input)
        linear_output = torch.relu(linear_output)
        gnn_input = torch.cat((sample_feature, linear_output[:,5,:]), 1)
        oxygen_pred = self.output_layer(gnn_input)
        return oxygen_pred 

class MLP_LSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(MLP_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp = MLP(15, self.hidden_dim, layer_num=2)
        self.temporal_model = nn.LSTM(input_size=1, hidden_size=16, num_layers = 2, bidirectional = True, batch_first=True)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim + 32, output_dim)

    def forward(self, x_sample, temporal_do):
        sample_feature = self.mlp(x_sample)
        gnn_output = sample_feature
        temporal_feature, (_, _) = self.temporal_model(temporal_do)
        gnn_input = torch.cat((sample_feature, temporal_feature[:,5,:]), 1)
        oxygen_pred = self.output_layer(gnn_input)
        return oxygen_pred 


class LSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()

        self.temporal_model = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers = num_layers, bidirectional = True, batch_first=True)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Transformer forward pass
        temporal_feature, (_, _) = self.temporal_model(x)
        x = temporal_feature[:,20,:]
        output = self.output_layer(x)
        return output
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        # Transformer layers
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first = True)
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Transformer forward pass
        x = self.transformer(x)
        x = x[:, 20, :]
        output = self.output_layer(x)
        return output
    