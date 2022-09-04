from turtle import forward
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GatedGraphConv
from torch import nn
from torch_geometric.nn.norm import GraphNorm, LayerNorm
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron model.
    """
    def __init__(self, input_dim, hidden_dims, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hidden_dims + [out_dim]
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            #self.mlp.add_module("dropout_{}".format(i), nn.Dropout(0.1))
            if i+2 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.LeakyReLU())
    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)



class GGNN(torch.nn.Module):
    """
    A Gated Graph Neural Network.
    """
    def __init__(self, features_in, features_hidden, features_out):
        super(GGNN, self).__init__()
         
        self.conv1 = GatedGraphConv(features_in, 50)
        
        
        
    def forward(self, data, batch):
        
        
        
        x, edge_index = data.x, data.edge_index
        ew = data.edge_weights
        
        
        o = self.conv1(x, edge_index, ew)
        o = torch.nn.functional.leaky_relu(o)
      
        o = global_add_pool(o, batch)
        
        return o


class ddGPredictor(torch.nn.Module):
    """
    The model which calculates the final ddG values.
    """
    def __init__(self,config = { "features_in" :15, "features_hidden":15, "gnn_features_out":15, "mlp_hidden_dim" : [15, 15, 15], "out_dim":1}):
        super(ddGPredictor, self).__init__()

        self.config = config
    
        self.mlp = MLP(config["gnn_features_out"], config["mlp_hidden_dim"], config["out_dim"])
       
        self.model_a = GGNN(config["features_in"], config["features_hidden"], config["gnn_features_out"])
        self.model_b = GGNN(config["features_in"], config["features_hidden"], config["gnn_features_out"])
      
    def forward(self, x, y):
        
        msg_x = self.model_a(x, x.batch)
        msg_y = self.model_b(y, y.batch)        

        out = msg_x- msg_y


        out = self.mlp(out)
        return out
    
    
    
class ProBindNN(torch.nn.Module):
    """
        The neural network designed for ddG prediction.
        config: dict
            The configuration dictionary with relevant parameters for the underlying architecture.
            features in: int
                Size of a feature vector
            layers: int
                Passes of the GRU in each GNN 
            gnn_features_out: int
                The feature vectors will be padded to this value, it has to be greater than features in
            out_dim: int
                The dimension of the values that we want to predict. In our case its just the ddG (1), but it can be extended for other descriptors.
            mlp_hidden_dim: List[int]
                A list of length of the layers in the MLP with does the final predictions, integers in the list are the number of neurons for each layer. 
    """
    def __init__(self, config={"features_in":15, "layers":30, "gnn_features_out":15, "out_dim":1, "mlp_hidden_dim":[15, 15, 15]}):
        super(ProBindNN, self).__init__()
    
        self.GGNN_a = GatedGraphConv(config["features_in"], config["layers"])
        self.GGNN_b = GatedGraphConv(config["features_in"], config["layers"])
        
        self.mlp = MLP(config["gnn_features_out"], config["mlp_hidden_dim"], config["out_dim"])
    
    def forward(self, x, y):
        out_a = self.GGNN_a(x.x, x.edge_index, x.edge_weights)
        out_b = self.GGNN_b(y.x, y.edge_index, y.edge_weights)
        
        out_a = torch.nn.functional.leaky_relu(out_a)
        out_b = torch.nn.functional.leaky_relu(out_b)
        
        out_a = global_add_pool(out_a, x.batch)
        out_b = global_add_pool(out_b, y.batch)
        
        out =out_a-out_b
        
        out = self.mlp(out)
        
        return out
    
    
