import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GatedGraphConv, GCNConv, GraphConv 
from torch import nn
from torch_geometric.nn.norm import GraphNorm, LayerNorm


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
                self.mlp.add_module('act_{}'.format(i), nn.ReLU())
    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)

from torch_geometric.nn import global_mean_pool

class GGNN(torch.nn.Module):
    """
    A Gated Graph Neural Network.
    """
    def __init__(self, features_in, features_hidden, features_out):
        super(GGNN, self).__init__()
         
        
        self.conv1 = GatedGraphConv(features_hidden, 4)
        self.gn1 = GraphNorm(features_hidden)

        self.conv2 = GatedGraphConv(features_hidden, 3)
        self.gn2 = GraphNorm(features_hidden)

        self.conv3 = GatedGraphConv(features_hidden, 2)
        self.gn3 = GraphNorm(features_hidden)

        self.conv4 = GatedGraphConv(features_hidden, 2)
        self.gn4 = GraphNorm(features_hidden)

        self.conv5 = GatedGraphConv(features_hidden, 1)
        
    
        
    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
       
        o = self.conv1(x, edge_index)
        o = self.gn1(o)
        o = torch.nn.functional.leaky_relu(o)
      

        o = self.conv2(o, edge_index)
        o = torch.nn.functional.leaky_relu(o)
        o = self.gn2(o)
        
        o = self.conv3(o, edge_index)
        o = torch.nn.functional.leaky_relu(o)
        o = self.gn3(o)

       

        o = self.conv4(o, edge_index)
        o = torch.nn.functional.leaky_relu(o)
        o = self.gn4(o)
        
        
        o = self.conv5(o, edge_index)
        o = torch.nn.functional.leaky_relu(o)

        o = global_mean_pool(o, batch)

        return o


class ddGPredictor(torch.nn.Module):
    """
    The model which calculates the final ddG values.
    """
    def __init__(self,config = { "features_in" :18, "features_hidden":30, "gnn_features_out":30, "mlp_hidden_dim" : [50, 50], "out_dim":1}):
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
