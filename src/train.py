from genericpath import isfile
from re import T
import sys
sys.path.append("./src")



from model import ProBindNN
from dataset import MutationDataset

from torch_geometric.loader import DataLoader
import torch
from torch import nn

from tqdm import tqdm
from IPython.display import clear_output
import copy
import os
import time
from datetime import datetime


from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def train(model, loaders, optimizer, loss_fn, scheduler, n_epochs=1000):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    
    model.train()
    
    t = time.time()
    tstamp = datetime.utcfromtimestamp(t).strftime('%Y_%m_%d_%H_%M_%S')
    path = "logs/tensorboard/RUN_{}".format(tstamp)
    writer = SummaryWriter(path)
    
    for epoch in tqdm(range(1, n_epochs)):
        epoch_loss = 0.
        best_loss = 1000.
        
        for loader in loaders.keys():
            if loader == "train_loader":
                model.train()
                
                for i, batch in enumerate(loaders[loader]):    
                    x, y = batch["mutated"].to(device), batch["non_mutated"].to(device)
                    ddg = x.ddg.to(device).squeeze()
                    optimizer.zero_grad()
                    out = model(x,y).squeeze()
                    loss = loss_fn(out, ddg)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    clear_output(wait=True)
                    
                epoch_loss/=len(loaders[loader])
                
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))
            elif loader == "val_loader":
        
                model.eval()
                
                val_loss = 0

                for i, batch in enumerate(loaders[loader]):
                    x, y = batch["mutated"].to(device), batch["non_mutated"].to(device)
                    ddg = x.ddg.to(device).squeeze()
                    out = model(x,y).squeeze()
                    loss = loss_fn(out, ddg)
                    val_loss+=loss.item()
                    
                val_loss /= len(loaders[loader]) 
                writer.add_scalar("Loss/val", val_loss, epoch)
                print("Validation loss:", val_loss)
        
        if epoch_loss<best_loss:
            
            best_model = copy.deepcopy(model)
            t = time.time()
            stamp = datetime.utcfromtimestamp(t).strftime('%Y_%m_%d_%H_%M_%S')
            best_model_path = "models/model_{}.pt".format(tstamp, stamp)
            torch.save(model.state_dict(), best_model_path)
            
        else:
            
            scheduler.step()
    return best_model, best_model_path
    
if __name__ == "__main__":

    #Create dataset and dataloaders
    dataset = MutationDataset(index_xlsx="index.xlsx", root="dataset")
    train_size = int(len(dataset)*0.9)
    val_size = (len(dataset)-train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    loaders = {"val_loader": val_loader, "train_loader":train_loader}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProBindNN().to(device)

    if os.path.isfile("pretrained_model.pt"):
        model.load_state_dict(torch.load("pretrained_model.pt"))
        
    print("Using {} device".format(device))

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = scheduler = ExponentialLR(optimizer, gamma=0.9)

    loss_fn =  nn.MSELoss()
    epochs = 1500

    train(model, loaders, optimizer, loss_fn, scheduler, n_epochs=1500)