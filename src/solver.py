import numpy as np
from math import isclose
from torch.nn import MSELoss
from torch.optim import Adam
import torch

class Solver(object):
    def __init__(self, model, train_dataloader, val_dataloader,
                 loss_func=MSELoss(), learning_rate=1e-3,
                 optimizer=Adam, verbose=True, print_every=1, lr_decay = 1.0,
                 **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.loss_func = loss_func

        self.opt = optimizer(model, loss_func, learning_rate)

        self.verbose = verbose
        self.print_every = print_every

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.current_patience = 0

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []
        self.current_patience = 0

    

    def train(self, epochs=100, patience = None):
        """
        Run optimization to train the model.
        """

        # Start an epoch
        for t in range(epochs):

            # Iterate over all training samples
            train_epoch_loss = 0.0

            for i, batch in enumerate(self.train_loader) :
                x, y = batch["mutated"].to(self.device), batch["non_mutated"].to(self.device)
                ddg = x.ddg.to(self.device).squeeze()
                self.opt.zero_grad()
                out = self.model(x,y).squeeze()
                train_loss = self.loss_func(out, ddg)
                train_loss.backward()
                self.opt.step()
                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

            train_epoch_loss /= len(self.train_dataloader)

            
            
            
            
            
            val_epoch_loss = 0.0

            for batch in self.val_dataloader:
                x, y = batch["mutated"].to(self.device), batch["non_mutated"].to(self.device)
                ddg = x.ddg.to(self.device).squeeze()
                out = self.model(x,y).squeeze()
                val_loss = self.loss_func(out, ddg)
                self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss

            val_epoch_loss /= len(self.val_dataloader)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                    t + 1, epochs, train_epoch_loss, val_epoch_loss))

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss)
            if patience and self.current_patience >= patience:
                print("Stopping early at epoch {}!".format(t))
                break

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

    def update_best_loss(self, val_loss, train_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss":val_loss, "train_loss":train_loss}
            self.best_params = self.model.params
            self.current_patience = 0
        else:
            self.current_patience += 1
