from torch.utils.data import DataLoader, Dataset

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np

class RandDataset(Dataset):
    
    def __init__(self, N, F):
        # N: data size
        # F: feature dim
        self.N = N
        self.F = F
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return torch.rand(self.F), torch.rand(1)
    
    
class MLP(pl.LightningModule):
    def __init__(self, M, F, epochs):
        # M: number of layers
        # F: feature dim
        super(MLP, self).__init__()
        self.epochs = epochs
        layers = []
        for i in range(M):
            layers.append(nn.Linear(F,F))
        layers.append(nn.Linear(F,1))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def training_step(self, batch, batch_index):
        x,y = batch
        yp = self.forward(x)
        loss = F.l1_loss(yp,y)
        self.log(f"mae", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)
        slr = torch.optim.lr_scheduler.CosineAnnealingLR(adam, self.epochs)
        return [adam], [slr]
        
        
if __name__ == '__main__':
    N = 10000
    F = 10
    M = 100
    epochs = 10000
    
    ds = RandDataset(N,F)
    x,y = ds[0]
    print(x)
    print(y)
    
    batch_size = 1024
    shuffle = True
    cpu_workers = 4
    drop_last = True
    dl = DataLoader(ds, batch_size=batch_size,
                    shuffle=shuffle, num_workers=cpu_workers,
                    drop_last=drop_last)
    
    x,y = next(iter(dl))
    print(x.shape, y.shape)
    
    model = MLP(M,F,epochs)
    yp = model(x)
    print(yp.shape)