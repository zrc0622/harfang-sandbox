import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, lr, stateDim, nActions, full1Dim, full2Dim, layerNorm, name):
        super(Actor,self).__init__()
        
        self.layerNorm = layerNorm
        
        self.full1 = nn.Linear(stateDim,full1Dim)
        nn.init.kaiming_uniform_(self.full1.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm1 = nn.LayerNorm(full1Dim)
        
        self.full2 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full2.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm2 = nn.LayerNorm(full2Dim)
        
        self.final = nn.Linear(full2Dim,nActions)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,x):
        if self.layerNorm:
            
            x = F.leaky_relu(self.layernorm1(self.full1(x)))
            x = F.leaky_relu(self.layernorm2(self.full2(x)))

        else:
            
            x = F.leaky_relu(self.full1(x))
            x = F.leaky_relu(self.full2(x))

        
        return torch.tanh(self.final(x))
    
    def saveCheckpoint(self,ajan):
        torch.save(self.state_dict(),'.\\' + self.name + '\\{}'.format(ajan)  + self.name)
        
    def loadCheckpoint(self,ajan):
        self.load_state_dict(torch.load('.\\' + self.name + '\\{}'.format(ajan) + self.name))

agent = Actor(lr=1e-3, stateDim=14, nActions=4, full1Dim=256, full2Dim=512, layerNorm=True, name='harfang')
