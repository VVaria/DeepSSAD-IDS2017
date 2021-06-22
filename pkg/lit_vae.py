import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class FCBatchNorm1d(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.BatchNorm1d(dim)
    def forward(self, x):
        return self.norm(x.transpose(-2, -1)).transpose(-2, -1)

class LitIDSVAE(pl.LightningModule):

    def __init__(self, sip_map, dip_map, z_dim, hidden_dim=512, continuous_dim=128, 
                 ip_dim = 12, port_dim = 16, protocol_dim = 2, lr = 3e-3, weight_decay = 1e-6, 
                warmup = 10, use_category = True):
        super().__init__()
        self.z_dim = z_dim
        self.continuous_dim = continuous_dim
        self.ip_dim = ip_dim
        self.port_dim = port_dim
        self.protocol_dim = protocol_dim
        self.use_category = use_category
        self.warmup = warmup
        self.beta = 0.0 if warmup > 0 else 1.0
        if self.use_category:
            self.categorical_dim = 2 * ip_dim + 2 * port_dim + protocol_dim
        else:
            self.categorical_dim = 0
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.save_hyperparameters('z_dim', 'hidden_dim', 'continuous_dim', 
                                  'ip_dim', 'port_dim', 'protocol_dim', 'lr', 'weight_decay',
                                  'use_category', 'warmup')
        self.continuous_emb = nn.Linear(67, continuous_dim)
        input_dim = continuous_dim
        
        if self.use_category:
            self.sip_emb = nn.Embedding(len(sip_map), ip_dim)
            self.dip_emb = nn.Embedding(len(dip_map), ip_dim)
            self.sport_emb = nn.Embedding(65536, port_dim)
            self.dport_emb = nn.Embedding(65536, port_dim)
            self.protocol_emb = nn.Embedding(3, protocol_dim)
            input_dim += 2 * ip_dim + 2 * port_dim + protocol_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            FCBatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            FCBatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            FCBatchNorm1d(hidden_dim // 4),
        )
        self.layer_mean = nn.Linear(hidden_dim // 4, z_dim, bias=False)
        self.layer_logvar = nn.Linear(hidden_dim // 4, z_dim, bias=False)
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            FCBatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            FCBatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            FCBatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )
        
        self.continuous_decoder = nn.Sequential(
            nn.Linear(continuous_dim, 67, bias=False),
            nn.Sigmoid()
        )
    def encode(self, x):
        h1 = self.encoder(x)
        return self.layer_mean(h1), self.layer_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.decoder(z)
        return h3

    def get_index_by_distance(self, matrix, vector):
        matrix = matrix / (torch.norm(matrix, dim=-1,keepdim=True) +1e-6)
        vector = vector / (torch.norm(vector, dim=-1,keepdim=True) + 1e-6)
        distance = (matrix - vector) ** 2
        return distance

    def decode_v2(self, z):
        out = self.decoder(z)
        continuous = self.continuous_decoder(out[...,self.categorical_dim:])
        if self.use_category:
            categories = out[...,:self.categorical_dim]
            sip_hat = categories[...,:self.ip_dim]
            dip_hat = categories[...,self.ip_dim:2 * self.ip_dim]
            sport_hat = categories[...,2 * self.ip_dim:2 * self.ip_dim + self.port_dim]
            dport_hat = categories[...,2 * self.ip_dim + self.port_dim:2 * self.ip_dim + 2 * self.port_dim]
            protocol_hat = categories[...,2 * self.ip_dim + 2 * self.port_dim:]        
            return sip_hat, dip_hat, sport_hat, dport_hat, protocol_hat, continuous
        else:
            return continuous

    def categorical_emb(self, cat, con):
        batch = cat.shape[0]
        continuous_y = self.continuous_emb(con)
        if self.use_category:
            cat_split = cat.split(1, dim=-1)
            sip_y = self.sip_emb(cat_split[0])
            dip_y = self.dip_emb(cat_split[1])
            sport_y = self.sport_emb(cat_split[2])
            dport_y = self.dport_emb(cat_split[3])
            protocol_y = self.protocol_emb(cat_split[4])
            return torch.cat([sip_y, dip_y, sport_y, dport_y, protocol_y, continuous_y], dim=-1)
        else:
            return continuous_y
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode_v2(z)
        return out, mu, logvar
    
    def distance_loss(self, distances, x_gt):
        loss = F.log_softmax(-distances, -2)
        loss = -1 * torch.gather(loss, -2, x_gt.reshape(-1,1,1))
        return loss
    def training_epoch_end(self, outputs):
        if self.warmup > 0 and self.beta != 1.:
            self.beta += 1 / self.warmup
    def training_step(self, batch, batch_idx):
        categorical = batch[0]
        continuous =  batch[1].unsqueeze(-2)
        x = self.categorical_emb(categorical, continuous) #.detach().cuda()
        x_hat, mu, logvar = self(x)
        
        kld = self.beta * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = -1))
        loss = kld
        if self.use_category:
            sip_hat, dip_hat, sport_hat, dport_hat, protocol_hat, continuous_hat = x_hat
            sip_dist = self.get_index_by_distance(self.sip_emb.weight.unsqueeze(0), sip_hat)
            dip_dist = self.get_index_by_distance(self.dip_emb.weight.unsqueeze(0), dip_hat)
            sport_dist = self.get_index_by_distance(self.sport_emb.weight.unsqueeze(0), sport_hat)
            dport_dist = self.get_index_by_distance(self.dport_emb.weight.unsqueeze(0), dport_hat)
            protocol_dist = self.get_index_by_distance(self.protocol_emb.weight.unsqueeze(0), protocol_hat)
            sip_loss = torch.mean(self.distance_loss(sip_dist, categorical[...,0]))
            dip_loss = torch.mean(self.distance_loss(dip_dist, categorical[...,1]))
            sport_loss = torch.mean(self.distance_loss(sport_dist, categorical[...,2]))
            dport_loss = torch.mean(self.distance_loss(dport_dist, categorical[...,3]))
            protocol_loss = torch.mean(self.distance_loss(protocol_dist, categorical[...,4]))
            loss += sip_loss + dip_loss + sport_loss + dport_loss + protocol_loss
            self.log('train/sip', sip_loss)
            self.log('train/dip', dip_loss)
            self.log('train/sport', sport_loss)
            self.log('train/dport', dport_loss)
            self.log('train/protocol', protocol_loss)
        else:
            continuous_hat = x_hat
            
        mse = F.mse_loss(continuous_hat, continuous, reduction='sum')
        loss = torch.mean(mse + loss)
        self.log('train/mse', mse)
        self.log('train/kld', kld)
        self.log('train/total', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    