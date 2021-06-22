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
    
class LitIDSVAE_SAD(pl.LightningModule):

    def __init__(self, sip_map, dip_map, z_dim, hidden_dim=512, continuous_dim=128, 
                 ip_dim = 12, port_dim = 16, protocol_dim = 2, eta = 1., 
                 lr = 3e-3, weight_decay = 1e-6,use_category = True, eps = 1e-6):
        super().__init__()
        self.z_dim = z_dim
        self.continuous_dim = continuous_dim
        self.ip_dim = ip_dim
        self.port_dim = port_dim
        self.protocol_dim = protocol_dim
        self.use_category = use_category
        if self.use_category:
            self.categorical_dim = 2 * ip_dim + 2 * port_dim + protocol_dim
        else:
            self.categorical_dim = 0
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta = eta
        self.eps = eps
        self.save_hyperparameters('z_dim', 'hidden_dim', 'continuous_dim', 
                                  'ip_dim', 'port_dim', 'protocol_dim', 
                                  'eta', 'eps', 'lr', 'weight_decay', 'use_category')
        
        self.continuous_emb = nn.Linear(67, continuous_dim)
        input_dim = continuous_dim
        if self.use_category:
            self.sip_emb = nn.Embedding(len(sip_map), ip_dim)
            self.dip_emb = nn.Embedding(len(dip_map), ip_dim)
            self.sport_emb = nn.Embedding(65536, port_dim)
            self.dport_emb = nn.Embedding(65536, port_dim)
            self.protocol_emb = nn.Embedding(3, protocol_dim)
            input_dim += 2 * ip_dim + 2 * port_dim + protocol_dim
        self.register_buffer('m_c', torch.zeros(z_dim).unsqueeze(0))
        self.register_buffer('v_c', torch.zeros(z_dim).unsqueeze(0))
    
 
        
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False),
#             nn.ReLU(inplace=True),
#         )
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
    
    def init_center_c(self, train_loader,  eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        m_c = torch.zeros(self.z_dim, device=self.device)
        v_c = torch.zeros(self.z_dim, device=self.device)
        
        self.eval()
        with torch.no_grad():
            for row in train_loader:
                # get the inputs of the batch
                categorical = row[0].to(self.device)
                continuous =  row[1].to(self.device).unsqueeze(-2)
                x = self.categorical_emb(categorical, continuous) #.detach().cuda()
                mu, var = self(x)
                n_samples += mu.shape[0]
                m_c += torch.sum(mu, dim=0).squeeze()
                v_c += torch.sum(var, dim=0).squeeze()
                

        m_c /= n_samples
        v_c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
#         v_c[(abs(v_c) < eps) & (v_c < 0)] = -eps
#         v_c[(abs(v_c) < eps) & (v_c > 0)] = eps
        self.m_c = m_c.unsqueeze(0)
        self.v_c = v_c.unsqueeze(0)
    
    def init_encoder(self, state_dict):
        net_dict = self.state_dict()

        # Filter out decoder network keys
        state_dict = {k: v for k, v in state_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(state_dict)
        # Load the new state_dict
        self.load_state_dict(net_dict)
        
    def encode(self, x):
        h1 = self.encoder(x)
        return self.layer_mean(h1), self.layer_logvar(h1)

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
        return  mu.squeeze(-2), logvar.squeeze(-2)
    
    
    def training_step(self, batch, batch_idx):
        categorical = batch[0]
        continuous =  batch[1].unsqueeze(-2)
        labels = batch[2].unsqueeze(-1)
        x = self.categorical_emb(categorical, continuous)
        mu_p, logvar_p = self(x)

#         dist = 0.5 * torch.sum((self.v_c - logvar_p  - 1 + (mu_p - self.m_c) * \
#                     (1 / self.v_c.exp()) * (mu_p - self.m_c) + \
#                     (1 / self.v_c.exp()) * logvar_p.exp()), dim = -1)
        dist = 0.5 * (logvar_p - self.v_c  - 1 + (self.m_c - mu_p) * \
                    (1 / logvar_p.exp()) * (self.m_c - mu_p) + \
                    (1 / logvar_p.exp()) * self.v_c.exp())
        losses = torch.where(labels == 0, dist, self.eta * ((dist + self.eps) ** -1))
        loss = torch.mean(losses)
        self.log('train/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    