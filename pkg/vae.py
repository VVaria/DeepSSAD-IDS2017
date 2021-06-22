import torch.nn.functional as F
import torch.nn as nn

class IDSVAE(nn.Module):
    def __init__(self, sip_map, dip_map, z_size, hidden=512, continuous_dim=128, 
                 ip_dim = 12, port_dim = 16, protocol_dim = 2):
        super().__init__()
        
        self.z_dim = z_size
        self.continuous_dim = continuous_dim
        self.ip_dim = ip_dim
        self.port_dim = port_dim
        self.protocol_dim = protocol_dim
        self.categorical_dim = 2 * ip_dim + 2 * port_dim + protocol_dim
        
        input_dim = continuous_dim + 2 * ip_dim + 2 * port_dim + protocol_dim
        
        self.sip_emb = nn.Embedding(len(sip_map), ip_dim)
        self.dip_emb = nn.Embedding(len(dip_map), ip_dim)
        self.sport_emb = nn.Embedding(65536, port_dim)
        self.dport_emb = nn.Embedding(65536, port_dim)
        self.protocol_emb = nn.Embedding(3, protocol_dim)
        self.continuous_emb = nn.Linear(67, continuous_dim)
        self.continuous_decoder = nn.Sequential(
            nn.Linear(continuous_dim, 67),
            nn.Sigmoid()
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, hidden // 4, bias=False),
            nn.ReLU(inplace=True),
        )
        self.layer_mean = nn.Linear(hidden // 4, z_size, bias=False)
        self.layer_logvar = nn.Linear(hidden // 4, z_size, bias=False)
        
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 4, hidden // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, input_dim, bias=False),
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
        matrix = matrix / torch.norm(matrix, dim=-1,keepdim=True)
        vector = vector / torch.norm(vector, dim=-1,keepdim=True)
        distance = (matrix - vector) ** 2
        return distance

    def decode_v2(self, z):
        out = self.decoder(z)
        
        categories = out[...,:self.categorical_dim]
        sip_hat = categories[...,:self.ip_dim]
        dip_hat = categories[...,self.ip_dim:2 * self.ip_dim]
        sport_hat = categories[...,2 * self.ip_dim:2 * self.ip_dim + self.port_dim]
        dport_hat = categories[...,2 * self.ip_dim + self.port_dim:2 * self.ip_dim + 2 * self.port_dim]
        protocol_hat = categories[...,2 * self.ip_dim + 2 * self.port_dim:]        
        continuous = self.continuous_decoder(out[...,self.categorical_dim:])
        return sip_hat, dip_hat, sport_hat, dport_hat, protocol_hat, continuous

    def categorical_emb(self, cat, con):
        batch = cat.shape[0]
        cat_split = cat.split(1, dim=-1)
        sip_y = self.sip_emb(cat_split[0])
        dip_y = self.dip_emb(cat_split[1])
        sport_y = self.sport_emb(cat_split[2])
        dport_y = self.dport_emb(cat_split[3])
        protocol_y = self.protocol_emb(cat_split[4])
        continuous_y = self.continuous_emb(con)
        return torch.cat([sip_y, dip_y, sport_y, dport_y, protocol_y, continuous_y], dim=-1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_v2(z), mu, logvar