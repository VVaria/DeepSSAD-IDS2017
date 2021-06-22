import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class IDSDataset(Dataset):
    def __init__(self, data_path, sip_map, dip_map, only_benign=False, only_anomaly=False, stat = None, transform=None):
        self.categorical_keys = ['Source IP', 'Destination IP', 'Source Port', 'Destination Port',
                                'Protocol']
        self.drop_keys = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk',
                         'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
                         'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                         'Timestamp', 'Fwd URG Flags', 'Fwd Header Length.1',
                         'Unnamed: 0', 'Unnamed: 0.1']
        self.label_key = ['Flow ID', 'Label']
        self.protocol_map = {0: 0, 6: 1, 17: 2}
        self.class_map = {
            'BENIGN': 0, 'DoS slowloris': 1, 'DoS Slowhttptest': 2, 'DoS Hulk': 3,
       'DoS GoldenEye': 4, 'Heartbleed': 5, 'FTP-Patator': 6, 'SSH-Patator': 7,
       'Web Attack \x96 Brute Force': 8, 'Web Attack \x96 XSS': 9,
       'Web Attack \x96 Sql Injection': 10, 'Infiltration': 11, 'Bot': 12, 'DDoS': 13,
       'PortScan': 14
        }
        self.data = pd.read_csv(data_path)
        self.data = self.data.rename(columns={s:s.strip() for s in self.data.keys()})
        self.data_labels =  self.data[self.label_key]
        self.data = self.data.drop(columns=self.drop_keys).replace(np.inf, np.nan).fillna(0)
        self.data = self.data.drop(columns=self.label_key)
        # self.data['Source IP'] = self.data['Source IP'].apply(lambda ip: int(''.join([hex(int(n)) for n in ip.split('.')]).replace('0x',''), 16))
        # self.data['Destination IP'] = self.data['Destination IP'].apply(lambda ip: int(''.join([hex(int(n)) for n in ip.split('.')]).replace('0x',''), 16))
        self.data['Source IP'] = self.data['Source IP'].apply(lambda ip: sip_map[ip])
        self.data['Destination IP'] = self.data['Destination IP'].apply(lambda ip: dip_map[ip])
        self.data['Protocol'] = self.data['Protocol'].apply(lambda pt: self.protocol_map[pt])
        self.continuous_keys = self.data.columns.difference(self.categorical_keys)
        self.data[self.continuous_keys] = self.normalize(self.data[self.continuous_keys], stat).replace(np.inf, np.nan).fillna(0)
        if only_benign and only_anomaly:
            pass
        elif only_benign:
            self.data = self.data[self.data_labels['Label'] == 'BENIGN']
        elif only_anomaly:
            self.data = self.data[self.data_labels['Label'] != 'BENIGN']
        
        self.transform = transform
    def __len__(self):
        return len(self.data)
    
    def normalize(self, pd_data, stat = None):
        if stat is None:
            stat = pd_data.describe()
        pd_data = (pd_data - stat.loc['min']) /  (stat.loc['max'] - stat.loc['min'])
        return pd_data
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        row_label = self.data_labels.iloc[index]
        categorical = row[self.categorical_keys].values
        continuous = row[self.continuous_keys].values
        label = self.class_map[row_label['Label']]
        #label = torch.eye(len(self.class_map))[self.class_map[row_label['Label']]]
        if self.transform:
            categorical = torch.from_numpy(categorical).type(torch.int64)
            continuous = torch.from_numpy(continuous).type(torch.float32)
        return categorical, continuous, label

