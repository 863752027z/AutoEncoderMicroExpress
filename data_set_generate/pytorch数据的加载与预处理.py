import torch
from torch.utils.data import Dataset
import pandas as pd


class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


file_path = 'F:/SAMM_FACE_CUT/median_benchmark.csv'
file_path = 'F:/SAMM_FACE_CUT/SAMM_Micro.csv'
ds_demo = BulldozerDataset(file_path)
print(ds_demo[0])
print(len(ds_demo))