import torch
from torch.utils.data import Dataset
import pandas as pd

class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].SalePrice

if __name__ == '__main__':
    csv_file = 'F:/ZLW/kaggle/median_benchmark.csv'
    ds_demo = BulldozerDataset(csv_file)
    print(ds_demo)
    print(len(ds_demo))
    dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)
    idata = iter(dl)
    print(next(idata))
    for i, data in enumerate(dl):
        print(i, data)