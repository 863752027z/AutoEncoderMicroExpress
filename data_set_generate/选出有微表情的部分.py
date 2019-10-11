import torch
import cv2
import shutil
import os
from torch.utils.data import Dataset
import pandas as pd


class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


file_path = 'F:/SAMM_FACE_CUT/SAMM_Micro.csv'
ds_demo = BulldozerDataset(file_path)

source = []
target = []
source_path = 'F:/SAMM_FACE_CUT_1/SAMM/'
target_path = 'F:/SAMM_FACE_CUT_1/MicroExpress/'

for i in range(len(ds_demo)):
    Filename = ds_demo[i]['Filename']
    OnsetFrame = ds_demo[i]['Onset Frame']
    Duration = ds_demo[i]['Duration']
    print(Filename, OnsetFrame, Duration)
    file1 = Filename[0:3]
    file2 = Filename[4:5]
    file3 = Filename[6:]
    path1 = source_path + file1 + '/' + file2 + '/'
    path2 = target_path + Filename + '/'
    if not os.path.exists(path2[:-1]):
        os.mkdir(path2[:-1])
    dirs = os.listdir(path1)
    mod = dirs[0]
    for j in range(Duration):
        path1 = source_path + file1 + '/' + file2 + '/'
        path2 = target_path + Filename + '/'
        curr_frame = OnsetFrame + j
        curr_frame = str(curr_frame)
        for k in range(len(mod)-8 - len(curr_frame)):
            curr_frame = '0' + curr_frame
        path1 = path1 + Filename[0:3] + '_' + curr_frame + '.jpg'
        path2 = path2 + Filename[0:3] + '_' + curr_frame + '.jpg'
        shutil.copyfile(path1, path2)
        print('copy to', path2)
        os.remove(path1)
        print('remove', path2)