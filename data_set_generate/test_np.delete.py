import numpy as np
import pandas as pd
import datetime
import torch

print(datetime.datetime.now())
path = 'F:/SaveFeature/test.xlsx'
sheet = 'page_1'
df = pd.read_excel(path, sheet)
M = np.array(df)
print(M)
M = np.delete(M, 0, axis=1)
print(M)
#print(M.shape)
#print(datetime.datetime.now())

a = torch.arange(8).view(1, 8)
b = torch.arange(8).view(1, 8)
c = torch.cat((a, b), 0)
print(c)
