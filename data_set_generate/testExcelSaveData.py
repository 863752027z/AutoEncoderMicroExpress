import pandas as pd
import numpy as np

def save_data_to_excel(data, path):
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer, 'page_1', float_format='%.5f') # float_format 控制精度
    writer.save()


x = np.arange(6).reshape(2, 3)
print(x)
path = 'F:/SaveFeature/test.xlsx'
save_data_to_excel(x, path)