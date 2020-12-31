import numpy as np
import pandas as pd

w = np.load('/public/lhy/wmy/dataset/Taian/slice_5min_month_2_3_4_5/adj_way_560.npy')
w = pd.DataFrame(w)
w.to_csv('./dataset/w.csv', index=False, header=None)

v = np.load('/public/lhy/wmy/dataset/Taian/slice_5min_month_2_3_4_5/way_volume_560.npy').squeeze(axis=-1)
v = pd.DataFrame(v)
v.to_csv('./dataset/v.csv', index=False, header=None)
