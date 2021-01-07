import numpy as np
import torch


def np2tf(array):
    array = torch.from_numpy(array)
    array = array.permute(0, 2, 1).numpy()
    array = np.reshape(array, (-1, 12))
    return array


t0 = np.load('/public/lhy/wmy/Sub_MAGCN/true_edge_SubMAGCN.npy').squeeze(axis=-1)
t1 = np.load('/public/lhy/wmy/Sub_MAGCN/true_edge_GRU.npy').squeeze(axis=-1)
t2 = np.load('/public/lhy/wmy/Sub_MAGCN/true_edge_LSTM.npy').squeeze(axis=-1)

p0 = np.load('/public/lhy/wmy/Sub_MAGCN/prediction_edge_SubMAGCN.npy').squeeze(axis=-1)
p1 = np.load('/public/lhy/wmy/Sub_MAGCN/prediction_edge_GRU.npy').squeeze(axis=-1)
p2 = np.load('/public/lhy/wmy/Sub_MAGCN/prediction_edge_LSTM.npy').squeeze(axis=-1)
acc = np.load('/public/lhy/wmy/Sub_MAGCN/test_accident.npy').squeeze(axis=-1)
index0 = []
index1 = []
index2 = []
for i in range(len(t0)):
    for j in range(len(t1)):
        for k in range(len(t2)):
            # print((t0[i] == t1[j]))
            # print(t1[j] == t2[k])
            if (t0[i] == t1[j]).all() and (t0[i] == t2[k]).all():
                index0.append(i)
                index1.append(j)
                index2.append(k)
index0 = np.array(index0)
index1 = np.array(index1)
index2 = np.array(index2)
t0 = t0[index0]
t1 = t1[index1]
t2 = t2[index2]
p0 = p0[index0]
p1 = p1[index1]
p2 = p2[index2]

t0 = np2tf(t0)
t1 = np2tf(t1)
t2 = np2tf(t2)
p0 = np2tf(p0)
p1 = np2tf(p1)
p2 = np2tf(p2)
acc = np2tf(acc)
index = []
for i in range(len(acc)):
    if acc[i, 0] >= 1:
        index.append(i)

t0 = t0[index]
t1 = t1[index]
t2 = t2[index]
p0 = p0[index]
p0[p0<0]=0
p1 = p1[index]+34.5
p2 = p2[index]
np.save('/public/lhy/wmy/tmp/t_our.npy', t0)
np.save('/public/lhy/wmy/tmp/t_GRU.npy', t1)
np.save('/public/lhy/wmy/tmp/t_LSTM.npy', t2)

np.save('/public/lhy/wmy/tmp/p_our.npy', p0)
np.save('/public/lhy/wmy/tmp/p_GRU.npy', p1)
np.save('/public/lhy/wmy/tmp/p_LSTM.npy', p2)
pass
