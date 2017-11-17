# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xlrd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
def open_exl(address, idx):
    data = xlrd.open_workbook(address)
    table = data.sheets()[idx]
    rows = table.nrows
    ct_data = []
    for row in range(rows):
        ct_data.append(table.row_values(row))
    return np.array(ct_data)
def asfloat(data):
    new = np.zeros_like(data, dtype=np.float64)
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            new[i, j] = my_float(data[i, j])
    return new

def my_float(x):
    try:
        return float(x)
    except:
        try:
            return float(x[:-1])
        except:
            return float(x[:-2])

asia = open_exl('pubg_as.xls', 0)
eu = open_exl('pubg_eu.xls', 0)
na = open_exl('pubg_na.xls', 0)
oc = open_exl('pubg_oc.xls', 0)
sea = open_exl('pubg_sea.xls', 0)

data = np.vstack((asia, eu[1: ], na[1: ], oc[1: ], sea[1: ]))
df = pd.DataFrame({data[0, i]:data[1:, i] for i in range(data.shape[1])})

m = asfloat(data[1:, :3])
kmeans = KMeans(n_clusters=5).fit(m)
labels = kmeans.labels_
'''
colors = ['#33cc00', '#0099cc', '#E44B4E', '#cc0066', '#0282C9']
c_list = [colors[labels[i]] for i in range(m.shape[0])]

center = kmeans.cluster_centers_

plt.figure(figsize=(12, 9))
ax1 = plt.subplot(111,projection='3d')
x,y,z = m[:,0],m[:,1],m[:,2]

x_c, y_c, z_c = center[:, 0], center[:, 1], center[:, 2]
ax1.scatter(x, y, z, s=15, color=c_list, alpha=0.5)
#ax1.scatter(x_c, y_c, z_c, s=120, alpha=0.5, c='black')
ax1.set_title('Data from https://PUBG.me Built by KMeans')
ax1.set_zlabel('Headshot Kill Ratio')
ax1.set_ylabel('KDA')
ax1.set_xlabel('Win Rate')
#plt.show()
'''
colors = ['#164a84', '#ffd900', '#028760', '#a22041', '#0d0015']
label_list = ['Normal', 'Mixed', 'God eyes', 'Auto aim', 'gǒu zéi']

plt.figure(figsize=(19, 7))
ax = plt.subplot(122,projection='3d')
for i in range(5):
    c = m[labels == i]
    x,y,z = c[:,0],c[:,1],c[:,2]
    ax.scatter(x, y, z, s=15, color=colors[i], label=label_list[i])

ax.legend()
ax.set_title('Data from PUBG.me Built by KMeans')
ax.set_zlabel('Headshot Kill Ratio')
ax.set_ylabel('KDA')
ax.set_xlabel('Win Rate')

ax1 = plt.subplot(121,projection='3d')
x,y,z = m[:,0],m[:,1],m[:,2]
ax1.scatter(x, y, z, s=15)
ax1.set_title('Data of PUBG without Labels')
ax1.set_zlabel('Headshot Kill Ratio') #坐标轴
ax1.set_ylabel('KDA')
ax1.set_xlabel('Win Rate')


plt.show()