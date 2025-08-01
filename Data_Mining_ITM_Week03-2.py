# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:29:47 2025

@author: user
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

s = np.array([2, 2, 4, 5, 5, 5, 8, 9, 9, 9, 12])
len(s)

np.mean(s)
np.median(s)
np.std(s, ddof=0)
np.std(s, ddof=1)
np.var(s, ddof=0)
np.var(s, ddof=1)

np.min(s)
np.max(s)

np.quantile(s, 0.5)
np.quantile(s, 0.25)
np.quantile(s, 0.25, method='inverted_cdf')
np.quantile(s, 0.75)
np.quantile(s, 0.75, method='inverted_cdf')

iqr = np.quantile(s, 0.75, method='inverted_cdf') - np.quantile(s, 0.25, method='inverted_cdf')
iqr

scipy.stats.iqr(s)

scipy.stats.skew(s)
scipy.stats.kurtosis(s, fisher=False)
scipy.stats.kurtosis(s, fisher=True)


x = np.random.normal(0,1, size=200)

cnt, bins = np.histogram(x, bins=10)

xmin, xmax = np.min(x), np.max(x)


np.linspace(0,1,11)
np.linspace(xmin-0.5, xmax+0.5, 12)

cnt, bins = np.histogram(x, bins=np.linspace(xmin-0.5, xmax+0.5, 12))

fig = plt.figure()
plt.hist(x, bins=20, density=True)
plt.show()

fig = plt.figure()
plt.hist(x, bins=np.linspace(xmin-0.5, xmax+0.5, 12), density=True)
plt.show()

plt.boxplot(x)
plt.show()


x = np.random.normal(0,1, size=200)
mu = np.mean(x)
sigma = np.std(x, ddof=1)

Z = (x-mu)/sigma

Z

x[(Z<-3)|(Z>3)] 

x2 = scipy.stats.t.rvs(1, size=200)

plt.boxplot(x2)
plt.show()

Q1 = np.quantile(x, 0.25)
Q3 = np.quantile(x, 0.75)

IQR = Q3 - Q1

lower_inner_fence = Q1 - 1.5 * IQR
upper_inner_fence = Q3 + 1.5 * IQR

outliers = x2[(x2>upper_inner_fence)|(x2<lower_inner_fence)]

lower_outer_fence = Q1 - 3 * IQR
upper_outer_fence = Q3 + 3 * IQR

ext_outliers = outliers[(outliers<lower_outer_fence)|(outliers>upper_outer_fence)]
mild_outliers = outliers[(outliers>=lower_outer_fence)&(outliers<=upper_outer_fence)]

len(outliers)
len(ext_outliers)
len(mild_outliers)

import pandas as pd

df = pd.DataFrame(np.random.rand(50,3), columns=['A','B','C'])

df.corr()
df.corr(method='pearson')
df.corr(method='spearman')
df.corr(method='kendall')


cbar = plt.imshow(df.corr())
plt.colorbar(cbar)
plt.show()

plt.scatter(df['A'], df['B'])
plt.show()

df['D'] = np.random.randint(0, 2, size=50)


freq = df['D'].value_counts()

freq.index
plt.bar(freq.index, freq.values)
plt.show()


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

X = [['Male',1],['Female', 3],['Female', 2]]

enc.fit(X)


enc.transform(X).toarray()

enc.categories_

enc.get_feature_names_out(['gender', 'group'])

from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = [[0, 0], [0, 0], [1,1], [1, 1]]
scaler = StandardScaler()

scaler.fit(data)

scaler.transform(data)
scaler.mean_
scaler.transform([[2,2]])

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(data)
minmax_scaler.transform(data)
minmax_scaler.transform([[2,2]])












































