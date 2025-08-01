import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sales=pd.read_csv('https://drive.google.com/uc?export=download&id=1n9SDdK2pFbM0H14ZSRLB8HpreFYBl6KH')
print(plt.scatter(sales['temperature'],sales['sales']))
plt.xlabel('TEMPERATURE')
plt.ylabel('SALES')
plt.show()
print(sales)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(sales['temperature'],sales['sales']) #Learning method is a fit.
#In this case x is in 2dim, y is in 1 dim
type(sales['temperature'])

reg.fit(sales[['temperature']],sales['sales'])
#series, dataframe, pannel(3 dimensional)

# reg.fit(sales['temperature'],sales['sales'])vs reg.fit(sales[['temperature']],sales['sales']) 
reg.coef_
reg.intercept_

reg.predict(sales[['temperature']]) #Estimate with the given input data matrix
sales['sales'].values


petrol=pd.read_csv('https://drive.google.com/uc?export=download&id=1R9B0D_fSjfCiSaS1WWEjEbSHFOXlVjQA')
print(petrol.columns)

reg.fit(petrol[['tax','income','highway','license']],petrol['consumption'])

reg.coef_
reg.intercept_

reg.predict([[9,3500,2000,0.6]])

# Linear Regression in respect to Ttest
#Caclulate statistics without any Library

#beta hat=(X'X)^-1X'Y
#Y=XB+e
#e=Y-XB
A=np.reshape(np.arrange(1,10),(3,3))
#A*A do elementwise multiplication
#A@A do matrix multiplication
np.matmul(A,A)
np.linalg.inv(A) #inverse of matrix
np.matmul(np.linalg.inv(A),A) #Identity matrix
np.matmul(np.linalg.inv(A),A) #Identity matrix



X=petrol[['tax','income','highway','license']].values
y=petrol['consumptioin'].values

np.ones(3)
np.ones((3,4))

n,p=X.shape
#Number of columns correspond to input variable
#number of rows correspond to number of observations(Samples)

n,p=X.shape #P is number of input varaiable
x1=np.c_[np.ones(n),X]

x1.shape
x1.T.shape

XtX=np.matmul(x1.T,x1)
XtX.shape

XtX_inv=np.linalg.inv(XtX)
np.matmul(XtX_inv,x1.T).shape
beta_hat=np.matmul(np.matmul(XtX_inv,x1.T),y)
beta_hat
beta=np.dot(np.matmul(XtX_inv,X.T),y)
print(beta)

print(beta,reg.coef_)

print(beta[0],reg.intercept_)

print(beta[1:],reg.coef_)

from scipy import stats
y_pred=reg.predict(X)
y_pred2=np.dot(x1,beta)

y-np.mean(y)

SST=np.sum((y-np.mean(y))**2)
print(SST)
SSE=(y-y_pred)**2
print(SSE)
SSR=np.sum((y_pred-np.mean(y))**2)
print(SSR)


MSE=SSE/(n-p-1) ##Mean Square Error Degree of Freedom 1 is mean 
MSR=SSR/p
f0=MSR/MSE
print(f0)
alpha=0.05
#If you want to get critical value
f_critical=stats.f.ppf(1-alpha,p,n-p-1) #Why 1-alpha? Because it is right tail test
print(f_critical)

#If you want to get P value;probability of observing the test statistic
p_value=1-stats.f.cdf(f0,p,n-p-1)
print(p_value)
#PDF => specify degree of freedom because 
#CDF => should specify degree of freedom because 
#PPF;Inverse function of CDF receive probability und return critical value


