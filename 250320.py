import numpy as np
import scipy.stats
s=np.array([2,2,4,5,5,5,8,9,9,9,12])
len(s)
np.mean(s)
np.median(s)
np.std(s,ddof=0)
np.std(s,ddof=1) #degree of freedom=1
np.var(s,ddof=0)#Population variance
np.var(s,ddof=1)#Sample variance
np.min(s)
np.max(s)
np.percentile(s,25)
np.percentile(s,75)
np.percentile(s,50)
np.percentile(s,75)-np.percentile(s,25) #IQR 

np.quantile(s,0.5)
np.quantile(s,0.25)
np.quantile(s,0.75)
np.quantile(s,0.75)-np.quantile(s,0.25) #IQR
#Quantile use linear interpolation Q?=(1-w)X[i]+wX[i+1]
np.quantile(s,0.25,interpolation='inverted_cdf')
np.quantile(s,0.75,interpolation='inverted_cdf')

iqr=np.quantile(s,0.75,'inverted_cdf')-np.quantile(s,0.25,method='inverted_cdf')

scipy.stats.iqr(s) #Use Linear interpolation



# To caclulate skeness
scipy.stats.skew(s)
scipy.stats.kurtosis(s,fisher=False)
scipy.stats.kurtosis(s,fisher=True) #Excess kurtosis, Negative value which means smaller kurtosis than normal distribution


#Frequency Table <= Histogram
cnt,bins=np.histogram(s,bins=4) #split into 10 bins # First array defines the frequency, second array defines the bin

#just set boundary of the bins
cnt,bins=np.histogram(s,bins=[0,3,6,9,12]) #split into 10 bins # First array defines the frequency, second array defines the bin
xmin,xmax=np.min(s),np.max(s)


np.linspace(xmin-0.5,xmax+0.5,12) #Define first und last, split into 12 bins

cnt,bins=np.histogram(s,bins=np.linspace(xmin-0.5,xmax+0.5,12)) #

fig=plt.figure()
plt.hist(x,bins=20)
plt.show()
#If you want to infer the probability from this histogram,
flg=plt.figure()
plt.hist(x,bins=20,density=True)
plt.show()


#Boxplot,lower upper inner fense
plt.boxplot(s)
plt.show()


mu=np.mean(s)
sigma=np.std(s,ddoff=1)
z=(x-mu)/sigma

#To check z score for checking Outlier,
s[(z<-3)|(z>3)] #Outlier





#Generate random number based on T dist
x2=scipy.statws.t.rvs(1,size=200)
plt.boxplot(x2)
plt.show

Q1=np.quantile(x,0.25)
Q3=np.quantile(x,0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
outlier=x2[(x2<lower)|(x2>upper)]
print(outlier)

lower_outer=Q1-3*IQR
upper_outer=Q3+3*IQR


outlier_outers=outlier[()]   #outlier_outer= 
#outlier_outer=x2[(x2<lower_outer)|(x2>upper_outer)]
#print(outlier_outer)

len(outlier)
len(outlier_outer)




#Want to find correlation coefficient
#Transpose matrix is same(change rows and columns), then symmetric

#df=pd.DataFrame({'A':[1,2,3,4,5],'B':[2,3,4,5,6],'C':[3,4,5,6,7]})
df=pd.DataFrame(np.random.rand(50,3),columns=['A','B','C'])
df.corr()
df.corr(method='pearson')
df.corr(method='spearman')
df.corr(method='kendall')
cbar=plt.imshow(df.corr())
plt.colorbar(cbar)
plt.show()


plt.scatter(df['A'],df['B'])
plt.show()

df['D']=np.random.randint(0,2,size=50)
#Categorical value's Distribution
freq=df['D'].value_counts()
freq.index
plt.bar(freq.index,freq.values)
plt.show()


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
X=[['Male',1],['Female',3],['Female',2]] #2nd columns are number that can be converted to categorical value
enc.fit(X)
enc.transform(X).toarray()

enc.categories_
enc.get_feature_names_out(['gender','group'])

from sklearn.preprocessing import StandardScaler,MinMaxScaler
data=[[0,0],[0,0],[1,1],[1,1]]
scaler=StandardScaler()
scaler.fit(data)
scaler.transform(data)

scaler.mean_
scaler.transform([[2,2]])

minmax_scaler=MinMaxScaler()
minmax_scaler.fit(data)
minmax_scaler.transform(data)
minmax_scaler.transform([[2,2]])    #Transform the data to 0-1 range


