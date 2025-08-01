import pandas as pd # The prediction step provide correct image path
import numpy as np

s=pd.Series([1,2,3,4,5,6,7,8,9,10])
print(s)
print(s.head(3))
print(s.tail(3))
#visualize it in table 

np.random.randn(6,4)
df=pd.DataFrame(np.random.randn(6,4),index=[11,12,14,15,16,17],columns=['A','B','C','D'])
print(df)

#df[0]
#df[:,0]

print(df['A'])
print(df[['A','B']])
print(df.dtypes)

print(df.tail())
print(df.tail(3))

print(df.index)
print(df.columns)
print(df.values)
print(df.describe())
print(df.T)
print(df.sort_index(axis=1,ascending=False))
print(df.sort_values(by='B'))
print(df.to_numpy())
print(df.T)#index and column transpost
print(df.sort_index(axis=1,ascending=False))
print(df.sort_values(by='B'))
print(df.to_numpy())

df2=pd.DataFrame(np.random.randint(0,3,size=(6,4)),index=['a','b','c','d','e','f'],columns=['A','B','C','D']) #Lowest,highest, #,index=[11,12,14,15,16,17],columns=['A','B','C','D'])
print(df2.sort_values(by='A'))
print(df2.sort_values(by='B'))
print(df2.sort_values(by=['A','B']))
print(df2.sort_values(by=['A','B'],ascending=False))
print(df2.sort_values(by=['A','B'],ascending=[False,True]))
print(df.loc[[11,12]])
print(df.loc[[11,12],['A','B']])
print(df.iloc[[0,1]])
print(df.iloc[[0,1],[0,1]])
print(df.iloc[0:2,0:2])

#logical operator; every element that variable a is greater than 0
print(df[df['A']>0])
print(df2[df2['B'].isin([0,1])])#+ "is in the list")

df2['E']=['one','one','two','three','four','three'] #add new column
print(df2)
df2['F']=df2['A']+df2['B'] #Add new column by adding two columns which can be used in Feature Generation
print(df2)

print(type(df['A']))
s=pd.Series([1,2,3,4,5,6,7,8,9,10],index=[11,12,14,15,16,17,18,19,20,21])
print(s)
print(s[11])
df2['F']=s
print(s.values)
#df2['F']=s.values 
#print(df2)

df2['E']=[1,2,3,np.nan,5,6] #add new column
print(df2)
print(df2.dropna()) #drop row
print(df.fillna(value=0)) #fill missing value
print(df2['E'].fillna(value=df2['E'].mean())) #fill missing value
#drop column


#Unobserved value 


df.mean(axis=1) #If you want to calculate mean of individual row
df.var(axis=1)
df.std(axis=1)
df.sum(axis=1)
df.quantile(0.5,axis=1)#median
df.quantile(0.25,axis=1)#Q1
df.quantile(0.75,axis=1)#Q3
df.max(axis=1)
df.min(axis=1)
print(df.describe())



#For the categorial variable, tto check the distribution, check the frequency of the variable
df2['E'].value_counts()
df2['E'].value_counts(normalize=True)
df2['E'].value_counts(normalize=True)*100
print(df2['E'].value_counts(normalize=True)*100)


pd.concat((df,df2)) # Difference bewteen pd.concat((df,df2)) und pd.concat((df,df2),axis=1) Joining horizontally and vertically


left=pd.DataFrame({'key':['one','two','three','four'],'x1':np.random.rand(4),'x2':np.random.rand(4)})#certain keyvariable is used to join the two dataframes
right=pd.DataFrame({'key':['one','three','five','six','seven'],'x3':np.random.rand(4),'x4':np.random.rand(4)})

left.merge(right,on='key',how='inner') #inner join
left.merge(right,on='key',how='outer') #outer join
left.merge(right,on='key',how='left') #left join
left.merge(right,on='key',how='right') #right join


#Groupby
df2.groupby(['E'])['A'].sum()


""" Image produce und editing
from keras.preprocessing import image
test_image = image.load_img('image path', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'Your class'"""