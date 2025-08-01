import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
"""from  keras.preprocessing import image
test_image = image.load_img('image path', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'Your class'"""
 
import pandas as pd
import numpy as np

# Provide the absolute path to the CSV file
file_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/2023-open-data-dfb-ambulance.csv'

try:
    table1 = pd.read_csv(file_path)
    print(table1)
    print(table1.head())
    print(table1.tail())
    print(table1.info())
    print(table1.describe())
    print(table1.columns)
    print(table1.shape)
    print(table1.dtypes)
    print(table1.isnull().sum())
except Exception as e:
    print(f"An error occurred: {e}")
#open it and change it to dataframe
print(table1)
print(table1.head())
print(table1.tail())
print(table1.info())
print(table1.describe())
print(table1.columns)
print(table1.shape)
print(table1.dtypes)
print(table1.isnull().sum())
print(table1['Station Name'].value_counts())
print(table1['Station Name'].value_counts(normalize=True)*100)
"""print(table1['Year'].value_counts())
print(table1['Year'].value_counts(normalize=True)*100)
print(table1['Year'].value_counts(normalize=True)*100)
print(table1['Year'].value_counts(normalize=True)*100)
print(table1['Year'].value_counts(normalize=True)*100)
print(table1['Year'].value_counts(normalize=True)*100)
"""

#By days,weeks,month,year analysis using groupby
#print(table1.groupby('Year').mean())

import pandas as pd
import numpy as np

# Provide the absolute path to the CSV file
import pandas as pd
import numpy as np

# Provide the absolute path to the CSV file
file_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/2023-open-data-dfb-ambulance.csv'

try:
    table1 = pd.read_csv(file_path)
    
    # Convert the Date column to datetime with the correct format
    table1['Date'] = pd.to_datetime(table1['Date'], format='%d/%m/%Y')
    
    # Extract Year, Quarter, Month, Week, and Day
    table1['Year'] = table1['Date'].dt.year
    table1['Quarter'] = table1['Date'].dt.quarter
    table1['Month'] = table1['Date'].dt.month
    table1['Week'] = table1['Date'].dt.isocalendar().week
    table1['Day'] = table1['Date'].dt.day_name()
    
    # Analysis
    print("Days count:")
    print(table1['Day'].value_counts())
    
    print("\nMonths count:")
    print(table1['Month'].value_counts())
    
    print("\nQuarters count:")
    print(table1['Quarter'].value_counts())
    
    print("\nYearly trend:")
    print(table1['Year'].value_counts().sort_index())
    
except Exception as e:
    print(f"An error occurred: {e}")
    
try:
    table1 = pd.read_csv(file_path)
    
    # Convert the Date column to datetime with the correct format
    table1['Date'] = pd.to_datetime(table1['Date'], format='%d/%m/%Y')
    
    # Extract Year, Quarter, Month, Week, and Day
    table1['Year'] = table1['Date'].dt.year
    table1['Quarter'] = table1['Date'].dt.quarter
    table1['Month'] = table1['Date'].dt.month
    table1['Week'] = table1['Date'].dt.isocalendar().week
    table1['Day'] = table1['Date'].dt.day_name()
    
    # Print unique years to check if there are multiple years
    print("Unique years in the dataset:")
    print(table1['Year'].unique())
    
    # Analysis
    print("\nDays count:")
    print(table1['Day'].value_counts())
    
    print("\nMonths count:")
    print(table1['Month'].value_counts())
    
    print("\nQuarters count:")
    print(table1['Quarter'].value_counts())
    
    print("\nYearly trend:")
    print(table1['Year'].value_counts().sort_index())
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Day', data=table1, order=table1['Day'].value_counts().index)
    plt.title('Count of Records by Day')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Month', data=table1, order=table1['Month'].value_counts().index)
    plt.title('Count of Records by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Quarter', data=table1, order=table1['Quarter'].value_counts().index)
    plt.title('Count of Records by Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Year', data=table1, order=table1['Year'].value_counts().sort_index().index)
    plt.title('Count of Records by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.show()
    # Time series decomposition
    table1.set_index('Date', inplace=True)
    table1.sort_index(inplace=True)
    
    # Assuming you want to analyze the count of records per day
    daily_counts = table1.resample('D').size()
    
    decomposition = seasonal_decompose(daily_counts, model='additive')
    
    # Plot the decomposition
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")
#By days,weeks,month,year analysis using groupby
#print(table1.groupby('Year').mean())
#print(table1.groupby('Month').mean())
#print(table1.groupby('Week').mean())
#print(table1.groupby('Day').mean())
#why is it printing all the name of station