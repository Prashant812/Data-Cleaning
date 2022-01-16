# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 08:46:31 2021

@author: Prashant Kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("landslide_data3_miss.csv")

#Plotting a graph of the attribute names (x-axis) with the number of missing values
attributes = list(df.columns)
values = []
missing = df.isnull().sum(axis = 0)
missing_df = pd.DataFrame(missing)
for i in attributes:
    print("No. of missing values in",i,"is",missing_df.loc[i,0])
    values.append(missing_df.loc[i,0])

fig = plt.figure(figsize = (10, 5))
plt.bar(attributes,values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Attributes")
plt.ylabel("No. of Missing values")
plt.title("Missing Values in each attributes")
plt.show()


#With target attribute “stationid”, the total number of tuples deleted having missing value in the target attribute :-
df1 = pd.read_csv("landslide_data3_miss.csv")
has_nan = df1['stationid'].isnull()
stationid_nan =  dict(df1['stationid'].loc[has_nan])
row_has_nan = list(stationid_nan.keys())
print(*row_has_nan)
print("Total numbers of tuples deleted :",len(row_has_nan))
df1 = df1.drop(row_has_nan)

#Deleting the tuples(rows) having equal to or more than one third of attributes with missing values:-
df2 = pd.read_csv("landslide_data3_miss.csv")
dict_has_nan = dict(df2.isnull().sum(axis=1))
indx = list(dict_has_nan.keys())
no_nan = list(dict_has_nan.values())
# for i in indx:
#     print("Row number :",i,
#           "Missing values :",no_nan[i],"\n")

count = 0
nan_rows = []
for i in indx:
    if no_nan[i]>=3: 
        nan_rows.append(i)
        count +=1
print("Total number of tuples to be deleted :",count)
#print(nan_rows)
df2 = df2.drop(nan_rows)
print("No. of rows in original data :",len(df.index))
print("No. of rows remaining after dropping the tuples :",len(df2.index))

#____
for i in attributes:
    print("No. of missing values remaing in",i,'is',df2[i].isnull().sum())

#Handling the Missing values
#Replacing the missing values by mean of their respective attribute
df3 = pd.read_csv("landslide_data3_miss.csv")
attributes_new = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in attributes_new: 
    mean_attr = df3[i].mean()
    df3[i] = df3[i].fillna(mean_attr)
    
df_original = pd.read_csv("landslide_data3_original.csv")

RMSE_value = []

for i in attributes_new:
    RMSE_value.append(((df_original[i] - df3[i]) ** 2).mean() ** .5)
    print("RMSE values for",i,"is %1.2f"%(((df_original[i] - df3[i]) ** 2).mean() ** .5))
    
fig = plt.figure(figsize = (10, 5))
plt.bar(attributes_new,RMSE_value, color ='maroon',
        width = 0.4)
 
plt.xlabel("Attributes")
plt.ylabel("RMSE Values")
plt.title("RMSE Values for each attribute")
plt.show()

#Replace the missing values in each attribute using linear interpolation technique

df5 = pd.read_csv("landslide_data3_miss.csv")
for i in attributes_new:
    df5[i] = df5[i].interpolate(method ='linear')

print(df5.isnull().sum())
RMSE_inter = []
for i in attributes_new:
    RMSE_inter.append(((df_original[i] - df5[i]) ** 2).mean() ** .5)
    print("RMSE values for",i,"is %1.2f"%(((df_original[i] - df5[i]) ** 2).mean() ** .5))
    
fig = plt.figure(figsize = (10, 5))
plt.bar(attributes_new,RMSE_inter, color ='maroon',
        width = 0.4)
 
plt.xlabel("Attributes")
plt.ylabel("RMSE Values")
plt.title("RMSE Values for each attribute after interpolation")
plt.show()

#Outlier detection

#Boxplot of Rain and Temperature attribute

#print(bxplot1)
print("Boxplot can be seen in the plot section\n")
fig = plt.figure(figsize =(10, 5))
plt.title("Boxplot for temperature")
plt.boxplot(df5['temperature'])
plt.xlabel("Temperature")
plt.show()


fig = plt.figure(figsize =(10, 5))
plt.title("Boxplot for Rain")
plt.boxplot(df5['rain'])
plt.xlabel("Rain")
plt.semilogy()

#List of outliers in Temperature and Rain attribute

def outliers(x):  #Function for outliers
    minimum=2.5*np.percentile(df5[x],25)-1.5*np.percentile(df5[x],75) #conditions for outliers
    maximum=2.5*np.percentile(df5[x],75)-1.5*np.percentile(df5[x],25)
    outliers_=pd.concat((df5[x][df5[x]< minimum],df5[x][df5[x]> maximum]))
    return outliers_

rain_out = list(outliers('rain').index)
temp_out = list(outliers('temperature').index)
print("Outliers in rain : ",*rain_out)
print("\nOutliers in Temperatures : ",*temp_out)


df5['rain'].replace(df5['rain'][rain_out],df5['rain'].median(),inplace=True)
df5['temperature'].replace(df5['temperature'][temp_out],df5['temperature'].median(),inplace=True)

print("Boxplot after replacing outliers\n")
fig = plt.figure(figsize =(10, 5))
plt.title("Boxplot for temperature after replacing outliers")
plt.boxplot(df5['temperature'])
plt.xlabel("Temperature")
plt.show()


fig = plt.figure(figsize =(10, 5))
plt.title("Boxplot for Rain after replacing outliers")
plt.boxplot(df5['rain'])
plt.xlabel("Rain")
plt.semilogy()
