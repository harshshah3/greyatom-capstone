#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
import seaborn as sns
import humanize
import re
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn_pandas import CategoricalImputer

path_invoice = r'E:/Mahindra-Capstone/greyatom-capstone/data/Final_invoice.csv'
path_JTD = r'E:/Mahindra-Capstone/greyatom-capstone/data/JTD.csv'
path_customer_data = r'E:/Mahindra-Capstone/greyatom-capstone/data/Customer_Data.xlsx'
path_plant_master = r'E:/Mahindra-Capstone/greyatom-capstone/data/Plant Master.xlsx'


# In[2]:


invoice_data_original = pd.read_csv(path_invoice)
JTD_data_original = pd.read_csv(path_JTD)

customer_data_original = pd.read_excel(path_customer_data)
plant_data_original = pd.read_excel(path_plant_master)


#print(customer_data_original.head(10))
#print(customer_data_original.describe())
#print(customer_data_original.shape)
#print(customer_data_original.info())


# In[3]:


# Function to identify numeric features
def numeric_features(df):
    numeric_col = df.select_dtypes(include=np.number).columns.tolist()
    return df[numeric_col].head()

# Function to identify categorical features
def categorical_features(df):
    categorical_col = df.select_dtypes(exclude=np.number).columns.tolist()
    return df[categorical_col].head()

# Function to check the datatypes of all the columns:
def check_datatypes(df):
    return df.dtypes

#Check for missing data
def missing_data(df):
    total = df.isnull().sum()
    percent = (round(total * 100 / len(df), 2))
    missing_data = pd.DataFrame({'Total': total,
                                'Percent': percent})
    missing_data = missing_data.sort_values(by = 'Total', ascending=False)
    return missing_data

def get_missing_cols(missing_data_df):
    missing_data_df = missing_data_df[missing_data_df['Total'] > 0]
    missing_data_cols = missing_data_df.index.tolist()
    return missing_data_cols

#Drop missing_data cols setting thresold 
def drop_missing(df, missing, value):
    df = df.drop((missing[missing['Percent'] > value]).index,axis= 1)
    print(df.isnull().sum().sort_values(ascending = False))
    return df 
    
#
def data_clean_formatt(df):
    missing_vals_df = missing_vals_df[missing_vals_df['missing_vals_%'] >= 20.00]
    # Plotting Missing values chart
    missing_vals_df.plot.barh(x='column_name', y='missing_vals_%', rot=0)
    plt.title("Column-wise Bar-Plot of Missing values in - " + get_df_name(df))
    plt.show()
    # Dropping columns with >70% of missing values
    #df_trimmed = df
    df_trimmed = df.dropna(thresh=df.shape[0] * 0.7, how='all', axis=1)
    # =============================================================================
    #     #Formatting Column-Names
    #     for col in df_trimmed.columns:
    #       col = re.sub(r'/.*|\\.*', r'', col)
    # =============================================================================

    return df_trimmed

#
def data_trimming(df):
    df_trimmed = df
    df_trimmed.columns = df_trimmed.columns.str.replace(' ', '_')
    df_trimmed.columns = df_trimmed.columns.str.strip('.')
    df_trimmed.columns = df_trimmed.columns.str.lower()
    
    return df_trimmed

#Get Name of DataFrame
def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name


# In[4]:


#Customer Data - customer_data_original

#Shape and info about data
print("===="*30)
print("Shape:===",customer_data_original.shape)
#print(customer_data_original.info())
print("===="*30)

#Numerical cols
numeric_columns = numeric_features(customer_data_original)
print("\nNumeric Features:")
print(numeric_columns)
print("===="*30)

#Categorical cols
categorical_columns = categorical_features(customer_data_original)
print("\nCategorical Features:")
print(categorical_columns)
print("===="*30)

#DataTypes
print(check_datatypes(customer_data_original))
print("===="*30)

#Missing data
cust_missing_data = missing_data(customer_data_original)
print("\nMissing Data:")
print(cust_missing_data)
print("===="*30)

#Missing data for rows
rows_percentage_1 = (1 - len(customer_data_original.dropna(thresh=4)) / len(customer_data_original)) * 100
print("Missing rows %:  {} with 4 columns-data null".format(rows_percentage_1))
print("===="*30)

#Dropping cols with 70% thresold
customer_data = drop_missing(customer_data_original,cust_missing_data,90)
print("===="*30)

print("Shape:===",customer_data.shape)
print("===="*30)


# In[5]:


#Invoice Data - invoice_data_original

#Shape and info about data
print("===="*30)
print("Shape:===",invoice_data_original.shape)
#print(invoice_data_original.info())
print("===="*30)

#Numerical cols
numeric_columns = numeric_features(invoice_data_original)
print("\nNumeric Features:")
print(numeric_columns)
print("===="*30)

#Categorical cols
categorical_columns = categorical_features(invoice_data_original)
print("\nCategorical Features:")
print(categorical_columns)
print("===="*30)

#DataTypes
print(check_datatypes(invoice_data_original))
print("===="*30)

#Missing data
inv_missing_data = missing_data(invoice_data_original)
print("Missing Data:")
print(inv_missing_data)
print("===="*30)

#Missing data for rows
rows_percentage_1 = (1 - len(invoice_data_original.dropna(thresh=40)) / len(invoice_data_original)) * 100
print("Missing rows %:  {} with 40 columns-data null".format(rows_percentage_1))
print("===="*30)

#Dropping cols with 70% thresold
invoice_data = drop_missing(invoice_data_original, inv_missing_data, 40)
print("===="*30)


print("Shape:===",invoice_data.shape)
print("===="*30)


# In[6]:


# ###############

# car_make_count_summary = invoice_data['make'].value_counts()
# print(type(car_make_count_summary))
# print(car_make_count_summary)
# car_make_count_summary = car_make_count_summary[:10, ]
# plt.figure(figsize=(15, 5))
# sns.barplot(car_make_count_summary.index, car_make_count_summary.values, alpha=0.8)
# plt.title('MAKE-wise Car-Counts serviced across the nation')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Make of Car', fontsize=12)
# plt.xticks(rotation=45)
# plt.show()

# ##################
# avg_spend_by_car_mfd = invoice_data.groupby(['make', 'total_amt_wtd_tax'])['total_amt_wtd_tax'].mean()


# avg_spend_by_car_mfd_2 = invoice_data.groupby(['make', 'model'])['total_amt_wtd_tax'].sum()
# avg_spend_by_car_mfd_2 = avg_spend_by_car_mfd_2[:15, ].unstack()
# print(avg_spend_by_car_mfd_2)
# # =============================================================================
# # avg_spend_by_car_mfd_arry = []
# # for val in avg_spend_by_car_mfd_2.values:
# # avg_spend_by_car_mfd_2.values[i] = humanize.intword(avg_spend_by_car_mfd_2.values[i])
# # print(humanize.intword(val))
# # avg_spend_by_car_mfd_arry.append(humanize.intword(val))
# # print(avg_spend_by_car_mfd_arry)
# # =============================================================================


# avg_spend_by_car_mfd_2.plot(kind='bar', subplots=True)
# #sns.barplot(avg_spend_by_car_mfd_2.index, avg_spend_by_car_mfd_2.values, alpha=0.8)
# plt.title('MAKE-wise Servicing cost for Cars')
# plt.ylabel('Total Amt of Servicing', fontsize=12)
# plt.xlabel('Make of Car', fontsize=12)
# plt.xticks(rotation=60)
# plt.show()


# In[7]:


customer_data_ = data_trimming(customer_data)
customer_data_ = customer_data_.drop(columns = ['business_partner'])
print(customer_data_.head())
print("...."*30)
print(customer_data_.tail())
#print(customer_data_['customer_no'].astype(str))


# In[8]:


invoice_data_ = data_trimming(invoice_data)
print(invoice_data_.columns.to_list())
print("=="*30)
invoice_data_ = invoice_data_.drop(columns = ['unnamed:_0', 'print_status'])
invoice_data_ = invoice_data_.rename(columns = {'area_/_locality':'area', 'plant_name1':'plant_name'})
print(invoice_data_.head())
print("...."*30)
print(invoice_data_.tail())


# In[9]:


from sklearn_pandas import CategoricalImputer


#Merging invoice_data_ & customer_data_
print(len(invoice_data_.columns.tolist()) + len(customer_data_.columns.tolist()))
print("==="*30)
customer_data_['customer_no'] = customer_data_['customer_no'].astype(str)
invoice_data_['customer_no'] = invoice_data_['customer_no'].astype(str)

data_merged = pd.merge(invoice_data_, customer_data_, on='customer_no', how='left')
#data_merged_1 = data_merged_1.sort_values(ascending = True, by = 'customer_no')

print(data_merged.head())
print("...."*30)
print(data_merged.tail())


# In[10]:


# print("==="*30)
# print(data_merged[data_merged['customer_no'] == "1"])
# print("==="*30)
# print(data_merged.shape)
# print("==="*30)


# In[11]:


# print(data_merged.isnull().sum().sort_values(ascending= False))
# print("==="*30)
# print(data_merged.isna().sum().sort_values(ascending= False))
# print("==="*30)
# print(data_merged[data_merged['city'].isnull()])
# print("==="*30)
# print(data_merged[data_merged['pin_code'] == 174303]['area','city','pin_code','customer_no'])


# In[12]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline


# In[55]:


df_missing_data = missing_data(data_merged)

missing_data_cols = get_missing_cols(df_missing_data)

missing_data_cols.remove('city')
missing_data_cols.remove('area')
missing_data_cols.remove('regn_no')
df_merged = data_merged[missing_data_cols]
df_merged['data_origin']= df_merged['data_origin'].astype(str).map(lambda x: x.lstrip('Z'))
df_merged.head()


# In[63]:


le = LabelEncoder()
len(df_merged.model.unique())
df_merged['model'] = le.fit_transform(df_merged['model'].astype(str))
# missing_data_cols
#df_merged.head()


# In[14]:


# transformer = FeatureUnion(
#                            transformer_list=[
#                            ('features', SimpleImputer(strategy='most_frequent')),
#                            ('indicators', MissingIndicator())])
# transformer = transformer.fit(data_merged)
# results = transformer.transform(data_merged)
# results.shape
missing_data_cols


# In[71]:


imp = IterativeImputer(add_indicator=False, estimator=None,
                 imputation_order='ascending', initial_strategy='most_frequent',
                 max_iter=10, max_value=None, min_value=None,
                 n_nearest_features=1,
                 random_state=0, sample_posterior=False, tol=0.001,
                 verbose=0)
imp.fit_transform(df_merged)  
df_merged.isnull().sum()

