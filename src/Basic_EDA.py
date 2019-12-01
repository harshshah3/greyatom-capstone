#!/usr/bin/env python
# coding: utf-8

# # Capstone - Mahindra First Choice Services (MFCS)

# ## Importing header files

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
import warnings
warnings.filterwarnings('ignore')

path_invoice = r'E:/Mahindra-Capstone/greyatom-capstone/data/Final_invoice.csv'
path_JTD = r'E:/Mahindra-Capstone/greyatom-capstone/data/JTD.csv'
path_customer_data = r'E:/Mahindra-Capstone/greyatom-capstone/data/Customer_Data.xlsx'
path_plant_master = r'E:/Mahindra-Capstone/greyatom-capstone/data/Plant Master.xlsx'


# ## Loading the data

# In[69]:


invoice_data_original = pd.read_csv(path_invoice)
JTD_data_original = pd.read_csv(path_JTD)

customer_data_original = pd.read_excel(path_customer_data)
plant_data_original = pd.read_excel(path_plant_master)


# ## Defining basic functions for EDA

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


# ### Customer_Data

# In[4]:


customer_data_original.head()


# In[5]:


customer_data_original.shape


# In[6]:


customer_data_original.info()


# In[7]:


#Numerical cols
numeric_columns = numeric_features(customer_data_original)
print(*numeric_columns.columns.tolist(), sep = " \t| ")


# In[8]:


#Categorical cols
categorical_columns = categorical_features(customer_data_original)
print(*categorical_columns.columns.tolist(), sep = " \t| ")


# In[9]:


#DataTypes
check_datatypes(customer_data_original)


# In[10]:


#Missing data
cust_missing_data = missing_data(customer_data_original)
cust_missing_data

# #Missing data for rows
# rows_percentage_1 = (1 - len(customer_data_original.dropna(thresh=4)) / len(customer_data_original)) * 100
# print("Missing rows %:  {} with 4 columns-data null".format(rows_percentage_1))
# print("===="*30)


# In[11]:


#Dropping cols with 90% thresold
customer_data = drop_missing(customer_data_original,cust_missing_data,90)
customer_data.shape


# ### Invoice_data

# In[12]:


invoice_data_original.head()


# In[68]:


invoice_data_original.shape


# In[29]:


#invoice_data_original.info()


# In[14]:


#Numerical cols
numeric_columns = numeric_features(invoice_data_original)
print(*numeric_columns.columns.tolist(), sep = "\t| ")


# In[15]:


#Categorical cols
categorical_columns = categorical_features(invoice_data_original)
print(*categorical_columns.columns.tolist(), sep = "\t| ")


# In[70]:


#Missing data
inv_missing_data = missing_data(invoice_data_original)
inv_missing_data
# #Missing data for rows
# rows_percentage_1 = (1 - len(invoice_data_original.dropna(thresh=40)) / len(invoice_data_original)) * 100
# print("Missing rows %:  {} with 40 columns-data null".format(rows_percentage_1))
# print("===="*30)


# In[71]:


#Dropping cols with 40% thresold
invoice_data = drop_missing(invoice_data_original, inv_missing_data, 40)
invoice_data.shape


# In[18]:


#DataTypes
check_datatypes(invoice_data)


# ### Plant Master Data

# In[19]:


plant_data_original.head()


# In[30]:


plant_data_original['Plant'].isin(plant_data_original['Valuation Area']).value_counts()


# In[54]:


plant_data_original.shape


# In[31]:


plant_data_original['Factory calendar'].value_counts()


# In[37]:


plant_data_original['Customer no. - plant'].isin(plant_data_original['Vendor number plant']).value_counts()


# In[110]:


plant_data_original
plant_missing_data = missing_data(plant_data_original)
plant_missing_data


# In[105]:


path_valid_pincodes = r'E:/Mahindra-Capstone/greyatom-capstone/data/Pincode_dir.csv'
valid_pincode_data = pd.read_csv(path_valid_pincodes, engine='python')


# In[107]:


valid_pincode_data.tail()


# In[96]:


valid_pincode_data = valid_pincode_data.sort_values(by='Pincode')
valid_pincodes = valid_pincode_data.Pincode.unique()
len(valid_pincodes)


# In[61]:


#ix = np.isin(plant_data_original['Postal Code'], valid_pincodes)
#unique_elements, counts_elements = np.unique(ix, return_counts=True)
#counts_elements
plant_data_original_1 = plant_data_original[np.isin(plant_data_original['Postal Code'], valid_pincodes) == True]
plant_data_original_1.head(10)


# In[102]:


invoice_data_copy = invoice_data
invoice_data_copy.shape


# In[103]:


invoice_data_copy.head()


# In[100]:


print(len(plant_data_original['Postal Code'].unique()))
print(len(invoice_data_copy['Pin code'].unique()))
invoice_data_1 = invoice_data_copy[np.isin(invoice_data_copy['Pin code'], valid_pincodes) == False]
print("Compared Invoice_Data Shape:", invoice_data_1.shape)
print("Invoice_Data Shape:", invoice_data.shape)
# invoice_data_1.head()
#plant_data_original['Postal Code'].value_counts().index.size


# In[138]:


invoice_data_2 = invoice_data_copy[np.isin(invoice_data_copy['Pin code'], valid_pincodes) == True]
invoice_data_2.drop(columns = ['Unnamed: 0', 'Print Status', 'Area / Locality'], axis=1, inplace= True)
invoice_data_2.shape


# ##### Dirty pincode data

# In[108]:


invoice_data_1.head(10)


# In[145]:


#Merge - Take city, Pincode from Plant_Data for accurate values.. left_on = 'Plant' , right_on = 'Plant'
invoice_columns = ['Cust Type', 'Customer No.', 'Gate Pass Time', 'Invoice Date', 'Invoice No', 'Invoice Time', 
                   'Job Card No', 'JobCard Date', 'JobCard Time', 'KMs Reading', 'Labour Total', 'Make', 'Misc Total', 
                   'Model', 'OSL Total', 'Order Type', 'Order Type', 'Parts Total', 'Plant', 'Recovrbl Exp', 
                   'Total Amt Wtd Tax.', 'User ID']
plant_columns = ['Plant', 'Name 1', 'Postal Code', 'City', 'State']
result_invoice_df = pd.merge(invoice_data_1[invoice_columns],
                             plant_data_original_1[plant_columns],
                             left_on= 'Plant',
                             right_on= 'Plant',
                             how= 'inner')
        
# invoice_columns = invoice_data_1.columns.tolist()
# print(*invoice_columns, sep = ' \t |')
# print("===="*30)
# plant_columns = plant_data_original_1.columns.tolist()
# print(*plant_columns, sep = ' \t |')
result_invoice_df.shape


# In[141]:


plant_data_original_1['Name 1'].value_counts().index.size
#invoice_data['Plant Name1'].value_counts().index.size


# In[142]:


inv_columns = result_invoice_df.columns.tolist()
print(*inv_columns, sep = ' \t |')


# In[146]:


result_invoice_df.rename({'Name 1':'Plant Name1', 'Postal Code':'Pin code','State':'District'})
result_invoice_df.shape


# In[151]:


res_inv_missing_data = missing_data(result_invoice_df)
res_inv_missing_data


# In[158]:


#merged_inv_df = pd.concat([invoice_data_2, result_invoice_df], ignore_index=True)
merged_inv_df = pd.concat([invoice_data_2.reset_index(drop=True), result_invoice_df.reset_index(drop=True)], axis=0)
#merged_inv_df = invoice_data_2.append(result_invoice_df)
merged_inv_df.shape


# In[155]:


merged_inv_df.head(10)


# In[123]:


result_invoice_missing_data = missing_data(result_invoice_df)
result_invoice_missing_data


# In[ ]:


customer_data_['customer_no'] = customer_data_['customer_no'].astype(str)
invoice_data_['customer_no'] = invoice_data_['customer_no'].astype(str)

data_merged = pd.merge(invoice_data_, customer_data_, on='customer_no', how='left')
#data_merged_1 = data_merged_1.sort_values(ascending = True, by = 'customer_no')

print(data_merged.head())


# ##### 1). Plant column has the same values as in Valuation Area - Discard 'Valuation Area'
# ##### 2). Factory calendar column is a single value column - OF NO USE

# In[20]:


plant_data_original.shape


# In[21]:


plant_data_original.info()


# In[22]:


#Numerical cols
numeric_columns = numeric_features(plant_data_original)
print(*numeric_columns.columns.tolist(), sep = "\t| ")


# In[23]:


#Categorical cols
categorical_columns = categorical_features(plant_data_original)
print(*categorical_columns.columns.tolist(), sep = "\t| ")


# In[24]:


#DataTypes
check_datatypes(plant_data_original)


# In[25]:


#Missing data
plant_missing_data = missing_data(plant_data_original)
plant_missing_data
# #Missing data for rows
# rows_percentage_1 = (1 - len(plant_data_original.dropna(thresh=40)) / len(plant_data_original)) * 100
# print("Missing rows %:  {} with 40 columns-data null".format(rows_percentage_1))
# print("===="*30)


# In[26]:


#Dropping cols with 90% thresold
plant_data = drop_missing(plant_data_original, plant_missing_data, 90)
plant_data.shape


# ### JTD_Data

# In[6]:


JTD_data_original.head()
JTD_data_original.shape
JTD_data_original.info()
#Numerical cols
numeric_columns = numeric_features(JTD_data_original)
print(*numeric_columns.columns.tolist(), sep = "\t| ")
#Categorical cols
categorical_columns = categorical_features(JTD_data_original)
print(*categorical_columns.columns.tolist(), sep = "\t| ")
#DataTypes
check_datatypes(JTD_data_original)
#Missing data
jtd_missing_data = missing_data(JTD_data_original)
jtd_missing_data
# #Missing data for rows
# rows_percentage_1 = (1 - len(invoice_data_original.dropna(thresh=40)) / len(invoice_data_original)) * 100
# print("Missing rows %:  {} with 40 columns-data null".format(rows_percentage_1))
# print("===="*30)
#Dropping cols with 70% thresold
jtd_data = drop_missing(JTD_data_original, jtd_missing_data, 40)
jtd_data.shape


# ## Data Trimming  
# 
# 
# ###### Trim Column Name, Discard unwanted columns, Strip unwanted chars from data

# In[27]:


customer_data = data_trimming(customer_data)
customer_data = customer_data.drop(columns = ['business_partner'])
customer_data.head()
#print(customer_data_['customer_no'].astype(str))


# In[28]:


invoice_data = data_trimming(invoice_data)
invoice_data = invoice_data.drop(columns = ['unnamed:_0', 'print_status'])
invoice_data.head()


# In[ ]:


plant_data = data_trimming(plant_data)
plant_data = plant_data.drop(columns = ['business_partner'])
plant_data.head()
#print(customer_data_['customer_no'].astype(str))


# In[167]:


###############

car_make_count_summary = invoice_data['make'].value_counts()
print(type(car_make_count_summary))
print(car_make_count_summary)
car_make_count_summary = car_make_count_summary[:10, ]
plt.figure(figsize=(15, 5))
sns.barplot(car_make_count_summary.index, car_make_count_summary.values, alpha=0.8)
plt.title('MAKE-wise Car-Counts serviced across the nation')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Make of Car', fontsize=12)
plt.xticks(rotation=45)
plt.show()

##################
avg_spend_by_car_mfd = invoice_data.groupby(['make', 'total_amt_wtd_tax'])['total_amt_wtd_tax'].mean()


avg_spend_by_car_mfd_2 = invoice_data.groupby(['make', 'model'])['total_amt_wtd_tax'].sum()
avg_spend_by_car_mfd_2 = avg_spend_by_car_mfd_2[:15, ].unstack()
print(avg_spend_by_car_mfd_2)
# =============================================================================
# avg_spend_by_car_mfd_arry = []
# for val in avg_spend_by_car_mfd_2.values:
# avg_spend_by_car_mfd_2.values[i] = humanize.intword(avg_spend_by_car_mfd_2.values[i])
# print(humanize.intword(val))
# avg_spend_by_car_mfd_arry.append(humanize.intword(val))
# print(avg_spend_by_car_mfd_arry)
# =============================================================================


avg_spend_by_car_mfd_2.plot(kind='bar', subplots=True)
#sns.barplot(avg_spend_by_car_mfd_2.index, avg_spend_by_car_mfd_2.values, alpha=0.8)
plt.title('MAKE-wise Servicing cost for Cars')
plt.ylabel('Total Amt of Servicing', fontsize=12)
plt.xlabel('Make of Car', fontsize=12)
plt.xticks(rotation=60)
plt.show()


# In[ ]:





# In[ ]:





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


# In[34]:


# print(data_merged.isnull().sum().sort_values(ascending= False))
# print("==="*30)
# print(data_merged.isna().sum().sort_values(ascending= False))
# print("==="*30)
# print(data_merged[data_merged['city'].isnull()])
# print("==="*30)
# print(data_merged[data_merged['pin_code'] == 174303]['area','city','pin_code','customer_no'])
a = data_merged.columns.tolist()
print(*a, sep = "\t| ")


# In[12]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline


# In[139]:


df_missing_data = missing_data(data_merged)

missing_data_cols = get_missing_cols(df_missing_data)

missing_data_cols


# In[170]:


# missing_data_cols.remove('city')
# missing_data_cols.remove('area')
# missing_data_cols.remove('regn_no')
# df_merged = data_merged[missing_data_cols]
# df_merged['data_origin']= df_merged['data_origin'].astype(str).map(lambda x: x.lstrip('Z'))
df_merged.head()


# In[14]:


le = LabelEncoder()
len(df_merged.model.unique())
df_merged['model'] = le.fit_transform(df_merged['model'].astype(str))
# missing_data_cols
#df_merged.head()


# In[15]:


# transformer = FeatureUnion(
#                            transformer_list=[
#                            ('features', SimpleImputer(strategy='most_frequent')),
#                            ('indicators', MissingIndicator())])
# transformer = transformer.fit(data_merged)
# results = transformer.transform(data_merged)
# results.shape
missing_data_cols


# In[16]:


imp = IterativeImputer(add_indicator=False, estimator=None,
                 imputation_order='ascending', initial_strategy='most_frequent',
                 max_iter=10, max_value=None, min_value=None,
                 n_nearest_features=1,
                 random_state=0, sample_posterior=False, tol=0.001,
                 verbose=0)
imp.fit_transform(df_merged)  
df_merged.isnull().sum()


# In[24]:


#data_merged['gender'] = data_merged['title'].astype(int)


# ### Using KNN for prediction of missing values of 'model'

# In[169]:


# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# missing_data_cols.remove('model')
# #Filter the data frame
# X = data_merged.drop(columns=missing_data_cols)
# print("X shape: ",X.shape)
# #X_train = X.dropna(axis= 0)
# #print("X_train shape: ",X_train.shape)
print(X.head())


# In[141]:


categorical_cols_df = categorical_features(X)
numerical_cols_df = numeric_features(X)


# In[142]:


numerical_cols_df.columns


# In[143]:


drop_cols_list = ['customer_no','gate_pass_time', 'invoice_time', 'plant_name', 'user_id', 'invoice_no','invoice_date', 'invoice_time', 'jobcard_date', 'job_card_no', 'jobcard_time']
X = X.drop(columns = drop_cols_list, axis=1)
X.head(5)


# In[166]:


cat_feature_df = categorical_features(X)
cat_cols = cat_feature_df.columns.tolist()
#cat_cols
cat_feature_df.isnull().sum()


# In[163]:


#Label Encoding the cat-features
def encode_col_by_col(df):
    encoders = dict()
    for col in df.columns:
        series = df[col]
        le = LabelEncoder()
        df[col] = pd.Series(
            le.fit_transform(series[series.notna()]),
            index=series[series.notna()].index
        )
        encoders[col] = le
    return encoders
#le = LabelEncoder()
encoders = encode_col_by_col(X[cat_cols])
#X[categorical_cols] = X[categorical_cols].apply(lambda x: le.fit_transform(x.astype(str)))
#encoders['cust_type'].inverse_transform(X['cust_type'])


# In[164]:





# In[156]:


X.head()


# In[67]:


X_train.isnull().sum()


# In[61]:


X_test = X[pd.isnull(X['model'])]
print("X_test shape: ",X_test.shape)


# In[68]:


#Train K-NN Classifier
clf = KNeighborsClassifier(5, weights='distance')
trained_model = clf.fit(X_train.drop(columns = 'model', axis = 1), X_train.loc[:,'model'])


# In[65]:


#Predicting Missing Values' Class --> 'Model'
imputed_values = trained_model.predict(X_test.drop(columns = 'model', axis = 1))
type(imputed_values)


# In[ ]:


# Join column of predicted class with their other features
X_with_imputed = X_test['model']
np.hstack((imputed_values.reshape(-1,1), X_train.drop(columns = 'model', axis = 1)))

# Join two feature matrices
np.vstack((X_with_imputed, X))

