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

# In[2]:


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


# ### Data Trimming  
# 
# 
# ###### Trim Column Name, Discard unwanted columns, Strip unwanted chars from data

# In[11]:


#Dropping cols with 90% thresold
customer_data = drop_missing(customer_data_original,cust_missing_data,90)
customer_data.shape


# In[12]:


customer_data = data_trimming(customer_data)
customer_data = customer_data.drop(columns = ['business_partner'])
customer_data.head()
#print(customer_data_['customer_no'].astype(str))


# ### Invoice_data

# In[13]:


invoice_data_original.head()


# In[14]:


invoice_data_original.shape


# In[15]:


#invoice_data_original.info()


# In[16]:


#Numerical cols
numeric_columns = numeric_features(invoice_data_original)
print(*numeric_columns.columns.tolist(), sep = "\t| ")


# In[17]:


#Categorical cols
categorical_columns = categorical_features(invoice_data_original)
print(*categorical_columns.columns.tolist(), sep = "\t| ")


# In[18]:


#Missing data
inv_missing_data = missing_data(invoice_data_original)
inv_missing_data
# #Missing data for rows
# rows_percentage_1 = (1 - len(invoice_data_original.dropna(thresh=40)) / len(invoice_data_original)) * 100
# print("Missing rows %:  {} with 40 columns-data null".format(rows_percentage_1))
# print("===="*30)


# ### Data Trimming  
# 
# 
# ###### Trim Column Name, Discard unwanted columns, Strip unwanted chars from data

# In[19]:


#Dropping cols with 40% thresold
invoice_data = drop_missing(invoice_data_original, inv_missing_data, 40)
invoice_data.shape


# In[20]:


#DataTypes
#check_datatypes(invoice_data)


# In[21]:


invoice_data = data_trimming(invoice_data)
invoice_data = invoice_data.drop(columns = ['unnamed:_0', 'print_status'])
invoice_data.head()


# ### Plant Master Data

# In[22]:


plant_data_original.head()


# In[23]:


plant_data_original.shape


# In[24]:


plant_data_original['Plant'].isin(plant_data_original['Valuation Area']).value_counts()


# ##### Identical columns. Hence 'Valuation Area' is of NO-USE.

# In[25]:


plant_data_original['Factory calendar'].value_counts()


# ##### Single Value Column. Hence 'Factory Calendar' is of NO-USE.

# In[26]:


plant_data_original['Customer no. - plant'].isin(plant_data_original['Vendor number plant']).value_counts()


# In[27]:


plant_data_original
plant_missing_data = missing_data(plant_data_original)
plant_missing_data


# ##### Vendor number plant has 99%+ missing vals. Hence is of NO-USE.

# ### Data Trimming  
# 
# 
# ###### Trim Column Name, Discard unwanted columns, Strip unwanted chars from data

# In[28]:


#Dropping cols with 90% thresold
plant_data = drop_missing(plant_data_original, plant_missing_data, 90)
plant_data = plant_data.drop(columns = ['Factory calendar'], axis=1)
plant_data.shape


# In[29]:


plant_data = data_trimming(plant_data)
plant_data.head()


# ## Importing Valid Pinocode Data of India
# ### Downloaded from [here](https://data.gov.in/catalog/all-india-pincode-directory)

# In[30]:


path_pincode_data = r'E:/Mahindra-Capstone/greyatom-capstone/data/Pincode_dir.csv'
gov_auth_pincode_data = pd.read_csv(path_pincode_data, engine='python')


# In[31]:


gov_auth_pincode_data.tail()


# In[32]:


gov_auth_pincode_data = gov_auth_pincode_data.sort_values(by='Pincode')
pincodes = gov_auth_pincode_data.Pincode.unique()
print("Unique Pin codes across India ::",len(pincodes))


# In[33]:


path_pincodes = r'E:/Mahindra-Capstone/greyatom-capstone/data/Pincode_City_data.csv'
pincode_data = pd.read_csv(path_pincodes)


# In[34]:


pincode_data.tail()


# In[35]:


pincode_data = pincode_data.sort_values(by='Pincode')
pincodes_list = pincode_data.Pincode.unique()
print("Unique Pin codes across India ::",len(pincodes_list))


# In[36]:


valid_pincodes = pincode_data[np.isin(pincode_data['Pincode'], pincodes) == True]
valid_pincodes_unique = valid_pincodes.Pincode.unique()
valid_cities_unique = valid_pincodes.City.unique()
print("Unique Cities across India ::",len(valid_cities_unique))


# In[37]:


valid_pincodes.head()


# In[38]:


invoice_data_copy = invoice_data
print(invoice_data_copy.shape)


# In[39]:


invoice_data_valid = invoice_data_copy[np.isin(invoice_data_copy['pin_code'], valid_pincodes_unique) == True]
print("Invoice_Data with Valid pincodes -- %:",invoice_data_valid.shape[0]/ invoice_data_copy.shape[0] * 100)


# ##### Out of 492314 rows, 386542 (78.5%) Pincode-Values are valid. So, replacing corresponding dirty value of city_name with a valid city_name from externally imported data. 

# In[40]:


#Cities' name replaced for every valid pincode
invoice_data_copy2 = invoice_data_copy.merge(valid_pincodes[['Pincode', 'City']],
                                         left_on= 'pin_code',
                                         right_on= 'Pincode')
print(invoice_data_copy2.shape)


# In[41]:


invoice_data_copy.job_card_no.value_counts()


# In[42]:


#Dropping duplicate rows
invoice_data_copy2= invoice_data_copy2.drop_duplicates(subset=['invoice_no'])
print(invoice_data_copy2.shape)


# In[43]:


#Preparing the dataframe-'invoice_data_copy3' with invalid pincodes to get merged and result in original df- 'invoice_data' with cleaned city name
invoice_data_copy3 = invoice_data_copy[np.isin(invoice_data_copy['invoice_no'], invoice_data_copy2['invoice_no'].values) == False]
print(invoice_data_copy3.shape)


# In[44]:


invoice_data_copy3['City'] = invoice_data_copy3.loc[:,['city']]
invoice_data_copy3['Pincode'] = invoice_data_copy3.loc[:,['pin_code']]
print(invoice_data_copy3.shape)


# In[45]:


invoice_data_copy4 = pd.concat([invoice_data_copy2, invoice_data_copy3], axis=0)
print(invoice_data_copy4.shape)


# In[46]:


invoice_data = invoice_data_copy4
invoice_data = invoice_data.drop(columns=['city','pin_code'], axis=1)
print(invoice_data.shape)


# ## MERGING Invoice-Plant_Master Data

# ##### Valid pincode-city data

# In[47]:


invoice_data_valid_cities = invoice_data[np.isin(invoice_data['City'], valid_cities_unique) == True]
invoice_data_valid_pinC = invoice_data[np.isin(invoice_data['Pincode'], valid_pincodes_unique) == True]
print("Invoice_Data with Valid Pincodes  -- %:",invoice_data_valid_pinC.shape[0]/ invoice_data.shape[0] * 100)
print("Invoice_Data with Valid City name -- %:",invoice_data_valid_cities.shape[0]/ invoice_data.shape[0] * 100)


# #### Dirty Pincode Data

# In[48]:


#invoice_data_1 --- INVALID Pincodes
invoice_data_1 = invoice_data_original[np.isin(invoice_data_original['Pin code'], valid_pincodes_unique) == False]
print("Invoice_Data with Invalid/Dirty pincodes -- %:", invoice_data_1.shape[0]/ invoice_data_original.shape[0] * 100)


# In[49]:


inv_missing_data = missing_data(invoice_data)
inv_missing_data


# ### Filling missing value with tag :- "no info"

# In[50]:


invoice_data[['model', 'City', 'regn_no', 'area_/_locality']] = invoice_data[['model', 'City', 'regn_no', 'area_/_locality']].fillna('NO_INFO')
invoice_data.isnull().sum()


# #### Merging - Take [City, Pincode, District, Plant_Name] from Plant_Data for accurate values.

# In[51]:


# invoice_columns = ['CITY','Cust Type', 'Customer No.', 'District', 'Gate Pass Time', 'Invoice Date', 'Invoice No', 'Invoice Time', 
#                    'Job Card No', 'JobCard Date', 'JobCard Time', 'KMs Reading', 'Labour Total', 'Make', 'Misc Total', 
#                    'Model', 'OSL Total', 'Order Type', 'Parts Total', 'Plant', 'Plant Name1', 'Recovrbl Exp','Regn No',
#                    'Total Amt Wtd Tax.', 'User ID']
# plant_columns = ['Plant', 'Name 2', 'Postal Code']

# df_merged_1 = pd.merge(invoice_data_1[invoice_columns],
#                      plant_data_original_1[plant_columns],
#                      left_on= 'Plant',
#                      right_on= 'Plant',
#                      how= 'inner')

# df_merged_1.rename({"City":"CITY", "Name 1":"Plant Name1", "State":"District", "Postal Code":"Pin code"},axis=1,inplace=True)
# df_merged_1.rename({"Postal Code":"Pin code"}, axis=1, inplace=True)
# df_merged_1.shape 

# invoice_columns = invoice_data_1.columns.tolist()
# print(*invoice_columns, sep = ' \t |')
# print("===="*30)
# plant_columns = plant_data_original_1.columns.tolist()
# print(*plant_columns, sep = ' \t |')


# In[52]:


customer_data['customer_no'] = customer_data['customer_no'].astype(str)
invoice_data['customer_no'] = invoice_data['customer_no'].astype(str)

data_merged = pd.merge(invoice_data, customer_data, on='customer_no', how='left')
#data_merged_1 = data_merged_1.sort_values(ascending = True, by = 'customer_no')

data_merged.head()


# In[53]:


df_missing_vals = missing_data(data_merged)
df_missing_vals


# In[54]:


data_merged[['title','data_origin','partner_type']] = data_merged[['title','data_origin','partner_type']].fillna('NO INFO', axis=1)


# ### JTD_Data

# In[55]:


JTD_data_original.head()


# In[56]:


JTD_data_original.shape
JTD_data_original.info()
#Numerical cols
numeric_columns = numeric_features(JTD_data_original)
print(*numeric_columns.columns.tolist(), sep = "\t| ")
#Categorical cols
categorical_columns = categorical_features(JTD_data_original)
print(*categorical_columns.columns.tolist(), sep = "\t| ")
#DataTypes
#check_datatypes(JTD_data_original)
#Missing data
jtd_missing_data = missing_data(JTD_data_original)
print(jtd_missing_data)
# #Missing data for rows
# rows_percentage_1 = (1 - len(invoice_data_original.dropna(thresh=40)) / len(invoice_data_original)) * 100
# print("Missing rows %:  {} with 40 columns-data null".format(rows_percentage_1))
# print("===="*30)


# In[57]:


#Data Trimming and filling missing vals with "NO INFO" tag
jtd_data = data_trimming(JTD_data_original)
jtd_data = jtd_data.fillna('NO INFO')


# In[58]:


JTD_data_original.shape


# In[59]:


#jtd_data = jtd_data.drop(columns=['unnamed:_0'], axis=1)
print(jtd_data.shape)
jtd_data.head()


# In[60]:


data_merged.tail()


# In[61]:


df_cols_list = data_merged.columns.tolist()
print(*df_cols_list, sep=' \t|')


# In[62]:


#Trial Cell
amt_0_mask = data_merged.total_amt_wtd_tax == 0.00
test_data_merged = data_merged[amt_0_mask]
test_data_merged1 = test_data_merged.loc[:,['job_card_no','labour_total','misc_total','recovrbl_exp']]
print(test_data_merged1.shape)
print(data_merged.shape)
test_data_merged1.head(10)


# #### Observation:
# Almost 52000 rows are with 0 total amount. Hence removing the rows..

# In[63]:


data_merged = data_merged.drop(test_data_merged.index, axis=0)
data_merged.shape


# In[64]:


# jtd_no_service_cost = jtd_data[np.isin(jtd_data['dbm_order'],test_data_merged.values) == True]
# jtd_no_service_cost.head()


# ## Top states and cities with highest revenue

# In[65]:


states_top_revenue = data_merged.groupby('district')['total_amt_wtd_tax'].sum().reset_index()
states_top_revenue = states_top_revenue.rename({'total_amt_wtd_tax':'total_sum_amt'},axis=1).reset_index()
states_top_revenue = states_top_revenue.sort_values(by='total_sum_amt', ascending=False)
#states_top_revenue = states_top_revenue[:15, ].unstack()
states_top_revenue.head()


# In[66]:


len(np.isin(data_merged['job_card_no'], jtd_data['dbm_order'].values) == True)


# In[67]:


data_merged.shape


# In[68]:


df_merged = data_merged.merge(jtd_data,
                             left_on='job_card_no',
                             right_on='dbm_order',how='left')
df_merged.shape


# In[69]:


#df_merged.drop_duplicates(subset=['job_card_no'])


# In[70]:


JTD_data_original.columns


# ## Total No of Services done per State

# In[71]:


total_services_states = data_merged.groupby('district')['invoice_no'].size()
total_services_states = total_services_states.sort_values(ascending=False)
total_services_states = total_services_states.rename('total_services')
total_services_states.head()


# ## Which Parts/Items in heavy demand?
# ## Which Kind of service is 

# In[72]:


top_trending_parts = jtd_data['description'].value_counts()
#top_trending_parts = top_trending_parts.sort_values(ascending=False)
top_trending_parts


# In[73]:


top_trending_item_category = jtd_data['item_category'].value_counts()
#top_trending_parts = top_trending_parts.sort_values(ascending=False)
top_trending_item_category


# In[74]:


category_parts_df = df_merged.groupby('item_category')['description'].agg()
#category_parts_df = category_parts_df.sort_values(ascending=False)
#category_parts_df = category_parts_df.rename('total_services')
category_parts_df


# In[ ]:


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


# ### Using KNN for prediction of missing values of 'model'

# In[264]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn_pandas import CategoricalImputer


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


# # Important Checkpoints
# Sunday 8/12 4.00PM - dataframe,missing vals done
# 
# Sunday 8/12 6.53PM - added plots
