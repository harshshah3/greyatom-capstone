#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import humanize
import re

path_invoice = 'data/Final_invoice.csv'
path_JTD = 'data/JTD.csv'
path_customer_data = 'data/Customer_Data.xlsx'
path_plant_master = 'data/Plant Master.xlsx'

# %%
invoice_data_original = pd.read_csv(path_invoice)
JTD_data_original = pd.read_csv(path_JTD)

customer_data_original = pd.read_excel(path_customer_data)
plant_data_original = pd.read_excel(path_plant_master)

#%%
print(customer_data_original.head(10))

#print(customer_data_original.describe())
#print(customer_data_original.shape)
ā
print(customer_data_original.info())

# %%
def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name


def data_clean_formatt(df):
    # Checking for Null/Missing Values
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    missing_vals_df = pd.DataFrame({'column_name': df.columns,
                                    'missing_vals_%': percent_missing})
    missing_vals_df = missing_vals_df.reset_index().drop(columns='index')
    missing_vals_df = missing_vals_df.sort_values(by = 'missing_vals_%', ascending= False)
    missing_vals_df = missing_vals_df[missing_vals_df['missing_vals_%'] >= 20.00]
    # Plotting Missing values chart
    missing_vals_df.plot.barh(x='column_name', y='missing_vals_%', rot=0)
    plt.title("Column-wise Bar-Plot of Missing values in - " + get_df_name(df))
    plt.show()
    # Dropping columns with >70% of missing values
    df_trimmed = df.dropna(thresh=df.shape[0] * 0.7, how='all', axis=1)
    # =============================================================================
    #     #Formatting Column-Names
    #     for col in df_trimmed.columns:
    #       col = re.sub(r'/.*|\\.*', r'', col)
    # =============================================================================
    df_trimmed.columns = df_trimmed.columns.str.replace(' ', '_')
    df_trimmed.columns = df_trimmed.columns.str.strip('.')
    df_trimmed.columns = df_trimmed.columns.str.lower()

    # Setting dtypes of columns

    # Filling NaN values

    return df_trimmed



# %%
print("-" * 80,'\n', customer_data_original.head(), "-" * 80,'\n\n')
customer_data = data_clean_formatt(customer_data_original)
print("-" * 80,'\n', customer_data.head(), "-" * 80,'\n\n')
print(customer_data.columns.to_list())


# %%
print("-" * 80,'\n', invoice_data_original.head(), "-" * 80,'\n\n')
invoice_data = data_clean_formatt(invoice_data_original)
print("-" * 80, invoice_data.head(), "-" * 80, end='\n\n')
print(invoice_data.columns.to_list())

# %%

print("-" * 80, JTD_data_original.head(), "-" * 80, end='\n\n')
JTD_data = data_clean_formatt(JTD_data_original)
print("-" * 80, JTD_data.head(), "-" * 80, end='\n\n')
print(JTD_data.columns.to_list())

# %%
print(customer_data.info(), "-" * 80, end='\n\n')
print(invoice_data.info(), "-" * 80, end='\n\n')
print(JTD_data.info(), "-" * 80, end='\n\n')

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

# %%

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

