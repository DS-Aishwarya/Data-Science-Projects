#!/usr/bin/env python
# coding: utf-8

# <p> <font color="Blue" face="Arial" size="+3">
# SMDM PROJECT:
#  <font></p> 

# <p> <font color="Red" face="Arial" size="+1">
# 
#  Problem 1:
# 
#     
# A wholesale distributor operating in different regions of Portugal has information on annual spending of several items in their stores across different regions and channels. The data consists of 440 large retailers’ annual spending on 6 different varieties of products in 3 different regions (Lisbon, Oporto, Other) and across different sales channel (Hotel, Retail).
#     <font></p>

# <p><font color="purple" face="Arial" size="+1" style="Bold">
#     
# This data set has 440 rows and 9 columns. The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories. This data set is recommended for learning and practicing your skills in exploratory data analysis, data visualization. The Following data dictionary gives more details on this data set:
# 
# 
# #### Description of variables is as follows:
# 
# FRESH : annual spending (m.u.) on fresh products (Continuous);
#     
# MILK : annual spending (m.u.) on milk products (Continuous);
#     
# GROCERY : annual spending (m.u.)on grocery products (Continuous);
#     
# FROZEN : annual spending (m.u.)on frozen products (Continuous);
#     
# DETERGENTS_PAPER : annual spending (m.u.) on detergents and paper products (Continuous);
#     
# DELICATESSEN : annual spending (m.u.)on and delicatessen products (Continuous);
#     
# CHANNEL : customers Channel - Hotel (Hotel/Restaurant/Cafe) or Retail channel (Nominal);
#     
# REGION : customers Region Lisnon, Oporto or Other (Nominal);
#     
# BUYER/SPENDER : it is showing running id number (assumption it is index) (Continuous);
#      <font> </p>

# ### The dataset gives data about sales of 6 category of products across 3 regions via 2 channel.
# 
# Region Frequency Region - total : 440 rows Lisbon 77 rows Oporto 47 rows Other 316 row
# 
# Channel Frequency Channel -total : 440 rows Hotel 298 rows Retail 142 rows

# <p> <font color="Blue" face="Arial" size="+1">
# IMPORT THE DATA:
#  <font></p> 

# In[2]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
import os
from   scipy.stats        import    chi2_contingency
from scipy.stats import   ttest_1samp,ttest_ind


# <p> <font color="Blue" face="Arial" size="+1">
# LOAD THE DATA:
#  <font></p>

# In[3]:


df=pd.read_csv('Wholesale_Customers_Data.csv')
df.head()


# <p> <font color="Blue" face="Arial" size="+1">
# SUMMARY OF THE DATASET:
#  <font></p>

# In[4]:


df.Region.unique()


# In[5]:


df.Channel.unique()


# In[6]:


df.describe().T


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


print("The number of rows are ",df.shape[0],"\nThe number of columns are",df.shape[1])


# In[10]:


Total=df.size
print("Total number of elements is", Total)


# <p> <font color="Blue" face="Arial" size="+1">
# CHECKING FOR MISSING VALUES:
#  <font></p>

# In[11]:


df.isnull().values.any()


# In[12]:


plt.figure(figsize=(15,10))
df["Region"].hist()
plt.ylabel("Count")
plt.xlabel("Region")
plt.show()
 


# <p> <font color="BLACK" face="Arial">
# 6 continuous types of feature ('Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen')
# 
# 2 categoricals features ('Channel', 'Region')
# 
# 1 continuous types of feature (Buyer/Spender) will be dropped as no use for our analysis
#     <font></p>

# In[13]:


wholesale_customer=df.copy()
wholesale_customer


# In[14]:


del wholesale_customer['Buyer/Spender']
wholesale_customer


# In[15]:


df


# <p> <font color="BLUE" face="Arial" size="+1">
# REGION COUNT:
# <font></p>

# In[16]:


wholesale_customer['Region'].value_counts()


# <p> <font color="BLUE" face="Arial" size="+1">
# CHANNEL COUNT:
# <font></p>

# In[17]:


wholesale_customer['Channel'].value_counts()


# In[18]:


def categorical_multi(i,j):
    pd.crosstab(wholesale_customer[i],wholesale_customer[j]).plot(kind='bar')
    plt.show()
    print(pd.crosstab(wholesale_customer[i],wholesale_customer[j]))
categorical_multi(i='Channel',j='Region')


# <p> <font color="BLUE" face="Arial" size="+2">
#        EDA :
# <font></p>

# We are going to start exploring our data with the Univariate analysis (each feature individually), before carrying the Bivariate analysis and compare pairs of features to find correlation between them.

# <p> <font color="Blue" face="Arial" size="+1">
# DESCRIPTIVE ANALYSIS OF DATA:
#  <font></p>

# In[19]:


print('Descriptive Statastics of our Data including Channel & Retail:')
wholesale_customer.describe(include='all').T


# # Univariate

# In[175]:


def plot_distribution(df, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(df.shape[1]) / cols)
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            g = sns.histplot(y=column, data=df)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.histplot(df[column])
            plt.xticks(rotation=25) 
            
plot_distribution(wholesale_customer, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)


# From the graphs on the distribution of product it seems that we have some outliers in the data, let's have a closer look before we decide what to do:

# In[102]:


# Let’s remove the categorical columns:
products = wholesale_customer[wholesale_customer.columns[+2:wholesale_customer.columns.size]]

#Let’s plot the distribution of each feature
def plot_distribution(df2, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(df2.shape[1]) / cols)
    for i, column in enumerate(df2.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        g = sns.boxplot(df2[column])
        plt.xticks(rotation=25)
    
plot_distribution(products, cols=3, width=20, height=10, hspace=0.45, wspace=0.5)


# Outliers are detected but not necessarily removed, it depends of the situation. Here I will assume that the wholesale distributor provided us a dataset with correct data, so I will keep them as is.

# # Bivariate :

# In[99]:


sns.set(style="ticks")
g = sns.pairplot(products,corner=True,kind='reg')
g.fig.set_size_inches(15,15)


# From the pairplot above, the correlation between the "detergents and paper products" and the "grocery products" seems to be pretty strong, meaning that consumers would often spend money on these two types of product. Let's look at the Pearson correlation coefficient to confirm this:

# In[103]:


# Compute the correlation matrix
corr = products.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .6},annot=True)

plt.title("Pearson correlation", fontsize =20)


# #### There is strong correlation (0.92) between the "detergents and paper products" and the "grocery products"

# ## 1.1. Use methods of descriptive statistics to summarize data.
# 
# ### Which Region and which Channel seems to spend less?

# In[24]:


print('Descriptive Statastics of our Data:')
wholesale_customer.describe().T


# In[25]:


print('Descriptive Statastics of our Data including Channel & Retail:')
wholesale_customer.describe(include='all').T


# Measure of Central Tendency - Mean, Median, mode Measure of Dispersion - Range, IQR, Standard Deviation
# 
# From the above two describe function, we can infer the following
# 
# Channel has two unique values, with "Hotel" as most frequent with 298 out of 440 transactions. i.e 67.7 percentage of spending comes from "Hotel" channel.
# 
# Retail has three unique values, with "Other" as most frequent with 316 out of 440 transactions. i.e.71.8 percentage of spending comes from "Other" region.
# 
# Fresh item (440 records),
# 
# has a mean of 12000.3, standard deviation of 12647.3, with min value of 3 and max value of 112151 .
# 
# The other aspect is Q1(25%) is 3127.75, Q3(75%) is 16933.8, with Q2(50%) 8504
# 
# range = max-min =112151-3=112,148 & IQR = Q3-Q1 = 16933.8-3127.75 = 13,806.05 (this helpful in calculating the outlier(1.5 IQR Lower/Upper limit))
# Milk item (440 records),
# 
# has a mean of 5796.27, standard deviation of 7380.38, with min value of 55 and max value of 73498.
# 
# The other aspect is Q1(25%) is 1533, Q3(75%) is 7190.25, with Q2(50%) 3627
# 
# range = max-min =73498-55=73443 & IQR = Q3-Q1 = 7190.25-1533 = 5657.25 (this helpful in calculating the outlier(1.5 IQR Lower/Upper limit))
# 
# Grocery item (440 records),
# 
# has a mean of 7951.28, standard deviation of 9503.16, with min value of 3 and max value of 92780.
# 
# The other aspect is Q1(25%) is 2153, Q3(75%) is 10655.8, with Q2(50%) 4755.5
# 
# range = max-min =92780-3=92777 & IQR = Q3-Q1 = 10655.8-2153 = 8502.8 (this helpful in calculating the outlier(1.5 IQR Lower/Upper limit))
# Frozen (440 records),
# 
# has a mean of 3071.93, standard deviation of 4854.67, with min value of 25 and max value of 60869.
# 
# The other aspect is Q1(25%) is 742.25, Q3(75%) is 3554.25, with Q2(50%) 1526
# 
# range = max-min =60869-25=60844 & IQR = Q3-Q1 = 3554.25-742.25 = 2812 (this helpful in calculating the outlier(1.5 IQR Lower/Upper limit))
# 
# Detergents_Paper (440 records),
# 
# has a mean of 2881.49, standard deviation of 4767.85, with min value of 3 and max value of 40827.
# 
# The other aspect is Q1(25%) is 256.75, Q3(75%) is 3922, with Q2(50%) 816.5
# 
# range = max-min =40827-3=40824 & IQR = Q3-Q1 = 3922-256.75 = 3665.25 (this helpful in calculating the outlier(1.5 IQR Lower/Upper limit))
# 
# Delicatessen (440 records),
# 
# has a mean of 1524.87, standard deviation of 2820.11, with min value of 3 and max value of 47943.
# 
# The other aspect is Q1(25%) is 408.25, Q3(75%) is 1820.25, with Q2(50%) 965.5
# 
# range = max-min =47943-3=47940 & IQR = Q3-Q1 = 1820.25-408.25 = 1412 (this helpful in calculating the outlier(1.5 IQR Lower/Upper limit))
# 
# 

# pd.crosstab(df['Channel'],df['Region'] )

# ### Which Region and which Channel seems to spend more?

# <p><font color="BLUE" face="Arial" size="+2" style="Bold">
# The Region and Channel spent most:
#     <font></p>

# In[26]:


wholesale_customer_max_spent =wholesale_customer.copy()
wholesale_customer_max_spent


# In[27]:


wholesale_customer_max_spent['Spending']=wholesale_customer['Fresh']+wholesale_customer['Milk']+wholesale_customer['Grocery']+wholesale_customer['Frozen']+wholesale_customer['Detergents_Paper']+wholesale_customer['Delicatessen']
wholesale_customer_max_spent


# In[28]:


region = wholesale_customer_max_spent.groupby('Region')['Spending'].sum()
print(region)


# In[29]:


channeldf = wholesale_customer_max_spent.groupby('Channel')['Spending'].sum()
print(channeldf)


# In[30]:


print("Highest spend in the Region is from Others and lowest spend in the region is from Oporto\nHighest spend in the Channel is from Hotel and lowest spend in the Channel is from Retail")


# In[31]:


region_channel_df = wholesale_customer_max_spent.groupby(['Region','Channel'])['Spending'].sum()
print(region_channel_df)


# #### Highest spend in the Region/Channel is from Others/Hotel
# 
# #### and lowest spend in the Region/Channel is from Oporto/Hotel

# In[32]:


df1 = wholesale_customer.drop(columns=['Region'])
mean1 = df1.groupby('Channel').mean()
mean1.round(2)


# In Channel "Hotel" Average Highest Spending in Fresh items and Lowest Spending in Detergents_Paper.
# 
# In Channel "Retail" Average Highest Spending in Grocery items and Lowest Spending in Frozen items.

# In[33]:


df2 = wholesale_customer.drop(columns=['Channel'])
mean2 = df2.groupby('Region').mean()
mean2.round(2)


# In Region "Lisbon" Average Highest Spending in Fresh and Lowest in Delicatessen items.
# 
# In Region "Oporto" Average Highest Spending in Fresh and Lowest in Delicatessen items.
# 
# In Region "Other" Average Highest Spending in Fresh and Lowest in Delicatessen items.

# ## 1.2 There are 6 different varieties of items that are considered. Describe and comment/explain all the varieties across Region and Channel? Provide a detailed justification for your answer.

# See Behaviour in all items across Channel and Region use Bar Plot. Here we see that they are different in Channel and Region.

# In[34]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="Channel", y="Fresh", hue ="Region", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Fresh')


# In[35]:


sns.catplot(x="Channel", y="Fresh", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Fresh')


# In[36]:


sns.catplot(x="Region", y="Fresh", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Fresh')


# ####  Based on the plot, Fresh item is sold more in the Retail channel

# In[37]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="Channel", y="Milk", hue ="Region", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Milk')


# In[38]:


sns.catplot(x="Channel", y="Milk", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Milk')


# In[39]:


sns.catplot(x="Region", y="Milk", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Milk')


# In[40]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="Channel", y="Grocery", hue ="Region", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Grocery')


# In[41]:


sns.catplot(x="Channel", y="Grocery", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Grocery')


# In[42]:


sns.catplot(x="Region", y="Grocery", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Grocery')


# In[43]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="Channel", y="Frozen", hue ="Region", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Frozen')


# In[44]:


sns.catplot(x="Channel", y="Frozen", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Frozen')


# In[45]:


sns.catplot(x="Region", y="Frozen", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Frozen')


# In[46]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="Channel", y="Detergents_Paper", hue ="Region", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Detergents_Paper')


# In[47]:


sns.catplot(x="Channel", y="Detergents_Paper", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Detergents_Paper')


# In[48]:


sns.catplot(x="Region", y="Detergents_Paper", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Detergents_Paper')


# In[49]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="Channel", y="Delicatessen", hue ="Region", kind="bar", ci=None, data=wholesale_customer)
plt.title('Delicatessen')


# In[50]:


sns.catplot(x="Channel", y="Delicatessen", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Delicatessen')


# In[51]:


sns.catplot(x="Region", y="Delicatessen", kind="bar", ci=None, data=wholesale_customer)
plt.title('Item - Delicatessen')


# ### 1.3. On the basis of the descriptive measure of variability, which item shows the most inconsistent behaviour?Which items shows the least inconsistent behaviour?

# In[52]:


products = wholesale_customer[wholesale_customer.columns[+2:wholesale_customer.columns.size]]
standard_deviation_items = products.std() #used standard deviation to check the measure of variabilty
standard_deviation_items.round(2)


# #### Fresh item have highest Standard deviation So that is Inconsistent.
# 
# #### Delicatessen item have smallest Standard deviation, So that is consistent.

# In[53]:


cv_fresh = np.std(products['Fresh']) / np.mean(products['Fresh'])
cv_fresh


# In[54]:


cv_milk = np.std(products['Milk']) / np.mean(products['Milk'])
cv_milk


# In[55]:


cv_grocery = np.std(products['Grocery']) / np.mean(products['Grocery'])
cv_grocery


# In[56]:


cv_frozen = np.std(products['Frozen']) / np.mean(products['Frozen'])
cv_frozen


# In[57]:


cv_detergents_paper = np.std(products['Detergents_Paper']) / np.mean(products['Detergents_Paper'])
cv_detergents_paper


# In[58]:


cv_delicatessen = np.std(products['Delicatessen']) / np.mean(products['Delicatessen'])
cv_delicatessen


# In[59]:


from scipy.stats import variation
print(variation(products, axis = 0))


# #### “Fresh” item have lowest coefficient of Variation So that is consistent.
# 
# #### “Delicatessen” item have highest coefficient of Variation, So that is Inconsistent.

# In[60]:


variance_items = products.var()
variance_items


# In[61]:


products.describe().T


# In[62]:


plt.style.use('seaborn-pastel')
products.plot.area(stacked=False,figsize=(11,5))
plt.grid()
plt.show()


# ### 1.4. Are there any outliers in the data?

# In[63]:


plt.figure(figsize=(15,8))
sns.boxplot(data=products, orient="h", palette="Set2")


# In[64]:


def plot_distribution(items, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(items.shape[1]) / cols)
    for i, column in enumerate(items.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        g = sns.boxplot(items[column])
        plt.xticks(rotation=25)
    
plot_distribution(products, cols=3, width=20, height=10, hspace=0.45, wspace=0.5)


# #### Yes there are outliers in all the items across the product range (Fresh, Milk, Grocery, Frozen, Detergents_Paper & Delicatessen)

# In[65]:


products.hist(figsize=(6,6));


# In[66]:


def out_std(s, nstd=3.0, return_thresholds=False):
    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]


# In[67]:


def out_iqr(s, k=1.5, return_thresholds=False):
    # calculate interquartile range
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
     # calculate the outlier cutoff
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_thresholds:
        return lower, upper
    else: # identify outliers
        return [True if x < lower or x > upper else False for x in s]


# In[68]:


# outlier_mask is a boolean list identifies the indices of the outliers
outlier_mask = out_std(products['Fresh'], nstd=3.0)
# first 10 elements
outlier_mask[:10]


# Identify the outliers, notice these values are on both low and high

# In[69]:


products['Fresh'][outlier_mask]


# In[70]:


plt.figure(figsize=(8,6))
sns.histplot(products['Fresh'], kde=False);
plt.vlines(products['Fresh'][outlier_mask], ymin=0, ymax=110, linestyles='dashed')


# In[71]:


# outlier_mask is a boolean list identifies the indices of the outliers
outlier_mask_Milk = out_std(products['Milk'], nstd=3.0)
# first 10 elements
outlier_mask_Milk[:10]


# In[72]:


products['Milk'][outlier_mask_Milk]


# In[73]:


plt.figure(figsize=(8,6))
sns.histplot(products['Milk'], kde=False);
plt.vlines(products['Milk'][outlier_mask_Milk], ymin=0, ymax=110, linestyles='dashed');


# In[74]:


# outlier_mask is a boolean list identifies the indices of the outliers
outlier_mask_Frozen = out_std(products['Frozen'], nstd=3.0)
# first 10 elements
outlier_mask_Frozen[:10]


# In[75]:


products['Frozen'][outlier_mask_Frozen]


# In[76]:


plt.figure(figsize=(8,6))
sns.histplot(products['Frozen'], kde=False);
plt.vlines(products['Frozen'][outlier_mask_Frozen], ymin=0, ymax=110, linestyles='dashed');


# In[77]:


# outlier_mask is a boolean list identifies the indices of the outliers
outlier_mask_Grocery= out_std(products['Grocery'], nstd=3.0)
# first 10 elements
outlier_mask_Grocery[:10]


# In[78]:


products['Grocery'][outlier_mask_Grocery]


# In[79]:


plt.figure(figsize=(8,6))
sns.histplot(products['Grocery'], kde=False);
plt.vlines(products['Grocery'][outlier_mask_Grocery], ymin=0, ymax=110, linestyles='dashed');


# In[80]:


# outlier_mask is a boolean list identifies the indices of the outliers
outlier_mask_Detergents_Paper= out_std(products['Detergents_Paper'], nstd=3.0)
# first 10 elements
outlier_mask_Detergents_Paper[:10]


# In[81]:


products['Detergents_Paper'][outlier_mask_Detergents_Paper]


# In[82]:


plt.figure(figsize=(8,6))
sns.histplot(products['Detergents_Paper'], kde=False);
plt.vlines(products['Detergents_Paper'][outlier_mask_Detergents_Paper], ymin=0, ymax=110, linestyles='dashed');


# In[83]:


# outlier_mask is a boolean list identifies the indices of the outliers
outlier_mask_Delicatessen = out_std(products['Delicatessen'], nstd=3.0)
# first 10 elements
outlier_mask_Delicatessen[:10]


# In[84]:


products['Delicatessen'][outlier_mask_Delicatessen]


# In[85]:


plt.figure(figsize=(8,6))
sns.histplot(products['Delicatessen'], kde=False);
plt.vlines(products['Delicatessen'][outlier_mask_Delicatessen], ymin=0, ymax=110, linestyles='dashed');


# In[86]:


def plot_cutoff(dataframe, col, nstd=2.0, color='red'):
    lower, upper = out_std(dataframe[col], nstd=nstd, return_thresholds=True)
    plt.axvspan(min(dataframe[col][dataframe[col] < lower], default=dataframe[col].min()), lower, alpha=0.2, color=color);
    plt.axvspan(upper, max(dataframe[col][dataframe[col] > upper], default=dataframe[col].max()), alpha=0.2, color=color);


# In[87]:


column = 'Fresh'
sns.histplot(products[column], kde=False)
plot_cutoff(products, column, nstd=2.0, color='red');
plot_cutoff(products, column, nstd=3.0, color='green');
plot_cutoff(products, column, nstd=4.0, color='yellow');


# In[88]:


column = 'Milk'
sns.histplot(products[column], kde=False)
plot_cutoff(products, column, nstd=2.0, color='red');
plot_cutoff(products, column, nstd=3.0, color='green');
plot_cutoff(products, column, nstd=4.0, color='yellow');


# In[89]:


column = 'Grocery'
sns.histplot(products[column], kde=False)
plot_cutoff(products, column, nstd=2.0, color='red');
plot_cutoff(products, column, nstd=3.0, color='green');
plot_cutoff(products, column, nstd=4.0, color='yellow');


# In[90]:


column = 'Frozen'
sns.histplot(products[column], kde=False)
plot_cutoff(products, column, nstd=2.0, color='red');
plot_cutoff(products, column, nstd=3.0, color='green');
plot_cutoff(products, column, nstd=4.0, color='yellow');


# In[91]:


column = 'Detergents_Paper'
sns.histplot(products[column], kde=False)
plot_cutoff(products, column, nstd=2.0, color='red');
plot_cutoff(products, column, nstd=3.0, color='green');
plot_cutoff(products, column, nstd=4.0, color='yellow');


# In[92]:


column = 'Delicatessen'
sns.histplot(products[column], kde=False)
plot_cutoff(products, column, nstd=2.0, color='red');
plot_cutoff(products, column, nstd=3.0, color='green');
plot_cutoff(products, column, nstd=4.0, color='yellow');


# In[105]:


wholesale_customer.groupby(['Channel', 'Region']).agg(['mean', 'std']).round(1)


# In[106]:


def hist_plot(column):
    fig = plt.figure()
    ax = fig.add_subplot(111) # stands for subplot(1,1,1)
    ax.hist(products[column], bins=25)
    plt.title('Histgram plot of ' + column)
    plt.show()

columns = ['Milk', 'Grocery', 'Detergents_Paper']
for c in columns:
    hist_plot(c)


# There are several paris of features exhibit correlations, such as Milk and Grocery, Milk and Detergents_Paper. It confirms my suspicions.
# 
# All these three features are not normally distributed. Most of the points lie on the left, closer to the minimal value. We can see the skewness of all variables are greater than 1, indicates they are righ skewed.

# ### 1.5 On the basis of your analysis, what are your recommendations for the business? How can your analysis help the business to solve its problem? Answer from the business perspective

# Solution :As per the analysis, I find out that there are inconsistencies in spending of different items (by calculating
# Coefficient of Variation), which should be minimized. The spending of Hotel and Retail channel are
# different which should be more or less equal. And also spent should equal for different regions. Need to
# focus on other items also than “Fresh” and “Grocery”
# 

# <p> <font color="Red" face="Arial" size="+1">
# Problem 2 
# 
# The Student News Service at Clear Mountain State University (CMSU) has decided to gather data about the undergraduate students that attend CMSU. CMSU creates and distributes a survey of 14 questions and receives responses from 62 undergraduates (stored in the Survey data set).
#     <font> </p>
#     

# In[108]:


survey = pd.read_csv('Survey.csv')
survey.head()


# ###  2.1. For this data, construct the following contingency tables (Keep Gender as row variable)
# ### 2.1.1. Gender and Major
# 

# In[109]:


survey_major = pd.crosstab(index=survey['Gender'],columns=survey['Major'])
survey_major['RowTotal'] = survey_major.sum(axis=1)
survey_major.loc['ColumnTotal'] = survey_major.sum()
survey_major


# ###  2.1.2. Gender and Grad Intention

# In[110]:


survey_gradint = pd.crosstab(index=survey['Gender'],columns=survey['Grad Intention'])
survey_gradint['RowTotal'] = survey_gradint.sum(axis=1)
survey_gradint.loc['ColumnTotal'] = survey_gradint.sum()
survey_gradint


# ###  2.1.3. Gender and Employment

# In[111]:


survey_emp = pd.crosstab(index=survey['Gender'],columns=survey['Employment'])
survey_emp['RowTotal'] = survey_emp.sum(axis=1)
survey_emp.loc['ColumnTotal'] = survey_emp.sum()
survey_emp


# ### 2.1.4. Gender and Computer

# In[112]:


survey_comp = pd.crosstab(index=survey['Gender'],columns=survey['Computer'])
survey_comp['RowTotal'] = survey_comp.sum(axis=1)
survey_comp.loc['ColumnTotal'] = survey_comp.sum()
survey_comp


# ### 2.2. Assume that the sample is representative of the population of CMSU. Based on the data, answer the following question:
# 
# #### 2.2.1. What is the probability that a randomly selected CMSU student will be male?

# In[113]:


print(f"P(male)={round(survey_major.loc['Male','RowTotal']/survey_major.loc['ColumnTotal','RowTotal']*100,2)}%")


# #### 2.2.2. What is the probability that a randomly selected CMSU student will be female?

# In[114]:


print(f"P(female)={round(survey_major.loc['Female','RowTotal']/survey_major.loc['ColumnTotal','RowTotal']*100,2)}%")


# ### 2.3. Assume that the sample is representative of the population of CMSU. Based on the data, answer the following question:

# In[116]:


survey_major


# #### 2.3.1. Find the conditional probability of different majors among the male students in CMSU.

# In[117]:


for i in survey_major.iloc[:,:-1]:
    print(f"P({i}|Male)={round(survey_major.loc['Male',i]/survey_major.loc['Male','RowTotal']*100,2)}%")


# #### 2.3.2 Find the conditional probability of different majors among the female students of CMSU.

# In[118]:


for i in survey_major.iloc[:,:-1]:
    print(f"P({i}|Female)={round(survey_major.loc['Female',i]/survey_major.loc['Female','RowTotal']*100,2)}%")


# ### 2.4. Assume that the sample is a representative of the population of CMSU. Based on the data, answer the following question:

# In[119]:


survey_gradint


# #### 2.4.1. Find the probability That a randomly chosen student is a male and intends to graduate.

# In[120]:


print(f"P(Male and Intends to Graduate)={round(survey_gradint.loc['Male','Yes']/survey_gradint.loc['Male','RowTotal']*100,2)}%")


# # 2.4.2 Find the probability that a randomly selected student is a female and does NOT have a laptop. 

# In[144]:


survey_comp


# In[145]:


print(f"P(Female and doesnot have laptop)={round(survey_comp.loc['Female',['Desktop','Tablet']].sum()/survey_comp.loc['Female','RowTotal']*100,2)}%")


# ### 2.5. Assume that the sample is representative of the population of CMSU. Based on the data, answer the following question:

# #### 2.5.1. Find the probability that a randomly chosen student is a male or has full-time employment?

# In[142]:


survey_emp


#     P(Male)=29/62
#     P(Full-Time Employment)=10/62
#     P(Male and Full-Time Employment)=7/62
#     P(Male or Full-Time Employment)=P(Male)+P(Full-Time Employment)-P(Male and Full-Time Employment)
#                                    =29/62+10/62-7/62 = 32/62

# In[143]:


print(f"""P(Male or Full-time employment)={round((survey_emp.loc['Male','RowTotal']+
                                        survey_emp.loc['ColumnTotal','Full-Time']-
                                        survey_emp.loc['Male','Full-Time'])/
                                        survey_emp.loc['ColumnTotal','RowTotal']*100,2)}%""")


# #### 2.5.2 Find the conditional probability that given a female student is randomly chosen, she is majoring in international business or management.

#     P(International_business or Management | Female) = ?
#     P(International_buinesss | Female) = 4/33
#     P(Management | Female) = 4/33
#     P(International_business or Management | Female) = P(International_buinesss | Female) + P(Management | Female)
#                                                      = 4/33 + 4/33 = 8/33
#     (Since chosing International business and Management are mutually exclusive events)

# In[125]:


survey_major


# In[126]:


print(f"""P(International Business or Management | Female) = {round((survey_major.loc['Female','International Business']+
                                                            survey_major.loc['Female','Management'])/
                                                            survey_major.loc['Female','RowTotal']*100,2)}%""")


# ### 2.6 Construct a contingency table of Gender and Intent to Graduate at 2 levels (Yes/No). The Undecided students are not considered now and the table is a 2x2 table. 

# In[127]:


survey_gradint_2 = pd.crosstab(index=survey['Gender'],
                    columns = survey[(survey['Grad Intention']=='Yes')|(survey['Grad Intention']=='No')]['Grad Intention'])
survey_gradint_2['RowTotal'] = survey_gradint_2.sum(axis=1)
survey_gradint_2.loc['ColumnTotal'] = survey_gradint_2.sum()
survey_gradint_2


# ##### Do you think graduate intention and being female are independent events?
# To answer this question, we compare the probability that a randomly selected student intents to graduate with the probability that a randomly selected female student intends to graduate. If these two probabilities are the same (or very close), we say that the events are independent. In other words, independence means that being female does not affect the likelihood of having the intention to graduate.
# 
# To answer this question, we compare:
# <ol><li>the unconditional probability: P(Intention to graduate)</li>
# <li>the conditional probability: P(Intention to graduate | female)</li></ol>
# If these probabilities are equal (or at least close to equal), then we can conclude that having the intention to graduate is independent of being a female. If the probabilities are substantially different, then we say the variables are dependent. 
#     
#     If P(A | B) =  P(A), then the two events A and B are independent

# In[128]:


print(f"""P(Intention to graduate) = {round(survey_gradint_2.loc['ColumnTotal','Yes']/
                                    survey_gradint_2.loc['ColumnTotal','RowTotal']*100,2)}%""")


# In[129]:


print(f"""P(Intention to graduate | Female) = {round(survey_gradint_2.loc['Female','Yes']/
                                            survey_gradint_2.loc['Female','RowTotal']*100,2)}%""")


# As we can see from above that the probabilities are not equal (or atleast close to equal). This means that graduate intention and being female are dependent events.

# ### 2.7 Note that there are four numerical (continuous) variables in the data set, GPA, Salary, Spending and Text Messages. Answer the following questions based on the data

# #### 2.7.1 If a student is chosen randomly, what is the probability that his/her GPA is less than 3?

# In[130]:


print(f"P(student with probability less than 3) = {round(survey[survey['GPA']<3].shape[0]/survey.shape[0]*100,2)}%")


# #### 2.7.2 Find conditional probability that a randomly selected male earns 50 or more. Find conditional probability that a randomly selected female earns 50 or more.

# In[131]:


print(f"""P(Earning 50 or more | Male) = {round(survey[(survey['Gender']=='Male') & (survey['Salary']>=50)].shape[0]/
                                        survey[survey['Gender']=='Male'].shape[0]*100,2)}%""")


# In[132]:


print(f"""P(Earning 50 or more | Female) = {round(survey[(survey['Gender']=='Female') & (survey['Salary']>=50)].shape[0]/
                                        survey[survey['Gender']=='Female'].shape[0]*100,2)}%""")


# ### 2.8 Note that there are four numerical (continuous) variables in the data set, GPA, Salary, Spending and Text Messages. For each of them comment whether they follow a normal distribution.

# In[133]:


import seaborn as sns
import matplotlib.pyplot as plt
for i in survey[['GPA', 'Salary', 'Spending', 'Text Messages']]:
    sns.histplot(survey[i],kde=True)
    plt.show()


# Based on the diagrams we can say that the variables 'GPA' and 'Salary' are normally distributed to an extend. Whereas 'Spending' and 'Text Messages' are right skewed and hence not normally distributed.

# <p> <font color="Red" face="Arial" size="+1">
# Problem 3
# 
# An important quality characteristic used by the manufacturers of ABC asphalt shingles is the amount of moisture the shingles contain when they are packaged. Customers may feel that they have purchased a product lacking in quality if they find moisture and wet shingles inside the packaging.   In some cases, excessive moisture can cause the granules attached to the shingles for texture and coloring purposes to fall off the shingles resulting in appearance problems. To monitor the amount of moisture present, the company conducts moisture tests. A shingle is weighed and then dried. The shingle is then reweighed, and based on the amount of moisture taken out of the product, the pounds of moisture per 100 square feet are calculated. The company would like to show that the mean moisture content is less than 0.35 pounds per 100 square feet.
# <font> </p>

# In[135]:


df3=pd.read_csv("A&B shingles.csv")
df3


# In[136]:


df3.shape


# In[137]:


df3.isnull().any()


# ### 3.1 Do you think there is evidence that means moisture contents in both types of shingles are within the permissible limits? State your conclusions clearly showing all steps.

# #### For the A shingles, 
# The null hypothesis states the population mean moisture content is less than 0.35 pound per 100 square feet.
# The alternative hypothesis states that the population mean moisture content is greater than 0.35 pound per 100 square feet.
#                  
#                   𝐻0 : 𝜇 ≤ 0.35  
#                   𝐻1 : 𝜇 > 0.35
#                   

# In[170]:


# one sample t-test
# null hypothesis: expected value = 0.35
t_statistic, p_value = ttest_1samp(df3, 0.35)
print('One sample t test \nt statistic: {0} p value: {1} '.format(t_statistic, p_value/2))


# #### Conclusion:
# Since pvalue > 0.05, do not reject H0 . There is not enough evidence to conclude that the mean moisture
# content for Sample A shingles is less than 0.35 pounds per 100 square feet. p-value = 0.0748. If the
# population mean moisture content is in fact no less than 0.35 pounds per 100 square feet, the probability
# of observing a sample of 36 shingles that will result in a sample mean moisture content of 0.3167 pounds
# per 100 square feet or less is .0748

# #### For the B shingles, 
# The null hypothesis states the population mean moisture content is less than 0.35 pound per 100 square feet.
# The alternative hypothesis states that the population mean moisture content is greater than 0.35 pound per 100 square feet.
#                  
#                   𝐻0 : 𝜇 ≤ 0.35  
#                   𝐻1 : 𝜇 > 0.35

# In[171]:


# one sample t-test
# null hypothesis: expected value = 0.35
t_statistic, p_value = ttest_1samp(df3, 0.35,nan_policy='omit' )
print('One sample t test \nt statistic: {0} p value: {1} '.format(t_statistic, p_value/2))


# #### Conclusion:
# Since pvalue < 0.05, reject H0 . There is enough evidence to conclude that the mean moisture content for
# Sample B shingles is not less than 0.35 pounds per 100 square feet. p-value = 0.0021. If the population mean
# moisture content is in fact no less than 0.35pounds per 100 square feet, the probability of observing a sample
# of 31 shingles that will result in a sample mean moisture content of 0.2735 pounds per 100 square feet or less
# is .0021.
# 

# ### 3.2 Do you think that the population mean for shingles A and B are equal? Form the hypothesis and conduct the test of the hypothesis. What assumption do you need to check before the test for equality of means is performed?

# ### Step 1: Define null and alternative hypotheses

# #### In testing whether the population mean of shingles A and B are equal.
# 
# The Null Hypothesis : The population mean of both shingles A and B are equal.
#                                      i.e $\mu{A}$ = $\mu{B}$
#                                      
# The Alternative Hypothesis : The population mean of both shingles A and B are different.i.e $\mu{A}$ $\neq$ $\mu{B}$

# ### Step 2: Decide the significance level

# Here we select  𝛼  = 0.05 and the population standard deviation is not known.

# ### Step 3: Calculate the p - value and test statistic

# In[140]:


t_statistic,p_value=ttest_ind(df3['A'],df3['B'],equal_var=True ,nan_policy='omit') 
print("t_statistic={} and pvalue={}".format(round(t_statistic,3),round(p_value,3)))


# In[141]:


# p_value < 0.05 => alternative hypothesis:
# they don't have the same mean at the 5% significance level
print ("two-sample t-test p-value=", p_value)

alpha_level = 0.05

if p_value < alpha_level:
    print('We have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
    print('We conclude that the population mean of both the shingles A and B are not same.')
else:
    print('We do not have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
    print('We conclude that the population mean of both the shingles A and B are same.')


# ### Conclusion:
# As the pvalue > α , do not reject H0; and we can say that population mean for shingles A and B are equal Test
# Assumptions When running a two-sample t-test, the basic assumptions are that the distributions of the two
# populations are normal, and that the variances of the two distributions are the same. If those assumptions are
# not likely to be met, another testing procedure could be use.
