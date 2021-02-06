# EXPLORATORY DATA ANALYSIS

import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import shap
import xgboost as xgb
%matplotlib inline

df = pd.read_csv('Data Analytics/train.tsv',sep = '\t')

msk = np.random.rand(len(df))<0.8
train = df[msk]
test = df[~msk]

print(train.shape)
print(test.shape)

train.head() #shiping 1 the retailer pays for shiping 0 means the consumer pays for shiping

train.info()

train.price.describe()

plt.subplot(1,2,1)
(train['price']).plot.hist(bins = 50, figsize=(12,6), edgecolor = 'white', range = [0,250])
plt.xlabel("price")
plt.title('Price Distribution')

plt.subplot(1,2,2)
np.log(train['price']+1).plot.hist(bins = 50, figsize = (12,6), edgecolor = 'white')
plt.xlabel("log(price+1)")
plt.title('Price Distribution')

#Percentage of data when buyer or seller pays the shipping price
train['shipping'].value_counts()/len(train) * 100

shipping_fee_by_buyer = train.loc[df['shipping']==0, 'price']
shipping_fee_by_seller = train.loc[df['shipping']==1,'price']

fig,ax = plt.subplots(figsize=(18,8))
ax.hist(shipping_fee_by_seller,color='#8CB4E1',alpha=1.0,bins=50,range=[0,100],
              label = 'Price when seller pays shipping')
ax.hist(shipping_fee_by_buyer,color='#007D00',alpha=0.7,bins=50,range=[0,100],
                            label = 'Price when buyer pays shipping')
plt.xlabel('price',fontsize=12)
plt.ylabel('frequency',fontsize = 12)
plt.title('Price Distribution by shipping Type',fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()

print("The average price is {}".format(round(shipping_fee_by_buyer.mean(),2)),"if buyer pays shipping");
print("The average price is {}".format(round(shipping_fee_by_seller.mean(),2)),"if seller pays shipping");

fig,ax = plt.subplots(figsize=(18,8))
ax.hist(np.log(shipping_fee_by_seller+1),color = '#8CB4E1',alpha = 1.0, bins = 50, label = 'Price when seller pays shipping')
ax.hist(np.log(shipping_fee_by_buyer+1),color = '#007D00',alpha = 0.7, bins = 50, label = 'Price when buyer pays shipping')
plt.xlabel('log(price+1)',fontsize=12)
plt.ylabel('frequency',fontsize=12)
plt.title('Price Distribution by shipping type', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()

train['category_name'].nunique()

#Top 10 category of data
print(train['category_name'].value_counts()[:10])

sns.boxplot(x = "item_condition_id", y = np.log(train['price']+1),data = train, palette=sns.color_palette('RdBu',5))

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

print("Missing Category name:%d"%train['category_name'].isnull().sum())
print("Missing brand name:%d"%train['brand_name'].isnull().sum())
print("Missing item description:%d"%train['item_description'].isnull().sum())

# DATA CLEANING
#filling the missing value
def handling_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing',inplace=True)
    dataset['brand_name'].fillna(value = 'missing',inplace=True)
    dataset['item_description'].replace('No description yet','missing',inplace = True)
    dataset['item_description'].fillna(value='missing',inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x:x.index!='missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand),'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x:x.index!='missing'].index[:NUM_CATEGORIES]

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

df = pd.read_csv('Data Analytics/train.tsv',sep = '\t')
msk = np.random.rand(len(df))<0.8
train = df[msk]
test = df[~msk]

test_new = test.drop('price',axis = 1)
y_test = np.log1p(test["price"])
nrow_train = train.shape[0]
y = np.log1p(train["price"])
merge:pd.DataFrame = pd.concat([train,test_new])

handling_missing_inplace(merge)
cutting(merge)
to_categorical(merge)

merge.head()

#CONVERTING THE DATA INTO A SPARSE MATRIX FOR LIGHTGBM.
cv = CountVectorizer(min_df=NAME_MIN_DF) #minimum data frequency
X_name = cv.fit_transform(merge['name'])

X_name

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])

#tf-idf : term frequency times inverse document-frequency  idf(t) = log [ n / df(t) ] + 1 ;  tf-idf(t, d) = tf(t, d) * idf(t);
#it is used to scale down the imapct of frequently occuring words.

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,ngram_range=(1,3), stop_words='english')
X_description = tv.fit_transform(merge['item_description'])

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id','shipping']],dtype=float).values)

sparse_merge = hstack((X_dummies,X_description,X_brand,X_category,X_name)).tocsr()

mask = np.array(np.clip(sparse_merge.getnnz(axis=0)-1,0,1),dtype=bool)
sparse_merge = sparse_merge[:,mask]


X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

train_X = lgb.Dataset(X,label=y)

params = {
'learning_rate':0.50,
'application':'regression',
'max_depth':5,
'num_leaves':100,
'verbosity':-1,
'metric':'RMSE',
}

#evals_result = {}
gbm = lgb.train(params, train_set = train_X, num_boost_round=3200,verbose_eval = False,keep_training_booster=True)

gbm.save_model('model_lightgbm.txt')

gbm.save_model('model_lightgbm_iter2.txt')

gbm.save_model('model_lightgbm_iter3.txt')

gbm.save_model('model_lightgbm_iter4.txt')

gbm = lgb.Booster(model_file='model_lightgbm.txt')

gbm = lgb.Booster(model_file='model_lightgbm_iter2.txt')

gbm_3dep = lgb.Booster(model_file='model_lightgbm_iter3.txt')

gbm = lgb.Booster(model_file='model_lightgbm_iter4.txt')

y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)

print('The RMSE of prediction is:',mean_squared_error(y_test,y_pred)**0.5)

lgb.plot_tree(gbm,figsize=(10,10),show_info=['internal_value','internal_count','leaf_count','leaf_weight','column_name'])

lgb.plot_tree(gbm_3dep,figsize=(20,20),show_info=['internal_value','internal_count','leaf_count','leaf_weight','column_name'])
