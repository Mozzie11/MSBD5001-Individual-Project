import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRFRegressor

df=pd.read_csv('/Users/zhouwen/Desktop/5001IndiProject/train.csv',
                       parse_dates=['purchase_date','release_date'])
df1=pd.read_csv('/Users/zhouwen/Desktop/5001IndiProject/test.csv',
                       parse_dates=['purchase_date','release_date'])
df,df1=df.set_index('id'),df1.set_index('id')#357rows*11columns
len,len1=df.shape[0],df1.shape[0]


#fill the null
df,df1=df.fillna(axis=0,method='ffill'),df1.fillna(axis=0,method='ffill')
quality,time=[],[]
quality1,time1=[],[]
for i in range(len):
    if df['total_negative_reviews'][i] == 0.0:
        df['total_negative_reviews'][i]=1.0
    quality.append(df['total_positive_reviews'][i]/df['total_negative_reviews'][i])
    time.append(df['purchase_date'][i].toordinal() - df['release_date'][i].toordinal())
    if time[i] < 0:
        time[i] = -time[i]

for i in range(len1):
    if df1['total_negative_reviews'][i] == 0.0:
        df1['total_negative_reviews'][i]=1.0
    quality1.append(df1['total_positive_reviews'][i]/df1['total_negative_reviews'][i])
    time1.append(df1['purchase_date'][i].toordinal() - df1['release_date'][i].toordinal())
    if time1[i] < 0:
        time1[i] = -time1[i]

df['quality'],df['time']=quality,time
df1['quality'],df1['time']=quality1,time1

##Deal with discrete feature
le=LabelEncoder()
df['genres'],df1['genres']=le.fit_transform(df['genres']),le.fit_transform(df1['genres'])
df['categories'],df1['categories']=le.fit_transform(df['categories']),le.fit_transform(df1['categories'])
df['tags'],df1['tags']=le.fit_transform(df['tags']),le.fit_transform(df1['tags'])

#Normalization
Norm=MinMaxScaler()
scale_features = ['price','time','genres','categories',
                  'tags','total_positive_reviews','total_negative_reviews']
df[scale_features],df1[scale_features]=\
    Norm.fit_transform(df[scale_features]),Norm.fit_transform(df1[scale_features])
df,df1=df.drop(['purchase_date','release_date'], axis=1),df1.drop(['purchase_date','release_date'],axis=1)
df['price'],df1['price']=df['price'].map(lambda x: x*1000),\
                         df1['price'].map(lambda x: x*1000)#100or1000orLog
df['is_free'],df1['is_free']=df['is_free'].astype('int'),df1['is_free'].astype('int')
#df.info()
#df1.info()

#data
y = df['playtime_forever']
df= df.drop(['playtime_forever'],axis=1)
X=np.array(df)
X_test=np.array(df1)

#model
rfr=XGBRFRegressor()
bagging_rfr=BaggingRegressor(rfr, n_estimators=20, max_samples=0.8,
                             max_features=1.0, bootstrap=True,
                             bootstrap_features=False, n_jobs=-1)
model=bagging_rfr.fit(X,y)

#test
predictions=bagging_rfr.predict(X_test)
result = pd.DataFrame({'id':df1.index,
                       'playtime_forever':predictions.astype(np.float64)}).set_index('id')
print(result)
a=result.shape[0]
print(a)

result.to_csv('/Users/zhouwen/Desktop/5001IndiProject/test.csv')#samplesubmission_test