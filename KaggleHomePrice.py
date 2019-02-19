#FINAL PROJECT
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import cross_validation,preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import sys
#%%
area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea','OpenPorchSF', 
             'EnclosedPorch', '3SsnPorch', 'ScreenPorch','WoodDeckSF', 'PoolArea','LowQualFinSF']
# Read Train
files=sys.argv[1:]
train=pd.read_csv(files[0],index_col=0)
print(train.head())
print('The Training Shape',train.shape)
#%%
#==============================================================================
# # # Study the var through the boxplot
# sns.boxplot(x = 'LotFrontage', hue = 'SalePrice',  data = train,orient='v')
# fig=plt.figure(figsize = (4, 5))	
# plt.title("Boxplot of “LotArea” respect “SalePrice”")
# sns.boxplot(x = 'LotArea', hue = 'SalePrice',  data = train,orient='v')
# plt.savefig('LotArea.png')
# sns.boxplot(x = 'MasVnrArea', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'BsmtFinSF1', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'BsmtFinSF2', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'BsmtUnfSF', hue = 'SalePrice',  data = train,orient='v')
# fig=plt.figure(figsize = (4, 5))
# plt.title("Boxplot of “TotalBsmtSF” respect “SalePrice”")
# sns.boxplot(x = 'TotalBsmtSF', hue = 'SalePrice',  data = train,orient='v')
# plt.savefig('TotalBsmtSF.png')
# sns.boxplot(x = '1stFlrSF', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = '2ndFlrSF', hue = 'SalePrice',  data = train,orient='v')
# fig=plt.figure(figsize = (4, 5))
# plt.title("Boxplot of “GrLivArea” respect “SalePrice”")
# sns.boxplot(x = 'GrLivArea', hue = 'SalePrice',  data = train,orient='v')
# plt.savefig('GrLivArea.png')
# sns.boxplot(x = 'GarageArea', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'OpenPorchSF', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'EnclosedPorch', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = '3SsnPorch', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'ScreenPorch', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'PoolArea', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'WoodDeckSF', hue = 'SalePrice',  data = train,orient='v')
# sns.boxplot(x = 'LowQualFinSF', hue = 'SalePrice',  data = train,orient='v')
#==============================================================================
#%%
train.drop(train[train["LotFrontage"] > 300].index, inplace=True)
train.drop(train[train["LotArea"] > 100000].index, inplace=True)
train.drop(train[train["MasVnrArea"] > 1000].index, inplace=True)
train.drop(train[train["BsmtFinSF1"] > 3000].index, inplace=True)
train.drop(train[train["BsmtFinSF2"] > 1200].index, inplace=True)
train.drop(train[train["BsmtUnfSF"] > 2300].index, inplace=True)
train.drop(train[train["TotalBsmtSF"] > 3000].index, inplace=True)
train.drop(train[train["1stFlrSF"] > 2800].index, inplace=True)
train.drop(train[train["2ndFlrSF"] > 2000].index, inplace=True)
train.drop(train[train["GrLivArea"] > 4000].index, inplace=True)
train.drop(train[train["GarageArea"] > 1300].index, inplace=True)
train.drop(train[train["OpenPorchSF"] > 400].index, inplace=True)
train.drop(train[train["EnclosedPorch"] > 350].index, inplace=True)
train.drop(train[train["3SsnPorch"] > 400].index, inplace=True)
train.drop(train[train["ScreenPorch"] > 350].index, inplace=True)
train.drop(train[train["PoolArea"] > 450].index, inplace=True)
train.drop(train[train["WoodDeckSF"] > 600].index, inplace=True)
train.drop(train[train["LowQualFinSF"] > 400].index, inplace=True)
print('Train dataset cleaned of outliers shape',train.shape)
# # The logaritm of SalePrice
ylog=np.log(train.pop('SalePrice').values)
test_ori=pd.read_csv(files[1],index_col=0)
print(test_ori.head())
print('Test Shape',test_ori.shape)

data=pd.concat([test_ori, train], axis=0)
print('New Concatenated Dataset Shape',data.shape)
#%%
# Remove variable unuseful in my opinion
data.drop(['Utilities'],axis=1,inplace=True)
data.shape

num=list(data.select_dtypes([np.number]).columns)
cat=list(data.select_dtypes([object]).columns)

#==============================================================================
# fig=plt.figure(figsize = (9,9))
# train_box=pd.read_csv("train.csv",index_col=0)
# train_n=train_box.select_dtypes([np.number])
# corr = train_n.corr()
# sns.heatmap(corr, vmax=1, square=True,cmap="RdBu_r")
# xt = plt.xticks(rotation=45)
# xt = plt.yticks(rotation=45)
# plt.title("Correlation of numerical variables")
# plt.savefig('Correlazione.png')
#==============================================================================

data_num=data[num]
#%%
# Work the features in the numerical dataset one by one
num_lab=pd.DataFrame(index=data_num.index)

num_lab['YrSold']=data_num.YrSold
num_lab['MoSold']=data_num.MoSold
num_lab['HalfBath']=data_num['HalfBath']
num_lab['FullBath']=data_num['FullBath']
num_lab['TotBath']=num_lab['FullBath']+num_lab['HalfBath']
num_lab['BsmtHalfBath']=data_num['BsmtHalfBath']
num_lab['BsmtHalfBath'].fillna(data_num.BsmtHalfBath.median(axis=0),inplace=True)
num_lab['BsmtFullBath']=data_num['BsmtFullBath']
num_lab['BsmtFullBath'].fillna(data_num.BsmtFullBath.median(axis=0),inplace=True)
num_lab['BsmtTotBath']=num_lab['BsmtFullBath']+num_lab['BsmtHalfBath']
num_lab['TotBathHouse']=num_lab['FullBath']+num_lab['HalfBath']+num_lab['BsmtFullBath']+num_lab['BsmtHalfBath']
num_lab['BsmtUnfSF']=data_num.BsmtUnfSF
num_lab['BsmtUnfSF'].fillna(data_num.BsmtUnfSF.median(axis=0),inplace=True)
num_lab['TotalBsmtSF']=data_num.TotalBsmtSF
num_lab['TotalBsmtSF'].fillna(data_num.TotalBsmtSF.median(axis=0),inplace=True)
num_lab['BsmtFinSF1']=data_num.loc[:,'BsmtFinSF1']
num_lab['BsmtFinSF1'].fillna(data_num.BsmtFinSF1.median(axis=0),inplace=True)
num_lab['BsmtFinSF2']=data_num.loc[:,'BsmtFinSF2']
num_lab['BsmtFinSF2'].fillna(data_num.BsmtFinSF2.median(axis=0),inplace=True)
num_lab['TotBsmtFinSF1']=num_lab['BsmtFinSF1']+num_lab['BsmtFinSF1']
num_lab['GarageCars']=data_num.GarageCars
num_lab['GarageCars'].fillna(data_num.GarageCars.median(axis=0),inplace=True)
num_lab['GarageArea']=data_num.GarageArea
num_lab['GarageArea'].fillna(data_num.GarageArea.median(axis=0),inplace=True)
# Given the high correlation, I create a new var to substitute
num_lab['GarageAreaPerCars']=num_lab['GarageCars']/num_lab['GarageArea']
num_lab['GarageAreaPerCars'].fillna(num_lab.GarageAreaPerCars.median(axis=0),inplace=True)
num_lab['LotFrontage']=data_num.LotFrontage.fillna(data_num.LotFrontage.median(axis=0))
num_lab['LotArea']=data_num.LotArea
num_lab['GarageYrBlt']=data_num.GarageYrBlt
num_lab['GarageYrBlt'].fillna(data_num.GarageYrBlt.median(axis=0),inplace=True)
num_lab['YearBuilt']=data_num.YearBuilt
num_lab['YearRemodAdd']=data_num.YearRemodAdd
num_lab['1stFlrSF']=data_num.loc[:,'1stFlrSF']
num_lab['2ndFlrSF']=data_num.loc[:,'2ndFlrSF']
num_lab['TotSF1&2Flr']=num_lab['1stFlrSF']+num_lab['2ndFlrSF']
num_lab['GrLivArea']=data_num.GrLivArea
num_lab['BedroomAbvGr']=data_num.BedroomAbvGr
num_lab['KitchenAbvGr']=data_num.KitchenAbvGr
num_lab['TotRmsAbvGrd']=data_num.TotRmsAbvGrd
num_lab['Fireplaces']=data_num.Fireplaces
num_lab['OpenPorchSF']=data_num.OpenPorchSF
num_lab['OpenPorchSF'].fillna(data_num.OpenPorchSF.median(axis=0),inplace=True)
num_lab['EnclosedPorch']=data_num.EnclosedPorch
num_lab['EnclosedPorch'].fillna(data_num.EnclosedPorch.median(axis=0),inplace=True)
num_lab['3SsnPorch']=data_num.loc[:,'3SsnPorch']
num_lab['3SsnPorch'].fillna(data_num.loc[:,'3SsnPorch'].median(axis=0),inplace=True)
num_lab['ScreenPorch']=data_num.ScreenPorch
num_lab['ScreenPorch'].fillna(data_num.ScreenPorch.median(axis=0),inplace=True)
num_lab['WoodDeckSF']=data_num.WoodDeckSF
num_lab['PoolArea']=data_num.PoolArea
num_lab['PoolArea'].fillna(data_num.PoolArea.median(axis=0),inplace=True)
num_lab['LowQualFinSF']=data_num.LowQualFinSF
num_lab["TotalArea"] = data_num[area_cols].sum(axis=1)
num_lab["TotalAreaPorch"] =data_num[area_cols[11:16]].sum(axis=1)
num_lab['OverallQual']=data_num.OverallQual
num_lab['OverallCond']=data_num.OverallCond
num_lab['MiscVal']=data_num.MiscVal
num_lab['MSSubClass']=data_num.MSSubClass
num_lab["MasVnrArea"]=data_num["MasVnrArea"].fillna(data_num.MasVnrArea.median(axis=0))
#%%
data_cat=data[cat]
col_num_to_cat=['MoSold','OverallQual','OverallCond','MSSubClass','MiscVal',
                'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                'PoolArea', 'LowQualFinSF','WoodDeckSF','2ndFlrSF']
le=preprocessing.LabelEncoder()
data_cat=pd.concat([data_cat,data_num[col_num_to_cat]],axis=1)
#%%
col_new=[]
cat_lab=pd.DataFrame(index=data_cat.index)
    
data_cat['Alley'].fillna('NoAlleyAccess',inplace=True)
data_cat['MSZoning'].fillna('RL',inplace=True)
data_cat['MasVnrType'].fillna('Not Found',inplace=True)
data_cat['Electrical'].fillna('SBrkr',inplace=True)
data_cat['GarageType'].fillna('No Garage',inplace=True)
data_cat['SaleType'].fillna('WD',inplace=True)

data_cat["Functional"].replace( {'Maj1':'Other','Maj2': 'Other','Sev': 'Other'},inplace=True)
data_cat['Functional'].fillna('Typ',inplace=True)

data_cat["MoSold"].replace( {1: 'Winter', 2: 'Winter', 3:'Spring', 4:'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11:  'Autumn', 12:  'Winter'},inplace=True)
col_num_to_cat.remove('MoSold')

data_cat["Heating"].replace( {'OthW':'Other','Floor': 'Other'},inplace=True)

data_cat["Exterior1st"].replace( {'AsbShng': 'Other', 'Stucco': 'Other','BrkComm': 'Other','AsphShn': 'Other','CBlock': 'Other','Stone': 'Other','ImStucc':'Other'},inplace=True)
data_cat['Exterior1st'].fillna('VinylSd',inplace=True)
data_cat["Exterior2nd"].replace( {'Wd Shng': 'WdShing','AsbShng': 'Other', 'Stucco': 'Other','Brk Cmn': 'Other','AsphShn': 'Other','CBlock': 'Other','Stone': 'Other','ImStucc':'Other'},inplace=True)
data_cat['Exterior2nd'].fillna('VinylSd',inplace=True)

data_cat.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3},inplace=True)
data_cat.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3},inplace=True)
cat_lab["NewOverallQual"] = data_cat.OverallQual
cat_lab["NewOverallCond"] = data_cat.OverallCond


#==============================================================================
# train_box=pd.read_csv("train.csv",index_col=0)
# fig=plt.figure(figsize = (12, 6))
# sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = train_box)
# xt = plt.xticks(rotation=45)
# plt.title("Boxplot of Neighborhood categories respect “SalePrice”")
# plt.savefig('Neighborhood.png')
#==============================================================================

data_cat["ClassNeighborhood"]=data_cat["Neighborhood"].replace( {'OldTown':'LowClass','NAmes': 'LowClass',
                                                           'Edwards':'LowClass','Sawyer': 'LowClass',
                                                           'BrkSide':'LowClass','IDOTRR': 'LowClass','BrDale':'LowClass',
                                                           'MeadowV':'LowClass','Gilbert': 'MedClass',
                                                           'SawyerW':'MedClass','NWAmes': 'MedClass',
                                                           'Mitchel':'MedClass','SWISU': 'MedClass',
                                                           'Blmngtn':'MedClass','NPkVill':'MedClass',
                                                           'Blueste':'MedClass',
                                                           'CollgCr':'Med-HighClass','Somerst': 'Med-HighClass',
                                                           'Crawfor':'Med-HighClass','Timber': 'Med-HighClass',
                                                           'ClearCr':'Med-HighClass','Veenker': 'Med-HighClass',
                                                           'NridgHt':'HighClass','NoRidge': 'HighClass',                                   
                                                           'StoneBr': 'HighClass'})

new=le.fit_transform(data_cat['ClassNeighborhood'])
cat_lab['ClassNeighborhood']=pd.DataFrame({'ClassNeighborhood':new},index=data_cat['ClassNeighborhood'].index)


#==============================================================================
# fig=plt.figure(figsize = (12, 6))
# sns.boxplot(x = 'MSSubClass', y = 'SalePrice',  data = train_box)
# xt = plt.xticks(rotation=45)
# plt.title("Boxplot of MSSubClass categories respect “SalePrice”")
# plt.savefig('MSSubClass.png')
#==============================================================================

test_ori.loc[2819] #has MSSubClass 150 non contain in train----->'MedClass'
data_cat["MSSubClassCat"]=data_cat["MSSubClass"].replace( {30:'LowClass',45: 'LowClass',
                                                           180:'LowClass',40: 'MedClass',
                                                           50:'MedClass',85: 'MedClass',
                                                           90:'MedClass',150: 'MedClass',
                                                           160: 'MedClass',190: 'MedClass',
                                                           70:'MedClass',80: 'MedClass',
                                                           75:'HighClass',60:'HighClass',
                                                           120: 'HighClass',20: 'HighClass'})

new=le.fit_transform(data_cat['MSSubClassCat'])
cat_lab['MSSubClassCat']=pd.DataFrame({'MSSubClassCat':new},index=data_cat['MSSubClassCat'].index)
new=le.fit_transform(data_cat['MSSubClass'])
cat_lab['CatMSSubClass']=pd.DataFrame({'CatMSSubClass':new},index=data_cat['MSSubClass'].index)

# Categorial to numeric
data_cat['Fence'].fillna('No Fence',inplace=True)
data_cat['BsmtFinType1'].fillna("No Basement",inplace=True)
data_cat['BsmtFinType2'].fillna('No Basement',inplace=True)
data_cat['BsmtExposure'].fillna("No Basement",inplace=True)
data_cat['PoolQC'].fillna("No Pool",inplace=True)
data_cat['KitchenQual'].fillna('TA',inplace=True)
data_cat['BsmtQual'].fillna('No Basement',inplace=True)
data_cat['BsmtCond'].fillna('No Basement',inplace=True)
data_cat['GarageFinish'].fillna('No Garage',inplace=True)
data_cat['GarageQual'].fillna('No Garage',inplace=True)
data_cat['GarageCond'].fillna('No Garage',inplace=True)
data_cat['FireplaceQu'].fillna('No Fireplace',inplace=True)

cat_lab['PavedDrive']=data_cat['PavedDrive'].replace({'Y':2,'P':1,'N':0})
cat_lab['Fence']=data_cat['Fence'].replace({'GdWo':2,'GdPrv':2,'MnWw':1,'MnPrv':1,'No Fence':0})
cat_lab['BsmtFinType1']=data_cat['BsmtFinType1'].replace({'No Basement': 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6})
cat_lab['BsmtFinType2']=data_cat['BsmtFinType2'].replace({'No Basement': 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6})
cat_lab['BsmtExposure']=data_cat['BsmtExposure'].replace({'Av':5,'Gd':4,'Mn':3,'No':1,'No Basement':0})
cat_lab['PoolQC']=data_cat['PoolQC'].replace({'Ex':4,'Gd':3,'Fa':1,'No Pool':0})
cat_lab['KitchenQual']=data_cat['KitchenQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
cat_lab['ExterQual']=data_cat['ExterQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2})
cat_lab['ExterCond']=data_cat['ExterCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
cat_lab['BsmtQual']=data_cat['BsmtQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Basement':0})
cat_lab['BsmtCond']=data_cat['BsmtCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Basement':0})
cat_lab['GarageFinish']=data_cat['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1,'No Garage':0})
cat_lab['GarageQual']=data_cat['GarageQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Garage':0})
cat_lab['GarageCond']=data_cat['GarageCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Garage':0})
cat_lab['HeatingQC']=data_cat['HeatingQC'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
cat_lab['FireplaceQu']=data_cat['FireplaceQu'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Fireplace':0})
#variable to binary
data_cat['MiscFeature'].fillna('None',inplace=True)

cat_lab['HasOpenPorchSF']=(data_cat.OpenPorchSF==0)*1
cat_lab['HasEnclosedPorch']=(data_cat.EnclosedPorch==0)*1
cat_lab['Has3SsnPorch']=(data_cat['3SsnPorch']==0)*1
cat_lab['HasScreenPorch']=(data_cat.ScreenPorch==0)*1
cat_lab['HasWoodDeckSF']=(data_cat.WoodDeckSF==0)*1
cat_lab['HasPool']=(data_cat.PoolArea==0)*1
cat_lab['HasLowQualFinSF']=(data_cat.LowQualFinSF==0)*1
cat_lab['HasMiscFeature']=(data_cat['MiscFeature']=='None')*1
cat_lab['LotShapeReg']=(data_cat['LotShape']=='Reg')*1
cat_lab['LandContour']=(data_cat['LandContour']=='Lv1')*1
cat_lab["Remodeled"] = (data_num["YearRemodAdd"] != data_num["YearBuilt"]) * 1
cat_lab['HasRoofMatl']=(data_cat.RoofMatl=='CompShg')*1
cat_lab['HasStreetPave']=(data_cat['Street']=='Pave')*1
cat_lab['HasPavedDrive']=(data_cat['PavedDrive']=='Y')*1
cat_lab['MiscHasVal']=(data_cat['MiscVal']==0)*1
cat_lab['CentralAir']=(data_cat['CentralAir']=='Y')*1
cat_lab['Has2ndFlr']=(data_cat['2ndFlrSF']==0.0)*1

col_new.extend(['NewOverallQual','NewOverallCond','CatMSSubClass','HasOpenPorchSF','HasEnclosedPorch',
               'Has3SsnPorch','HasScreenPorch','HasWoodDeckSF','HasPool','HasLowQualFinSF','HasMiscFeature',
               'LotShapeReg','LandContour','Remodeled','HasRoofMatl',
               'HasStreetPave','HasPavedDrive','MiscHasVal','CentralAir','Has2ndFlr'])

num_lab=pd.concat([num_lab, cat_lab], axis=1)

#%%
#log transform skewed numeric features:
skewed_var = num_lab.apply(lambda x: skew(x.dropna())) #compute skewness
skewed_var = skewed_var[skewed_var > 0.75].index
num_lab[skewed_var] = np.log1p(num_lab[skewed_var])
#Scale the data so that each feature has zero mean and unit std
scaled = preprocessing.StandardScaler().fit_transform(num_lab)
data_num_f=pd.DataFrame(scaled,index=num_lab.index,columns=list(num_lab.columns))

data_cat.drop(col_num_to_cat,axis=1,inplace=True)
data_cat_f=pd.get_dummies(data_cat)

for col in col_new:
   temp=pd.get_dummies(cat_lab[col], prefix="_" + col)
   data_cat_f=pd.concat([data_cat_f,temp],axis=1)

data=pd.concat([data_cat_f, data_num_f], axis=1)
# Split the dataset into two parts  temporarily for the last fixes
X1=data.loc[train.index]
test1=data.loc[test_ori.index]
#%%
cat_final=list(data_cat_f.columns)
# Columns that not contain value in the X--> dropped
colX_to_drop=X1[cat_final].columns[(X1[cat_final].sum(axis=0) < 1)]
data.drop(colX_to_drop,axis=1,inplace=True)
for col in colX_to_drop:
    cat_final.remove(col)
# Columns that not contain value in the test--> dropped
colTest_to_drop=test1[cat_final].columns[(test1[cat_final].sum(axis=0) < 1)]
data.drop(colTest_to_drop,axis=1,inplace=True)
print('The Final Dataset Shape',data.shape)
#%%
print("######### PREDICTION #########")
X=data.loc[train.index].values
test=data.loc[test_ori.index]
# PREDICTION 
ridge = RidgeCV(alphas = np.logspace(1,1.8,100))
ridge.fit(X, ylog)
print("Best alpha :", ridge.alpha_)
    
clf = Ridge(alpha=ridge.alpha_)
clf.fit(X, ylog)
coef = pd.Series(clf.coef_, index = data.columns) 
print("R^2 of the model is:",clf.score(X,ylog))

imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")

# CROSS VALIDATION
clf_cross= Ridge(alpha=ridge.alpha_)
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,ylog,test_size=0.2)
clf_cross.fit(X_train,y_train)
print("In the cross validation the R^2 of the train set is:",clf_cross.score(X_train,y_train))
print("In the cross validation the R^2 of the test set is:",clf_cross.score(X_test,y_test))

# Create dataset for the prediction
predsRidge=pd.DataFrame({'SalePrice':clf.predict(test)},index=test_ori.index)
predsRidge.head()
predsRidge.SalePrice=np.exp(predsRidge.SalePrice)
predsRidge.head()
predsRidge.to_csv("pred.csv")













