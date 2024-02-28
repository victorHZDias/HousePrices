import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline  # atenção para Pipeline do imblearn
import time
import warnings
import os
import sys
from xgboost import XGBRegressor

baseTrain=pd.read_excel('baseTreinoTratada.xlsx')
baseTest=pd.read_excel('baseTestTratada.xlsx')

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    
RANDOM_STATE = 1


def AplicaMelhormodelo(X, y,x_test, modelo):

    pipeline = Pipeline([
        ("modelo", modelo),
    ])

    tempo_inicio = time.time()
    pipeline.fit(X, y)
    y_pred = pipeline.predict(x_test.values)
    tempo_fim = time.time()
    
    tempo_execucao = tempo_fim - tempo_inicio 

    x_test['SalePrice']=y_pred
    
    tempo_execucao = tempo_fim - tempo_inicio 

    return x_test


X_test=pd.read_excel('baseTestTratada.xlsx')

X=baseTrain.drop('SalePrice',axis=1)
X=X[0:len(X_test)]
y=baseTrain['SalePrice'][0:len(X_test)]

# X_test['GarageArea']=X_test['GarageArea'].fillna(X_test['GarageArea'].median())
# x_test,resultado=AplicaMelhormodelo(X, y,X_test, XGBRegressor(random_state=RANDOM_STATE))

modeloTeste = AplicaMelhormodelo(X, y, X_test, XGBRegressor(random_state=RANDOM_STATE,colsample_bytree= 0.8, learning_rate= 0.1, max_depth= 3, n_estimators= 200, subsample= 0.8))

modeloTeste[['Id','SalePrice']].to_csv('Submission10.csv',sep=",",index=False)