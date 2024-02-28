import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline  # atenção para Pipeline do imblearn
import time
import warnings
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

baseTrain=pd.read_excel('baseTreinoTratada.xlsx')
baseTest=pd.read_excel('baseTestTratada.xlsx')

baseTrain.shape
baseTest.shape

x=baseTrain.drop('SalePrice',axis=1)
y=baseTrain['SalePrice']

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    
RANDOM_STATE = 1

#Avaliar qual o melhor modelo para o projeto

def Aplicamodelos(X, y, modelo):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, shuffle=True)

    pipeline = Pipeline([
        ("modelo", modelo),
    ])

    tempo_inicio = time.time()
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test.values)
    tempo_fim = time.time()
    
    tempo_execucao = tempo_fim - tempo_inicio 

    resultado = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "tempo_segundos": tempo_execucao,
    }

    return resultado

Resultado_modelos = {
    "LinearRegression": Aplicamodelos(x, y, LinearRegression()),
    "Ridge": Aplicamodelos(x, y, Ridge(random_state=RANDOM_STATE)),
    "Lasso": Aplicamodelos(x, y, Lasso(random_state=RANDOM_STATE)),
    "ElasticNet": Aplicamodelos(x, y, ElasticNet(random_state=RANDOM_STATE)),
    "DecisionTreeRegressor": Aplicamodelos(x, y, DecisionTreeRegressor(random_state=RANDOM_STATE)),
    "RandomForestRegressor": Aplicamodelos(x, y, RandomForestRegressor(random_state=RANDOM_STATE)),
    "GradientBoostingRegressor": Aplicamodelos(x, y, GradientBoostingRegressor(random_state=RANDOM_STATE)),
    "SVR": Aplicamodelos(x, y, SVR()),
    "KNeighborsRegressor": Aplicamodelos(x, y, KNeighborsRegressor()),
    "XGBRegressor": Aplicamodelos(x, y, XGBRegressor(random_state=RANDOM_STATE)),
    "LGBMRegressor": Aplicamodelos(x, y, LGBMRegressor(random_state=RANDOM_STATE)),
    "CatBoostRegressor": Aplicamodelos(x, y, CatBoostRegressor(random_state=RANDOM_STATE))
}

from pprint import pprint
pprint(Resultado_modelos)

#Após verificação do melhor modelo, aplica e e altera hiperparâmetros

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

# x_test[x_test['LotFrontage'].isna()]

modeloTeste.to_excel('modeloPrevisão.xlsx')
