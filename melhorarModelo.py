from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd

baseTrain=pd.read_excel('baseTreinoTratada.xlsx')
baseTest=pd.read_excel('baseTestTratada.xlsx')

# Leitura dos dados
X_test = pd.read_excel('baseTestTratada.xlsx')
X = baseTrain.drop('SalePrice', axis=1)
X = X[0:len(X_test)]
y = baseTrain['SalePrice'][0:len(X_test)]

# Definindo o modelo XGBRegressor
xgb_model = XGBRegressor(random_state=RANDOM_STATE)

# Definindo um grid de parâmetros reduzido
param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Criando um objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

tempo_inicio = time.time()

# Executando a busca em grade
grid_search.fit(X, y)

tempo_fim = time.time()
tempo_execucao = tempo_fim - tempo_inicio

# Exibindo os melhores parâmetros encontrados
print("Melhores Parâmetros:", grid_search.best_params_)

# Exibindo a melhor pontuação (neg_mean_squared_error)
print("Melhor Pontuação:", grid_search.best_score_)
print("Tempo de execução:", tempo_execucao, "segundos")
