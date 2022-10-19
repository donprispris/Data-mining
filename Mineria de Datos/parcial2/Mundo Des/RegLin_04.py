# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:28:11 2022

@author: RGAMBOAH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


import os
import sys
# =============================================================================
#
# =============================================================================

cadsep = "=" * 60

pd.set_option('display.width',200,
              'display.max_columns',25,
              'display.max_rows',500,
              'expand_frame_repr',True,
              'display.min_rows',50,
              'display.max_seq_items',200)

np.set_printoptions(linewidth =  200,
                    suppress  = None,
                    edgeitems = 2000,
                    threshold = 5000)


def salida(letrero):
   print(cadsep)
   print('saliendo:' + letrero)
   print(cadsep)
   sys.exit(0)

# =============================================================================
#
# =============================================================================
tray = os.path.abspath(os.path.dirname(__file__)) + '/'

traynomArch = tray + "MundoDes.csv"
df = pd. read_csv(traynomArch,encoding="CP1250")

df =df.drop(labels="región",axis = 1)

df["Log10PNBPerCapita"] = np.log10(  df.PNBperCapita)
#df['const'] = 1

# print(cadsep)
# print("                    statsmodels.api as sm")
# print(cadsep)

# mdl = sm.OLS(df['tasaNat'],df[['const','tasaMort','mortInf','espVidaHom','espVidaMuj','Log10PNBPerCapita']],
#              hasconst=True).fit()

# print(mdl.summary())
# print(cadsep)
# print("                      Estimaciones")
# print(cadsep)
# yt = mdl.predict()
# print(yt)

# =============================================================================
#                                   Regresión Lineal
# =============================================================================
print(cadsep)
print("                   statsmodels.formula.api as smf")
print('                  datos centrados y estandarizados')
print(cadsep)
dfstd = (df -  df.mean())/df.std()
# =============================================================================
#   Gráfica de celosía para aprciar dependencias entre variables cuantitativas
# =============================================================================
#_ = sns.pairplot(dfstd, kind="reg", diag_kind="kde")
#plt.show()
# =============================================================================
#              Tabla de correlaciones de los datos estandaizados
# =============================================================================
print(dfstd.corr())
#
strFormula = 'tasaNat ~ tasaMort + mortInf + espVidaHom + espVidaMuj + Log10PNBPerCapita -1'
#strFormula = 'tasaNat ~ tasaMort + espVidaMuj - 1'
resultado  = smf.ols(strFormula,dfstd).fit()
print(resultado.summary())
#print(cadsep)
#print("                Estimaciones")
#print(cadsep)
#yt = resultado.predict()
#print(yt)
#print(cadsep)
cobertura = 0.95
print(cadsep)
print("                Estimaciones e intervalos de confianza")
print(cadsep)
df_yt_int_conf = resultado.get_prediction().summary_frame(alpha=1-cobertura)
print(df_yt_int_conf)
x = df.tasaNat
###x = df_yt_int_conf['mean']
##plt.plot(x,yt,'o',
##         x,df_yt_int_conf.obs_ci_lower,'+',
##         x,df_yt_int_conf.obs_ci_upper,'.')
##plt.title('tasaNat: Real vs Estimado e int conf ' + str(100*cobertura)+ '%')
##plt.xlabel("tasaNat")
##plt.ylabel("estimacion tasaNat")
##plt.show()
print(cadsep)
#
# =============================================================================
#   Obteniendo X y Y
# =============================================================================
X         = dfstd[['tasaMort','mortInf','espVidaHom','espVidaMuj','Log10PNBPerCapita']]
Y         = dfstd['tasaNat']
# =============================================================================
#                    Regresión por Gradiente Descendente Estocástico
# =============================================================================
alfa_sgd = 1e-3
lr_sgd   = 'invscaling'
eta0_sgd = 0.01

reg_sgd = SGDRegressor(alpha=alfa_sgd,fit_intercept = False,learning_rate=lr_sgd,tol=1e-5,eta0=eta0_sgd)
res_sgd = reg_sgd.fit(X,Y.ravel())

print(cadsep)
print('         Regresión Lineal por Gradiente Estocástico Descendente')
print('                            datos estandarizados')
print(cadsep)
print('                     alpha:',alfa_sgd)
print('             learning_rate:',lr_sgd)
print('                      eta0:',eta0_sgd)
print('              Coeficientes:',reg_sgd.coef_)
print('Coeficiente de correlación:',reg_sgd.score(X,Y))
print('                tolerancia:',reg_sgd.tol)
print('     Número de iteraciones:',reg_sgd.n_iter_)

# =============================================================================
#                             Regresión Ridge
# =============================================================================
alfa_ridge   = 2
solver_ridge = 'lsqr'

ridge = Ridge(alpha=alfa_ridge,tol=1e-4,solver=solver_ridge,max_iter=1000,fit_intercept = False)
res_ridge = ridge.fit(X,Y.ravel())
print(cadsep)
print('                               Regresión Ridge')
print('                            datos estandarizados')
print(cadsep)
print('                     alpha:',alfa_ridge)
print('                    solver:',solver_ridge)
print('              Coeficientes:',ridge.coef_)
print('Coeficiente de correlación:',ridge.score(X,Y))
print('     Número de iteraciones:',ridge.n_iter_[0])
# =============================================================================
#                             Regresión Lasso
# =============================================================================
alfa_lasso   = 0.015 # probar 0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04

lasso     = Lasso(alpha=alfa_lasso,tol=1e-4,max_iter=1000,fit_intercept = False)
res_lasso = lasso.fit(X,Y.ravel())
print(cadsep)
print('                               Regresión Lasso')
print('                            datos estandarizados')
print(cadsep)
print('                     alpha:',alfa_lasso)
print('              Coeficientes:',lasso.coef_)
print('Coeficiente de correlación:',lasso.score(X,Y))
print('     Número de iteraciones:',lasso.n_iter_)
# =============================================================================
#                    Regresión Elastica (por Gradiente Descendente Estocástico) 
# =============================================================================
alfa_ela   = 0.025
modo_ela   = 'elasticnet' #'elasticnet' # l1 para Lasso y l2 para Ridge
prop_ela   = 0.5         # proporción entre Lasso y Ridge.
lr_ela     =  'constant' #'adaptive' #'constant' #'invscaling'
eta0_ela   = 0.01

reg_ela = SGDRegressor(alpha=alfa_ela,penalty = modo_ela, l1_ratio = prop_ela,
                       fit_intercept = False,learning_rate=lr_ela,tol=1e-5,eta0=eta0_ela)
res_ela = reg_ela.fit(X,Y.ravel())

print(cadsep)
print('         Regresión Elástica (por Gradiente Descendente Estocático)')
print('                            datos estandarizados')
print(cadsep)
print('                     alpha:',alfa_ela)
print('                   penalty:',modo_ela)
print('             learning_rate:',lr_ela)
print('                      eta0:',eta0_ela)
print('                  l1_ratio:',prop_ela)
print('              Coeficientes:',reg_ela.coef_)
print('Coeficiente de correlación:',reg_ela.score(X,Y))
print('                tolerancia:',reg_ela.tol)
print('     Número de iteraciones:',reg_ela.n_iter_)
# =============================================================================
#                               Fin del código
# =============================================================================
