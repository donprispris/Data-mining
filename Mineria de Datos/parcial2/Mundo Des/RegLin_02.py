# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:28:11 2022

@author: RGAMBOAH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import os
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
#
# =============================================================================
print(cadsep)
print("                   statsmodels.formula.api as smf")
print(cadsep)
strFormula = 'tasaNat ~ tasaMort + mortInf + espVidaHom + espVidaMuj + Log10PNBPerCapita'
resultado  = smf.ols(strFormula,df).fit()
print(resultado.summary())
print(cadsep)
print("                Estimaciones")
print(cadsep)
yt = resultado.predict()
print(yt)
print(cadsep)
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

# =============================================================================
#  Obteniendo los coeficientes vía las ecuaciones normales
# =============================================================================
df['uno'] = 1
X         = df[['tasaMort','mortInf','espVidaHom','espVidaMuj','Log10PNBPerCapita']]
Xmean     = X.mean()
Xstd      = X.std()
X         = (X - X.mean())/X.std()
Y         = df['tasaNat']
Ymean     = Y.mean()
Tstd      = Y.std()
Y         = ( Y - Y.mean())/Y.std()
XtX       = X.T.dot(X)
invXtX    = np.linalg.inv(XtX)
XtY       = X.T.dot(Y)
A         = invXtX.dot(XtY)
cols      = X.columns 
for k in range(len(A)):
   print((cols[k]+'.'*20)[:20],'\t',A[k])
ecp = ((X.dot(A)-Y)**2).mean()
print('ecp:',ecp)
   

# =============================================================================
#  Cálculo por gradiente descendiente
# =============================================================================
def gradDesc(X,Y,Ak,const_apren,traza):
    gD   = (2.0/91.0)*X.T.dot(X.dot(Ak) - Y)
    Akm1 = Ak - const_apren * gD
    if traza:
        print(cadsep)
        print(Ak)
        print(gD)
        print(Akm1)
        ecp = ((X.dot(Akm1)-Y)**2).mean()
        print(ecp)
    return Akm1.values

Ag    = []
Ag.append(0.5*(-1+2*np.random.rand(len(A)))) #(A) #(1+2*np.random.rand(len(A))))
const_apren = 0.1 # 0.25
for k in range(2001):
    traza = k % 100 == 0 
    if traza:
        print(cadsep)
        print('            ',k)
    Ag.append(gradDesc(X,Y,Ag[k],const_apren,traza))
print(cadsep)
##print(cadsep)
##for k in range(len(Ag)):
##    print(k,Ag[k])
print(cadsep)
print('                Solución por ecuaciones normales')
print(cadsep)
for k in range(len(A)):
   print((cols[k]+'.'*20)[:20],'\t',A[k])
ecp = ((X.dot(A)-Y)**2).mean()
print('ecp:',ecp)              
print(cadsep)
