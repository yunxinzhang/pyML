# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:37:42 2018

@author: zyx
"""

import pandas as pd
import numpy as np

data = pd.read_csv("50_Startups.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# categorical_features
le = LabelEncoder()
x[:,-1] = le.fit_transform(x[:,-1])
ohe = OneHotEncoder(categorical_features=[3])
x = ohe.fit_transform(x).toarray()          # toarray()

# don't need all
x = x[:,1:]

# multiple_linear_regression
from sklearn.model_selection import train_test_split

xtr, xt, ytr, yt = train_test_split(x, y)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(xtr,ytr)
yp = lr.predict(xt)
lr.score(xt, yt)

#backward elimilation
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((x.shape[0],1)), values=x, axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
ols.summary()

'''
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     169.9
Date:                Mon, 27 Aug 2018   Prob (F-statistic):           1.34e-27
Time:                        16:14:51   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1063.
Df Residuals:                      44   BIC:                             1074.
Df Model:                           5        
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
'''

x_opt = x[:,[0,1,3,4,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
ols.summary()

'''
OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.946
Method:                 Least Squares   F-statistic:                     217.2
Date:                Mon, 27 Aug 2018   Prob (F-statistic):           8.49e-29
Time:                        16:17:37   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1061.
Df Residuals:                      45   BIC:                             1070.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
Omnibus:                       14.758   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.172
Skew:                          -0.948   Prob(JB):                     2.53e-05
Kurtosis:                       5.563   Cond. No.                     1.40e+06
==============================================================================
'''

x_opt = x[:,[0,3,4,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
ols.summary()

'''
 OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     296.0
Date:                Mon, 27 Aug 2018   Prob (F-statistic):           4.53e-30
Time:                        16:19:08   Log-Likelihood:                -525.39
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      46   BIC:                             1066.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
Omnibus:                       14.838   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
Skew:                          -0.949   Prob(JB):                     2.21e-05
Kurtosis:                       5.586   Cond. No.                     1.40e+06
==============================================================================
'''

x_opt = x[:,[0,3,5]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
ols.summary()

'''
  OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     450.8
Date:                Mon, 27 Aug 2018   Prob (F-statistic):           2.16e-31
Time:                        16:20:13   Log-Likelihood:                -525.54
No. Observations:                  50   AIC:                             1057.
Df Residuals:                      47   BIC:                             1063.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================
'''

x_opt = x[:,[0,3]]
ols = sm.OLS(endog=y, exog=x_opt).fit()
ols.summary()

xtr2, xt2, ytr2, yt2 = train_test_split(x_opt, y)
lr2 = LinearRegression()
lr2.fit(xtr2,ytr2)
yp2 = lr2.predict(xt2)
lr2.score(xt2, yt2)