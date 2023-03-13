# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:40:30 2023

@author: HP
"""

import pandas as pd
import numpy as np
data=pd.read_csv("50_Startups.csv")
data.info()
data.isnull().sum()
data.describe()
data.duplicated().sum()
data1=data.rename({"R&D Spend":"rds","Administration":"adms",
                  "Marketing Spend":"mkts"},axis=1)
data1[data1.duplicated()]
data1.corr()
import seaborn as sns
sns.set_style(style="darkgrid")
sns.pairplot(data1)
import statsmodels.formula.api as smf
model=smf.ols("Profit~rds+adms+mkts",data=data1).fit()
model.params 
model.tvalues , np.round(model.pvalues,3)
model.rsquared , model.rsquared_adj
slr_a=smf.ols("Profit~adms",data=data1).fit()
slr_a.tvalues , slr_a.pvalues
slr_m=smf.ols("Profit~mkts",data=data1).fit()
slr_m.tvalues , slr_m.pvalues
mlr_am=smf.ols("Profit~adms+mkts",data=data1).fit()
mlr_am.tvalues , np.round(mlr_am.pvalues,3)
rsq_r=smf.ols("rds~adms+mkts",data=data1).fit().rsquared
vif_r=1/(1-rsq_r)
rsq_a=smf.ols("adms~rds+mkts",data=data1).fit().rsquared
vif_a=1/(1-rsq_a)
rsq_m=smf.ols("mkts~rds+adms",data=data1).fit().rsquared
vif_m=1/(1-rsq_m)
data2={'Variables':['rds','adms','mkts'],'Vif':[vif_r,vif_a,vif_m]}
vif_df=pd.DataFrame(data2)
import statsmodels.api as sm
sm.qqplot(model.resid,line='q')
import matplotlib.pyplot as plt
plt.title("Normal Q-Q plot of residuals")
plt.show()
list(np.where(model.resid<-30000))
def standard_values(vals) : 
    return (vals-vals.mean())/vals.std()
plt.scatter(standard_values(model.fittedvalues),
            standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'rds',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'adms',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'mkts',fig=fig)
plt.show()
(c,_)=model.get_influence().cooks_distance
c
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
np.argmax(c) , np.max(c)
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()
k=data1.shape[1]
n=data1.shape[0]
leverage_cutoff=(3*(k+1))/n
data1[data1.index.isin([49])]
data2=data1.drop(data1.index[[49]],axis=0).reset_index(drop=True)
while np.max(c)>0.5 :
    model=smf.ols("Profit~rds+adms+mkts",data=data2).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    data2=data2.drop(data2.index[[np.argmax(c)]],
                     axis=0).reset_index(drop=True)
else:
    final_model=smf.ols("Profit~rds+adms+mkts",data=data2).fit()
    final_model.rsquared , final_model.aic
    print(final_model.rsquared)
final_model.rsquared
new_data=pd.DataFrame({'rds':70000,"adms":90000,"mkts":140000},index=[0])
final_model.predict(new_data)
pred_y=final_model.predict(data2)
d2={'Prep_Models':['Model','Final_Model'],
    'Rsquared':[model.rsquared,final_model.rsquared]}
table=pd.DataFrame(d2)
