# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:25:37 2023

@author: HP
"""

import pandas as pd 
import numpy as np
df=pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
df
df.head()
df.info()
df.isnull().sum()
data=pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],
                df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)
data
data.corr()
data2=data.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},
                  axis=1)
data2[data.duplicated()]
data3=data2.drop_duplicates().reset_index(drop=True)
data3.describe()
data3.corr()
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(data3)
import statsmodels.formula.api as smf
model=smf.ols("Price~Age+KM+HP+CC+Doors+Gears+QT+Weight",
              data=data3).fit()
model.params , np.round(model.pvalues,3)
model.rsquared , model.rsquared_adj 
model.aic
slr_c=smf.ols('Price~CC',data=data3).fit()
slr_c.tvalues , slr_c.pvalues
slr_d=smf.ols('Price~Doors',data=data3).fit()
slr_d.tvalues , slr_d.pvalues
slr_cd=smf.ols('Price~CC+Doors',data=data3).fit()
slr_cd.tvalues , slr_cd.pvalues
rsq_Age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',
                data=data3).fit().rsquared
vif_Age=1/(1-rsq_Age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',
               data=data3).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',
               data=data3).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',
               data=data3).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',
               data=data3).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',
               data=data3).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',
               data=data3).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',
               data=data3).fit().rsquared
vif_WT=1/(1-rsq_WT)
print(vif_Age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT)
import statsmodels.api as sm
sm.qqplot(model.resid,line='q')
import matplotlib.pyplot as plt
plt.title("Normal Q-Q plot of residuals")
plt.show()
list(np.where(model.resid>6000))
list(np.where(model.resid<-6000))
def standard_values(vals) : 
    return (vals-vals.mean())/vals.std() 
plt.scatter(x=standard_values(model.fittedvalues),
            y=standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()
(c,_)=model.get_influence().cooks_distance
c
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data3)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
np.argmax(c) , np.max(c)
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()
k=data3.shape[1]
n=data3.shape[0]
leverage_cutoff = (3*(k+1))/n
data3[data3.index.isin([80])] 
new_data=data3.copy()
data4=new_data.drop(new_data.index[[80]],axis=0).reset_index(drop=True)
while np.max(c)>0.5 :
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',
                  data=data4).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    data4=data4.drop(data4.index[[np.argmax(c)]],
                     axis=0).reset_index(drop=True)
else:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',
                        data=data4).fit()
    final_model.rsquared , final_model.aic
    print(final_model.rsquared)
if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',
                  data=data4).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    data4=data4.drop(data4.index[[np.argmax(c)]],
                     axis=0).reset_index(drop=True)
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',
                        data=data4).fit()
    final_model.rsquared , final_model.aic
    print(final_model.rsquared)
final_model.rsquared
final_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,
                       "Gears":5,"QT":69,"Weight":1012},index=[0])
final_model.predict(final_data)
pred_y=final_model.predict(data4)
print(final_data,pred_y)









