"""
"""

import pandas as pd 
df=pd.read_csv("Salary_Data.csv")
df
df.shape
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt
plt.scatter(df['YearsExperience'],df['Salary'])
plt.show()
df.corr()
df.describe()
import seaborn as sns
sns.distplot(df["YearsExperience"])
sns.regplot(x=df['YearsExperience'],y=df['Salary'])
import statsmodels.formula.api as smf
model1=smf.ols("Salary~YearsExperience",data=df).fit()
model1.summary()
model1.params  
model1.tvalues , model1.pvalues
model1.rsquared 
pred=model1.predict(df["YearsExperience"])
pred
import numpy as np
rmse_lin=np.sqrt(np.mean((np.array(df["Salary"])-
                          np.array(pred))**2))
model2 = smf.ols("Salary~np.log(YearsExperience)",data=df).fit()
model2.summary()
pred2 = model2.predict(pd.DataFrame(df['YearsExperience'])) 
pred2.corr(df["Salary"])
rmse_log=np.sqrt(np.mean((np.array(df['Salary'])-np.array(pred2))**2)) 
model3=smf.ols("np.log(Salary)~YearsExperience",data=df).fit()
model3.summary()
model3.params
pred_log = model3.predict(pd.DataFrame(df['YearsExperience']))
pred3=np.exp(pred_log) 
rmse_exp = np.sqrt(np.mean((np.array(df['Salary'])-np.array(pred3))**2)) 
df["YearsExperience_sq"]=df["YearsExperience"]*df["YearsExperience"]
model3_quad= smf.ols("np.log(Salary)~YearsExperience+YearsExperience_sq",
                 data=df).fit()
model3_quad.summary()
model3_quad.params
pred_quad=model3_quad.predict(df)
pred4=np.exp(pred_quad)  
rmse_quad=np.sqrt(np.mean((np.array(df['Salary'])-np.array(pred4))**2))

plt.scatter(df["YearsExperience"],df["Salary"],c="b")
plt.plot(df["YearsExperience"],pred4,"r") 
plt.plot(np.arange(30),model3_quad.resid_pearson)
plt.axhline(y=0,color='red')
plt.xlabel("Observation Number")
plt.ylabel("Standardized Residual")   
data = {"MODEL":pd.Series(["rmse_lin","rmse_log","rmse_exp","rmse_quad"]),
        "RMSE_Values":pd.Series([rmse_lin,rmse_log,rmse_exp,rmse_quad]),
        "Rsquare":pd.Series([model1.rsquared,model2.rsquared,
                             model3.rsquared,model3_quad.rsquared])}
table=pd.DataFrame(data)
table 