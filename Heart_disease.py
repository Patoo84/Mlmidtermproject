#!/usr/bin/env python
# coding: utf-8

# In[63]:


import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import mlxtend
from colorama import Fore, Back, Style 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj

init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

get_ipython().run_line_magic('matplotlib', 'inline')

import xgboost
import lightgbm
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


# In[62]:


pip install catboost


# In[2]:


df = pd.read_csv('/Users/mohamed_romdhan.GROUPEVSC/Downloads/archive (5)/heart_failure_clinical_records_dataset.csv')


# In[3]:


df.head()


# In[4]:


df.describe().round()


# # Setting up the validation framework

# In[7]:


from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2)
df_train, df_val = train_test_split(df_full_train, test_size=0.25)


# In[8]:


len(df_full_train), len(df_test),len(df_val)


# In[10]:


y_train = df_train.DEATH_EVENT.values
y_val = df_val.DEATH_EVENT.values
y_test = df_test.DEATH_EVENT.values


# In[11]:


del df_train['DEATH_EVENT']
del df_val['DEATH_EVENT']
del df_test['DEATH_EVENT']


# # EDA
Check missing values
look at the target variable (DEATH_EVENT)
look at numerical and categorical variables
# In[12]:


df_full_train.isnull().sum()


# In[13]:


df_full_train.DEATH_EVENT.value_counts(normalize=True)


# In[14]:


df_full_train.DEATH_EVENT.mean()


# In[15]:


global_DEATH_EVENT_rate = df_full_train.DEATH_EVENT.mean()
round(global_DEATH_EVENT_rate,2)


# In[16]:


df_full_train.dtypes


# In[24]:


Death_female = df_full_train[df_full_train.sex==0].DEATH_EVENT.mean()
Death_female

Sex - Gender of patient Male = 1, Female =0
Age - Age of patient
Diabetes - 0 = No, 1 = Yes
Anaemia - 0 = No, 1 = Yes
High_blood_pressure - 0 = No, 1 = Yes
Smoking - 0 = No, 1 = Yes
DEATH_EVENT - 0 = No, 1 = Yes
# In[25]:


Death_male = df_full_train[df_full_train.sex==1].DEATH_EVENT.mean()
Death_male


# In[26]:


global_death = df_full_train.DEATH_EVENT.mean()
global_death


# In[23]:


df_full_train.sex.value_counts()


# In[27]:


global_death - Death_female


# In[28]:


from IPython.display import display


# In[30]:


Features_columns = ['time','ejection_fraction','serum_creatinine','sex','diabetes','anaemia','high_blood_pressure','smoking']


# In[31]:


for c in Features_columns:
    print (c)
    df_group = df_full_train.groupby(c).DEATH_EVENT.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - global_death
    df_group['risk'] = df_group['mean']/global_death
    display(df_group)
    print()
    print()


# In[32]:


from sklearn.metrics import mutual_info_score


# In[33]:


mutual_info_score(df_full_train.DEATH_EVENT,df_full_train.sex)


# In[34]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.DEATH_EVENT)


# In[35]:


mi = df_full_train[Features_columns].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)


# In[36]:


df_full_train[Features_columns].corrwith(df_full_train.DEATH_EVENT).abs()


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix


# In[40]:


Features = ['time','ejection_fraction','serum_creatinine']
x = df[Features]
y = df["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)


# In[41]:


accuracy_list = []


# In[44]:


# logistic regression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
accuracy_list.append(100*log_reg_acc)


# In[46]:


print("Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))


# In[56]:


cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[64]:


sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = accuracy_score(y_test, sv_clf_pred)
accuracy_list.append(100* sv_clf_acc)


# In[65]:


print(Fore.GREEN + "Accuracy of SVC is : ", "{:.2f}%".format(100* sv_clf_acc))


# In[66]:


cm = confusion_matrix(y_test, sv_clf_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVC Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[67]:


# K Neighbors Classifier

kn_clf = KNeighborsClassifier(n_neighbors=6)
kn_clf.fit(x_train, y_train)
kn_pred = kn_clf.predict(x_test)
kn_acc = accuracy_score(y_test, kn_pred)
accuracy_list.append(100*kn_acc)


# In[68]:


print(Fore.GREEN + "Accuracy of K Neighbors Classifier is : ", "{:.2f}%".format(100* kn_acc))


# In[69]:


cm = confusion_matrix(y_test, kn_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("K Neighbors Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[70]:


# Decision Tree Classifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = accuracy_score(y_test, dt_pred)
accuracy_list.append(100*dt_acc)


# In[71]:


print(Fore.GREEN + "Accuracy of Decision Tree Classifier is : ", "{:.2f}%".format(100* dt_acc))


# In[72]:


cm = confusion_matrix(y_test, dt_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Decision Tree Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[73]:


# RandomForestClassifier

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = accuracy_score(y_test, r_pred)
accuracy_list.append(100*r_acc)


# In[74]:


print(Fore.GREEN + "Accuracy of Random Forest Classifier is : ", "{:.2f}%".format(100* r_acc))


# In[75]:


cm = confusion_matrix(y_test, r_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[76]:


# GradientBoostingClassifier

gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)
gradientboost_clf.fit(x_train,y_train)
gradientboost_pred = gradientboost_clf.predict(x_test)
gradientboost_acc = accuracy_score(y_test, gradientboost_pred)
accuracy_list.append(100*gradientboost_acc)


# In[77]:


print(Fore.GREEN + "Accuracy of Gradient Boosting is : ", "{:.2f}%".format(100* gradientboost_acc))


# In[78]:


cm = confusion_matrix(y_test, gradientboost_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Gredient Boosting Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[79]:


model_list = ['Logistic Regression', 'SVC','KNearestNeighbours', 'DecisionTree', 'RandomForest',
              'GradientBooster']


# In[80]:


plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[84]:


import pickle
# save the model to disk
filename = 'Heart_disease.sav'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




