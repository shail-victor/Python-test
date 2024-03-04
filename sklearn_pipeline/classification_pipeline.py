
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


data = pd.read_csv('../input/Iris.csv')
data.head()
data.info()
data.drop('Id',axis=1,inplace=True)


sns.pairplot(data, hue='Species', size=3)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data['Species'] = LabelEncoder().fit_transform(data['Species'])
data.iloc[[0,1,-2,-1],:]


pipeline = Pipeline([
    ('normalizer', StandardScaler()), #Step1 - normalize data
    ('clf', LogisticRegression()) #step2 - classifier
])
pipeline.steps


#Seperate train and test data
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1].values,
                                                   data['Species'],
                                                   test_size = 0.4,
                                                   random_state = 10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



from sklearn.model_selection import cross_validate

scores = cross_validate(pipeline, X_train, y_train)
scores

scores['test_score'].mean()


