
# **Bank Project**

## **Data Preprocessing**

###Import Libraries
"""

import numpy as np
import pandas as pd

"""### Import the Dataset"""

dataset = pd.read_csv('bank.csv', delimiter = ";" )
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(x)

print(y)

"""### Encoding categorical data"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1,2,3,4,6,7,8,10,15])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

print(x)

"""### Encoding the dependent variable"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

"""### Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)

print(x_test)

"""## **Training the Decision Tree Classification model on the training set**"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train,y_train)

"""## **Predicting the test set result**"""

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## **Making the Confusion Matrix**"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
