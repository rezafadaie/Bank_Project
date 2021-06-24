# **Bank Project**

## **Data Preprocessing**

###Import Libraries
"""

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

"""### Import the Dataset"""

dataset = pd.read_csv('bank-full.csv', delimiter = ";" )
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

"""## **Building ANN**

### Initializing the ANN
"""

ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

"""### Adding the second hidden layer"""

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

"""### Adding the output layer"""

ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

"""## **Training the ANN**

### Compiling the ANN
"""

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the ANN on the Training set"""

ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

"""## **Predicting the Test set results**"""

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## **Making the Confusion Matrix**"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
