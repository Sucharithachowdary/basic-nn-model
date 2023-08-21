# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
Developed By: Sucharitha . K
Reference Number : 212221240021
~~~
### Importing Required Packages :
~~~
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
~~~
### Authentication and Creating DataFrame From DataSheet :
~~~
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
~~~
### Assigning X and Y values :
~~~
X = df[['INPUT']].values
Y = df[['OUTPUT']].values
~~~
### Normalizing the data :
~~~
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
~~~
### Creating and Training the model :
~~~
model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=2200)
~~~
### Plot the loss :
~~~
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
~~~
### Evaluate the Model :
~~~
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)
~~~
### Prediction for a value :
~~~
X_n1 = [[20]]
X_n1_1 value = Scaler.transform(X_n1)
model.predict(X_n1_1 value)
~~~

## Dataset Information
![262069281-1d023f8c-2198-4abc-9c59-d978eed0f0dd](https://github.com/Sucharithachowdary/basic-nn-model/assets/94166007/b1c75067-9cbc-49a6-a1b4-7da3afb20b0f)


## OUTPUT

### Training Loss Vs Iteration Plot

![262072687-2b673fa3-8798-46a9-96e5-a909ee93b891](https://github.com/Sucharithachowdary/basic-nn-model/assets/94166007/4aef49fc-3529-49e5-bade-a790cfef5fa2)


### Test Data Root Mean Squared Error
![262072465-43e38082-b4c1-4e4c-b48a-003e3e86a077](https://github.com/Sucharithachowdary/basic-nn-model/assets/94166007/c917b268-7f5f-4c27-b81c-86125d3fb059)


### New Sample Data Prediction
![262072825-be42d87e-ede4-4077-848e-03f61e176a12](https://github.com/Sucharithachowdary/basic-nn-model/assets/94166007/bc87b5ff-1f26-4d7e-ad38-338a8c4071b7)


## RESULT
Thus the neural network regression model for the given dataset is executed successfully.
