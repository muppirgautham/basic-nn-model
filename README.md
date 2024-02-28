# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/muppirgautham/basic-nn-model/assets/94810884/020b6a5b-967b-4351-9833-1b2d6e87b89a)

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
### Name: M Gautham
### Register Number: 212221230027
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('exp1dl').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'output':'float'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('exp1dl').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()


X = dataset1[['Input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)
ex1_model = Sequential([
    Dense(units = 3, activation = 'relu', input_shape=[1]),
    Dense(units = 2, activation = 'relu'),
    Dense(units = 1)
    ])
ex1_model.compile(optimizer = 'rmsprop', loss = 'mse')
ex1_model.fit(X_train1,y_train,epochs = 5000)
ex1_model.summary()
loss_df = pd.DataFrame(ex1_model.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

ex1_model.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

ex1_model.predict(X_n1_1)

```
## Dataset Information

![image](https://github.com/muppirgautham/basic-nn-model/assets/94810884/b85f615f-8757-4678-ac8f-93e3a4546fe6)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/muppirgautham/basic-nn-model/assets/94810884/275e985c-4971-48cc-93e7-6fa26e6fc183)

### Test Data Root Mean Squared Error
![image](https://github.com/muppirgautham/basic-nn-model/assets/94810884/0f875435-c0e8-4ccf-917f-d7922647e06b)


### New Sample Data Prediction
![image](https://github.com/muppirgautham/basic-nn-model/assets/94810884/c9763ba6-0c23-47ad-abc2-c89e8298fcae)


## RESULT

Thus, The Process of developing a neural network regression model for the created dataset is successfully executed.


