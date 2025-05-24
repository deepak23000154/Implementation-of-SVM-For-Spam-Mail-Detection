# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2. Analyse the data. 

3. Use modelselection and Countvectorizer to preditct the values. 

4. Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: GAUTHAM KRISHNA S
RegisterNumber:  212223240036
*/
```
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>



## Output:

### data
![328855338-4a522776-209c-4329-932c-be8f8102c5ba](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/1cd23b36-148e-4c3d-9611-31f9a37c6440)


### data.shape()
![328855480-f8f741f4-3206-4526-92fd-c890f6ecb1e5](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/55503fe2-7a86-4c16-9a40-b9f9dc9c87c5)


### x.shape()
![328855694-2d7d8faa-ef77-405b-aedb-3009855bfeb9](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/0cb00b08-2fe8-40ec-a21f-1fbbcc989f5c)

### y.shape()  
![328855981-d3439f11-7e22-4ade-b5c6-917b3352cb8d](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/edd4a8a2-719b-4e6c-b7f0-fc3d91f21540)

### x_train
![328856257-67edd510-0d60-49f7-bde7-cd13ba895357](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/edc3b651-98f0-4a74-9d64-0396e9964eb3)


### x_train.shape()
![328856601-e9f7eb9a-89b8-4d67-b58c-29e8bd8668e0](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/9d207b3e-dcd6-48f4-bedc-ac8330836604)

### y_pred
![328856854-d98bdfde-aa7b-46a0-814a-020100201f28](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/0702e763-a932-48a3-b7c4-016980d2bf20)

### acc (accuracy)
![328857052-da3dd64d-b341-4b5b-824e-dd621396b816](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/bf25aae0-c1f5-4b70-a5b0-e1f87235e533)


### con (confusion matrix)
![328857298-43ae6fb1-8477-4118-abea-e8b2891123aa](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/96c8e632-5837-4d94-a02e-7db33ddfc123)


### cl (classification report)
![328857416-c1a9e002-dc90-4f21-bb0d-daf799640c92](https://github.com/gauthamkrishna7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/141175025/93d05231-8e93-4866-9d05-30f66368ce71)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
