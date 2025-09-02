# Ex-02 Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Load the Iris dataset using a suitable library.

### STEP 2:
Preprocess the data by handling missing values and normalizing features.

### STEP 3: 
Split the dataset into training and testing sets.

### STEP 4: 
Train a classification model using the training data.

### STEP 5: 
Evaluate the model on the test data and calculate accuracy.

### STEP 6: 
Display the test accuracy, confusion matrix, and classification report.



### Name:PAVITHRA S

### Register Number: 212223230147

## PROGRAM

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
df['target']=y
df.tail()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)
train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)
train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
test_loader=DataLoader(test_data,batch_size=16)
class IrisClassifier(nn.Module):
  def __init__(self,input_size):
    super(IrisClassifier,self).__init__()
    self.fc1=nn.Linear(input_size,16)
    self.fc2=nn.Linear(16,8)
    self.fc3=nn.Linear(8,3)

  def forward(self,x):
    x= F.relu(self.fc1(x))
    x= F.relu(self.fc2(x))
    return self.fc3(x)
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
    if(epoch+1)%10==0:
      print(f'Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}')
model=IrisClassifier(input_size=X_train.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
train_model(model,train_loader,criterion,optimizer,epochs=100)
model.eval()
predictions,actuals=[],[]
with torch.no_grad():
  for X_batch,y_batch in test_loader:
    outputs=model(X_batch)
    _,predicted=torch.max(outputs,1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())
accuracy=accuracy_score(actuals,predictions)
conf_matrix=confusion_matrix(actuals,predictions)
class_report=classification_report(actuals,predictions,target_names=iris.target_names)
print(f'Test Accuracy: {accuracy:.2f}\n')
print("Classification Report:\n",class_report)
print("Confusion Matrix:\n",conf_matrix)
sns.heatmap(conf_matrix,annot=True,cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
sample_input=X_test[5].unsqueeze(0)
with torch.no_grad():
  output=model(sample_input)
  predicted_class_index=torch.argmax(output[0]).item()
  predicted_class_label=iris.target_names[predicted_class_index]
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')
```

### Dataset Information
<img width="945" height="246" alt="image" src="https://github.com/user-attachments/assets/fc035259-2b07-4b14-add9-c002229c22cd" />
<img width="944" height="237" alt="image" src="https://github.com/user-attachments/assets/5060f994-1634-4357-badf-5578cab3e3a1" />
<img width="377" height="226" alt="image" src="https://github.com/user-attachments/assets/70a4eb4d-5de2-410a-975e-511959be6e98" />


### OUTPUT

## Confusion Matrix

<img width="723" height="662" alt="image" src="https://github.com/user-attachments/assets/d1f7f2f0-3a86-4158-ab8e-b15042f757f3" />


## Classification Report
<img width="616" height="288" alt="image" src="https://github.com/user-attachments/assets/8e406113-dbc9-4136-a68e-81ebeeab638a" />


### New Sample Data Prediction
<img width="405" height="43" alt="image" src="https://github.com/user-attachments/assets/6fb7293d-2e36-4933-a08b-f521ed654f47" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch
