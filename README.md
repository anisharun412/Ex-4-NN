<H3>NAME : Arunsamy D</H3>
<H3>REGISTER NO : 212224240016</H3>
<H3>EX. NO.4</H3>

<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>
<H3>Aim:</H3>
To implement a Multilayer Perceptron for Multi classification
<H3>Theory</H3>

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)

<H3>Algorithm:</H3>

1. Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5. In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6. Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7. In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

<H3>Program:</H3> 

### Include packages and builtin classes
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

### Read the csv file to be considered for Multi-classification
```py
df = pd.read_csv("Iris_data.csv")

df.head()
```

### Preporcessing
```py
print("Before : ")
print(df.isnull().sum())

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
  df[col] = df[col].fillna(df[col].median())

print("\nAfter : ")
print(df.isnull().sum())
```
### Seperate the input features and target from the dataset
```py
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X.head())
print("\nClasses:\n")
print(y.unique())
     
```

### Transform the categorial into numerical values
```py
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

print("Encoded Classes:")
print(dict(zip(le.classes_, le.transform(le.classes_))))
```
### Split the data  for training  and testing
```py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(" X_train shape : ", X_train.shape)
print(" X_test  shape : ", X_test.shape)
     
 X_train shape :  (120, 4)
 X_test  shape :  (30, 4)
```
### Normalize the input features - Feature Scaling
```
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print(X_train[:5])
```

### Define the MLP classifier
```py
mlp = MLPClassifier(
    hidden_layer_sizes=(10,10),
    activation='relu',
    max_iter=1000,
    random_state=42
  )
``` 

### Train the classifier
```py
mlp.fit(X_train, y_train)
print("Training Completed")
```
### Evaluation of algorithm performance in classifying.
```py
y_pred = mlp.predict(X_test)
print("Predicted Output : ",y_pred)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy =", accuracy * 100,"%")

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix : \n",cm)

report = classification_report(
    y_test,
    y_pred
)

print("\n\nClassification Report :\n",report)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
```

<H3>Output:</H3>

### Read the csv file to be considered for Multi-classification
<img width="1384" height="230" alt="image" src="https://github.com/user-attachments/assets/797f5f66-9d31-4de2-becc-0d572b36eb1c" />

### Preporcessing

<img width="1022" height="309" alt="image" src="https://github.com/user-attachments/assets/f6d1009b-bb02-44df-8878-7751d8872727" />

### Seperate the input features and target from the dataset
<img width="940" height="219" alt="image" src="https://github.com/user-attachments/assets/48fa956a-67f6-4001-a437-1cee6788efc9" />

### Transform the categorial into numerical values
<img width="1023" height="66" alt="image" src="https://github.com/user-attachments/assets/8d400b25-e950-44e6-be00-e2bb453c1bc4" />

### Split the data  for training  and testing
<img width="780" height="64" alt="image" src="https://github.com/user-attachments/assets/05e0bbad-36cf-4089-aef9-db362ad9dc5f" />

### Normalize the input features - Feature Scaling
<img width="1041" height="132" alt="image" src="https://github.com/user-attachments/assets/cbb1e188-c33f-48a0-b9d8-b4f449c80b36" />

### Train the classifier 
<img width="781" height="145" alt="image" src="https://github.com/user-attachments/assets/a18b5572-0ded-4a20-9b35-5ed652ef4f3a" />

###  Evaluation of algorithm performance in classifying.
<img width="1275" height="427" alt="image" src="https://github.com/user-attachments/assets/96a5422b-2bc4-48f0-a0ba-d8ad1212010a" />

<img width="503" height="435" alt="download" src="https://github.com/user-attachments/assets/c81e4079-30c6-424e-894f-e8e0535a2997" />

<H3>Result:</H3>
Thus, MLP is implemented for multi-classification using python.
