# Gradient Descent

<img width="634" alt="Screen Shot 2019-09-24 at 9 23 52 AM" src="https://user-images.githubusercontent.com/46575719/65515302-0a98b800-dead-11e9-8e12-8023f3d3ad98.png">
 
 Now we learned that in order to minimize the error function, we need to take some derivatives. 
 So let's get our hands dirty and actually compute the derivative of the error function. 
 The first thing to notice is that the sigmoid function has a really nice derivative.

<img width="798" alt="Screen Shot 2019-09-24 at 9 25 35 AM" src="https://user-images.githubusercontent.com/46575719/65515577-86930000-dead-11e9-9841-68e2a82323a6.png">

<img width="791" alt="Screen Shot 2019-09-24 at 9 27 03 AM" src="https://user-images.githubusercontent.com/46575719/65515581-86930000-dead-11e9-97ac-934a0cb09813.png">

<img width="800" alt="Screen Shot 2019-09-24 at 9 26 23 AM" src="https://user-images.githubusercontent.com/46575719/65515580-86930000-dead-11e9-8101-e039a9c83a6f.png">

<img width="802" alt="Screen Shot 2019-09-24 at 9 26 43 AM" src="https://user-images.githubusercontent.com/46575719/65515579-86930000-dead-11e9-90c0-8b6d0ce2cbaf.png">


### Logistic Regression Algorithm

<img width="780" alt="Screen Shot 2019-09-24 at 9 44 58 AM" src="https://user-images.githubusercontent.com/46575719/65516989-028e4780-deb0-11e9-83ef-cd2f6233e686.png">

### Implementing the Gradient Discent Algorithm

```python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Some helper functions for plotting and drawing lines

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

```

#### Reading an Plotting the Data

```python

data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
plot_points(X,y)
plt.show()

```


#### Basic Functions for implementing Gradient Descent

<img width="819" alt="Screen Shot 2019-09-24 at 9 55 38 AM" src="https://user-images.githubusercontent.com/46575719/65517955-8694ff00-deb1-11e9-9592-af72e976c721.png">


#### My Original Answer

```python

# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):
    return 1/ (1+ np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(weights*features + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return -y*np.log(output) - (1-y)*np.log(1-output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    return weights + learnrate*(y - output_formula(x, weights, bias))*x, bias - learnrate*(y - output_formula(x, weights, bias))
```

#### Actual Solution:

```python

# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias

```
