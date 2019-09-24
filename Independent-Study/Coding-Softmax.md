## Softmax

Equivalent of the sigmoid activation function but when the problem has three or more classes.

#### Multi-Class Classification & Softmax.


<img width="756" alt="Screen Shot 2019-09-23 at 11 23 42 PM" src="https://user-images.githubusercontent.com/46575719/65478843-3ee28900-de59-11e9-8e2d-5bb61d279cd6.png">



```python

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expElement = np.exp(L)
    summation = sum(expElement)
    result = []
    for j in expElement :
        result.append( j*1.0  / summation )
    return result

```
