## Softmax

Equivalent of the sigmoid activation function but when the problem has three or more classes.

#### Multi-Class Classification & Softmax.

<img width="724" alt="Screen Shot 2019-09-23 at 11 26 29 PM" src="https://user-images.githubusercontent.com/46575719/65478945-9a147b80-de59-11e9-96b7-5a9a677c9a02.png">


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
