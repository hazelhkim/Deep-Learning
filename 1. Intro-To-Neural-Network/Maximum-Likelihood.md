## Maximum Likelihood
- Our quest is still for an algorithm that will help us pick the best model that separates our data. 
- Maximum Likelihood is that we pick the model that gives the existing labels the highest probability. Thus, by maximizing the probability, we can pick the best possible model.

<img width="514" alt="Screen Shot 2019-09-24 at 12 31 44 AM" src="https://user-images.githubusercontent.com/46575719/65481289-b79a1300-de62-11e9-87a0-4a84f6a3bdb8.png">

<img width="535" alt="Screen Shot 2019-09-24 at 12 32 19 AM" src="https://user-images.githubusercontent.com/46575719/65481371-0d6ebb00-de63-11e9-8453-4e4191a33442.png">

<img width="493" alt="Screen Shot 2019-09-24 at 12 33 48 AM" src="https://user-images.githubusercontent.com/46575719/65481374-0f387e80-de63-11e9-8de9-c67f9e6b7d4e.png">

<img width="524" alt="Screen Shot 2019-09-24 at 12 35 06 AM" src="https://user-images.githubusercontent.com/46575719/65481399-2d05e380-de63-11e9-95c7-57591a8d81e5.png">


#### Maximizing Probabilities

Could it be that maximizing the probability is equivalent to minimizing the error function? Maybe!

<img width="718" alt="Screen Shot 2019-09-24 at 12 56 37 AM" src="https://user-images.githubusercontent.com/46575719/65482410-0184f800-de67-11e9-8b71-5d699e400269.png">

We want to stay away from the products because it would lead the output which would be resulting from a number of inputs to a drastically tiny value.
Let's do sums instead of products. => logarithms!

#### Cross-Entropy
= That sums up negatives of logarithms of the probabilities.

![Screen Shot 2019-09-24 at 1 03 39 AM](https://user-images.githubusercontent.com/46575719/65482489-4446d000-de67-11e9-88f6-b704c2c886db.png)

- The points that are correctly classified have smaller values than those mis-classified. The reason for this is that a correctly classified point will have a probability that has close to 1, which when we take the negative of the logarithm, we'll get a small value.
- Thus, we can think of the negatives of these logarithms as errors at each point.
- Points that are correctly classified wlll have small errors and points that are mis-classified will have large errors. 
- And now we've concluded that our cross entropy will tell us if a model is good or bad.
- So now our goal has changed from maximizing a probability to minimizing a cross entropy to get a better model.
- The error function we were looking for is exactly the "cross entropy."

<br />

= a connection between probabilities and error functions.
  - How likely is it that those *events* happen based on the *probabilities*?
      - If it's very likely, then we have a small cross entropy.
      - If it's unlikely, then we have a large cross entropy.
  
```python
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y*np.log(P)+ (1-Y)*np.log(1-P))
```    
 
<img width="547" alt="Screen Shot 2019-09-24 at 1 35 12 AM" src="https://user-images.githubusercontent.com/46575719/65484834-12853780-de6e-11e9-8872-882eabb38d29.png">

<img width="530" alt="Screen Shot 2019-09-24 at 1 36 05 AM" src="https://user-images.githubusercontent.com/46575719/65484828-11540a80-de6e-11e9-8eeb-e48106f8d3d2.png">

