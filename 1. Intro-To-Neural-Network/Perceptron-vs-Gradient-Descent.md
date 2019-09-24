# Perceptron vs. Gradient Descent

#### Gradient Discent Algorithm
We take the weights and all of points change.

#### Perceptron Algorithm
Not every point changes weights, but only the misclassified ones. 

<img width="678" alt="Screen Shot 2019-09-24 at 12 18 12 PM" src="https://user-images.githubusercontent.com/46575719/65530165-725afd00-dec5-11e9-83f9-ffceb5dd667c.png">


- You'll realize that teh right and the left are exactly the same thing.
- The only difference is that in the left, y-hat can take any number between zero and one,
whereas in the right, y-hat can take only the values zero or one.

#### Let's study Gradient Descent even more carefully.

- Both in the Perceptron algorithm and the Gradient Descent Algorithm, a point that is misclassified tells a line to come closer because eventually, it wants the line to surpass it
so it can be in the correct side.
- But if a point is correctly classified, the Perceptron algorithm says do absolutely nothing. 
In the Gradient Descent algorithm, you are changing the weights. (-> What it does is that the point is telling the line to go father away.)
  - Because if you're correctly classified -- if you're a blue point in the blue region --, you'd like to be even more into the blue region, so your prediction is even closer to one and your error is even smaller.
  - Similarly for a red point in the red region.
- The Gradient Descent: the misclassified points ask the line to come closer and the correctly classified points ask the line to go farther away. The line listens to all the points and takes steps in such a way that it eventually arrives to a pretty good solution.

## Perceptron ( Continuous Re-cap )

<img width="575" alt="Screen Shot 2019-09-24 at 5 05 52 PM" src="https://user-images.githubusercontent.com/46575719/65550483-a9dd9f80-deed-11e9-9d2f-48d4edd0e9d5.png">

<img width="593" alt="Screen Shot 2019-09-24 at 5 06 10 PM" src="https://user-images.githubusercontent.com/46575719/65550482-a9dd9f80-deed-11e9-9c14-eddd6355b388.png">
