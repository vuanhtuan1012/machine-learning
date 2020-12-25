# Machine Learning

## Supervised Learning

### Problem

Given the training set <img src="https://render.githubusercontent.com/render/math?math=%28x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29_%7Bi%3D1%7D%5Em">
- *m* : number of training set
- <img src="https://render.githubusercontent.com/render/math?math=X%20%3D%20%5Cleft%20%5C%7B%20x%5E%7B%28i%29%7D%20%5Cright%20%5C%7D_%7Bi%3D1%7D%5Em"> : input values
- <img src="https://render.githubusercontent.com/render/math?math=Y%20%3D%20%5Cleft%20%5C%7B%20x%5E%7B%28i%29%7D%20%5Cright%20%5C%7D_%7Bi%3D1%7D%5Em"> : output values

### Objective

Find the function to predict <img src="https://render.githubusercontent.com/render/math?math=Y"> from <img src="https://render.githubusercontent.com/render/math?math=X"> so that <img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29"> is minimal.
- <img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29"> measures the difference between predicted values <img src="https://render.githubusercontent.com/render/math?math=%5Cwidehat%7BY%7D"> and <img src="https://render.githubusercontent.com/render/math?math=Y">.
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20Cost%28%5Cwidehat%7By%7D%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29">
</p>

- The formula of <img src="https://render.githubusercontent.com/render/math?math=Cost%28%5Cwidehat%7By%7D%2Cy%29"> depends on the type of problem.
    - Linear Regression:
    <p align="center"><img src="https://render.githubusercontent.com/render/math?math=Cost%28%5Cwidehat%7By%7D%2C%20y%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28%5Cwidehat%7By%7D-y%29%5E2"></p>

-
    - Logistic Regression:
    <p align="center"><img src="https://render.githubusercontent.com/render/math?math=Cost%28%5Cwidehat%7By%7D%2C%20y%29%20%3D%20-ylog%5Cwidehat%7By%7D%20-%20%281-y%29log%281%20-%20%5Cwidehat%7By%7D%29"></p>
- <img src="https://render.githubusercontent.com/render/math?math=%5Cwidehat%7By%7D%20%3D%20h_%5Ctheta%28x%29"> : predicted value.
- <img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29"> : hypothesis function. Its formula depends on the type of problem.
    - Linear Regression:
    <p align="center"><img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29%20%3D%20%5CTheta%5ETX"></p>

-
    - Logistic Regression:
    <p align="center"><img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-%5CTheta%5ETX%7D%7D"></p>

- <img src="https://render.githubusercontent.com/render/math?math=X"> : feature vector
- <img src="https://render.githubusercontent.com/render/math?math=%5CTheta"> : parameter vector
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=X%20%3D%20%5Cbegin%7Bbmatrix%7D%0Ax_0%5C%5C%20%0Ax_1%5C%5C%20%0A%5Cvdots%5C%5C%20%0Ax_n%0A%5Cend%7Bbmatrix%7D">
<img src="https://render.githubusercontent.com/render/math?math=%5CTheta%20%3D%20%5Cbegin%7Bbmatrix%7D%0A%5Ctheta_0%5C%5C%20%0A%5Ctheta_1%5C%5C%20%0A%5Cvdots%5C%5C%20%0A%5Ctheta_n%0A%5Cend%7Bbmatrix%7D"></p>

- <img src="https://render.githubusercontent.com/render/math?math=n"> : number of features.

### Gradient Descent

Gradient Descent is an algorithm to find <img src="https://render.githubusercontent.com/render/math?math=%5Ctheta"> so that <img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29"> is minimal.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20Cost%28%5Cwidehat%7By%7D%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29">
</p>

**Algorithm:**

repeat until <img src="https://render.githubusercontent.com/render/math?math=%7CJ%28%5Ctheta%29%5E%7B%28i%2B1%29%7D%20%20-%20J%28%5Ctheta%29%5E%7B%28i%29%7D%7C%20%5Cleq%20%5Cepsilon"> {

<img src="https://render.githubusercontent.com/render/math?math=%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta_j%7DJ%28%5Ctheta%29"> <img src="https://render.githubusercontent.com/render/math?math=%28j%20%3D%20%5Coverline%7B1%2C%20n%7D%29">

}  (simultaneously update <img src="https://render.githubusercontent.com/render/math?math=%5Ctheta_j">)

- <img src="https://render.githubusercontent.com/render/math?math=n"> : number of features
- <img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20"> : learning rate
- <img src="https://render.githubusercontent.com/render/math?math=%5Cepsilon%20"> : convergence condition


From the formula of <img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29">, they found the formula below is correct in both cases of linear and logistic regression.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_j%7DJ%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%28%5Cwidehat%7By%7D%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D">
</p>

Do **vectorization**, we have the algorithm gradient descent is:

repeat until <img src="https://render.githubusercontent.com/render/math?math=%7CJ%28%5Ctheta%29%5E%7B%28i%2B1%29%7D%20%20-%20J%28%5Ctheta%29%5E%7B%28i%29%7D%7C%20%5Cleq%20%5Cepsilon"> {

<img src="https://render.githubusercontent.com/render/math?math=%5CTheta%20%3A%3D%20%5CTheta%20-%20%5Cfrac%7B%5Calpha%20%7D%7Bm%7DX%5ET%28%5Cwidehat%7BY%7D-Y%29">

}

## Linear Regression

In this section, I'll talk about the programming exercise of week 2. The code source is rewritten in Python at [linear regression](linear%20regression.ipynb). If you want to refer to Matlab codes, they are in the diretory [w2](w2/).

In general, the workflow of two assignments (linear regression with one variable and linear regression with multiple variables) is presented in the figure below.

<p align="center">
<img src="images/linear_regression_workflow.svg">
</p>

It consists of four steps:
- Load Data: load data from text files, `ex1data1.txt` and `ex1data2.txt`.
- Define functions: define functions to predict outputs, to compute cost, to normalize features, and to carry out the algorithm gradient descent.
- Prepare Data: add column of ones to variables, do normalize features if needed.
- Training: initialize weights, learning rate, and number of iterations (called *epochs*) then lauch gradient descent.

Finally, I do one more step visualization to figure out the result obtained.

### Linear Regression with one variable

In this assignment, you need to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. They are presented in the figure below. A negative value for profit indicates a loss.

<p align="center">
<img src="images/linear_regression_restaurant.svg">
</p>