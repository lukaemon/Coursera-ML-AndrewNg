# Coursera ML MOOC
Andrew's class may be the common sense among ML practitioners.  

I don't want to fool myself.  
Even I have read some api doc of [sklearn](http://scikit-learn.org/stable/modules/classes.html) and know how to call them, I don't know the soul of machine learning. I have to get the basics right. So I implement every exercise of the [Coursera ML class](https://www.coursera.org/learn/machine-learning/home/welcome) using numpy, scipy and tensorflow.  

The reason I choose python over matlab is purely practical concern. This cs224d [Intro to TensorFlow](http://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf) ([video](https://www.youtube.com/watch?v=L8Y2_Cq2X5s&index=7&list=PLmImxx8Char9Ig0ZHSyTqGsdhb9weEGam)) presents very good explanation of why python may be the right choice to do ML.  

All these learning about theories and coding are preparation of real world application. Although the learning itself is it's own reward, I also want to create useful application that solves real world problems and create values to the community. This project is the very tiny step toward the goal. I learned so much.  

The more I learn, the more I respect all those great scientific adventures before me that paves the way I have right now. Andrew's class is very good overview of general ML. It's hands on approach  encourages new people like me keep moving, even some details are purposefully ignored. On the other hand, I found it very useful to pick up theories while doing these exercises. This book [Learning from Data](http://amlbook.com/) gives me so many aha moment about learning theories. This is my feeble foundation of ML theories.

Generally, Andrew's class shows me mostly **what** to do, and **how** to do it. The book shows me **why**. Theory and practice goes hand in hand. I couldn't express how happy I am when I read something in the book and suddenly understand the reason about what I was coding last night. Eureka!

## Project structure
* Each exercise has it's own folder. In each folder you will find:
  1. pdf that guide you through the project
  2. a series of Jupyter notebook
  3. data
* each notebook basically follows the logic flow of project pdf. I didn't present all codes in notebook because I personally think it's very messy. So you will only see visualization, project logic flows, simple experiments, equations and results in notebooks.
* In [helper](https://github.com/icrtiou/coursera-ML/tree/master/helper) folder, it has modules of different topics. This is where you can find details of model implementation, learning algorithm, and supporting functions.

## Go solo with python or go with built-in Matlab project?
The Matlab project is guiding students to finish the overall project goal, be it implementing logistic regression, or backprop NN. It includes many supporting function to help you do `visualization`,  `gradient checking`, and so on.  
The way I do it is to focus on pdf that tells you what is this project about, then figure out how to achieve those objectives using `Scipy` stack. Most of time I don't even bother looking into original `.m` files. Just need their data. 

Without those supports, I have to do:

1. **visualization** : `seaborn`, `matplotlib` are very handy  
2. **vetorized implementation** of ML model and gradient function use `numpy`'s power to manupulate `ndarray`  
3. **optimization** : figure out how to use `scipy` optimizer to fit you parameters  
4. **support functions** : nobody is loading, parsing, normalize data for you now, DIY  

By doing those, I learn more, which is even better.

## Supporting materials
I am learning by doing, not tools hoarding. Here is the list that helps me along the way.  
* Intuitions of Linear Algebra, [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), this is the best source to my knowledge, for intuition.
* [Python, numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
* More math behind the scene. [CS 229 Machine Learning Course Materials](http://cs229.stanford.edu/materials.html), basically Coursera ML is water down version of this cs229. The link has very good linear algebra review ,and probability theroy review. 
* [Quoc Le’s Lectures on Deep Learning](http://www.trivedigaurav.com/blog/quoc-les-lectures-on-deep-learning/): videos with perfect lecture notes. 
* [Learning from Data](http://amlbook.com/): learning theory in less than 300 pages, God.

# You can read all Jupyter notebooks here: [nbviewer](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/)

> * acknowledgement: Thank you [John Wittenauer](https://github.com/jdwittenauer?tab=overview&from=2016-08-01&to=2016-08-31&utf8=%E2%9C%93)! I shamelessly steal lots of your code and idea. [here](https://github.com/jdwittenauer/ipython-notebooks)    
> * if you want to run notebooks locally, you could refer to requirement.txt for libraries I've been using.  
> tensorflow is a little bit tricky to install. you could find the instructions [here](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html).  
> * I'm using `python 3.5.2` for those notebooks. You will need it because I use `@` operator for matrix multiplication extensively.  

### [ex1-linear regression](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex1-linear%20regression/)
Special thing I did in this project is I implement the linear regression model in [TensorFlow](https://www.tensorflow.org/). This is my first tf experience. Looking forward to learn more when I move into Deep Learning. code: [linear_regression.py](https://github.com/icrtiou/coursera-ML/blob/master/helper/linear_regression.py)
### [ex2-logistic regression](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex2-logistic%20regression/)
### [ex3-neural network](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex3-neural%20network/)
### [ex4-NN back propagation](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex4-NN%20back%20propagation/)
### [ex5-bias vs variance](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex5-bias%20vs%20variance/)
### [ex6-SVM](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex6-SVM/)
### [ex7-kmeans and PCA](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex7-kmeans%20and%20PCA/)
### [ex8-anomaly detection and recommendation](http://nbviewer.jupyter.org/github/icrtiou/coursera-ML/tree/master/ex8-anomaly%20detection%20and%20recommendation/)
