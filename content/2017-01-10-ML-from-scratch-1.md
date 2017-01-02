Title: Machine Learning From Scratch - Pt. I
Slug: ml-from-scratch
Date: 2017-01-03
Category: Blog

## A self-lead refresher course in basic ML algorithms

I'm in the process of [implementing various machine learning algorithms from scratch](https://github.com/jarfa/ML_from_scratch) and will write a short series of blogs posts about this project. This first post will be about what motivated this project and a bit about what I've learned. Further posts will compare my algorithms' performance vs. that of the algorithms in [scikit-learn](http://scikit-learn.org/stable/), and will attempt to walk the reader through how these algorithms work.

For now the algorithms include:

*  Regression (logistic and least squares) via gradient descent

*  Decision Trees

*  Random Forests  

I'll be testing these algorithms on the [handwritten digits dataset](http://scikit-learn.org/stable/datasets/index.html#optical-recognition-of-handwritten-digits-data-set) that ships with scikit-learn. The code as written only does binary classification, so for each digit 0-9 I'll define one digit as the target and the rest as non-targets.

********** 
## Part I - Motivation

This project was inspired by a hackathon in the Spring of 2016 when I was working on the engineering team at Magnetic. At Magnetic I had been working on our machine learning systems that used [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit), and in this hackathon we implemented a similar logistic regression solver in the Go language - which we called Gowpal Wabbit (a large part of projects at Magnetic involved arguing about how to construct a witty/punny name, Gowpal Wabbit was not our most clever). My colleague Dan Crosta wrote a great blog post about [what he learned about logistic regression](https://late.am/post/2016/04/22/demystifying-logistic-regression.html) from the process.

While I had learned a lot about the most commonly used algorithms in grad school and at work, writing logistic regression from scratch and teaching a team of software engineers the math and intuition beyond the gradient descent solver made me think much harder about the various choices that go into writing a working implementation. It was a surprisingly educational experience.

Since then I've changed jobs, and after a traveling a lot in the Summer and Fall of 2016 I've found some free time again. I am (intermittently) writing some of the more commonly used ML algorithms from near-scratch and comparing their performance (both in terms of predictive power and computational efficiency) versus scikit-learn.

Note: these algorithms exist for my re-education and very little else. My algorithms will hopefully be just as good at prediction as scikit-learn's options, but theirs are more fully-featured and are much faster (since they're written in [Cython](http://cython.org/)). There is no reason anybody should be using my algorithms unless they find my code educational.

I had a few self-imposed rules for this project:

*  Along with writing the machine learning algorithms, rewrite any necessary utility functions - within reason. For example, I used numpy instead of creating my own linear algebra library, but wrote from scratch performance metrics (LogLoss, ROC AUC), feature normalization, etc.

*  Don't just look at scikit-learn's code and copy what they do, I should implement things my way. But my algorithms' variable and method names are similar to theirs to simplify my demonstration code, and in some cases I looked through their code after my code was already working.

* Make my code user-friendly - it should have a sane interface and readable code.

* Don't go crazy - I'm not going to rewrite my algorithms in Cython or make the code [PyPy](http://pypy.org/) compatible in order to match scikit-learn's speed. The value of this project to me is about making sure I know how to implement these algorithms efficiently, not about minimizing the actual time they require.

Forcing yourself to rewrite from scratch algorithms you think you already know is full of fun discoveries. Things I learned include:

*  You don't need to regularize the bias (a.k.a. intercept) term when fitting a regression model via gradient descent, especially when your input data is normalized (transformed so that the mean is 0 and the standard deviation is 1.) I learned this from colleagues of mine on the [FAIR](https://research.fb.com/category/facebook-ai-research-fair/) team - it's pretty great to work in the same building as Machine Learning luminaries.

*  Something that should have been obvious beforehand - in gradient descent your learning rate and regularization rate should depend greatly on the minibatch size.

*  When I worked on Magnetic's bidding models I used [Vowpal Wabbit (VW)](https://github.com/JohnLangford/vowpal_wabbit/wiki) extensively. In an attempt to mimic VW's behavior, my first implementation of regression SGD used a hash table (the Python dictionary) to store the input data. While this allowed a lot of flexibility when dealing with sparse or categorical data, it resulted in a massive speed hit - especially when regularizing the coefficients. I could've made the hash table work well had I pre-allocated a large array of coefficients and hashed features for the user like VW does. For now I just require the user to turn their data into an array of floats before training the model, but a feature hasher would be easy to implement.

*  Writing a fast decision tree algorithm is hard. My implementation is orders of magnitude slower than scikit-learn's, and it could stand to be looked over carefully and optimized. Perhaps I'll go back and do that before I implement new algorithms.

*  Writing a [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) implementation that runs in O(N^2) time is quite easy - but with a bit of thought, you can make it run in [O(N) time](https://github.com/jarfa/ML_from_scratch/blob/12d91ee4109410855b09aa7da9df345ae79e117d/util.py#L40).

*  [R's ggplot2](http://ggplot2.org/) package still has no peer among Python packages for graphics. For a while I've been doing most of my coding in Python, then moving my results to R for plotting with ggplot2. I wanted to use [Python's reimplmentation of ggplot](http://ggplot.yhathq.com/) in this project to chart algorithm performance, but it's missing many crucial features and cannot yet replace R's version. [Seaborn](http://seaborn.pydata.org/) is similar to ggplot2 in many ways, but is not quite as flexible. And while [matplotlib](http://matplotlib.org/) is incredibly flexible, the the brevity and power of R's ggplot2 is far more appealing to me.

<!-- *  Gradient boosted trees sound so simple to implement - just iteratively build trees on the gradients! But there is apparently some some subtlety. My code for this algorithm will go on GitHub once I figure out what I'm doing wrong. -->
