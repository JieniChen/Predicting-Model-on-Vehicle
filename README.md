# Predicting-Model-on-Vehicle

Author: Jieni Chen

## This Model build up a predicting model based on the vehicle history data

Predictive modeling is the general concept of building a model that is capable of making predictions. Typically, such a model includes a machine learning algorithm that learns certain properties from a training dataset in order to make those predictions.
Predictive modeling can be divided further into two sub areas: Regression and pattern classification. Regression models are based on the analysis of relationships between variables and trends in order to make predictions about continuous variables, e.g., the prediction of the maximum temperature for the upcoming days in weather forecasting.
In contrast to regression models, the task of pattern classification is to assign discrete class labels to particular observations as outcomes of a prediction. To go back to the above example: A pattern classification task in weather forecasting could be the prediction of a sunny, rainy, or snowy day.

To not get lost in all possibilities, the main focus of this article will be on “pattern classification”, the general approach of assigning predefined class labels to particular instances in order to group them into discrete categories. The term “instance” is synonymous to “observation” or “sample” and describes an “object” that consists of one or multiple features (synonymous to “attributes”).

The goal is to predict the departure action and give the driver a warning



## Desicion Tree and K-Nearest Neighbors

### Desicion Tree
Decision tree learning is a method commonly used in data mining.[1] The goal is to create a model that predicts the value of a target variable based on several input variables. An example is shown in the diagram at right. Each interior node corresponds to one of the input variables; there are edges to children for each of the possible values of that input variable. Each leaf represents a value of the target variable given the values of the input variables represented by the path from the root to the leaf.

A decision tree is a simple representation for classifying examples. For this section, assume that all of the input features have finite discrete domains, and there is a single target feature called the classification. Each element of the domain of the classification is called a class. A decision tree or a classification tree is a tree in which each internal (non-leaf) node is labeled with an input feature. The arcs coming from a node labeled with an input feature are labeled with each of the possible values of the target or output feature or the arc leads to a subordinate decision node on a different input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes.
Left: A partitioned two-dimensional feature space. These partitions could not have resulted from recursive binary splitting. Middle: A partitioned two-dimensional feature space with partitions that did result from recursive binary splitting. Right: A tree corresponding to the partitioned feature space in the middle. Notice the convention that when the expression at the split is true, the tree follows the left branch. When the expression is false, the right branch is followed.

A tree can be "learned" by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. See the examples illustrated in the figure for spaces that have and have not been partitioned using recursive partitioning, or recursive binary splitting. The recursion is completed when the subset at a node has all the same value of the target variable, or when splitting no longer adds value to the predictions. This process of top-down induction of decision trees (TDIDT) is an example of a greedy algorithm, and it is by far the most common strategy for learning decision trees from data.

![alt text](https://github.com/JieniChen/Predicting-Model-on-Vehicle/blob/master/Images/desicionTree.png)


The Model Compare the Desicion tree and K-Nearest Neighboors Algorithm base on the presicion and recall



Also the Model generate the Precision and Recall Curve 




Ref:
https://en.wikipedia.org/wiki/Decision_tree_learning
http://sebastianraschka.com/Articles/2014_intro_supervised_learning.html#machine-learning-and-pattern-classification

