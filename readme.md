

## Dataset Analysis
An initial look at the dataset suggests that it is a sparse dataset. That means each new entry in the dataset interacts with only a small number of features. It can also be safely assumed that since every new event deals with a limited number of features, it is not a time-series data and each row can be considered independent of each other.
It was also seen that the data does not follow a linear model and therefore, nonlinear numerical methods need to be employed to classify an event between the two classes.

### Dimensionality reduction
#### PCA
PCA suggests that most of the information in the data is contained in three reduced dimensional feature set. It is also seen that into the projected subspace, features with maximum variance are not linearly seperable and have too much overlap in between the two classes that is 0 and 1. Therefore, other non-linear techniques need to be applied to decide whether the dataset can be classified or not. 

### Normalization
Upon normalizing the dataset using mean and standard deviation of each column, it can be seen that the reduced feature set are much better seperated in between the two classes

#### LDA
With LDA, we see that a linear projection into a subspace is possible where the two classes are seperated with one class being centered around a mean of 2 and the other one around a mean of -2


## Approach
From initial investigation after dimensionality reduction on the normalized dataset, it is clear that a linearly seperable sub-space exists and therefore, different approach such as logistic regression, LDA or decision trees will be effective in classifying the dataset between the two classes. All three methods with 5 fold cross-validation and a train-test split of 70-30 show that over 70-80 percent accuracy can be achieved. Using these methods, it can also be told that some feature such as X52, X7, X53, X24 and X25 are the most important features available in the dataset. More complex algorithms such as neural networks need not be used for the classification problem at hand since much more interpretable machine learning methods are already providing high accuracy.

