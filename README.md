# Fundamentals-of-data-science

Kaggle competition "Sold! How do home features add up to its price tag?", whose goal is to predict house prices using a set of 79
features.

At first, I did details exploration of data analysing the variables in order to see and remove eventually outliers.
For discover outliers I used a python’s package named seaborn, I confronted all the variables, principally
features that refer to square feet areas , with the SalePrice (target variable) and decided, one by one, where
fix the threshold beyond which delete the rows of the training data. There are also outliers on test set but
obviously, I could not drop these. Successively I log transformed the target variable allowing that the variables
are more normally distributed and this improve linear regression. I started definitely the project combining
the test and training set along the rows creating a new dataset with 2867 rows and 79 columns. I ignored
Utilities because all records are “AllPub” except for one “NoSeWa” in the train set while in the test set there
are 2 NA. Then I worked the numerical and categorical feature in separate way. I created, for a clearly
comprehension, a new data frame called “num_lab” where I transpose every numeric variable, this enabled
me to change singularly them if necessary. During this step, I added many new columns for a better prediction
supported by an improvement score in the leaderboard on kaggle. Studying the description of the numerical
variables, I thought to transform some of these features to categorical ones, leaving anyway these in the
numerical dataset. I putted always the median for replace the NA because the median suffers less than the
average the outliers, in fact; I see an improvement of the score in the leaderboard. For the categorical half I
have done roughly speaking the same think, I paid particular attention for some variables with many
categories - like “Neighborhood” and “MSSubClass” - creating a new type of these in according to the social
class, which were determined, arbitrarily, by comparing each categories with the target variable ‘SalePrice’.
All the ratings features are transformed in numerical in a range from 0 to 5, where a higher number means
higher quality. Some variables, instead, have been converted to binary variables maybe because there were
many values of one single categories of a feature and few in many other. I filled the NA, in categorical case,
conforming to ‘data_description’ of the competitions, how indicated; Where not indicated I filled with mode
if there were few missing values and with “Not Found” if the missing value are many more. Then I searched
all the variables with a skewness greater than 0.75 and I log transformed them for the above-mentioned
reasons, finally I scaled the data. Before the prediction, I dropped all the columns present in the test but not
in the training set and vice versa. To choose which model to use I tried three different model Linear, Ridge
and Lasso Regression. I discarded the linear regression because through the cross validation I saw that the
model goes on overfitting. Given that the dataset contained many variables, I thought to use the other two
method to alleviate multicollinearity and I definitely choose the Ridge regression in according to an
improvement of the score. Through a package named “RidgeCV” I founded the best alpha (19.5393). The
coefficient of determination R^2 of my prediction is 0.9426 and I reached a score of 0.11661
