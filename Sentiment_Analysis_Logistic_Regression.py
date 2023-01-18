# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
import math
from nrclex import NRCLex 
import re
from nltk.tokenize import word_tokenize
import nltk
from itertools import chain
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.stats import loguniform
from sklearn import svm

nltk.download()

# Goal: predict the number of stars in the review from the data.\
# We will be using sentiment analysis to see whether we can predict the number of stars from the text review.
# Dataset can be downloaded from: https://www.kaggle.com/datasets/whenamancodes/amazon-reviews-on-women-dresses?resource=download

# Pre-processing
# ratings_Books.csv can be found in the GitHub
ratings = pd.read_csv("WomensDressReviews.csv", delimiter=',', header = 0)
ratings.head()

ratings = ratings[['review_text', 'rating']]
ratings.head()

# We will now clean our reviews by reducing them down to the key words that convey the sentiment of the user. First, we remove puncutation and reduce the review down to its core words. 
# We do this using the $re.sub$ package that uses regular expression operations to remove puncutation and leave just the text.

def remove_punct(review):
    review = re.sub('[^A-Za-z]+', ' ', str(review))
    return review

ratings['Cleaned Reviews'] = ratings['review_text'].apply(remove_punct)

ratings.head()

# Now we will tokenize our data by breaking it down into categories. Since we are trying to extract sentiments from the reviews, we will be doing work tokeniation using the function $\textit{word_tokenize}$ from the $\textit{nltk tokenize}$ package.

tokenized_review = []
for i in range(len(ratings['Cleaned Reviews'])):
     tokenized_review.append(word_tokenize(ratings['Cleaned Reviews'][i]))

ratings['tokenized_review'] = tokenized_review
ratings.head()

# Now we find the top emotions from each review based on the sentiment analysis from the word tokenization we did for each review. Based on these top emotions, we can create numerical training data for our model and then optimize our hyperparameters to increase model accuracy as much as possible.

ratings['tokenized_review'] = tokenized_review
clean = ratings['Cleaned Reviews']
emotion = [0]*len(clean)
for i in range(len(clean)):
        emotion[i] = NRCLex(clean[i]).top_emotions
    
ratings['emotion'] = emotion
ratings.head()

# For each review, we add the top emotion and the probability of the emotion occurring in the review based on our sentiment analysis.

pos = 0
core = []
prob = []
for i in range(ratings.shape[0]):
        if((ratings['emotion'][i][0][0]) == 'positive'): 
            core.append('positive')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'negative'):
            core.append('negative')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'joy'):
            core.append('joy')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'trust'):
            core.append('trust')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'fear'):
            core.append('fear')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'anger'):
            core.append('anger')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'sadness'):
            core.append('sadness')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'disgust'):
            core.append('disgust')
            prob.append(ratings['emotion'][i][0][1])
        elif(ratings['emotion'][i][0][0] == 'surprise'):
            core.append('surprise')
            prob.append(ratings['emotion'][i][0][1])
        else:
            core.append(0)
            prob.append(ratings['emotion'][i][0][1])

ratings['core'] = core
ratings['prob'] = prob

ratings.head()

# Now we begin fitting our model. We being by split the data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(ratings['prob'], ratings['rating'], random_state=42)

y_train = y_train - 1
y_test = y_test - 1

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

# We standarize both our training and testing datasets to have a standard deviation of 1.

scaler = StandardScaler().fit(X_train.reshape(-1,1))
X_train = scaler.transform(X_train)
scaler = StandardScaler().fit(X_test.reshape(-1,1))
X_test = scaler.transform(X_test)

for i in range(X_train.shape[1]):
    print(np.std(X_train[:,i]))
for i in range(X_test.shape[1]):
    print(np.std(X_test[:,i]))

# We use the $\text{RandomizedSearchCV}$ function and the $\text{RepeatedStratifedKFold}$ function in sklearn to conduct a randomized search on hyperparameters and then conduct k-fold cross validation with 5 folds. This is to ensure we find the best set of hyperparameters to optimize our logistic regression model, which is also from $\text{sklearn}$.

%%capture --no-stderr
# Use k-fold cross validation with 5 folds
model = LogisticRegression(multi_class = 'multinomial')
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3,random_state=1)
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5,100)
search = RandomizedSearchCV(model, space, n_iter=500, scoring = 'accuracy', n_jobs=-1, cv=cv, random_state=1)
result = search.fit(X_train, np.squeeze(y_train))
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

result.best_params_

model = LogisticRegression(C = result.best_params_['C'], penalty = result.best_params_['penalty'], solver = result.best_params_['solver'], max_iter = 500).fit(X_train, np.squeeze(y_train))
score = model.score(X_test, np.squeeze(y_test))
print('The accuracy of the model on the test data set is ' + str(score) + ' or ' + str(round(score*100,2)) + '%.')

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = clf.score(X_test, np.squeeze(y_test))
print('The accuracy of the model on the test data set is ' + str(score) + ' or ' + str(round(score*100,2)) + '%.')

plt.hist(ratings['rating']) # skewed right, lots of five-star reviews, little one-star reviews
plt.ylabel('Value Counts')
plt.xlabel('Star Rating (from 1-5)')
plt.title('Histogram of Amazon Dataset Ratings')

# Upon looking at the distribution of star ratings, this explains why support vector machine performs poorly. We have a highly unnbalanced dataset, which is a large weakness of support vector machines. With this type of model, because the soft margin is weak, the decision hyperplanes become skewed towards the minority class when the model is training.

# Some ways to mitigate this is to balance the dataset by resampling so that there is an even distribution of ratings or to use class weights that reflect the proportion of each rating in the training dataset.

# This was an implementation of sentiment analysis, logistic regression on a dataset to classify text reviews into star ratings, hyperparameter optimization, and some comparison of different classificaation models from a publicily available dataset at: https://www.kaggle.com/datasets/whenamancodes/amazon-reviews-on-women-dresses?resource=download.

