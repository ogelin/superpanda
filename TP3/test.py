import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SoftmaxClassifier import SoftmaxClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt



# load dataset
data,target =load_iris().data,load_iris().target

# split data in train/test sets
X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.33, random_state=42)

# standardize columns using normal distribution
# fit on X_train and not on X_test to avoid Data Leakage
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

# import the custom classifier
cl = SoftmaxClassifier()

# train on X_train and not on X_test to avoid overfitting
train_p = cl.fit_predict(X_train,y_train)
test_p = cl.predict(X_test)

# display precision, recall and f1-score on train/test set
print("train : " + str(precision_recall_fscore_support(y_train, train_p, average = "macro")))
print("test : " + str(precision_recall_fscore_support(y_test, test_p, average = "macro")))
plt.plot(cl.losses_)
plt.show()