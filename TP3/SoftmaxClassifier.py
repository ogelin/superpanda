from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """A softmax classifier"""

    def __init__(self, lr=0.05, alpha=100, n_epochs=1000, eps=1.0e-5, threshold=1.0e-10, regularization=False,
                 early_stopping=True):

        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping

    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """

    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """

    def fit(self, X, y=None):

        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))

        X_bias = np.ones((np.shape(X)[0], np.shape(X)[1] + 1))
        X_bias[:, :-1] = X

        self.theta_ = np.random.rand(X.shape[1] + 1, self.nb_classes)

        i = 0

        for epoch in range(self.n_epochs):
            i += 1

            logits = np.dot(X_bias, self.theta_)
            probabilities = np.clip(self._softmax(logits), self.eps, 1 - self.eps)

            loss = self._cost_function(probabilities, y)
            self.theta_ = self.theta_ - self.lr * self._get_gradient(X_bias, y, probabilities)

            self.losses_.append(loss)
            #print(loss)

            if self.early_stopping:
                if np.abs(loss-prev_loss) < self.threshold:
                    print("early stop after ", i, " epochs")
                    return self

            prev_loss = loss

        return self

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilitides
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        X_bias = np.ones((np.shape(X)[0], np.shape(X)[1] + 1))
        X_bias[:, :-1] = X

        logits = np.dot(X_bias, self.theta_)
        probs = self._softmax(logits)

        return probs

        """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """

    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        probs = self.predict_proba(X)
        res = np.argmax(probs, axis=1)
        return res


    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)

    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """

    def score(self, X, y=None):
        probabilities = self.predict(X , y)
        self.regularization = False

        loss = self._cost_function(probabilities, y)

        return loss

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax

        Do:
        One-hot encode y, vector with correct classes for each example
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out:
        Probabilities
    """

    def _cost_function(self, probabilities, y):
        y_one_hot = self._one_hot(y)
        probs = np.clip(probabilities, self.eps, 1 - self.eps)

        l2 = 0

        if self.regularization:
            l2 = self._calculate_regularization()

        double_sum = -np.sum(np.sum(y_one_hot * np.log(probs), axis=1))

        log_loss = (double_sum + l2) / y.shape[0]

        return log_loss

    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    def _one_hot(self, y):
        number_of_classes = len(np.unique(y))
        number_of_instances = len(y)
        # initialize empty one hot matrix with 0s
        one_hot_matrix = [[0 for col in range(number_of_classes)] for row in range(number_of_instances)]

        for i in range(0, len(y)):
            one_hot_matrix[i][y[i]] = 1

        return one_hot_matrix

    """
        In :
        Logits: (self.nb_features +1) * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """

    def _softmax(self, z):

        e_y = np.exp(z - np.max(z))
        res = e_y / np.reshape( e_y.sum(axis=1), (e_y.shape[0], 1))

        return res

    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """

    def _get_gradient(self, X, y, probas):
        y_one_hot = self._one_hot(y)

        delta = probas - y_one_hot

        grad = np.dot(X.T, delta) / y.shape[0]

        if self.regularization:
            grad += self._calculate_regularization_derivative() / y.shape[0]

        return grad

    """
    l2 = alpha * sum(sum( w^2))
    """
    def _calculate_regularization(self):
        sq = np.square(self._theta_without_bias())
        l2 = np.sum(sq)
        l2 = self.alpha * l2
        return l2

    """
    l2' = alpha * 2 * w
    """
    def _calculate_regularization_derivative(self):
        l2_derivative = self.alpha*2*self._theta_without_bias()
        return l2_derivative

    """
    Return theta matrix with last row replaced with 0s
    """
    def _theta_without_bias(self):
        theta_prime = self.theta_
        theta_prime[-1, :] = 0
        return theta_prime
