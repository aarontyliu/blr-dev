import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from optim import hmc, sghmc


class BayesianLogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Bayesian Logistic Regression.
    """
    def __init__(
        self,
        penalty="l2",
        tol=1e-4,
        fit_intercept=True,
        class_weight=None,
        random_state=None,
        solver="hmc",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        n_jobs=None,
        batch_size=64,
        available_solvers=("hmc", "sghmc"),
    ):

        self.penalty = penalty
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.available_solvers = available_solvers

    def fit(self, X, y, M="I"):

        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if self.fit_intercept:
            self.X_ = np.insert(self.X_, 0, 1, 1)

        if M == "I":
            self.M = np.eye(self.X_.shape[1])

        assert (
            self.solver in self.available_solvers
        ), f"Please use available solvers: {available_solvers}"
        if self.solver == "hmc":
            self.parameters_ = hmc(
                self.X_,
                self.y_,
                self.M,
                max_iter=self.max_iter,
                m=400,
                eps=1e-4,
                verbose=self.verbose,
            )
        elif self.solver == "sghmc":
            self.parameters_ = sghmc(
                self.X_,
                self.y_,
                self.M,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                m=100,
                eps=1e-3,
                verbose=self.verbose,
            )
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.fit_intercept:
            X_ = np.insert(X, 0, 1, 1)
        y_prob = self.sigmoid(X_ @ self.parameters_).flatten()

        return y_prob

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    def predict_log_prob(self, X):
        return np.log(self.predict_proba(X))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
