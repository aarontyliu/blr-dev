import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from blr_sghmc.optim import hmc, sghmc, original_sghmc


class BayesianLogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Bayesian Logistic Regression.
    """

    def __init__(
        self,
        fit_intercept=True,
        solver="sghmc",
        max_iter=100,
        verbose=0,
        batch_size=64,
    ):

        self.fit_intercept = fit_intercept
        self.available_solvers = {
            "hmc": hmc,
            "sghmc": sghmc,
            "original_sghmc": original_sghmc,
        }
        assert (
            solver in self.available_solvers
        ), f"Please use available solvers: {available_solvers.keys()}"
        self.solver = self.available_solvers[solver]
        self.max_iter = max_iter
        self.verbose = verbose
        self.batch_size = batch_size

    def fit(self, X, y, M="I"):

        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        if self.fit_intercept:
            self.X_ = np.insert(self.X_, 0, 1, 1)

        if M == "I":
            self.M = np.eye(self.X_.shape[1])

        self.parameters_ = self.solver(
            self.X_,
            self.y_,
            self.M,
            max_iter=self.max_iter,
            m=400,
            eps=1e-4,
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
