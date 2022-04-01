# blr-dev (bayesian logistic regression with stochastic gradient hamiltonian monte carlo)

This repository contains an implementation of bayesian logistic regression using a method developed by Chen et al. in 2014 (Stochastic Gradient Hamiltonian Monte Carlo).


Installation
```bash
pip install -i https://test.pypi.org/simple/ blr-sghmc
```



Example:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from blr_sghmc import BayesianLogisticRegression

X, y = make_classification()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
model = BayesianLogisticRegression(max_iter=100, verbose=0, solver='sghmc', batch_size=16, eps=1e-3)
model.fit(X_train, y_train)
model.score(X_val, y_val)
```
