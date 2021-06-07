# Multilevel Survival Modeling with Structured Penalties for Disease Prediction from Imaging Genetics data

This code is a complement to the corresponding article. It has been tested which Python 3.8 and Numpy 1.18.5 on a MacBook Pro (Retina, 13-inch, Late 2013).

## Structure of the code

The code is mainly composed of the fit method:
* `fit`: Fit w in a Cox model h(t) = h_0(t) exp(w*x)

and three other helper methods:
* `scaler`: Scales features matrix (without E, T)
* `gradient`: Compute log-likelihood gradient in a Cox model with respect to w.
* `log_likelihood`: Compute log-likelihood in a Cox model with respect to w.
