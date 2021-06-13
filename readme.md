# Multilevel Survival Modeling with Structured Penalties for Disease Prediction from Imaging Genetics data

This code corresponds to the multilevel survival modelling approach proposed in the following paper:

> Lu P and Colliot O, Multilevel Survival Modeling with Structured Penalties for Disease Prediction from Imaging Genetics data, *Submitted to IEEE Journal of Biomedical and Health Informatics*, 2021.

It is written in Python and the only dependency is Numpy. This code is thus expected to work in most environments even though it has been developed with Python 3.8 and Numpy 1.18.5 on a MacBook Pro (Retina, 13-inch, Late 2013).

## Structure of the code

The code is mainly composed of the fit method:
* `fit`: Fits the multilevel survival model.

and three other helper methods:
* `scaler`: Scales the feature matrix.
* `gradient`: Compute the log-likelihood gradient.
* `log_likelihood`: Compute the log-likelihood.

Please refer to the comments of each function for details about their specific arguments.

The file `genetic_groups.py` contains the list of SNPs used for our experiments, the mapping of SNPs to Gene (one SNP can belong to several genes) and the list of genetic groups where keys are genes and values are positions of SNPs (snp_indices).
