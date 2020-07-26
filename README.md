# eBNSLNoisyOR
Structure Learning algorithms for Bayesian Networks with Noisy-OR relations.

An effective gradient descent algorithm to score candidate parent sets represented with noisy-OR using the BIC score and with pruning rules that allow structure learning for Bayesian Networks to successfully scale to medium sized networks.

Run ./runScoring to get noisy-OR score files for all nodes.

Bayesian Networks are learned with [GOBNILP](https://www.cs.york.ac.uk/aig/sw/gobnilp/). Inference on the networks is performed with JavaBayes, using a [modified version](https://github.com/CharupriyaSharma/JavaBayes8) for scripting and compilation in Java 8.
