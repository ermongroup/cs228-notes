---
layout: post
title: Contents
---
{% newthought 'These notes'%} form a concise introductory course on probabilistic graphical models{% sidenote 1 'Probabilistic graphical models are a subfield of machine learning that studies how to describe and reason about the world in terms of probabilities.'%}.
They are based on Stanford [CS228](http://cs.stanford.edu/~ermon/cs228/index.html), taught by [Stefano Ermon](http://cs.stanford.edu/~ermon/), and are written and maintained by [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov).
{% marginnote 'mn-id-whatever' 'These notes are still **under construction**!
They are about 70% complete, and probably contain a lot of typos.
We are in the process of finalizing them; in the meantime feel free to submit your fixes on [Github](https://github.com/ermongroup/cs228-notes).'%}
Free to contribute your improvements on [Github](https://github.com/ermongroup/cs228-notes).

## Preliminaries

1. [Introduction](preliminaries/introduction/) What is probabilistic graphical modeling? Overview of the course.

2. [Review of probability theory](preliminaries/probabilityreview): Probability distributions. Conditional probability. Random variables.

2. [Examples of real-world applications](preliminaries/applications): Image denoising. RNA structure prediction. Syntactic analysis of sentences. Optical character recogition.

## Representation

1. [Bayesian networks](representation/directed/): Definitions. Representations via directed graphs. Independencies in directed models.

2. [Markov random fields](representation/undirected/): Undirected vs directed models. Independencies in undirected models. Conditional random fields.

## Inference

1. [Variable elimination](inference/ve/) The inference problem. Variable elimination. Complexity of inference.

2. [Belief propagation](inference/jt/): The junction tree algorithm. Exact inference in arbitrary graphs. Loopy Belief Propagation.

3. [MAP inference](inference/map/): Max-sum message passing. Graphcuts. Linear programming relaxations. Dual decomposition.

4. [Sampling-based inference](inference/sampling/): Monte-Carlo sampling. Importance sampling. Markov Chain Monte-Carlo. Applications in inference.

5. Variational inference: Variational lower bounds. Mean Field. Marginal polytope and its relaxations.

## Learning

1. Learning in directed models: Maximum likelihood estimation. Learning theory basics. Maximum likelihood estimators for Bayesian networks.

2. Learning in undirected models: Exponential families. Gradient descent maximum likelihood estimation.

3. Learning in latent-variable models: Gaussian mixture models. Expectation maximization.

4. Bayesian learning: Bayesian paradigm. Conjugate priors. Examples.

5. Structure learning: Chow-Liu algorithm. Akaike information criterion. Bayesian information criterion. Bayesian structure learning.

