---
layout: post
title: Contents
---
<span class="newthought">These notes</span> form a concise introductory course on probabilistic graphical models{% include sidenote.html id="note-pgm" note="Probabilistic graphical models are a subfield of machine learning that studies how to describe and reason about the world in terms of probabilities." %}.
They are based on Stanford [CS228](https://cs228.stanford.edu/), and are written by [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov) and [Stefano Ermon](http://cs.stanford.edu/~ermon/), with the [help](https://github.com/ermongroup/cs228-notes/commits/master) of many students and course staff.
{% include marginnote.html id='mn-construction' note='The notes are still **under construction**! Although we have written up most of the material, you will probably find several typos. If you do, please let us know, or submit a pull request with your fixes to our [GitHub repository](https://github.com/ermongroup/cs228-notes).'%}
You too may help make these notes better by submitting your improvements to us via [GitHub](https://github.com/ermongroup/cs228-notes).

This course starts by introducing probabilistic graphical models from the very basics and concludes by explaining from first principles the [variational auto-encoder](extras/vae), an important probabilistic model that is also one of the most influential recent results in deep learning.

## Preliminaries

1. [Introduction](preliminaries/introduction/): What is probabilistic graphical modeling? Overview of the course.

2. [Review of probability theory](preliminaries/probabilityreview): Probability distributions. Conditional probability. Random variables (*under construction*).

3. [Examples of real-world applications](preliminaries/applications): Image denoising. RNA structure prediction. Syntactic analysis of sentences. Optical character recognition (*under construction*).

## Representation

1. [Bayesian networks](representation/directed/): Definitions. Representations via directed graphs. Independencies in directed models.

2. [Markov random fields](representation/undirected/): Undirected vs directed models. Independencies in undirected models. Conditional random fields.

## Inference

1. [Variable elimination](inference/ve/) The inference problem. Variable elimination. Complexity of inference.

2. [Belief propagation](inference/jt/): The junction tree algorithm. Exact inference in arbitrary graphs. Loopy Belief Propagation.

3. [MAP inference](inference/map/): Max-sum message passing. Graphcuts. Linear programming relaxations. Dual decomposition.

4. [Sampling-based inference](inference/sampling/): Monte-Carlo sampling. Forward Sampling. Rejection Sampling. Importance sampling. Markov Chain Monte-Carlo. Applications in inference.

5. [Variational inference](inference/variational/): Variational lower bounds. Mean Field. Marginal polytope and its relaxations.

## Learning

1. [Learning in directed models](learning/directed/): Maximum likelihood estimation. Learning theory basics. Maximum likelihood estimators for Bayesian networks.

2. [Learning in undirected models](learning/undirected/): Exponential families. Maximum likelihood estimation with gradient descent. Learning in CRFs

3. [Learning in latent variable models](learning/latent/): Latent variable models. Gaussian mixture models. Expectation maximization.

4. [Bayesian learning](learning/bayesian/): Bayesian paradigm. Conjugate priors. Examples (*under construction*).

5. [Structure learning](learning/structure/): Chow-Liu algorithm. Akaike information criterion. Bayesian information criterion. Bayesian structure learning (*under construction*).

## Bringing it all together

1. [The variational autoencoder](extras/vae): Deep generative models. The reparametrization trick. Learning latent visual representations.

2. [List of further readings](extras/readings): Structured support vector machines. Bayesian non-parametrics.
