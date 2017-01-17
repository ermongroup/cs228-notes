---
layout: post
title: The variational auto-encoder
---

In this chapter, we are going to use various ideas that we have learned in the class in order to present a very influential recent probabilistic model called the *variational autoencoder*.

Variational autoencoders (VAEs) are a deep learning technique for learning latent representations. They have also been used to [draw images](https://arxiv.org/pdf/1502.04623.pdf), achieve state-of-the-art results in [semi-supervised learning](https://arxiv.org/pdf/1602.05473.pdf), as well a generate [latent representations for sentences](https://arxiv.org/abs/1511.06349).

There are many online tutorials on VAEs. 
Our presentation will probably be a bit more technical than the average, since our goal will be to highlight connections to ideas seen in class as well as to show how they can be used in a research paper.

## Deep generative models

Consider a [directed](../../representation/directed) [latent-variable](../../learning/latent) model of the form
{%math%}
p(x,z) = p(x|z)p(z)
{%endmath%}
with observed $$x \in \mathcal{X}$$, where $$\mathcal{X}$$ can be either continuous or discrete, as well as latent $$z \in \mathbb{R}^k$$. 

To make things concrete, you may think of $$x$$ as an image (e.g. a human face), and $$z$$ are various latent factors (we don't see them at training time) that explain features of the face. For example one coordinate of $$z$$ can encode whether the face is happy or sad, another one whether the face is more masculine or feminine, etc.

We may also be interested in models with many layers, e.g. $$p(x \mid z_1)p(z_2 \mid z_3)\cdots p(z_{m-1}\mid z_m)p(z_m)$$. These are often called *deep generative models*, and can learn hierarchies of latent representations.
In this chapter, we will assume for simplicity that there is only one latent layer.

### Learning deep generative models

Suppose now that we are given a dataset $$D = \{x^1,x^2,...,x^n\}$$. We are interested in the following inference and learning tasks:

- Learning the parameters $$\theta$$ of $$p$$.
- Approximate posterior inference over $$z$$: given an image $$x$$, what are its latent factors?
- Approximate marginal inference over $$x$$: given an image $$x$$ with missing parts, how do we fill-in these parts?

We are also going to make the following additional assumptions

- *Intractability*: computing the posterior probability $$p(z\mid x)$$ is intractable.
- *Big data*: the dataset $$D$$ is too large to fit in memory; we can only work with small, subsampled batches of $$D$$.

Many interesting models fall in this class; the variational auto-encoder will be one such model.

## Trying out the standard approaches

In this class, we have learned several techniques that could be used to solve our three tasks. Let's try them out.

The EM algorithm can be used to learn latent-variable models. Recall, however, that performing the E step requires computing the approximate posterior $$p(z\mid x)$$, which we have assumed to be intractable. In the M step, we learn the $$\theta$$ by looking at the entire dataset{%sidenote 1 'Note, however, that there exists a generalization called online EM, which performs the M-step over mini-batches.'%}, which we have assumed to be too large to hold in memory.

To perform approximate inference, we may use the mean field approach. 
Recall, however, that one step of mean field requires us to compute an expectation whose time complexity scales exponentially with the size of  the Markov blanket of the target variable. 

What is the size of the Markov blanket for $$z$$? If we assume that at least one component of $$x$$ depends on each component of $$z$$, then this introduces a V-structure into the graph of our model (the $$x$$, which are observed, are explaining away the differences among the $$z$$). Thus, we know that all the $$z$$ variables will depend on each other and the Markov blanket of some $$z_i$$ will contain all the other $$z$$-variables. This will make mean-field intractable{% sidenote 1 'The authors refer to this when they say "the required integrals for any reasonable mean-field VB algorithm are also intractable". They also discuss the limitations of EM and sampling methods in the introduction and the methods section.'%}. An equivalent (and simpler) explanation is that $$p$$ will contain a factor $$p(x_i \mid z_1,...,z_K)$$, in which all the $$z$$ variables are tied.

Another approach would be to use sampling-based methods. In the VAE paper, the authors compare against these kinds of algorithms, but they find that they don't scale well to large datasets. In addition, techniques such as Metropolis-Hastings require a hand-crafted proposal distribution, which might be difficult to choose.

## Auto-encoding variational Bayes

In the rest of the chapter, we will introduce Auto-encoding variational Bayes (AEVB), an algorithm for solving our three inference and learning tasks; the variational auto-encoder will be one instantiation of this algorithm. 

AEVB is based on ideas from [variational inference](../../inference/variational/). Recall that in variational inference, we are interested in 
maximizing the evidence lower bound (ELBO)
{%math%}
\mathcal{L}(p_\theta,q_\phi) = \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x,z) - \log q_\phi(z|x) \right]
{%endmath%}
over the space of all $$q_\phi$$. The ELBO satisfies the equation
{%math%}
\log p_\theta(x) = KL(q_\phi(z|x) || p(z|x)) + \mathcal{L}(p_\theta,q_\phi).
{%endmath%}
Note that since $$x$$ is fixed, we can define $$q(z\mid x)$$ to be conditioned on $$x$$. This means we are effectively choosing a different $$q(z)$$ for every $$x$$, which will produce a better posterior approximation than always choosing the same $$q(z)$$.

How exactly do we optimize over $$q(z\mid x)$$? We may use [mean field](../../inference/variational/), but this turns out to be insufficiently accurate for our purposes. The assumption that $$q$$ is fully factored is too strong, and the coordinate descent optimization algorithm is too simplistic.

### Black-box variational inference

The first important idea used in the AEVB algorithm is a general purpose approach for optimizing $$q$$ that works for large classes of $$q$$ (that are more complex than in mean field). We later combine this algorithm with specific choices of $$q$$.

This approach -- called *black-box variational inference* -- consists in maximizing the ELBO using gradient descent over $$\phi$$ (instead of e.g., using a coordinate descent algorithm like mean field). Hence, it only assumes that $$q_\phi$$ is differentiable in its parameters $$\phi$$.

Furthermore, instead of just doing inference, we will simultaneously  perform learning via gradient descent on both $$\phi$$ and $$\theta$$ (jointly). Optimization over $$\phi$$ will make keep ELBO tight around $$p(x)$$; optimization over $$\theta$$ will keep pushing the lower bound (and hence $$p(x)$$) up. This is somewhat reminiscent of how the [EM algorithm](../../learning/latent/) optimizes a lower bound on the marginal likelihood.

### The score function gradient estimator 

To perform black-box variational inference, we need to compute the gradient
{%math%}
\nabla_{\theta, \phi} \mathbb{E}_{q_\phi(z)} \left[ \log p_\theta(x,z) - \log q_\phi(z) \right]
{%endmath%}

Taking the expectation with respect to $$q$$ in closed form will often not be possible. Instead, we may take [Monte Carlo](../../inference/sampling/) estimates of the gradient by sampling from $$q$$. This is easy to do for the gradient of $$p$$: we may swap the gradient and the expectation and estimate the following expression via Monte Carlo
{%math%}
\mathbb{E}_{q_\phi(z)} \left[ \nabla_{\theta} \log p_\theta(x,z) \right]
{%endmath%}

However, taking the gradient with respect to $$q$$ is more difficult. Notice that we cannot swap the gradient and the expectation, since the expectation is being taken with respect to the distribution that we are trying to differentiate!

One way to estimate this gradient is via the so-called score function estimator:

{%math%}
\nabla_{\phi} \mathbb{E}_{q_\phi(z)} \left[ \log p_\theta(x,z)  - \log q_\phi(z) \right] = \mathbb{E}_{q_\phi(z)} \left[ \left(\log p_\theta(x,z)  - \log q_\phi(z) \right) \nabla_{\phi} \log  q_\phi(z) \right]
{%endmath%}

This follows from some basic algebra and calculus and takes about half a page to derive. We leave it as an exercise to the reader, but for those interested, the full derivation may found in Appendix B of this [paper](https://www.cs.toronto.edu/~amnih/papers/nvil.pdf).

The above identity places the gradient inside the expectation, which we may now evaluate using Monte Carlo. We refer to this as the *score function* estimator of the gradient.

Unfortunately, the score function estimator has an important shortcoming: it has a high variance. What does this mean? Suppose you are using Monte Carlo to estimate an expectation whose mean is 1. If your samples are $$0.9, 1.1, 0.96, 1.05,..$$ and are close to 1, then after a few samples, you will get a good estimate of the true expectation. If on the other hand you sample zero 99 times out of 100, and you sample 100 once, the expectation is still correct, but you will have to take a very large number of samples to figure out that the true expectation is actually one. We refer to the latter case as being high variance.

This is the kind of problem we often run into with the score function estimator. In fact, it is so high variance, that it does not let us learn many models.

The key contribution of the VAE paper is to propose an alternative estimator that is much better behaved in terms of variance.

This is done in two steps: we first reformulate the ELBO so that parts of it can be computed in closed form (without Monte Carlo), and then we use an alternative gradient estimator, based on the so-called reparametrization trick.

### The SGVB estimator

The reformulation of the ELBO is as follows.
{%math%}
\log p(x) \geq \mathbb{E}_{q_\phi(z)} \left[ \log p_\theta(x|z) \right] - KL(q_\phi(z) || p(z))
{%endmath%}
It is straightforward to verify that this is the same using ELBO using some algebra.

This reparametrization has a very interesting interpretation. 
The left-hand side is made of two terms; both involve taking a sample $$z \sim q(z\mid x)$$, which we can interpret as a code describing point $$x$$. We also call $$q$$ the *encoder* network.

Then, in the first term, we use $$p(x\mid z)$$ to form a probability over observed variables $$x$$ given the sampled $$z$$ and
we compute the log-likelihood of the original $$x$$. This term is maximized when $$p(x\mid z)$$ assigns high probability to the original $$x$$. It is trying to reconstruct $$x$$ given the code $$z$$; for that reason we call $$p(x\mid z)$$ the *decoder* network and the term is called the *reconstruction error*.

The second term is the divergence between $$q(z\mid x)$$ and the prior $$p(z)$$, which we will fix to be a unit Normal. It encourages the codes $$z$$ to look Gaussian. We call it the *regularization* term. It prevents $$q(z\mid x)$$ from simply encoding an identity mapping, and instead forces it to learn some more interesting representation, such as the facial features in the case of images.

Thus, our optimization objective is trying to fit a $$q(z\mid x)$$ that will map $$x$$ into a useful latent space $$z$$ from which we are able to reconstruct $$x$$ via $$p(x\mid z)$$. This type of objective is reminiscent of *auto-encoder* neural networks. This is where the paper takes its name: auto-encoding variational Bayes

### The reparametrization trick

As we have seen earlier, optimizing our objective requires a good estimate of the gradient. The score function estimator is one approach, but it suffers from the problem of variance.

By rewriting the ELBO in its auto-encoding form, we may already reduce the gradient variance, because in many cases there exist closed form expressions the $$KL$$ term, and we only need to estimate the reconstruction term using Monte-Carlo.

However, the contribution of the paper is a way of reducing variance via an alternative gradient estimator based on the *reparametrization trick*.

Under certain mild conditions (section 2.4 in the paper) for a chosen 
we may can reparameterize the distribution $$q_\phi(z\mid x)$$ using (1) a differentiable transformation $$g_\phi(\epsilon, x)$$ of a noise variable $$\epsilon$$.
{%math%}
z = g(\epsilon, x)
{%endmath%}
for $$\epsilon \sim p(\epsilon)$$, where $$p(\eppilon)$$ is some simple distribution like the unit Normal.

Gaussian variables provide the simplest example of the reparametrization trick. Instead of writing $$x \sim \mathcal{N}(\mu, \sigma)$$ for $$x \in \mathbb{R}$$ we may write
{%math%}
 x = mu + \epsilon \cdot \sigma,
{%endmath%}
where $$\epsilon \sim \mathcal{N}(0,1)$$. The two formulations for $$x$$ have the same distribution.

The reparametrization trick can be applied to a very large class of distributions, including the Exponential, Cauchy, Logistic, and many others. The biggest advantage of this approach is that we many now write the gradient of an expectation with respect to $$q(z)$$ (for any $$f$$) as
{%math%}
 \nabla_\phi \mathbb{E}_{z \sim q(z\mid x)}\left[ f(x,z) \right] = \nabla_\phi \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ f(x,g(z,\epsilon)) \right] = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ \nabla_\phi f(x,g(z,\epsilon)) \right].
{%endmath%}
Hence, we no longer need to take the gradient of the parameters of the distribution that is also used to compute the expectation. The gradient inside the expectation is fully deterministic and we may estimate using standard Monte-Carlo. This will have much lower variance than the score estimator, and will enable us to learn models that we otherwise couldn't learn.

### Choosing $$q$$ and $$p$$

Up to now, we did not specify the form of $$p$$ or $$q$$, except that these can be arbitrary functions.

How should we choose $$q$$ and $$p$$? The best $$q(z\mid x)$$ should be able to approximate the true posterior $$p(z\mid x)$$  Similarly, the $$p(x)$$ should be flexible enough to represent the richness of the data.

Motivated by these considerations, we are going to choose to parametrize $$q$$ and $$p$$ by *neural networks*. These are extremely expressive function approximators that can be efficiently optimization over big datasets, and can benefit from GPU acceleration. This choice also very elegantly bridges the gap between classical machine learning methods like variational inference and modern neural net techniques.

But what does it mean to parametrize a distribution with a neural network? Let's assume again that $$q(z\mid x)$$ and $$p(x\mid z)$$ are Normal distributions; we may write them as
{%math%}
 q(z\mid x) = \mathcal{N}(z; \vec\mu(x), \vec \sigma(x) \odot I)
{%endmath%}
where $$\vec\mu(x), \vec \sigma(x)$$ are deterministic vector-values functions of $$x$$ parametrized by an arbitrary complex neural network.

Thus, each $$q(z\mid x)$$ places a Normal distribution in the space of $$x$$ and points are sampled from that distribution. More generally, the same technique can be applied to any exponential family distribution by parameterizing the sufficient statistics by a function of $$x$$.

## The variational auto-encoder

We are now ready to take the above define the AEVB algorithm (which is the main method proposed in the paper), and the variational autoencoder, its most popular instantiation.

The AEVB algorithm is simply the combination of (1) the auto-encoding ELBO reformulation, (2) the black-box variational inference approach, and (3) the reparametrization-based low-variance gradient estimator.

A variational auto-encoder in addition parametrizes $$q(z \mid x)$$ and $$p(x \mid z)$$ using neural networks. Specifically, both are normal distributions of the form that we have defined above (we parameterize the natural parameters with a neural network).

This choice of $$p$$ and $$q$$ allows us to further simplify the auto-encoding ELBO. In particular, we can use a closed form expression to compute the regularization term, and we only use Monte-Carlo estimates for the reconstruction term. These expressions are given in the paper.

### Experimental results

The result is a model that can be applied to images $$x$$ in order to learn a latent representation. The Kingma and Welling paper contains a few examples on the Frey face dataset on the MNIST digits. On the face dataset, by interpolating between latent variables, we can generate new faces with various facial expressions that are combine the interpolated ones (e.g. we can generate smooth transitions between "angry" and "surprised"). On the MNIST dataset, we can similarly interpolate between numbers.

The authors also compare their methods against three alternative approaches: the wake-sleep algorithm, Monte-Carlo EM, and hybrid Monte-Carlo. The latter two methods are sampling-based approaches; they are quite accurate, but don't scale well to large datasets. Wake-sleep is a variational inference algorithm that scales much better; however it does not use the exact gradient of the ELBO (it uses an approximation), and hence it is not as accurate as AEVB. The paper illustrates this by plotting learning curves.