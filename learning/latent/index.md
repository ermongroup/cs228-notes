---
layout: post
title: Learning in latent variable models
---

Up to now, we have assumed that when learning a directed or an undirected model, we are given examples of every single variable that we are trying to model.

However, that may not always be the case. Consider for example a probabilistic language model of news articles{% sidenote 1 'A language model $$p$$ assigns probabilities to sequences of words $$x_1,...,x_n$$. We can, among other things, sample from $$p$$ to generate various kinds of sentences.'%}. Each article $$x$$ typically focuses on a specific topic $$t$$, e.g. finance, sports, politics. 
Using this prior knowledge, we may build a more accurate model $$p(x|t)p(t)$$, in which we have introduced an additional, unobserved variable $$t$$. The model postulates than an article is created by first choosing the topic $$t$$, and then by sampling words given that topic.

However, since $$t$$ is unobserved, we cannot directly use the learning methods that we have so far. In fact, the unobserved variables make learning much more difficult; in this chapter, we will look at how to use and how to learn models that involve latent variables.

## Latent variable models

More formally, a latent variable model (LVM) $$p$$ is a probability distribution over two sets of variables $$x, z$$:
{%math%}
p(x, z; \theta),
{%endmath%}
where the $$x$$ variables are observed at learning time in a dataset $$D$$ and the $$z$$ are never observed. 

The model may be either directed or undirected. There exist both discriminative and generative LVMs, although here we will focus on the latter (the key ideas hold for discriminative models as well).

### Example: Gaussian mixture models

Gaussian mixture models (GMMs) are a latent variable model that is also one of the most widely used models in machine learning. {% marginfigure 'gmm1' 'assets/img/gmm1.png' 'Example of a dataset that is best fit with a mixture of two Gaussians. Mixture models allow us to model clusters in the dataset.' %}

In a GMM, each data point is a tuple $$(x_i, z_i)$$ with $$x_i \in \mathbb{R}^d$$ and $$z_i \in {1,2,...,K}$$ ($$z_i$$ is discrete). The joint $$p$$ is a directed model
{%math%}
p(x, z) = p(x|z)p(z),
{%endmath%}
where $$p(z = k) = \pi_k$$ for some vector of class probabilities $$\pi \in \Delta_{K-1}$$ and
{%math%}
p(x|z=k) = \mathcal{N}(x; \mu_k, \Sigma_k)
{%endmath%}
is a multivariate Gaussian with mean and variance $$\mu_k, \Sigma_k$$.

This model postulates that our observed data is comprised of $$K$$ clusters with proportions specified by $$\pi_1,...,\pi_K$$; the distribution of each cluster is a Gaussian. This is best seen by looking at the marginal probability of a point $$x$$:
{%math%}
p(x) = \sum_{k=1}^K p(x|z=k)p(z=k) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k).
{%endmath%}
To generate a new data point, we would sample a cluster $$k$$ and then sample its Gaussian $$\mathcal{N}(x; \mu_k, \Sigma_k)$$.

{% maincolumn 'assets/img/gmm2.png' 'Example of a Gaussian mixture model, consisting of three components with different class proportions (a). The true class of each point is unobserved, so the distribution over $$x$$ looks like in (b); it is both multi-modal and non-Gaussian. Visualizing it in 3D shows the effects of class proportions on the magnitudes of the modes.' %}

### Why are latent variable models useful?

There are two reasons why we might want to use latent variable models.

The simplest reason is that some data might be naturally unobserved. For example, if we are modeling a clinical trial, then some patients may drop out, and we won't have their measurements. The methods in this chapter can be used to learn with this kind of missing data.

Perhaps, the most important reason for studying LVMs is that they enable us to leverage our prior knowledge about the problem. Our topic modeling example from the introduction already illustrates this. We know that our set of news articles is actually a mixture of $$K$$ distinct distributions (one for each topic); LVMs allow us to design a model that captures this.

LVMs can also be viewed as increasing the expressive power of our model. In the case of GMMs, the distribution that we can model using a mixture of Gaussian components is much more expressive than what we could have modeled using a single component.

### Marginal likelihood training

How do we train an LVM? Our goal is still to fit the marginal distribution $$p(x)$$ over the visible variables $$x$$ to that observed in our dataset $$D$$. Hence our previous discussion about KL divergences applies here as well and by the same argument, we should be maximizing the *marginal log-likelihood* of the data
{%math%}
\log p(D) = \sum_{x \in D} \log p(x) = \sum_{x \in D} \log \left( \sum_z p(x|z) p(z) \right).
{%endmath%}

This optimization objective however is considerably more difficult than regular log-likelihood, even for directed graphical models. For one, we can see that the summation inside the log makes it impossible to decompose $$p(x)$$ into a sum of log-factors. Hence, even if the model is directed, we can no longer derive a simple closed form expression for the parameters.

{% marginfigure 'gmm1' 'assets/img/mixture.png' 'Exponential family distributions (gray lines) have concave log-likelihoods. However, a weighted mixture of such distributions is no longer concave (black lines).' %}
Looking closer at the distribution of a data point $$x$$, we also see that it is actually a mixture 
{%math%}
p(x) = \sum_z p(x|z) p(z)
{%endmath%}
of distributions $$p(x|z)$$ with weights $$p(z)$$. Whereas when we had a single exponential family distribution $$p(x)$$ results in a concave log-likelihood (as we have seen in our discussion of undirected models), the log of a weighted mixture of such distributions is no longer concave or convex.

This non-convexity requires the developments of specialized learning algorithms.


## Learning latent variable models

Since the objective is non-convex, we will resort to approximate learning algorithms. These methods are widely used in practice and are quite effective. Note however, that (quite surprisingly), many latent variable models (like GMMs) admit algorithms that can compute the global optimum of the maximum likelihood objective, even though it is not convex. Such methods are covered at the end of CS229T.

### The Expectation-Maximization algorithm

The Expectation-Maximization (EM) algorithm is a hugely important and widely used algorithm for learning directed latent-variable graphical models $$p(x,z; \theta)$$ with parameters $$\theta$$ and latent $$z$$.

The EM algorithm relies on two simple observations. 

- If the latent $$z$$ were fully observed, then we could optimize the log-likelihood exactly using our previously seen closed form solution for $$p(x,z)$$.
- Knowing the weights, we can often efficiently compute the posterior $$p(z\mid x; \theta)$$ (this is an assumption; it is not true for some models).

The EM follows a simple iterative two-step strategy: given an estimate $$\theta_t$$ of the weights, compute $$p(z\mid x)$$ and use it to "hallucinate" values for $$z$$. Then, find a new $$\theta_{t+1}$$ by optimizing the resulting (tractable objective). This process will eventually converge.

We haven't exactly defined what we mean by "hallucinating" the data. The full definition is a bit technical, but its instantiation is very intuitive in most models like GMMs.

By "hallucinating" the data, we mean computing the expected log-likelihood
{%math%}
\mathbb{E}_{z \sim p(z|x)} \log p(x,z; \theta).
{%endmath%}
This expectation is what gives the EM algorithm half of its name. If $$z$$ is not too high-dimensional (e.g. in GMMs it is a one-dimensional categorical variable), then we can compute this expectation. Since the summation is now outside the log, we can learn this objective as we did for directed graphical models.

We can formally define the EM algorithm as follows:

- Starting at an initial $$\theta_0$$, repeat until convergence for $$t=1,2,...$$:
- *E-Step*: Compute the posterior $$p(z\mid x; \theta_{t})$$
- *M-Step*: Compute new weights
{%math%}
\theta_{t+1} = \arg \max_{\theta} \mathbb{E}_{z \sim p(z|x; \theta_{t})} \log p(x,z; \theta).
{%endmath%}

### Example: Gaussian mixture models

Let's look at this algorithm in the context of GMMs. In the E-step, we may compute the posterior as follows
{%math%}
p(z \mid x; \theta_{t}) = \frac{p(z, x; \theta_{t})}{p(x; \theta_{t})} = \frac{p(x | z; \theta_{t}) p(z; \theta_{t})}{\sum_{k=1}^K p(x | z_k; \theta_{t}) p(z_k; \theta_{t})}.
{%endmath%}

Note that each $$ p(x \mid z_k; \theta_{t}) p(z_k; \theta_{t}) $$ is simply the probability that $$x$$ originates from component $$k$$ given the current set of parameters $$\theta$$. After normalization, these form the $$K$$-dimensional vector of probabilities $$p(z \mid x; \theta_{t})$$.

Recall that in the original model, $$z$$ is an indicator variable that chooses a component for $$x$$; we may view this a "hard" assignment of $$x$$ to one component. The result of the $$E$$ step is a $$K$$-dimensional vector (whose components sum to one) that specifies a "soft" assignment to components. In that sense, we have "hallucinated" a "soft" instantiation of $$z$$ that has a simple, intuitive interpretation.

At the M-step,... 

### EM as variational inference

Why exactly does this procedure converge? Actually, we can easily understand the behavior of EM through the lens of variational inference, which we have seen earlier.

Consider the inference problem over the $$z$$ variables in the model $$p(z,x; \theta)$$, with the $$x$$ variables held fixed as evidence.

Recall that the variational approach maximizes the evidence lower bound (ELBO)
{%math%}
\mathcal{L}(p,q) = \mathbb{E}_{q(z)} \left[ \log p(x,z) - \log q(z) \right]
{%endmath%}
over the space of all $$q$$. The ELBO satisfies the equation
{%math%}
\log p(x) = KL(q(z) || p(z|x)) + \mathcal{L}(p,q).
{%endmath%}
Hence, we see that for a fixed $$p$$, \mathcal{L}(p,q) is maximized when $$q=p(z|x)$$; in that case the KL term becomes zero, and 
{%math%}
\log p(x; \theta) = \mathcal{L}(p,q) = \mathbb{E}_{p(z|x; \theta_t)} \log p(x,z; \theta) - \mathbb{E}_{p(z|x; \theta)} \log p(z|x; \theta_t)
{%endmath%}
The right hand side of the equation, is precisely our EM maximization objective
{%math%}
\mathbb{E}_{z \sim p(z|x; \theta_{t})} \log p(x,z; \theta),
{%endmath%}
plus an addition term that does not depend on the free parameter $$\theta$$. When $$\theta = \theta_t$$, that expression precisely equals $$\log p(x; \theta)$$; in other cases, it remains a lower bound. Hence, at the M step, we will certainly have increased the log-likelihood of the data. Then we do another E step, which does not increase the true log-likelihood, but which tightens the bound. This allows us to maximize the log-likelihood again using another M step, and so on.

### Properties of EM

From our above discussion, it follows that EM has the following properties:

- After an EM cycle, the marginal likelihood increases by a step
- Since the marginal likelihood is bounded by its true global maximum, and it increases at every step, the procedure must eventually converge.

However, since we optimizing a non-convex objective, we have no guarantee to find the global optimum of the solution. In fact, EM in practice converges almost always to a local optimum, and moreover, that optimum heavily depends on the choice of initialization. Different initial $$\theta_0$$ can lead to very different solutions, and so it is very common to use multiple restarts of the algorithms and choose the best one in the end. In fact EM is so sensitive to the choice of initial parameters, that techniques for choosing these parameters are still an active area of research.

In summary, the EM algorithm is a very popular technique for optimizing latent variable models that is also often very effective. It's main downside are its difficulties with local minima.