---
layout: post
title: Learning in latent variable models
---

In this chapter, we are going to try to combine the various ideas that we have learned so far in the class in order to understand a very influential recent probabilistic model called the *variational autoencoder*.

Variational autoencoders (VAEs) are widely used in deep learning to learn latent representations for images. Closely related approaches have also been used to draw images, achieve state-of-the-art results in semi-supervised learning, as well a generate latent representation for sentences.

Although there exist many online resources that explain variational autoencoders, our presentation will try to follow more closely the original paper by Kingma and Welling and will try to highlight how their work relies on ideas from probabilistic graphical modeling. It may be more technical than usual, but will help you see how ideas from different parts of graphical modeling combine in this work.

## Latent variable learning with intractable posteriors

<!---
Interestingly, although it forms one of the key recent results in deep learning, the paper of Kingma and Welling is written very much from a classical probabilistic modeling perspective. They propose a general algorithm for PGMs that only at the very end is instantiated with neural networks.

After a high-level introduction, Kingma and Welling start by defining the general problem that they will try to address.
-->

Consider a directed latent-variable model of the form
{%math%}
p(x,z) = p(x|z)p(z)
{%endmath%}
with observed $$x \in \mathcal{X}$$, where $$\mathcal{X}$$ can be either continuous or discrete, as well as latent $$z \in \mathbb{R}^k$$. Crucially, the $$z$$ are continuous. We are given a dataset $$D$$ and are interested in learning parameters $$\theta$$ of $$p$$.

We are also going to make the following additional assumptions

- *Intractability*: computing the posterior probability $$p(z\mid x)$$ is intractable. Also, the Markov blanket of each variable is very large.
- *Big data*: the dataset $$D$$ is too large to fit in memory; we can only work with small batches of data subsampled at random from the large dataset.

The authors argue that many interesting models fall in this class, and towards the end of the paper, they present a powerful but intractable class of models involving neural networks, and show how their proposed algorithms enables learning in that setting.

First, however, we are going to develop a practical learning and inference algorithm for this class of models, focusing on three tasks:

- Approximate maximum-likelihood inference of the parameters $$\theta$$
- Approximate posterior inference over $$z$$
- Approximate marginal inference over $$x$$

### Existing approaches

As we have seen, learning is intractable in general in latent-variable models, and inference relative to $$p(z \mid x)$$ is intractable as well. However, we have also seen several approximate inference methods for these tasks.

The EM algorithm can be used to learn latent-variable models. Recall, however, that performing the E step requires computing the approximate posterior $$p(z|x)$$, which we have assumed to be intractable. The M step also performs optimization relative to entire set {%sidenote 1 'Note, however, that there exists a generalization called online EM, which can handle such datasets'%}, which we have assumed to be too large to optimize over in one step.

To perform approximate inference, we have seen approaches such as mean field. However, by our assumption on the the Markov blanket of each variable{% sidenote 1 'The authors refer to this as "the required integrals for any reasonable mean-field VB algorithm are also intractable".', this is also not possible. Recall that the time complexity of mean field is exponential in the size of the Markov blanket.

Another potential approach would be to use sampling based-methods. However, such methods are often slow and the Metropolis-Hastings approach would require a proposal distribution, that we would not necessarily know how to learn.

## Auto-encoding variational Bayes

The approach we will instead consider will be one based on approximate inference.

Recall that in approximate inference, we are interested in 
maximizing the evidence lower bound (ELBO)
{%math%}
\mathcal{L}(p_\theta,q_\phi) = \mathbb{E}_{q_\phi(z)} \left[ \log p_\theta(x,z) - \log q_\phi(z) \right]
{%endmath%}
over the space of all $$q_\phi$$. The ELBO satisfies the equation
{%math%}
\log p_\theta(x) = KL(q_\phi(z) || p(z|x)) + \mathcal{L}(p_\theta,q_\phi).
{%endmath%}

In the variational inference chapter, we have seen the mean field algorithm for optimizing the evidence lower bound. However, this approach assumes a very simple (fully factored) form for the approximate distribution $$q$$, which may not be sufficiently expressive for our purposes, especially if we have a lot of data. We have also proposed an coordinate descent optimization algorithm; however, it is only applicable to simple models like mean field.

Instead, we are going to adopt here a new approach to inference. First, we will no longer make any assumption on $$q_\phi$$, besides that it is differentiable in its parameters $$\phi$$. Next, instead of using a coordinate descent algorithm, we will maximize the ELBO using gradient descent over $$\phi$$.

In addition, instead of just performing inference, we will perform gradient descent on both $$\psi$$ and $$\theta$$. Thus, we will be performing jointly inference and learning. Optimization over $$\phi$$ will make keep ELBO tight around $$p(x)$$, while optimization over $$\theta$$ will keep pushing the lower bound (and hence $$p(x)$$) up. Recall that we have already seen this idea of maximizing the marginal likelihood via the lower bound in the section on the EM algorithm; in this case however, none of the steps will be exact; however, our method will be applicable to a much larger class of models.

### Black-box variational inference

We need to optimize the objective using the gradient
{%math%}
\nabla_{\theta, \psi} \mathbb{E}_{q_\phi(z)} \left[ \log p_\theta(x,z) - \log q_\phi(z) \right]
{%endmath%}

But how to we compute this objective? Notice that obtaining its exact value requires inference with respect to $$q$$, which we said could take  an arbitrary form. Furthermore, we not only have to perform this inference, but we must also differentiate with respect to this distribution. How can we do something like that?

One way to estimate this gradient (mentioned in Section 2.2 in Kingma and Welling) is as follows.

REINFORCE GOES HERE

We have now reformulated the gradient as an expectation; assuming that we can sample from $$q_\phi$$, we can obtain an estimate of the gradient using Monte Carlo sampling. We refer to this as the *score function* estimator of the gradient.

Unfortunately, the score function estimator has an important shortcoming: it has a high variance. This means that even though the average our Monte Carlo will eventually equal the true gradient, their spread around the mean will be very large. For example, if the true gradient value is 100, we may sample 0 99% of the time and 10000 1% of the time. We will need a lot of samples to estimate the gradient this way.

The key contribution of the Kingma and Welling is to propose an alternative estimator that is much better behaved in terms of variance.

This is done in two steps: we first reformulate the ELBO so that parts of it can be computed in closed form (without Monte Carlo), and then we use an alternative gradient estimator, based on the so-called reparametrization trick.

### The SVGB estimator

The reformulation of the ELBO is as follows.
{%math%}
\log p(x) \geq \mathbb{E}_{q_\phi(z)} \left[ \log p_\theta(x|z) \right] - KL(q_\phi(z) || p(z))
{%endmath%}
It is straightforward to verify that this is the same using ELBO using some algebra.

This reparametrization has a very interesting interpretation. 
The left-hand side is made of two terms; both involve taking a sample $$z \sim q(z|x)$$, which we can interpret as a code describing point $$x$$. We also call $$q$$ the *encoder* network.

Then, in the first term, we use $$p(x|z)$$ to form a probability over observed variables $$x$$ given the sampled $$z$$ and
we compute the log-likelihood of the original $$x$$. This term is maximized when $$p(x|z)$$ assigns high probability to the original $$x$$. It is trying to reconstruct $$x$$ given the code $$z$$; for that reason we call $$p(x|z)$$ the *decoder* network and the term is called the *reconstruction error*.

The second term is the divergence between $$q(z|x)$$ and the prior $$p(z)$$, which we will fix to be a unit Normal. It encourages the codes $$z$$ to look Gaussian. We call it the *regularization* term. It prevents $$q(z|x)$$ from simply encoding an identity mapping, and instead forces it to learn some more interesting representation, such as the facial features in the case of images.

Thus, our optimization objective is trying to fit a $$q(z|x)$$ that will map $$x$$ into a useful latent space $$z$$ from which we are able to reconstruct $$x$$ via $$p(x|z)$$. This type of objective is reminiscent of *auto-encoder* neural networks. This is where the paper takes its name: auto-encoding variational Bayes

### The reparametrization trick

As we have seen earlier, optimizing our objective requires a good estimate of the gradient. The score function estimator is one approach, but it suffers from the problem of variance.

By rewriting the ELBO in its auto-encoding form, we may already reduce the gradient variance, because in many cases there exist closed form expressions the $$KL$$ term, and we only need to estimate the reconstruction term using Monte-Carlo.

However, the contribution of the paper is a way of reducing variance via an alternative gradient estimator based on the *reparametrization trick*.

Under certain mild conditions (section 2.4 in the paper) for a chosen 
we may can reparameterize the distribution $$q_\phi(z|x)$$ using (1) a differentiable transformation $$g_\phi(\epsilon, x)$$ of a noise variable $$\epsilon$$.
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
 \nabla_\phi \mathbb{E}_{z \sim q(z|x)}\left[ f(x,z) \right] = \nabla_\phi \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ f(x,g(z,\epsilon)) \right] = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[ \nabla_\phi f(x,g(z,\epsilon)) \right].
{%endmath%}
Hence, we no longer need to take the gradient of the parameters of the distribution that is also used to compute the expectation. The gradient inside the expectation is fully deterministic and we may estimate using standard Monte-Carlo. This will have much lower variance than the score estimator, and will enable us to learn models that we otherwise couldn't learn.

### Choosing $$q$$ and $$p$$

Up to now, we did not specify the form of $$p$$ or $$q$$, except that these can be arbitrary functions.

How should we choose $$q$$ and $$p$$? The best $$q(z\mid x)$$ should be able to approximate the true posterior $$p(z\mid x)$$  Similarly, the $$p(x)$$ should be flexible enough to represent the richness of the data.

Motivated by these considerations, we are going to choose to parametrize $$q$$ and $$p$$ by *neural networks*. These are extremely expressive function approximators that can be efficiently optimization over big datasets, and can benefit from GPU acceleration. This choice also very elegantly bridges the gap between classical machine learning methods like variational inference and modern neural net techniques.

But what does it mean to parametrize a distribution with a neural network? Let's assume again that $$q(z\mid x)$$ and $$p(x\mid z)$$ are Normal distributions; we may write them as
{%math%}
 q(z|x) = \mathcal{N}(z; \vec\mu(x), \vec \sigma(x) \odot I)
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

The authors also compare their methods against two alternative approaches: the wake-sleep algorithm, which 


