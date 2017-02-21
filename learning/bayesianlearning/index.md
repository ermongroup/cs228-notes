---
layout: post
title: Bayesian Learning
---

Generally speaking, Bayesian learning is the method of selecting the best hypothesis $$h \in H$$  in terms of how well it can explain the observed training data $$D$$. In this learning method, we use Bayes' theorem to update the probability for a hypothesis as more information or data get available. 
## The Bayesian Paradigm 
The main idea behind the Bayesian paradigm is that any uncertainty can be modeled in a probabilistic manner. The probability model we build for this uncertainty reflects our beliefs or any prior experience we may have. And since this prior belief can differ from person to person, our probability model can be pretty crude. 

Now, lets say the probability model we use to express uncertainty is described by parameter $$\theta$$. In the Bayesian paradigm, we treat this parameter $$\theta$$ as if it were a random variable $$\Theta$$ whose distribution describes the uncertainty. Here we are in no means bound to hold our initial belief regarding the uncertainty. In fact, we modify our belief as we acquire more data and information. And the key behind this updating procedure is Bayes' theorem. 

Bayes' theorem states that:

$$P(\theta \mid x) = \frac{P(x \mid \theta) \, P(\theta)}{P(x)} = \frac{P(x \mid \theta) \, P(\theta)}{\int P(x | \theta) P(\theta) d\theta }$$


To motivate Bayesian learning, we can read this in the following way: "the probability of the model given the data, $$P(\theta \mid x)$$, is the probability of the data given the model, $$P(x \mid \theta)$$, times the prior probability of the model, $$P(\theta)$$, divided by the probability of the data $$P(x)$$. Under Bayesian paradigm, we treat degrees of belief exactly in the same way as we treat probabilities. In the above equation, the prior $$P(\theta)$$ represents how much we believe model $$\theta$$ to be the true model that generates the data $$x$$, before we actually observe the data $$x$$. The posterior $$P(\theta \mid x)$$ represents how much we believe model $$\theta$$ after observing the data. 

Informally speaking, in Bayesian learning, we start out by enumerating all reasonable models of the data and assigning our prior belief $$P(\theta)$$ to each of these models. Once we observe the data $$x$$, we evaluate how probable the observed data $$x$$ was under each of these models, i.e. we compute $$P(x \mid \theta)$$. We then multiply this likelihood $$P(x \mid \theta)$$ by the prior $$P(\theta)$$ and normalize it yielding posterior probability over models $$P(\theta \mid x)$$. This posterior probability encapsulates everything that we have learned from the data regarding the models we are considering. 
## Conjugate Priors
When calculating posterior distribution using Bayes' rule, as in the above, it should be pretty straightforward to calculate the numerator. But to calculate the denominator $$P(x)$$, we are required to compute the integral. This might cause us trouble, since for an arbitrary distribution, computing the integral is likely to be intractable.

To tackle this issue, we use a conjugate prior. A parametric family $$\varphi$$ is conjugate for the likelihood $$P(x \mid \theta)$$ if:

$$P(\theta) \in \varphi \Longrightarrow P(\theta \mid x) \in \varphi$$

This is convenient because if we know the normalizing constant of $$\varphi$$, then we get the denominator in Bayes' rule "for free". Thus it essentially reduces the computation of the posterior from a tricky numerical integral to some simple algebra. 

To see conjugate prior in action, let's consider an example. Suppose we are given a sequence of $N$ coin tosses, $$D = \{X_{1},...,X_{N}\}$$. We want to infer the probability of getting heads which we denote by $$\theta$$.  Now, we can model this as a sequence of Bernoulli trials with parameter $$\theta$$. A natural conjugate prior in this case is the beta distribution with

$$P(\theta) = Beta(\theta \mid \alpha_{H}, \alpha_{T}) = \frac{\theta^{\alpha_{H} -1 }(1-\theta)^{\alpha_{T} -1 }}{B(\alpha_{H},\alpha_{T})}$$

where the normalization constant $$B(\cdot)$$ is the beta function. Here $$\alpha = (\alpha_{H},\alpha_{T})$$ are called the hyperparameters of the prior. The expected value of $$\theta$$ is $$\frac{\alpha_{H}}{\alpha_{H}+\alpha_{T}}$$. Here the sum of the hyperparameters $$(\alpha_{H}+\alpha_{T})$$ can be interpreted as a measure of confidence in the expectations they lead to. Intuitively, we can think of $$\alpha_{H}$$ as the number of heads we have observed before the current dataset. 

Out of $$N$$ coin tosses, if the number of heads and the number of tails are $$N_{H}$$
and $$N_{T}$$ respectively, then it can be shown that the posterior is:

$$P(\theta \mid N_{H}, N_{T}) = \frac{\theta^{N_{H}+ \alpha_{H} -1 }(1-\theta)^{ N_{T}+ \alpha_{T} -1 }}{B(N_{H}+ \alpha_{H},N_{T}+ \alpha_{T})}$$

which is another Beta distribution with parameters $$(N_{H}+ \alpha_{H},N_{T}+ \alpha_{T})$$. We can use this posterior distribution as the prior for more samples with the hyperparameters simply adding each extra piece of information as it comes from additional coin tosses. 


{% maincolumn 'assets/img/beta.png' 'Here the exponents $$(3,2)$$ and $$(30,20)$$ can both be used to encode the belief that $$\theta$$ is $$0.6$$. But the second set of exponents imply a stronger belief as they are based on a larger sample.' %}


