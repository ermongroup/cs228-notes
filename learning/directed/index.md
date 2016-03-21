---
layout: post
title: Learning in directed models
---
We now turn our attention to the third and last part of the course: *learning*. Given a dataset, we would like to fit a model that will make useful prediction on various tasks that we care about.

We will consider this problem in two types of models: directed and undirected. It turns out that the former will admit easy, closed form solutions, while the latter will involve potentially intractable numerical optimization techniques. In later section, we will look at latent variable models (which contain unobserved latent variables that succinctly model the event of interest) as well as Bayesian learning, a general approach to statistics that offers certain advantages to more standard approaches.

Before, we start our discussion of learning, let's first reflect on what it means it fit a model, and what is a desirable objective for this task.