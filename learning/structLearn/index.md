---
layout: post
title: Structure Learning for Bayesian Networks
---
## Structure learning for Bayesian networks

The task of structure learning for Bayesian networks refers to learn the structure of the directed acyclic graph (DAG) from data. There are two major approaches for the structure learning: score-based approach and constraint-based approach . 

### Score-based approach

The score-based approach first defines a criterion to evaluate how well the Bayesian network fits the data, then searches over the space of DAGs for a structure with maximal score. In this way, the score-based approach is essentially a search problem and consists of two parts: the definition of score metric and the search algorithm. 

### Score metrics

The score metrics for a structure $$\mathcal{G}$$ and data $$D$$ can be generally defined as: 
$$Score(G:D)= LL(G:D) - \phi(|D|) \|G\|.$$ 

Here $$LL(G:D)$$ refers to the log-likelihood of the data under the graph structure $$\mathcal{G}.$$  The parameters in Bayesian network $$G$$ are estimated based on MLE and the log-likelihood score is calculated based on the estimated parameters. If we consider only the log-likelihood in the score function, we will end up with an overfitting structure (namely, a complete graph.) That is why we have the second term in the scoring function. $$|D|$$ is the number of sample and $$\|G\|$$ is the number of parameters in the graph 
$$ \mathcal{G}.$$ With this extra term, we will penalize the over-complicated graph structure and avoid overfitting.  For AIC the function $$\phi(t) = 1, $$ while for BIC $$\phi(t) =  \log(t)/2.$$ It is important to note that in BIC, the influence of model complexity will decrease as M grows, allowing the log-likelihood term to eventually dominate the score. 

There is another family of Bayesian score function called BD (Bayesian Dirichlet) score. For BD score, if first defines the probability of data $$D$$ conditional on the graph structure $$\mathcal{G}$$ as 

{%math%}
P(D|\mathcal{G})=\int P(D|\mathcal{G}, \Theta_{\mathcal{G}})P(\Theta_{\mathcal{G}}|\mathcal{G})d\Theta_{\mathcal{G}},
{%endmath%}

where $$P(D\mid| \mathcal{G} \Theta_{\mathcal{G}})$$ is the probability of the data given the network structure and parameters, and $$P(\Theta_{\mathcal{G}}\mid \mathcal{G})$$ is the prior probability of the parameters. When the prior probability is specified as a Dirichlet distribution,

{%math%}
P(D|\Theta_{\mathcal{G}}) = \prod_{i} \prod_{\pi_i} \left[ \frac{\Gamma(\sum_j N'_{i,\pi_i,j})}{\Gamma(\sum_j N'_{i,\pi_i,j} + N_{i,\pi_i,j} )} \prod_{j}\frac{\Gamma(N'_{i,\pi_i,j} + N_{i,\pi_i,j})}{\Gamma(N'_{i,\pi_i,j})}\right].
{%endmath%}

Here $$\pi_i$$ refers to the parent configuration of the variable $$i$$ and $$N_{i,\pi_i,j}$$ is the count of variable $$i$$ taking value $$j$$ with parent configuration $$\pi_i$$. $$N'$$ represents the counts in the prior respectively.

With a prior for the graph structure $$P(\Theta_{\mathcal{G}})$$ (say, a uniform one), the BD score is defined as 

{%math%}
\log P(D|\Theta_{\mathcal{G}}) + \log P(\Theta_{\mathcal{G}}).
{%endmath%}

Notice there is no penalty term appending to the BD score due to that it will penalize the overfitting implicitly via the integral over parameter space.

### Search algorithms

The most common choice for search algorithms are local search and  greedy search. 

For local search algorithm, it starts with an empty graph or a complete graph. At each step, it attempts to change the graph structure by a single operation of adding an edge, removing an edge or reversing an edge. (Of course, the operation should preserve the acyclic property.) If the score increases, then it adopts the attempt and does the change, otherwise it makes another attempt. 

For greedy search (namely the K3 algorithm), we first assume a topological order of the graph. For each variable, we restrict its parent set to the variables with a higher order. While searching for parent set for each variable, it takes a greedy approach by adding the parent that increases the score most until no improvement can be made.  

Although both approach are computational tractable, neither of them have a guarantee of the quality of the graph that we end up with. The graph space is highly "non-convex" and both algorithm might get stuck at some sub-optimal regions.


### Constraint-based approach

The constraint-based case employs the independence test to identify a set of edge constraints for the graph and then finds the best DAG that satisfies the constraints. For example, we could distinguish V-structure and fork-structure by doing an independence test for the two variables on the sides conditional on the variable in the middle. This approach works well with some other prior (expert) knowledge of structure but requires lots of data samples to guarantee testing power. So it is less reliable when the number of sample is small.

### Recent Advances

In this section, we will briefly introduce two recent algorithms for graph search: order-search (OS) approach and integer linear programming (ILP) approach.

The OS approach, as the name refers, conducts a search over the topological orders and the search over graph space at the same time. The K3 algorithm assumes a topological order in advance and do the search only over the graphs that obey the topological order. When the order specified is a poor one, it may end with a bad graph structure (with a low graph score). The OS algorithm resolves this problem by doing search over orders at the same time. It shifts two adjacent variable in an order at each step and employs the K3 algorithm as a sub-routine. 

The ILP approach encodes the graph structure, scoring and the acyclic constraints into a linear programming problem. Thus it can utilize the state-of-art integer programming solver. But this approach requires a bound on the maximum number of node parents in the graph (say to be 4 or 5). Otherwise, the number of constraints in the ILP will explode and the computation will be intractable.


<br/>

|[Index](../../) | [Previous](../bayesianlearning) |  [Next](../../extras/vae)|