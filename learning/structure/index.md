---
layout: post
title: Structure learning for Bayesian networks
---

We consider estimating the graphical structure for a Bayesian network from a dataset.
The task is challenging for at least two reasons.
First, the set of directed acyclic graphs is exponentially large in the number of variables.
Second, the graph structure need not be _identifiable_.
In other words, two different graphs may by _I-equivalent_ and hence induce the same set of conditional independence assumptions.

This second challenge is closely related to the fact that we do not apply causal interpretations to the edges learned, since the techniques we consider here are statistical in nature and can only detect association in the distribution or dataset of interest. 
One way to keep this subtlety in mind is to remember that two Bayesian networks with different edge orientations may still represent the same distribution.

Before discussing approaches, we emphasize the contrast between these challenges and our pleasant results on parameter learning for a Bayesian network _given_ the directed acyclic graph (see [Learning in directed models](../directed/)).
There we supposed that we had elicited a graph from a domain expert, constructed it using our own (causal) intuition, or asserted it to simplify learning and inference.
We will see that this last point---the accuracy-efficiency trade-off for learning and inference---is also relevant for structure learning.

## Approaches

We briefly touch on two broad approaches to structure learning: (1) constraint-based methods and (2) score-based methods.
Constraint-based approaches use the dataset to perform statistical tests of independence between variables and construct a graph accordingly.
Score-based approaches search for network structures to maximize the likelihood of the dataset while controlling the complexity of the model.

Our modeling goal often guides the choice of approach.
A useful distinction to make is whether we want to estimate parameters of the conditional probability distributions in addition to the graphical structure.
Sometimes, we are primarily interested in the qualitative statistical associations between the variables---namely, the graph itself and the conditional independence assertions it encodes.
Such structure learning is sometimes called _knowledge discovery_.
Since constraint-based techniques can avoid estimating parameters, they are natural candidates.
On the other hand, score based techniques tend to be natural when we also want parameter estimates.
In the sequel, we briefly touch upon constraint-based approaches before turning to score-based approaches.

### Constraint-based approaches for knowledge discovery

Here we briefly describe one simple and natural approach to constraint-based structure learning.
The method extends an algorithm for finding a minimal _I_-map to the case in which we do not know the conditional independence assertions, but estimate them using statistical tests of independence.

First we recall the algorithm for finding a minimal _I_-map.
Suppose $$X_1, \dots, X_n$$ is an ordering of $$n$$ random variables variables satisfying a set of conditional independence assertions $$\mathcal{I}$$.
For $$i = 1,\dots, n$$, define $$\mathbf{A}_i$$ to be a minimal subset of $$\{ X_1, \dots, X_{i-1}\}$$ satisfying 

$$
p(X_i | X_1, \dots, X_{i-1}) = p(X_i | \mathbf{A}_i).
$$

Then the directed acyclic graph $$G$$ defined by the parent function $$\text{pa}(X_i) = A_i$$ is a minimal _I_-map for $$\mathcal{I}$$.

There is a natural modification to this procedure for the case in which we have a dataset rather than a set of conditional independence assertions.
First, we select some ordering of the variables either arbitrarily or using domain knowledge.
Second, for subsets $$\mathbf{U}$$ of the set of variables $$\{X_1, \dots, X_{i-1}\}$$, the algorithm uses a hypothesis test to decide if 

$$
    X_i \perp (\{X_1, \dots, X_{i-1}\} \setminus \mathbf{U}) \; | \; \mathbf{U}.
$$

The test is performed under the null hypothesis that the conditional independence holds and is usually based on some statistical measure of deviance. 
For example, a $$\chi^2$$ statistic or empirical mutual information.

As usual, the reliability of such techniques suffers when we have limited data.
This situation is exacerbated when the number of variables involved in the test is large.
These approaches tend to work better when domain knowledge is incorporated in deciding the ordering of the variables or asserting conditional independence properties.

### Score-based approaches for simultaneous structure and parameter learning

Suppose $$\mathcal{D} = x^{(1)}, x^{(2)}, \dots, x^{(m)}$$ is a dataset of samples from $$n$$ random variables and $$\mathcal{G}$$ is a nonempty set of directed acyclic graphs.
Employing the principle of maximum likelihood, it is natural to be interested in finding a distribution $$p$$ and graph $$G$$ to

$$
\begin{aligned}
    \underset{p \text{ and } G}{\text{maximize}} \quad & \frac{1}{m} \sum_{i = 1}^{m} \log p(x^{(i)})  \\
    \text{subject to} \quad & p \text{ factors according to } G \in \mathcal{G} \\
\end{aligned}
$$

In other words, among structures in $$\mathcal{G}$$, we are interested in finding the one for which, with an appropriate choice of parameters, we maximize the likelihood of data.


_An approximation perspective._ 
Denote the _empirical (data) distribution_ by $$\hat{p}$$ and the usual Kullback-Leibler (KL) divergence between $$\hat{p}$$ and $$p$$ by $$D_{\text{KL}}(\hat{p}, p)$$.
It can be shown that the above problem is equivalent to 

$$
\begin{aligned}
    \underset{p \text{ and } G}{\text{maximize}} \quad & D_{\text{KL}}(\hat{p}, p)  \\
    \text{subject to} \quad & p \text{ factors according to } G \in \mathcal{G} \\
\end{aligned}
$$

To see this, express the KL-divergence as the likelihood of the dataset plus the entropy of $$\hat{p}$$.
Consequently, we can given an alternative interpretation of the original problem.
It finds the best _approximation_ of the empirical distribution, among those which factor appropriately.

There is always a solution to this optimization problem, but its quality often depends on how one constrains the set $$\mathcal{G}$$.
To see this, suppose $$\mathcal{G}$$ is the set of _all_ directed acyclic graphs.
In this case, _any_ complete directed acyclic graph will be optimal because it encodes no conditional independence assumptions.
In general, given an optimal $$p^\star$$ and $$G^\star$$, any 
graph $$G' \in \mathcal{G}$$ satisfying $$\mathcal{I}(G') \subseteq \mathcal{I}(G^\star)$$ is also optimal. 
The reason is that $$p^\star$$ _also_ factors according to $$G'$$.
Unfortunately, a complete graph (or generally any _dense_ graph) is often an undesirable solution because it models no (or few) conditional independence assertions and has many parameters to estimate.
It may be prone to overfitting and inference may be intractable.

These considerations, coupled with the accuracy-efficiency trade-off, make it natural to control the complexity of $$p$$ by restricting the class $$\mathcal{G}$$ or by adding regularization to the log-likelihood objective.
In other words, we can replace the average log likelihood in the problem above with a real-valued _score_ function $$\text{Score}(G, \mathcal{D})$$ which may trade off between a measure of model fit with a measure of model complexity.
Before discussing methods for solving the general (and difficult) score-based problem, we consider a famous tractable example in which the class $$\mathcal{G}$$ is taken to be the set of directed trees.


## The Chow-Liu algorithm

Here we discuss the celebrated Chow-Liu algorithm, proposed in 1968.

_A bit of history._
Chow and Liu were interested in fitting distributions over binary images of hand-written digits for the purposes of optical character recognition.
As we have seen, the number of parameters grows exponentially in the number of pixels and so they naturally became interested in parsimonious representations.
Specifically, they considered the set of distributions which factor according to some directed tree.
Roughly speaking, they showed that for this class of graphs the aforementioned problem of maximizing likelihood reduces to a maximum spanning tree problem. 
Such problems are famously _tractable_.

_A note on identifiability._ 
If a distribution $$p$$ factors according to a rooted tree with root $$X_r$$, where $$r \in \{1, \dots, n\}$$, then it factors according to the same tree rooted at any other variable $$X_i$$, where $$i = 1, \dots, n$$ and $$i \neq r$$.
In other words, two such rooted trees with the same skeleton but different roots are _I-equivalent_. To see this, notice that they have the same skeleton and same _v-structures_.
Hence, we say that the root of the graph is not _identifiable_.
By this we mean that two different choices of root give the same distribution, and so from the distribution alone we can not determine the root.
This is related to the fact that we do not apply _causal_ interpretations to the edges in structure learning. 
These techniques only involve statistical association apparent in the distribution.
We see below that, in Chow and Liu's formulation, the choice of root is immaterial to maximizing the likelihood.

### Chow and Liu's solution

Suppose we have a dataset $$x^{(1)}, \dots, x^{(m)}$$ in some finite set $$\mathcal{S} = \prod_{i = 1}^{n} S_i$$, where $$S_i$$ are each finite sets for $$i = 1, \dots, n$$.
As usual, we define the _empirical distribution_ $$\hat{p}$$ on $$\mathcal{S}$$ so that $$\hat{p}(x)$$ is the number of times $$x$$ appears in the dataset.
We consider the above optimization for the case in which $$\mathcal{G}$$ is the set of directed trees.
Chow and Liu's solution has two steps.

_Step 1: optimal distribution for a given tree._ 
First, we fix a directed tree $$T$$ and then maximize the log likelihood among all distributions that factor according $$T$$.
The solution to this problem (see [Learning in directed models](../directed/)) is to pick the conditional probabilities to match the empirical distribution.
Denote the optimal distribution by $$p^\star_T$$. Then

$$
    p^\star_T(X) = \hat{p}(X_r) \prod_{i \neq r} \hat{p}(X_i | X_{\text{pa}(i)})
$$

Here $$X_r$$, with $$r \in \{1, \dots, n\}$$, is the root of the tree and $$i$$ ranges from $$1, \dots, n$$ except $$r$$.

_Step 2: optimal tree._ 
Second, we substitute $$p^\star_T$$ into the objective and optimize $$T$$.
The first step is to express the log likelihood in terms of the empirical distribution as

$$
\begin{aligned}
    \frac{1}{m} \sum_{i = 1}^{m} \log p^\star_T(x) 
    &= \sum_{x \in \mathcal{S}} \hat{p}(x) \log p^\star_T(x) 
\end{aligned}
$$

The right hand side is the negative _cross-entropy_ of $$p^\star_T$$ with respect to $$\hat{p}$$.
Next we can re-write the negative cross-entropy

$$
\begin{aligned}
    \sum_{x \in \mathcal{S}} \hat{p}(x) \log p^\star_T(x)  
    &= -H_{\hat{p}}(X_r) + \sum_{i \neq r} \sum_{x \in \mathcal{S}} \hat{p}(x) \log \hat{p}(x_i | x_{\text{pa}(i)})  \\
    &= -H_{\hat{p}}(X_r) + \sum_{i \neq r} \sum_{x \in \mathcal{S}} \hat{p}(x) \log \frac{\hat{p}(x_i , x_{\text{pa}(i)})}{\hat{p}(x_{\text{pa}(i)})} \frac{\hat{p}(x_i)}{\hat{p}(x_i)} \\
    &= \sum_{i \neq r} I_{\hat{p}}(X_i, X_{\text{pa}(i)}) - \sum_{i = 1}^{n} H_{\hat{p}}(X_i)
\end{aligned}
$$

where $$H_{\hat{p}}(X_i) = -\sum_{x_i \in S_i} \hat{p}(x_i) \log \hat{p}(x_i)$$ is the _entropy_ of the random variable $$X_i$$ and $$I_{\hat{p}}(X_i, X_j) = D_{KL}(\hat{p}(X_i,X_j), \hat{p}(X_i)\hat{p}(X_j))$$ is the mutual information between random variables $$X_i$$ and $$X_j$$, under the distribution $$\hat{p}$$.

The key insight is that the first sum is over the edges of $$T$$, and the second sum of entropies _does not depend on T_.
Thus, all directed trees with the same skeleton have the same objective.
Consequently, we need only find an _undirected_ tree with a set of edges $$E$$ to 

$$
\begin{aligned}
    \text{maximize} \quad & \sum_{\{i, j\} \in E} I_{\hat{p}}(X_i, X_j)  \\
    \text{subject to} \quad & G = (\{1, \dots, n\}, E) \text{ is a tree}
\end{aligned}
$$

We recognize this as a maximum spanning tree problem. 
It has several well-known algorithms for its solution, each with a runtime which grows quadratically in the number of verticies.
Two famous examples include Kruskal's algorithm and Prim's algorithm.
Any such maximum spanning tree, with any node its root, is a solution.

### Chow and Liu's algorithm

1. Compute the mutual information for all pairs of variables $$X_i,X_j$$, where $$i \neq j$$:

    $$
    I_{\hat{p}}(X_i, X_j) =\sum_{x_i,x_j} \hat p(x_i,x_j)\log \frac{\hat{p}(x_i,x_j)}{\hat p(x_i) \hat{p}(x_j)}
    $$

    This symmetric function is an information theoretic measure of the association between $$X_i$$ and $$X_j$$.
    It is zero if $$X_i$$ and $$X_j$$ are independent.
    Recall $$\hat{p}(x_i, x_j)$$ is the proportion of all data points $$x^{(k)}$$ with $$x^{(k)}_i = x_i$$ _and_ $$x^{(k)}_j = x_j$$.

    Suppose we have four random variables $$A, B, C, D$$. 
    Then we may visualize these mutual information weights as follows:

    {% include maincolumn_img.html src='assets/img/mi-graph.png' %}


2. Find the _maximum_ weight spanning tree: the _undirected_ tree which connects all vertices in the graph and has the highest weight. 

    Again, we may visualize this with four random variables $$A, B, C, D$$
    {% include maincolumn_img.html src='assets/img/max-spanning-tree.png' %}
 
3. Pick any node to be the *root variable*. Direct arrows away from root to obtain a directed tree.
   The conditional probability parameters are chosen as usual, to match those of the empirical distribution.
    We visualize two choices of the four possible roots below:

    {% include maincolumn_img.html src='assets/img/chow-liu-tree.png' %}
 


_A note on complexity._ 
The Chow-Liu Algorithm has a runtime complexity which grows quadratically in the number of variables $$n$$.
To see this, notice that we must compute the mutual information between $$O(n^2)$$ variables. 
Given these weights, we can find a maximum spanning tree using any standard algorithm with runtime $$O(n^2)$$.

## General score-based approach

As we mentioned earlier, every distribution factors according to a complete directed graph.
Thus, the complete graph, if it is a member of $$\mathcal{G}$$, is always optimal.
However, complete graphs are undesirable because (1) they make no conditional independence assertion, (2) their tree width is $$n-1$$---making inference computationally expensive, and (3) they require many parameters---and so suffer from overfitting.
Consequently, we often regularize the log likelihood optimization problem by restricting the class of graphs considered, as in the Chow-Liu approach, or by penalizing the log likelihood objective.

Given a dataset $$\mathcal{D} = x^{(1)}, \dots, x^{(m)}$$, set of graphs $$\mathcal{G}$$, and a score function mapping graphs and datasets to real values, we want to find a graph $$G$$ to

$$
\begin{aligned}
\text{maximize} & \quad \text{Score}(G, \mathcal{D}) \\
\text{subject to} & \quad G \in \mathcal{G}
\end{aligned}
$$

Here we did not include the distribution $$p$$ factoring according to $$G$$ as an optimization variable.
This is because the standard practice is to associate the distribution which maximizes the dataset likelihood with $$G$$ (see [Learning in directed models](../directed))---the likelihood value obtained by this distribution is often used in computing the score.

In general, this is a difficult problem.
In the absence of additional structure, one must exhaustively search the set $$\mathcal{G}$$, which may be large.
Such an exhaustive, so-called brute-force search, is often infeasible.
As a result, heuristic search algorithms for exploring the set $$\mathcal{G}$$ are usually employed.

To summarize, score-based approaches are often described by specifying the score metric and a search algorithm. 

### Score metrics


Denote the log-likelihood obtained by the maximum likelihood distribution factoring according to $$G$$ by by $$\text{LL}(D \mid G)$$.
Often, score metrics take the form

$$ \text{Score}(G, \mathcal{D}) = \underbrace{\text{LL}(\mathcal{D} \mid G)}_{\text{fit}} - \underbrace{R(G, \mathcal{D})}_{\text{complexity}} $$

where the function $$R$$ is a regularizer measuring the complexity of the model.

_Common regularizers._
Often $$R$$ is a function of the size of the dataset and the number of parameters.
The former is denoted $$\lvert \mathcal{D} \rvert$$ and the latter is denoted by $$\lVert \mathcal{G} \rVert$$, since two categorical Bayes nets with the same graph structure have the same number of parameters.
The regularizer often has the form

$$
    R(G, \mathcal{D}) = \psi(\lvert \mathcal{D} \rvert) \lVert G \rVert
$$

where $$\psi$$ is a real-valued function of the dataset size.
The choice $$\psi(\lvert D \rvert) = 1$$ is called the _Akaike Information Criterion_ (AIC) and the choice $$\psi(\lvert \mathcal{D} \rvert) = \ln(n)/2$$ is called the _Bayesian Information Criterion_ (BIC).
In the former, the log-likelihood function grows linearly in the dataset size and will dominate the penalty term, and so the model complexity criterion will only be used to distinguish model with similar log-likelihoods.
In the latter, the the influence of model complexity grows logarithmically in the dataset size, which is heavier than in the AIC, but still allows the log-likelihood term to dominate as the dataset size grows large.
There are several desiderata associated with these scores.

### Search algorithms

In the absence of additional structure in $$\mathcal{G}$$, and in light of the computational infeasibility of exhaustive search, a local search algorithm is often employed.
Although such methods are not guaranteed to provide globally optimal graph structures, they are often designed to be fast and may perform well in practice.
We briefly outline two approaches here.

_Local structure search._
One such approach begins with a given graph, and at each step of an iterative procedure, modifies the graph by (a) adding an edge (b) removing an edge or (c) flipping an edge. 
Here these operations are only considered if the modified graph remains acyclic.
If the score of the new structure improves upon the current, the new structure is adopted. 
Otherwise, a different operation is attempted.
The procedure can be terminated via a variety of stopping criterion; for example, once no operation exists which improves the score.
Since the conditional probability tables only change locally, recomputing the score may be fast at each iteration.

_K3 algorithm._
A second approach, the K3 algorithm, takes as input an ordering of the variables.
In this order, it searches for a parent set for variable $$X_i$$ from within the variables $$\{X_1, \dots, X_{n-1}\}$$.
A greedy approach may be used which builds the parent set by iteratively adding the next parent which most increases the score, until no further improvement can be made or until a maximum number of parents have been added.
This approach is evidently sensitive to the initial variable ordering, and depends on the tractability of finding the parent set, but may still perform well in practice. 

### Other methods

In this section, we briefly mention two other methods for graph search: an order-search (OS) approach and an integer linear programming (ILP) approach.

The OS approach, as its name suggests, conducts a search over the topological orders and the graph space at the same time. The K3 algorithm assumes a topological order in advance and searches only over the graphs that obey the topological order. When the order specified is a poor one, it may end with a bad graph structure (with a low graph score). The OS algorithm resolves this problem by performing a search over orders at the same time. It swaps the order of two adjacent variables at each step and employs the K3 algorithm as a sub-routine.

The ILP approach encodes the graph structure, scoring and the acyclic constraints into a linear programming problem. Thus it can utilize a state-of-art integer programming solver. That said, this approach requires a bound on the maximum number of parents any node in the graph can have (say to be 4 or 5). Otherwise, the number of constraints in the ILP will explode and the computation will become intractable.

<br/>

|[Index](../../) | [Previous](../bayesian) | [Next](../../extras/vae)|
