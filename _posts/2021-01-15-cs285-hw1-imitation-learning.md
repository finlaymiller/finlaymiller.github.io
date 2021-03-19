---
title: "CS285: Imitation Learning"
date: 2021-01-15T15:34:30-04:00
categories:
  - blog
tags:
  - CS285
  - Machine Learning
  - Deep Reinforcement Learning
mathjax: true
---

## Introduction

I'm following along with Berkeley's [CS 285 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) as the start of a self-guided study into Reinforcement Learning. 
This post has my notes on the first four lectures and the first homework assignment.

## Imitation Learning

### Terminology

![tiger](/assets/images/cs285/hw1/tiger.png)

* $t$ -- Time step
* $s_t$ -- State of the system at a given time
* $o_t$ -- Observation resulting from the state. Not necessarily the true state (e.g. cheetah and gazelle and car)
* $a_t$ -- Action (often a distribution)l
* $\pi_\theta(a_t \mid o_t)$ -- Policy based on $o_t$ (or $s_t$) which outputs a distribution over $a_t$ given $o_t$ ($a_t \mid o_t$)
* $\theta$ -- Policy parameters e.g. in a NN, $\theta$ = weights

_Markov Property_: Given a state, one can figure out the distribution over the next state without knowing the previous state. 
Given $s_t$ we can find $p(s_{t+1} \mid s_t, a_t)$ without $s_{t-1}$. 
It is worth noting that observations do not necessarily have the Markov Property.

### Behavior Cloning

![behavior cloning](/assets/images/cs285/hw1/behavior_cloning.png)

* $p_{data}(o_t)$ -- The distribution of observations seen in the training data

Often doesn't work because small errors in the learned model compound and it doesn't know how to recover (e.g. NVIDIA car). 
The mathematical explanation is that when running the expected trajectory we're sampling from the distribution $\pi_\theta(a_t \mid o_t)$, which was trained on the data distribution $p_{data}(o_t)$. 
When errors compound $p_{data}(o_t) \neq p_{\pi_\theta}(o_t)$.

### Data Aggregation [DAgger]

We need $p_{\pi_\theta}(o_t) = p_{data}(o_t)$ to minimize the errors seen in behavior cloning. 
What if, instead of optimizing $p_{\pi_\theta}(o_t)$, we optimized $p_{data}(o_t)$? 
We want to collect training date from $p_{\pi_\theta}(o_t)$ instead of $p_{data}(o_t)$, which we can do by running $\pi_\theta(a_t \mid o_t)$ once we have $a_t$.

1. Train $\pi_\theta(a_t \mid o_t)$ on data from a human $D = \{ o_1, a_1, \dots, o_N, a_N \}$
2. Run $\pi_\theta(a_t \mid o_t)$ to get dataset $D_\pi = \{ o_1,\dots, o_M\}$
3. Get human to label $D_\pi$ with optimal actions $a_t$
4. Merge the two datasets $D \leftarrow D \cup D_\pi$
5. Start again from 1. with new $D$

The problematic step here is 3.

It's possible to make a model without the distributional drift problem, but first: why might we fail to fit the expert?

1. Even if $o_t$ is Markovian ($\pi_\theta(a_t \mid o_t$), the expert (the human) might exhibit non-Markovian behavior ($\pi_\theta(a_t \mid o_1,\dots, o_t$). This can be solved by using an RNN with LSTM
2. The demonstrator might inconsistently choose between different modes in the distribution, which is hard to imitate. The average of two good actions could be a bad action! This can be solved with three techniques:
   1. We could represent the distribution as a mixture of Gaussians  $\pi(a \mid 0)=\sum_iw_iN(\mu_i,\Sigma_i)$. This can get complicated quickly as you add more mixture elements.
   2. We could input a latent variable $\xi ~ N(0,I)$ into the model alongside our images, essentially injecting random noise. Can theoretically represent any distribution, but can be hard to train.
   3. Autoregressive discretization is a happy medium between the two options listed above. It takes advantage of the fact that discrete actions are easily representable by a softmax distribution, sidestepping the multi-modality problem.

How can we "score" actions mathematically? Cost/reward functions! If we wanted to minimize cost (Note that the cost is the negative of the reward) (we do), we would minimize the expectation under a distribution over sequences of states and actions of the sum of the cost function.

$$
 min_\theta \ (E_{s_{1:T},a_{1:T}} \left[ \sum_t c(s_{t},a_{t}) \right])
$$

[
