---
title: "Actor Critic Algorithms"
permalink: /cs285/hw2/actor-critic-algorithms
date: 2021-02-07T10:00:00-04:00
excerpt: "Learning actor critic algorithms"
categories:
  - CS285
tags:
  - CS285
  - Machine Learning
  - Deep Reinforcement Learning
toc: true
toc_sticky: true
sidebar:
  nav: "cs285"
---

## Improving the Policy Gradient

Recall $$\hat{Q}_{i,t}$$ from the last lecture. $$\hat{Q}_{i,t}$$ is the estimate of the expected reward if you take action $$\mathbf{a}_{i, t}$$ in state $$\mathbf{s}_{i, t}$$. We can improve this estimate by replacing it with its estimate:

$$
    \hat{Q}_{i,t} \approx \sum_{t'=t}^T E_{\pi_\theta} \left[ r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) \mid \mathbf{s}_{t}, \mathbf{a}_{t} \right]
$$

If we plug this back into the reward from last lecture, we can still use the baseline function (this time adapted to be the average $$\hat{Q}_{i,t}$$ value $$ b_t = \frac{1}{N} \sum_i Q(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}) $$) to perform better than average. The baseline depending on action leads to bias, but we can improve even further by making it depend on the state. This leads to the following function for the baseline:

$$
    V(\mathbf{s}_t) = E_{\mathbf{a}_t \sim \pi_\theta (\mathbf{a}_t \mid \mathbf{s}_t)} Q(\mathbf{s}_t, \mathbf{a}_t)
$$

This is the average reward over all the possibilities that start in a given state--the value function! Plugging it back in we get the equation below, where $$Q - V$$ makes sense because it represents your estimate of how much better $$\mathbf{a}_{i, t}$$ is than the average action in $$\mathbf{s}_{i, t}$$.

$$
    \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \left[ \nabla_\theta\log \pi_\theta (\mathbf{a}_{i,t} \mid \mathbf{s}_{i,t})(Q(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}) - V(\mathbf{s}_{i, t})) \right]
$$

$$Q - V$$ is so important that it is called the Advantage function. $$Q$$, $$V$$, and $$A$$ are often written with the superscript $$\pi$$ to denote that they rely on the policy $$\pi$$. To recap the three:

1. $$Q^\pi (\mathbf{s}_{t}, \mathbf{a}_{t})$$: Total reward from taking $$\mathbf{a}_{t}$$ in $$\mathbf{s}_{t}$$
2. $$V^\pi (\mathbf{s}_{t})$$: Total reward from $$\mathbf{s}_{t}$$
3. $$A^\pi (\mathbf{s}_{t}, \mathbf{a}_{t})$$: How much better $$\mathbf{a}_{t}$$ is

This provides us with a large reduction in variance with the cost of a small increase in bias.

## Policy Evaluation

Which of $$Q^\pi$$, $$V^\pi$$, and $$A^\pi$$ should we fit and what should we fit it to? We'll choose $$V^\pi$$ since it's only dependent on $$\mathbf{s}_{t}$$ and the other two functions can be approximated by it as follows:

$$
\begin{align}
    Q^\pi (\mathbf{s}_{t}, \mathbf{a}_{t}) &\approx r(\mathbf{s}_{t}, \mathbf{a}_{t}) + V^\pi(s_{t+1}) \\
    A^\pi (\mathbf{s}_{t}, \mathbf{a}_{t}) &\approx r(\mathbf{s}_{t}, \mathbf{a}_{t}) + V^\pi(s_{t+1}) - V^\pi(s_t)
\end{align}
$$

Fitting $$V^\pi(s_t)$$ is called policy evaluation. $$J(\theta)$$ can be expressed as

$$
    J(\theta) = E_{\mathbf{s}_{1}} \sim p(\mathbf{s}_{1}) \left[ V^\pi(\mathbf{s}_{1}) \right]
$$

How can we perform policy evaluation? We can use a NN function approximator ($$\phi$$) for $$V^\pi$$ which uses the supervised regression algorithm $$\mathcal{L}(\phi) = \frac{1}{2} \sum_i \left\Vert \hat{V}_\phi^\pi(\mathbf{s}_i) - y_i \right\Vert ^2 $$ on the training data $$\left{\left( s_{i, t}, y_{i, t} \right)\right}. How can we find $$y_{i, t}$$?

### Monte Carlo Method

The Monte Carlo target estimates it using a single sample: $$ y_{i, t} = \sum_{t'=t}^T r(\mathbf{s}_{i, t'}, \mathbf{a}_{i, t'}) $$

### Ideal Target

The ideal value is:

$$
\begin{align}
    y_{i, t} &= \sum_{t'=t}^T E_{\pi_\theta} \left[ r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) \ mid \mathbf{s}_{i,t)} \right] \\
    &\approx r(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}) + V^\pi (\mathbf{s}_{i, t+1}) \\
    &\approx r(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}) + \hat{V}^\pi_\phi (\mathbf{s}_{i, t+1})
\end{align}
$$

Note that we're directly using the previous fitted value function $$\hat{V}^\pi_\phi (\mathbf{s}_{i, t+1})$$, which reduces variance but slightly lowers accuracy. This is called the _bootstrapped_ estimate.
