---
title: "Policy Gradients"
permalink: /cs285/hw2/policy-gradients
date: 2021-01-30T09:00:00-04:00
excerpt: "Exploring policy gradients"
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

## REINFORCE

Given the finite-horizon case of the RL objective:

$$
    \theta^* = \textrm{argmax}_\theta \sum_{t=1}^T E_{(s_t,a_t) \sim p_\theta(s_t,a_t)} \left[  r(s_t, a_t) \right]
$$

We can rewrite the expectation as:

$$
  J(\theta) = E_{(s_t,a_t) \sim p_\theta(s_t,a_t)} \left[  r(s_t, a_t) \right] \approx \frac{1}{N}\sum_i\sum_tr(s_{i,t},a_{i_t})
$$

This can be approximated by making rollouts of our policy, running the policy $$N$$ times to collect $$N$$-sampled trajectories resulting in the approximation above. A bigger $$N$$ yields a more accurate estimation of the expected value. Now, we don't just want to estimate the objective, we actually want to _improve_ it, by estimating it's derivative.

**Math Note**: $$\sum_{t=1}^T r(s_{i,t},a_{i_t})$$ will be simplified as $$r(\tau)$$ from here on out.
{: .notice--info}

If continuous, we can expand the expectation out to the integral (sum, if discrete) of the products of the probability and the value:

$$
  J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ r(\tau) \right] = \int p_\theta(\tau)r(\tau)d\tau
$$

The gradient of which is:

$$  
  \nabla_\theta J(\theta) = \int\nabla_\theta p_\theta(\tau)r(\tau)d\tau
$$

**Math Note**: The following identity is useful at several points in this lecture $$x\nabla_\theta log x = x \frac{\nabla_\theta x}{x}$$.
{: .notice--info}

The identity above can be used to simplify the above equation to:

$$
  \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \left( \sum_{i=1}^T \nabla_\theta \log\pi_\theta(a_t \mid s_t) \right) \left( \sum_{i=1}^T r(a_t, s_t)  \right) \right]
$$

Recall the maximum likelihood equation used in supervised learning:

$$
  \nabla_{\theta} J_{\mathrm{ML}}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} | \mathbf{s}_{i, t}\right)\right)
$$

Our equation for the gradient is a reward-weighted maximum likelihood objective!

Thus, we come to the REINFORCE algorithm:

1. Sample $$\{\tau^i\}$$ from $$\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)$$ (run the policy).
2. Use the simplified equation above to approximate $$\nabla_\theta J(\theta)$$.
3. Take a step of gradient descent $$\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)$$.

This algorithm does not require the initial state distribution or the transition probabilities.
{: .notice--warning}

This algorithm does not actually use the Markov property, so it can be used in partially observed MDPs.
{: .notice--warning}

## Reducing Variance

The main problem with policy gradient methods is high variance. In the image below where rewards are denoted with green lines we would expect to see the reward function move from the blue line to the blue dotted line.

![variance movement 1](/assets/img/cs285/hw1/variance1.png)

If we added an offset to the rewards we would expect the behavior below:

![variance movement 2](/assets/img/cs285/hw1/variance2.png)

We can use the concept of _causality_ to reduce the variance of the policy gradient. Causality is the idea that the policy at time $$t'$$ cannot affect the reward at time $$t$$ when $$t < t'$$. This allows us to simplify the equation for $$\nabla_\theta J(\theta)$$ to:

$$
  \nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t}) \hat{Q}_{i,t}
$$

Where $$\hat{Q}_{i,t}$$ is called the "_reward to go_":

$$
  \hat{Q}_{i,t} = \sum_{t'=t}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} | \mathbf{s}_{i, t}\right)
$$

## Baselines

We want to center rewards around $$0$$, and to do so we'll subtract a quantity $$b$$ from them:

$$
  \nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \log p_{\theta}(\tau)[r(\tau) - b]
$$

Where $$b$$ is the average reward:

$$
  b = \frac{1}{N} \sum_{i=1}^{N} r(\tau)
$$

Subtracting this baseline does not bias the expectation. The average reward, though good, is not actually the best baseline. The best baseline is the expected reward, weighted  by magnitude values:

$$
  b* = \frac{\mathbb{E}[(\nabla_{\theta} \log p_{\theta}(\tau))^2r(\tau)]}{\mathbb{E}[(\nabla_{\theta} \log p_{\theta}(\tau))^2]}
$$
