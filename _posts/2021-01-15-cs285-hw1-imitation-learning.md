---
title: "CS285: Imitation Learning"
date: 2021-01-15T15:34:30-04:00
categories:
  - blog
tags:
  - CS285
  - Machine Learning
  - Deep Reinforcement Learning
---

## Introduction

I'm following along with Berkeley's [CS 285 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) as the start of a self-guided study into Reinforcement Learning. This post has my notes on the first four lectures and the first homework assignment.

## Imitation Learning

### Terminology

![tiger](/assets/images/cs285/hw1/tiger.png)

* $$t$$ -- Time step
* $$s_t$$ -- State of the system at a given time
* $$o_t$$ -- Observation resulting from the state. Not necessarily the true state (e.g. cheetah and gazelle and car)
* $$a_t$$ -- Action (often a distribution)l
* $$\pi_\theta(a_t \mid  o_t)$$ -- Policy based on $$o_t$$ (or $$s_t$$) which outputs a distribution over $$a_t$$ given $$o_t$$ ($$a_t \mid  o_t$$)
* $$\theta$$ -- Policy parameters e.g. in a NN, $$\theta$$ = weights

_Markov Property_: Given a state, one can figure out the distribution over the next state without knowing the previous state. 
Given $$s_t$$ we can find $$p(s_{t+1} \mid  s_t, a_t)$$ without $$s_{t-1}$$. It is worth noting that observations do not necessarily have the Markov Property.

### Behavior Cloning

![behavior cloning](/assets/images/cs285/hw1/behavior_cloning.png)

* $$p_{data}(o_t)$$ -- The distribution of observations seen in the training data

Often doesn't work because small errors in the learned model compound and it doesn't know how to recover (e.g. NVIDIA car). 
The mathematical explanation is that when running the expected trajectory we're sampling from the distribution $$\pi_\theta(a_t \mid  o_t)$$, which was trained on the data distribution $$p_{data}(o_t)$$. When errors compound $$p_{data}(o_t) \neq p_{\pi_\theta}(o_t)$$.

### Data Aggregation (DAgger)

We need $$p_{\pi_\theta}(o_t) = p_{data}(o_t)$$ to minimize the errors seen in behavior cloning. What if, instead of optimizing $$p_{\pi_\theta}(o_t)$$, we optimized $$p_{data}(o_t)$$? We want to collect training date from $$p_{\pi_\theta}(o_t)$$ instead of $$p_{data}(o_t)$$, which we can do by running $$\pi_\theta(a_t \mid  o_t)$$ once we have $$a_t$$.

1. Train $$\pi_\theta(a_t \mid  o_t)$$ on data from a human $$D = \{ o_1, a_1, \dots, o_N, a_N \}$$
2. Run $$\pi_\theta(a_t \mid  o_t)$$ to get dataset $$D_\pi = \{ o_1,\dots, o_M\}$$
3. Get human to label $$D_\pi$$ with optimal actions $$a_t$$
4. Merge the two datasets $$D \leftarrow D \cup D_\pi$$
5. Start again from 1. with new $$D$$

The problematic step here is 3.

It's possible to make a model without the distributional drift problem, but first: why might we fail to fit the expert?

1. Even if $$o_t$$ is Markovian ($$\pi_\theta(a_t \mid  o_t$$), the expert (the human) might exhibit non-Markovian behavior ($$\pi_\theta(a_t \mid  o_1,\dots, o_t$$). This can be solved by using an RNN with LSTM
2. The demonstrator might inconsistently choose between different modes in the distribution, which is hard to imitate. The average of two good actions could be a bad action! This can be solved with three techniques:
   1. We could represent the distribution as a mixture of Gaussians  $$\pi(a \mid  0)=\sum_iw_iN(\mu_i,\Sigma_i)$$. This can get complicated quickly as you add more mixture elements.
   2. We could input a latent variable $$\xi ~ N(0,I)$$ into the model alongside our images, essentially injecting random noise. Can theoretically represent any distribution, but can be hard to train.
   3. Autoregressive discretization is a happy medium between the two options listed above. It takes advantage of the fact that discrete actions are easily representable by a softmax distribution, sidestepping the multi-modality problem.

### Cost Functions

How can we "score" actions mathematically? Cost/reward functions! If we wanted to minimize cost (Note that the cost is the negative of the reward) (we do), we would minimize the expectation under a distribution over sequences of states and actions of the sum of the cost function.

$$ min_\theta \ (E_{s_{1:T},a_{1:T}} \left[ \sum_t c(s_{t},a_{t}) \right]) $$

A reasonable reward function could be the log probability of an expert's action:

$$ r(s,a) = log\;p(a = \pi^* (s) \mid  s) $$

$$\pi^*$$ is the unknown expert policy. Another cost function could be a 0-1 loss function, which assigns a 0 if we perfectly match the expert's actions and a 1 in every other case (harsh!):

$$
c(s,a) =
\begin{cases}
0 \;\textrm{if}\; a = \pi^*(s) \\
1 \;\textrm{otherwise} \\
\end{cases}
$$

If we assume that $$\pi_\theta(a\neq\pi^*(s)\mid  s)\leq\epsilon$$ for all $$s\in D_{train}$$\footnote{This is naively assuming that for all states in the training set, the probability of making a mistake at those steps is $\leq\epsilon$, where $\epsilon$ is a small number.} (remember the tightrope walker scenario) then the probability function is:

$$ p_\theta(s_t) = (1-\epsilon)^t p_{train}(s_t)+(1-(1-\epsilon)^t)p_{mistake}(s_t) $$

then, using the identity $(1-\epsilon)^t\geq1-\epsilon t$ for $\epsilon\in [0,1]$ we can simplify:

$$
\begin{split}
    \mid p_\theta(s_t) - p_{train}(s_t)\mid  &= (1 - (1 - \epsilon)^t)\mid p_{mistake}(s_t) - p_{train}(s_t)\mid  \\
    &\leq 2(1 - (1 - \epsilon)^t) \\
    &\leq 2 \epsilon t
\end{split}
$$

the cost then becomes:

$$
\begin{split}
    \sum_t E_{p_\theta(s_t)}[c_t] &= \sum_t \sum_{s_t} p_\theta(s_t)c_t(s_t) \\
    &\leq \sum_t \sum_{s_t} \left[ p_{train}(s_t)c_t(s_t) + \mid p_\theta(s_t) - p_{train}(s_t)\mid c_{max} \right] \\
    &\leq \sum_t(\epsilon + 2 \epsilon t ) \\
    &\leq \epsilon T + 2 \epsilon T^2
\end{split}
$$

Which is $$O(\epsilon T^2)$$, whereas the expectation with DAgger $$p_{train}(s) \rightarrow p_\theta(s)$$ is is $$O(T)$$.

## Reinforcement Learning

### Terminology

* $$M = \{ S, T \}$$ --  Markov chain where:
  * $$S$$: State space ($$s \in S$$)
  * $$T$$: Transition operator (matrix or tensor) denoting the probability of transition between states
* $$M = \{ S, A, T, r \}$$ -- Markov decision process where:
  * $$A$$: Action space ($$a \in A$$)
  * $$r$$: Reward function which is a mapping from the state and action spaces into a real number $$r\: : \: S \times A \rightarrow \Re$$
* $$M= \{ S, A, O, T, E, r \}$$ -- Partially observed Markov decision process where:
  * $$O$$: Observation space space ($$o \in O$$)
  * $$E$$: Emission (or observation) probability $$p(o_t\mid s_t)$$

### The Objective

![reinforcement learning](/assets/images/cs285/hw1/reinforcement_learning.png)

When looking at the system above, we can write down a probability distribution over _trajectories_, where trajectories are sequences of paired actions and states (e.g. $$(s_1, a_1)$$, $$(s_2, a_2)$$ and so on).

$$
\begin{split}
    p_\theta(s_1, a_1, \dots, s_T, a_T) &= p(s_1) \prod^T_{t=1} \pi_\theta(a_t \mid s_t)p(s_{t+1} \mid s_t,a_t) \\
    p_\theta(s_1, a_1, \dots, s_T, a_T) &= p_\theta(\tau) \\
\end{split}
$$

Remember that $$\pi_\theta(a_t \mid s_t)$$ is the probability of an action, and $$p(s_{t+1} \mid s_t,a_t)$$ is the probability of a state transition.

![markov](/assets/images/cs285/hw1/markov.png)

Now we can define an objective for reinforcement learning as an expected value_under_ the trajectory distribution:

$$
    \theta^* = argmax_\theta E_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(s_t, a_t) \right]
$$

We want to maximize the expected value of the sum of rewards over the trajectory. If we augment the Markov chain by bundling states and actions so that $$p((s_{t+1}, a_{t+1}\mid(s_t,a_t)) = p(s_{t+1}\mid s_t,a_t)\pi_\theta(a_{t+1}\mid s_{t+1})$$ we can use the linearity of expectation to rewrite the objective such that:

$$
    \theta^* = argmax_\theta \sum_{t=1}^T E_{(s_t,a_t) \sim p_\theta(s_t,a_t)} \left[ r(s_t, a_t) \right]
$$

Which we can use to find out what happens when $$T=\infty$$.

![markov](/assets/images/cs285/hw1/sato.png)

Does $$p(s_t,a_t)$$ converge to a single distribution as $$k\leftarrow \infty$$? If we assume [aperiodicity and ergodicity](https://towardsdatascience.com/the-intuition-behind-markov-chains-713e6ec6ce92) we can write $$\mu = \mathcal{T}\mu$$, where $$\mu$$ is a stationary distribution. It follows that $$(\mathcal{T}-\mathbf{I})\mu = 0$$ so $$\mu$$ is an eigenvector of $$\mathcal{T}$$ with an eigenvalue of $$1$$. As $$T \leftarrow \infty$$ the sum of the expectations of the marginals becomes dominated by the stationary distribution. We can divide the expectation by $$T$$ to find the average:

$$
    \theta^* = \textrm{argmax}_\theta \frac{1}{T} \sum_{t=1}^T E_{(s_t,a_t) \sim p_\theta(s_t,a_t)} \left[  r(s_t, a_t) \right]
$$

Then take the limit as $$T \leftarrow \infty$$ we get the expected value of the reward under the stationary distribution, where the stationary distribution $$\mu = p_\theta(s,a)$$:

$$
    \theta^* = E_{(s,a) \sim p_\theta(s,a)} \left[  r(s_t, a_t) \right]
$$

Expected values can be continuous in the parameters of corresponding distributions even when the function that we're taking the expectation of is discontinuous. If we have a discontinuous reward function $$r(x)$$, we could have a probability distribution over some action $$\pi_\theta(a)=\theta$$ which is a Bernoulli random variable with parameter $\theta$. The expected value of the reward, $$E_{\pi_\theta} \left[  r(x) \right]$$, is actually smooth in $$\theta$$! The takeaway is that expected values of non-smooth and non-differentiable functions under differentiable and smooth probability distributions, are themselves smooth and differentiable.

### Algorithms

Reinforcement Learning algorithms all generally have the same three part structure:

* _Generate Samples_: Run the policy in the environment, have it interact with your Markov Decision Process, and collect samples (sampled trajectories from the policy-defined trajectory distribution (i.e. run your policy in the environment)).
* _Fit Your Model_: Estimate how well your policy is doing.
* _Improve Your Policy_: Make your policy better given the metric calculated in step 2.

![RLA](/assets/images/cs285/hw1/RLA.png)

Regarding the costs of these steps:


* _Generate Samples_: Depends greatly on the system in question. Real systems generate samples in real time, simulated systems can often run many times faster than real time
* _Fit Your Model_: Also depends on the system! If you're doing a sum of rewards (as you would with a Policy Gradient Algorithm) it can be very fast. If you're fitting an entire neural net (as you would with reinforcement learning with back-propagation), it can be expensive.
* _Improve Your Policy_: Same as 2.

## Q and Value Functions

### The Q Function

To recap: A reinforcement learning objective is often an expectation, defined as a sum over time for every state-action marginal. (Recall that given a joint distribution of two discrete random variables $$X$$ and $$Y$$, the marginal distribution of $$X$$ is the probability distribution of $$X$$ without taking into account $$Y$$.). The expectation can be written as:

$$
    E_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right]
$$

We can write this out recursively as a series of nested expectations, expanding it into this monster:

$$
    E_{s_1 \sim p(s_1)} \left[ E_{a_1 \sim \pi(a_1 \mid  s_1)} \left[ r(s_1, a_1) + E_{s_2 \sim p(s_2 \mid  s_1, a1)} \left[ E_{a_2 \sim \pi(a_2 \mid  s_2)} \left[ r(s_2, a_2) + \dots \mid  s_2 \right] \mid  s_1, a_1 \right] \mid  s1 \right] \right]
$$

Why on earth would anyone do this? Well, what if we knew what went inside the second expectation, defined by some function called $Q$?

$$
    Q(s_1, a_1) = r(s_1, a_1) + E_{s_2 \sim p(s_2 \mid  s_1, a1)} \left[ E_{a_2 \sim \pi(a_2 \mid  s_2)} \left[ r(s_2, a_2) + \dots \mid  s_2 \right] \mid  s_1, a_1 \right]
$$

That monstrous expansion from above becomes

$$
    E_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right] = E_{s_1 \sim p(s_1)} \left[ E_{a_1 \sim \pi(a_1 \mid  s_1)} \left[ Q(s_1, a_1) \mid  s1 \right] \right]
$$

If we know this $$Q$$ function then optimizing the policy $$\pi(a_1 \mid  s_1)$$ at $$t=1$$ is easy, we can just maximize the value of $$Q$$. A more formal definition of the $$Q$$ function is:

$$
   Q^\pi(s_t,a_t) = \sum_{t'=t}^T E_{\pi_\theta} \left[ r(s_{t'}, a_{t'})\mid s_t, a_t \right]
$$

This means that you will get the expected sum of rewards if you roll out your policy from $$s_t, a_t$$.

### The Value Function

Defined below, the value function is essentially the same, except it's conditional only on the state, rather than the state and the action:

$$
   V^\pi(s_t) = \sum_{t'=t}^T E_{\pi_\theta} \left[ r(s_{t'}, a_{t'})\mid s_t \right]
$$

This means that the equation below is the entirety of the reinforcement learning objective!

$$
   E_{s_1\sim p(s_1)} = \left[ V^\pi(s_t) \right]
$$

### Using Q and Value Functions

* If we have a policy $$\pi$$ and we know $$Q^\pi(s,a)$$ then we can improve $$\pi$$ by setting $$\pi'(a\mid s)=1$$ if $$a=argmax_a Q^\pi(s, a)$$. This means that regardless of what $$\pi$$ is, $$\pi'$$ is at least as good, and is probably better.
* We could also compute the gradient to increase the probability of good actions $$a$$ given that, if $$Q^\pi(s,a) > V^\pi(s)$$, then $$a$$ is better than average. We would do this by modifying $$\pi(a\mid s)$$ to increase the probability of $$a$$ if $$Q^\pi(s,a) > V^\pi(s)$$.

## Types of RL Algorithms

### Policy Gradients

Directly differentiate the objective w.r.t. the policy

![policy_gradient](/assets/images/cs285/hw1/policy_gradient.png)

### Model Based

![model_based](/assets/images/cs285/hw1/model_based.png)

After estimating the transition model $$p$$ there are several options for improving the policy:

* Use the model to plan without a policy
  * Trajectory optimization; essentially backpropagation to optimize over actions (mainly continuous)
  * Discrete planning in discrete action spaces e.g. Monte Carlo tree search
* Backpropagation: use the learned model to calculate derivatives of the reward function w.r.t. the policy.
  * Can be tricky with regards to numerical stability
* Use the model to learn a Value or Q function, then use the learned function to improve the policy.

### Value Function Based

$$V(s)$$ and $$Q(s, a)$$ are often NNs

![value_function](/assets/images/cs285/hw1/value_function.png)

[
