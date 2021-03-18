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

I'm following along with Berkeley's [CS 285 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) as the start of a self-guided study into Reinforcement Learning. This post has my notes on the first four lectures and the first homework assignment.

## Imitation Learning

### Terminology

![tiger](/assets/images/tiger.png)

1. $t$ -- Time step
2. $s_t$ -- State of the system at a given time
3. $o_t$ -- Observation resulting from the state. Not necessarily the true state (e.g. cheetah and gazelle and car)
4. $a_t$ -- Action (often a distribution)
5. $\pi_\theta( a_t | o_t )$
6. $\theta$ -- Policy parameters e.g. in a NN, $\theta$ = weights

-- Policy based on $o_t$ (or $s_t$ ) which outputs a distribution over $a_t$ given $o_t$ ( $a_t|o_t$ )

[
