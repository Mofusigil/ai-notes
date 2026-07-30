---
title: 我的第一篇博客
published: 2026-07-30
description: 介绍这个博客以及接下来计划记录的内容
image: ""
tags: [Blog]
category: 随笔
draft: false
---

## 为什么创建这个博客

这个博客主要记录我的学习过程。

## 学习方向

Transformer 的注意力复杂度为 $O(n^2d)$。

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

```python title="attention.py" {3-4}
import torch

scores = query @ key.transpose(-2, -1)
attention = torch.softmax(scores, dim=-1)
output = attention @ value
```