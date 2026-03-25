# Micrograd — Scalar Autograd Engine & Neural Network from Scratch

A from-scratch implementation of a **scalar-valued automatic differentiation engine** and a **Multi-Layer Perceptron (MLP)** built on top of it, all in pure Python. This project is designed as an educational deep-dive into how neural networks and backpropagation work under the hood.

---

## Repository Structure

```
micrograd/
│
├── micrograd.ipynb                        # Core autograd engine + single neuron demo
└── micrograd-multi-layer-perceptron.ipynb # Full MLP built on top of the Value engine
```

---

## Notebook 1: `micrograd.ipynb` — Autograd Engine

### Overview

Implements a minimal `Value` class that wraps scalar numbers, tracks mathematical operations to build a **computational graph**, and supports **reverse-mode automatic differentiation** (backpropagation).

### The `Value` Class

Each `Value` object stores:

| Field | Description |
|---|---|
| `data` | The scalar value |
| `grad` | The gradient ∂output/∂self, initialized to 0 |
| `_backward` | Function that propagates gradients to child nodes |
| `_prev` | Set of child nodes that produced this value |
| `_op` | The operation that created this node (for visualization) |

### Supported Operations

| Operation | Method |
|---|---|
| Addition | `a + b`, `__add__`, `__radd__` |
| Multiplication | `a * b`, `__mul__`, `__rmul__` |
| Power | `a ** n`, `__pow__` |
| Division | `a / b`, `__truediv__` |
| Negation | `-a`, `__neg__` |
| Subtraction | `a - b`, `__sub__` |
| Exponentiation | `a.exp()` |
| Tanh activation | `a.tanh()` |

### Backpropagation

Calling `.backward()` on the output node:
1. Builds a **topological ordering** of all nodes in the graph
2. Sets the output node's gradient to `1`
3. Traverses nodes in **reverse topological order**, applying each node's `_backward()` to accumulate gradients via the chain rule

### Single Neuron Demo

```python
# Inputs
x1 = Value(2.0,  label='x1')
x2 = Value(0.0,  label='x2')

# Weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0,  label='w2')

# Bias
b  = Value(6.8814, label='b')

# Forward pass: n = x1*w1 + x2*w2 + b
n = x1*w1 + x2*w2 + b

# Tanh activation (manual via exp)
e = (2 * n).exp()
o = (e - 1) / (e + 1)

# Backward pass
o.backward()
```

**Result:** `o = 0.7071`, with gradients propagated back to all inputs and weights.

### Computation Graph Visualization

Uses **Graphviz** to render the full computation graph as an SVG, displaying each node's `data`, `grad`, and the operation that produced it.

```python
draw_dot(o)  # renders the computation graph inline
```

---


## Key Concepts

- **Computational graph** — every operation creates a node that tracks how values depend on each other
- **Reverse-mode autodiff** — gradients are computed in a single backward pass using the chain rule
- **Topological sort** — ensures gradients accumulate in the correct order during backpropagation
- **Tanh activation** — implemented manually via `exp` to show how composite functions are differentiated
- **MLP from scratch** — `Neuron`, `Layer`, and `MLP` classes built purely on `Value`, with no external ML library
- **Gradient descent** — weights updated manually as `p.data += -lr * p.grad`

---

## References

- [Andrej Karpathy — micrograd (GitHub)](https://github.com/karpathy/micrograd)
- [Andrej Karpathy — Neural Networks: Zero to Hero (YouTube)](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

