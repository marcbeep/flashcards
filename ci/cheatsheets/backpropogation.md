# Backpropogation Examples

> Reference Tutorial 8

# QUESTION 1

---

## Part A

---

## Problem restatement

**Architecture**

- **Inputs:** $x_{1},x_{2}$
- **Hidden neuron:** $a^{1}$
- **Output neuron:** $a^{2}$ – which is also the network output $y$

$$
\begin{aligned}
a^{1} &= \phi_{1}\Bigl(w^{1}_{1}x_{1}+w^{1}_{2}x_{2}\Bigr) \\
a^{2} &= \phi_{2}\bigl(w^{2}a^{1}\bigr) \\
y &= a^{2}
\end{aligned}
$$

- $\phi_{1}(x)=\alpha x$ (linear)
- $\phi_{2}(x)=\dfrac{1}{1+e^{-x}}$ (sigmoid)

**Parameters & input**

$$
w^{1}_{1}=1,\; w^{1}_{2}=0.1,\; w^{2}=0.1,\;
\alpha=0.5,\;
(x_{1},x_{2})=(1,2)
$$

**Task** – For each neuron, compute its **induced local field** $v$ (the net input) and its **output** $a$.

---

## Step-by-step plan

1. **Hidden neuron local field** $v^{1}=w^{1}_{1}x_{1}+w^{1}_{2}x_{2}$
2. **Hidden neuron output** $a^{1}=\phi_{1}(v^{1})=\alpha v^{1}$
3. **Output neuron local field** $v^{2}=w^{2}a^{1}$
4. **Output neuron output** $a^{2}=\phi_{2}(v^{2})=\dfrac{1}{1+e^{-v^{2}}}$

---

## Calculations

| Step | Expression                     | Numerical value |
| ---- | ------------------------------ | --------------- |
| 1    | $v^{1}=1\cdot1+0.1\cdot2$      | $1.2$           |
| 2    | $a^{1}=0.5\times1.2$           | **0.6**         |
| 3    | $v^{2}=0.1\times0.6$           | 0.06            |
| 4    | $a^{2}=\dfrac{1}{1+e^{-0.06}}$ | **≈ 0.5150**    |

---

## Results

| Neuron           | Induced field $v$ | Output $a$ |
| ---------------- | ----------------- | ---------- |
| Hidden $a^{1}$   | 1.2               | 0.6        |
| Output $a^{2}=y$ | 0.06              | **0.515**  |

So, given the input $(1,2)$ the network produces an output of **≈ 0.515**.

---

## Part B

Back-propagation for the single training pair $(x_{1},x_{2})=(1,2),\;d=1$

---

### Plan for the gradient calculation (words first)

1. **Output-layer error signal**
   $\displaystyle\delta^{2}=\frac{\partial E}{\partial v^{2}}          =(y-d)\,\phi_{2}'(v^{2})$

2. **Gradient for the second-layer weight**
   $\displaystyle\frac{\partial E}{\partial w^{2}}=\delta^{2}\,a^{1}$

3. **Back-propagate to hidden layer**
   $\displaystyle\delta^{1}=\delta^{2}\,w^{2}\,\phi_{1}'(v^{1})$

4. **Gradients for the first-layer weights**
   $\displaystyle
     \frac{\partial E}{\partial w^{1}_{1}}=\delta^{1}x_{1},\qquad
     \frac{\partial E}{\partial w^{1}_{2}}=\delta^{1}x_{2}$

_(Here $\phi_{1}'(x)=\alpha$ and $\phi_{2}'(x)=\sigma(x)\,[1-\sigma(x)]$.)\_

---

### Numeric evaluation

| Step                    | Formula                                         | Result        |
| ----------------------- | ----------------------------------------------- | ------------- |
| Output-layer derivative | $ \delta^{2}=(y-d)\,y(1-y)$                     | $-0.121142$   |
| Gradient $w^{2}$        | $\partial E/\partial w^{2}=\delta^{2}a^{1}$     | **−0.072685** |
| Hidden-layer signal     | $\delta^{1}=\delta^{2}\,w^{2}\,\alpha$          | $-0.006057$   |
| Gradient $w^{1}_{1}$    | $\partial E/\partial w^{1}_{1}=\delta^{1}x_{1}$ | **−0.006057** |
| Gradient $w^{1}_{2}$    | $\partial E/\partial w^{1}_{2}=\delta^{1}x_{2}$ | **−0.012114** |

---

### Weight-update with learning-rate $\eta = 0.1$

---

## Part C

### Update rule

$$
w^{\text{new}} \;=\; w^{\text{old}} \;-\; \eta \,\frac{\partial E}{\partial w}
$$

(the “ minus ” sign moves weights **against** the gradient).

---

| Weight      | Old value | Gradient $\dfrac{\partial E}{\partial w}$ | Increment $-\eta\,\partial E/\partial w$ | **New value** |
| ----------- | --------- | ----------------------------------------- | ---------------------------------------- | ------------- |
| $w^{1}_{1}$ | 1         | $-0.006057$                               | $+0.000606$                              | **1.000606**  |
| $w^{1}_{2}$ | 0.100     | $-0.012114$                               | $+0.001211$                              | **0.101211**  |
| $w^{2}$     | 0.100     | $-0.072685$                               | $+0.007269$                              | **0.107269**  |

_(Values are rounded to 6 d.p. for readability.)_

So after one gradient-descent update with $\eta = 0.1$:

$$
\boxed{w^{1}_{1}=1.000606,\;w^{1}_{2}=0.101211,\;w^{2}=0.107269}
$$

# QUESTION 2

## Part A

---

## 🎯 **Q2 Problem Restatement**

### 🧩 Network structure

**Inputs:** $x_1, x_2$
**Hidden layer neurons:** $a^1_1, a^1_2$
**Output neuron:** $a^2 = y$

Equations:

$$
\begin{aligned}
a^1_1 &= \phi_1(w^1_{11} x_1 + w^1_{21} x_2) \\
a^1_2 &= \phi_1(w^1_{12} x_1 + w^1_{22} x_2) \\
a^2 &= \phi_2(w^2_1 a^1_1 + w^2_2 a^1_2) \\
y &= a^2
\end{aligned}
$$

---

### 🧮 **Activation functions**

- Hidden layer $a^1_1, a^1_2$ : **sigmoid**

$$
\phi_1(x) = \frac{1}{1 + e^{-x}}
$$

- Output layer $a^2$ : **ReLU**

$$
\phi_2(x) =
\begin{cases}
x & x \geq 0 \\
0 & x < 0
\end{cases}
$$

---

### 📋 **Weights and inputs**

| Weight     | Value |
| ---------- | ----- |
| $w^1_{11}$ | 1     |
| $w^1_{12}$ | 0.1   |
| $w^1_{21}$ | 0.3   |
| $w^1_{22}$ | 0.4   |
| $w^2_1$    | 0.1   |
| $w^2_2$    | 0.2   |

Inputs:

$$
x_1 = 1,\; x_2 = 2
$$

---

## ✅ **Plan for solving (Q2a)**

1. **Compute induced local fields $v^1_1, v^1_2$** for hidden neurons:

$$
v^1_1 = w^1_{11} x_1 + w^1_{21} x_2
$$

$$
v^1_2 = w^1_{12} x_1 + w^1_{22} x_2
$$

2. **Compute outputs $a^1_1, a^1_2$** using sigmoid:

$$
a^1_1 = \phi_1(v^1_1),\; a^1_2 = \phi_1(v^1_2)
$$

3. **Compute induced field $v^2$** for output neuron:

$$
v^2 = w^2_1 a^1_1 + w^2_2 a^1_2
$$

4. **Compute output $a^2 = y$** using ReLU:

$$
a^2 = \phi_2(v^2)
$$

---

You’ve got it.
Let’s go through **Q2(b)** step by step **with full calculations**, like before.

---

## 🎯 **RECAP of values from (a)**

| Value              | Result |
| ------------------ | ------ |
| $a^1_1$            | 0.8320 |
| $a^1_2$            | 0.7109 |
| $a^2 = y$          | 0.2254 |
| Desired output $d$ | 0.5    |

---

## Part B

## ✅ **Backpropagation plan**

### 💡 Notes:

- **Output layer:** ReLU activation
    $\phi_2'(x) = 1$ if $x > 0$, else $0$. (Since $v^2 = 0.2254 > 0$, $\phi_2'(v^2)=1$)
- **Hidden layer:** sigmoid
    $\phi_1'(x) = a(1 - a)$, where $a = \phi_1(x)$

We will calculate:

1. **Output error signal** $\delta^2$
2. **Output layer weight gradients** $\partial E/\partial w^2_1, w^2_2$
3. **Hidden neuron error signals** $\delta^1_1, \delta^1_2$
4. **Hidden layer weight gradients** $\partial E/\partial w^1_{ij}$

---

## ✅ **Step 1: Output error signal**

$$
E = \tfrac{1}{2}(d - y)^2 = \tfrac{1}{2}(0.5 - 0.2254)^2 = 0.03767
$$

$$
\delta^2 = (y - d) \cdot \phi_2'(v^2) = (0.2254 - 0.5) \cdot 1 = -0.2746
$$

---

## ✅ **Step 2: Output weight gradients**

$$
\frac{\partial E}{\partial w^2_1} = \delta^2 \cdot a^1_1 = -0.2746 \cdot 0.8320 = -0.2284
$$

$$
\frac{\partial E}{\partial w^2_2} = \delta^2 \cdot a^1_2 = -0.2746 \cdot 0.7109 = -0.1953
$$

---

## ✅ **Step 3: Hidden neuron error signals**

### For hidden neuron 1:

$$
\phi_1'(v^1_1) = a^1_1(1 - a^1_1) = 0.8320(1 - 0.8320) = 0.1395
$$

$$
\delta^1_1 = \delta^2 \cdot w^2_1 \cdot \phi_1'(v^1_1) = -0.2746 \cdot 0.1 \cdot 0.1395 = -0.00383
$$

### For hidden neuron 2:

$$
\phi_1'(v^1_2) = a^1_2(1 - a^1_2) = 0.7109(1 - 0.7109) = 0.2056
$$

$$
\delta^1_2 = \delta^2 \cdot w^2_2 \cdot \phi_1'(v^1_2) = -0.2746 \cdot 0.2 \cdot 0.2056 = -0.01129
$$

---

## ✅ **Step 4: Hidden layer weight gradients**

### For neuron 1 weights:

$$
\frac{\partial E}{\partial w^1_{11}} = \delta^1_1 \cdot x_1 = -0.00383 \cdot 1 = -0.00383
$$

$$
\frac{\partial E}{\partial w^1_{21}} = \delta^1_1 \cdot x_2 = -0.00383 \cdot 2 = -0.00766
$$

### For neuron 2 weights:

$$
\frac{\partial E}{\partial w^1_{12}} = \delta^1_2 \cdot x_1 = -0.01129 \cdot 1 = -0.01129
$$

$$
\frac{\partial E}{\partial w^1_{22}} = \delta^1_2 \cdot x_2 = -0.01129 \cdot 2 = -0.02258
$$

---

## 🎯 **FINAL ANSWER: Gradients**

| Weight     | Gradient |
| ---------- | -------- |
| $w^2_1$    | -0.2284  |
| $w^2_2$    | -0.1953  |
| $w^1_{11}$ | -0.00383 |
| $w^1_{21}$ | -0.00766 |
| $w^1_{12}$ | -0.01129 |
| $w^1_{22}$ | -0.02258 |

---

## Part C

Now let’s apply the **weight update rule** to all weights using $\eta = 0.1$.

### ✅ **Weight update formula**

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial E}{\partial w}
$$

---

### 🎯 **Original weights**

| Weight     | Value |
| ---------- | ----- |
| $w^2_1$    | 0.1   |
| $w^2_2$    | 0.2   |
| $w^1_{11}$ | 1     |
| $w^1_{21}$ | 0.3   |
| $w^1_{12}$ | 0.1   |
| $w^1_{22}$ | 0.4   |

---

### ✅ **Apply updates**

| Weight     | Old | Gradient | Change    | New        |
| ---------- | --- | -------- | --------- | ---------- |
| $w^2_1$    | 0.1 | -0.2284  | +0.02284  | **0.1228** |
| $w^2_2$    | 0.2 | -0.1953  | +0.01953  | **0.2195** |
| $w^1_{11}$ | 1   | -0.00383 | +0.000383 | **1.0004** |
| $w^1_{21}$ | 0.3 | -0.00766 | +0.000766 | **0.3008** |
| $w^1_{12}$ | 0.1 | -0.01129 | +0.001129 | **0.1011** |
| $w^1_{22}$ | 0.4 | -0.02258 | +0.002258 | **0.4023** |

(_values rounded to 4 decimal places for clarity_)

---

### 🎉 **Final updated weights**

$$
\boxed{
w^2_1 = 0.1228,\;
w^2_2 = 0.2195,\;
w^1_{11} = 1.0004,\;
w^1_{21} = 0.3008,\;
w^1_{12} = 0.1011,\;
w^1_{22} = 0.4023
}
$$

---
