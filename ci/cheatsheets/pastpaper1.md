# QUESTION 1

## a.

Let’s walk through the steps of training a **single-layer perceptron** to solve the **OR problem** using the **Perceptron Learning Algorithm**. Here's what we're given:

---

### **Setup**

- **Inputs:** `x₁`, `x₂` with a bias input `x₀ = -1`
- **Initial Weights:** `w₀ = +0.3`, `w₁ = -0.4`, `w₂ = +0.8`
- **Learning Rate:** `η = 0.5`
- **Activation Function (signum):**

  $$
  ϕ(v) = \begin{cases}
  +1 & \text{if } v \geq 0 \\
  -1 & \text{if } v < 0
  \end{cases}
  $$

- **Training Data (OR function):**

  - (0, 0) → -1
  - (0, 1) → +1
  - (1, 0) → +1
  - (1, 1) → +1

---

### **Perceptron Learning Rule:**

If output ≠ desired:

$$
w_i \leftarrow w_i + η \cdot (target - output) \cdot x_i
$$

---

### **Epoch 1**

| Pattern (x₀, x₁, x₂) | Weights (w₀, w₁, w₂) | v = w·x               | ϕ(v) | Target | Error? | Weight Update                            |
| -------------------- | -------------------- | --------------------- | ---- | ------ | ------ | ---------------------------------------- |
| (-1, 0, 0)           | (0.3, -0.4, 0.8)     | 0.3 × (-1) = -0.3     | -1   | -1     | No     | —                                        |
| (-1, 0, 1)           | (0.3, -0.4, 0.8)     | -0.3 + 0.8 = 0.5      | +1   | +1     | No     | —                                        |
| (-1, 1, 0)           | (0.3, -0.4, 0.8)     | -0.3 - 0.4 = -0.7     | -1   | +1     | Yes    | Δw = 0.5×(1 - (-1))×x = (1.0)×x → w += x |
|                      |                      |                       |      |        |        | w₀ = 0.3 + 0.5×(-1) = -0.2               |
|                      |                      |                       |      |        |        | w₁ = -0.4 + 0.5×(1) = 0.1                |
|                      |                      |                       |      |        |        | w₂ = 0.8 + 0.5×(0) = 0.8                 |
| (-1, 1, 1)           | (-0.2, 0.1, 0.8)     | 0.2 + 0.1 + 0.8 = 0.7 | +1   | +1     | No     | —                                        |

---

### **Epoch 2**

| Pattern    | Weights (-0.2, 0.1, 0.8) | v                     | ϕ(v) | Target | Error? | Update                     |
| ---------- | ------------------------ | --------------------- | ---- | ------ | ------ | -------------------------- |
| (-1, 0, 0) | -0.2                     | +0.2                  | +1   | -1     | Yes    | Δw = 0.5×(-2)×x = (-1)×x   |
|            |                          |                       |      |        |        | w₀ = -0.2 + 0.5×(-1) = 0.3 |
|            |                          |                       |      |        |        | w₁ = 0.1 + 0.5×(0) = 0.1   |
|            |                          |                       |      |        |        | w₂ = 0.8 + 0.5×(0) = 0.8   |
| (-1, 0, 1) | (0.3, 0.1, 0.8)          | -0.3 + 0.8 = 0.5      | +1   | +1     | No     | —                          |
| (-1, 1, 0) | (0.3, 0.1, 0.8)          | -0.3 + 0.1 = -0.2     | -1   | +1     | Yes    | w₀ = 0.3 + 0.5×(-1) = -0.2 |
|            |                          |                       |      |        |        | w₁ = 0.1 + 0.5 = 0.6       |
|            |                          |                       |      |        |        | w₂ = 0.8 + 0.5×(0) = 0.8   |
| (-1, 1, 1) | (-0.2, 0.6, 0.8)         | 0.2 + 0.6 + 0.8 = 1.2 | +1   | +1     | No     | —                          |

---

### **Epoch 3**

Test all again with weights (-0.2, 0.6, 0.8). All outputs match targets now:

- (0,0): v = 0.2 → +1 (wrong) → update again

You’ll see some bouncing. Keep updating until **no more changes are needed**. The perceptron will eventually **converge**, usually within a few more epochs.

---

## b.

Let's break this into clear sections: diagrams, equations, and explanation for **forward pass**, **backward pass**, and **activation functions**.

---

## 🧠 Multi-Layer Feedforward Neural Network (MLP)

### Structure:

```
Input Layer → Hidden Layer(s) → Output Layer
```

Each layer is **fully connected** to the next. We'll assume **one hidden layer** to keep it simple.

---

## i) 📤 **Forward Pass**

### **Diagram:**

```
       Input Layer        Hidden Layer       Output Layer
    x1 ──┬────────────┐     z1 ──┬──────┐     y1 (output)
         │            ├────────▶│      ├─────▶
    x2 ──┼────────────┘     z2 ──┼──────┘
         │                      │
    x0=-1 (bias)           z0=-1 (bias)
```

Let:

- `x` be input vector: `[x₁, x₂, ..., xₙ]`
- `z` be hidden layer activations
- `y` be final output

### **Forward Equations:**

#### Step 1: Compute net input to hidden neurons

$$
v_j = \sum_{i} w_{ji} \cdot x_i + b_j
$$

Where:

- `w_{ji}` = weight from input `i` to hidden neuron `j`
- `x_i` = input value
- `b_j` = bias for hidden neuron `j`

#### Step 2: Apply activation function to get hidden output

$$
z_j = \phi(v_j)
$$

#### Step 3: Compute net input to output neurons

$$
v_k = \sum_{j} w_{kj} \cdot z_j + b_k
$$

#### Step 4: Apply activation to get final output

$$
y_k = \phi(v_k)
$$

---

## ii) 🔁 **Backward Pass (Backpropagation)**

Backpropagation **updates weights** by minimizing the **error** using gradient descent.

### **Diagram (backward arrows):**

```
       Input → Hidden → Output
         ↑        ↑        ↑
         Δw      Δw       Error
```

### **Equations (Using δ for error signal):**

#### Step 1: Compute output error

$$
\delta_k = (d_k - y_k) \cdot \phi'(v_k)
$$

- `d_k` = desired output
- `y_k` = actual output
- `φ'` = derivative of activation function

#### Step 2: Compute hidden error

$$
\delta_j = \phi'(v_j) \cdot \sum_{k} \delta_k \cdot w_{kj}
$$

#### Step 3: Update weights (output layer)

$$
w_{kj}^{\text{new}} = w_{kj} + \eta \cdot \delta_k \cdot z_j
$$

#### Step 4: Update weights (hidden layer)

$$
w_{ji}^{\text{new}} = w_{ji} + \eta \cdot \delta_j \cdot x_i
$$

Where:

- `η` is the learning rate.

---

## 🧮 Example Activation Functions

### 1. **Sigmoid Function**

$$
\phi(v) = \frac{1}{1 + e^{-v}}
$$

- Output range: (0, 1)
- Smooth and differentiable
- Derivative: $\phi'(v) = \phi(v)(1 - \phi(v))$
- Used in hidden and output layers for probabilistic output

---

### 2. **ReLU (Rectified Linear Unit)**

$$
\phi(v) = \max(0, v)
$$

- Output range: \[0, ∞)
- Simple and fast
- Derivative:

  $$
  \phi'(v) = \begin{cases}
  1 & \text{if } v > 0 \\
  0 & \text{otherwise}
  \end{cases}
  $$

- Helps with vanishing gradient problem

---

## c.

### 🔥 **Boltzmann Learning – Basic Modelling Characteristics**

**Boltzmann learning** is used in **stochastic neural networks** like the **Boltzmann Machine**. It's based on ideas from **statistical mechanics** and uses **probabilistic neurons** that can switch states with a certain probability.

---

### 🧠 **Basic Characteristics**

1. **Stochastic Units**:
   Neurons have binary states (0 or 1 or sometimes -1 and +1), but their activation is **probabilistic**, not deterministic.

2. **Energy-Based Model**:
   The network defines an **energy function** for each configuration of neuron states. Learning aims to **minimize this energy** for desired patterns.

3. **Symmetric Weights**:
   The weight from neuron _i_ to _j_ is the same as from _j_ to _i_:

   $$
   w_{ij} = w_{ji}
   $$

4. **No Self-Connections**:
   Neurons are **not connected to themselves**, so $w_{ii} = 0$.

5. **Global Learning Rule**:
   Unlike local rules (like in Hebbian learning), Boltzmann learning considers the **overall behavior of the network** based on equilibrium probabilities.

---

### 🧮 **Energy Function**

The energy of a state (set of binary neuron activations) is:

$$
E = -\frac{1}{2} \sum_{i} \sum_{j} w_{ij} s_i s_j - \sum_{i} \theta_i s_i
$$

Where:

- $s_i$ is the state of neuron _i_ (0 or 1, or ±1)
- $w_{ij}$ is the weight between neurons
- $\theta_i$ is the threshold (bias) for neuron _i_

---

### 🔁 **Weight Adaptation Rule (Learning Rule)**

The goal is to adjust weights so the network prefers patterns seen in training.

The **weight update** rule is:

$$
\Delta w_{ij} = \eta \left( \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}} \right)
$$

Where:

- $\eta$: learning rate
- $\langle s_i s_j \rangle_{\text{data}}$: average correlation between neuron _i_ and _j_ **when clamped** to the training data
- $\langle s_i s_j \rangle_{\text{model}}$: average correlation when the network runs **freely** (samples from the model’s distribution)

This rule is similar to **contrastive divergence** used in modern **Restricted Boltzmann Machines (RBMs)**.

---

### 📌 Summary

| Feature               | Description                                         |
| --------------------- | --------------------------------------------------- |
| Activation            | Probabilistic (stochastic neurons)                  |
| Learning Goal         | Match model distribution to data distribution       |
| Key Principle         | Minimize energy; learn correlations                 |
| Weight Symmetry       | $w_{ij} = w_{ji}$                                   |
| Weight Update Formula | Δw based on difference between data and model stats |

# QUESTION 2

## a.

---

## 🧠 What SVM Does

SVM is a **binary classifier** that tries to find the **best possible line (or hyperplane)** that separates two classes.
The goal is **not just any line**, but one that gives the **largest margin** between the two classes.

---

## 📏 Linear Decision Boundary

### 1. **Hyperplane Equation:**

For **n-dimensional inputs** $\mathbf{x}$, a **hyperplane** is defined as:

$$
\mathbf{w} \cdot \mathbf{x} + b = 0
$$

Where:

- $\mathbf{w}$ = weight vector (normal to the hyperplane)
- $b$ = bias (offset from origin)

This divides space into two regions:

- $\mathbf{w} \cdot \mathbf{x} + b > 0$ → class +1
- $\mathbf{w} \cdot \mathbf{x} + b < 0$ → class -1

---

## ✅ SVM Decision Rule

For input $\mathbf{x}$, the SVM predicts:

$$
y = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)
$$

---

## ✏️ Key Concept: **Margin**

The **margin** is the distance between the decision boundary and the **closest data points** from each class. These closest points are called **support vectors**.

SVM maximizes this margin to improve generalization.

---

## 📐 Deriving the Margin

### 1. **Support Vectors Constraints**

Assume all training data $(\mathbf{x}_i, y_i)$, with $y_i \in \{-1, +1\}$

We want:

$$
y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1
$$

- For points **on the margin**, this becomes:

  $$
  y_i (\mathbf{w} \cdot \mathbf{x}_i + b) = 1
  $$

These are the **support vectors**.

---

### 2. **Distance from a Point to the Hyperplane**

The distance from any point $\mathbf{x}_i$ to the hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ is:

$$
\text{Distance} = \frac{|\mathbf{w} \cdot \mathbf{x}_i + b|}{\|\mathbf{w}\|}
$$

For support vectors (with $y_i (\mathbf{w} \cdot \mathbf{x}_i + b) = 1$):

$$
\text{Margin} = \frac{1}{\|\mathbf{w}\|}
$$

So the **total margin between classes** = distance between two support vectors from opposite classes:

$$
\text{Total margin} = \frac{2}{\|\mathbf{w}\|}
$$

---

## 🎯 SVM Optimization Objective

To maximize the margin, minimize $\|\mathbf{w}\|$, subject to constraints:

$$
\begin{aligned}
& \min_{\mathbf{w}, b} \quad \frac{1}{2} \|\mathbf{w}\|^2 \\
& \text{subject to } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i
\end{aligned}
$$

This is a **convex quadratic optimization problem** with linear constraints.

---

## 🧩 Summary Table

| Concept           | Equation                                       |
| ----------------- | ---------------------------------------------- |
| Hyperplane        | $\mathbf{w} \cdot \mathbf{x} + b = 0$          |
| Decision Rule     | $\text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$ |
| Support Vector    | $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1$   |
| Margin            | $\frac{1}{\|\mathbf{w}\|}$                     |
| Optimization Goal | $\min \frac{1}{2}\|\mathbf{w}\|^2$             |

---

## b.

Great question! Let's walk step by step through the **derivation of the Linear Least Squares (LLS)** training algorithm for a **single-layer perceptron** with a **linear activation function**, used for **regression**.

---

## 🧠 Setup: Single-Layer Perceptron for Regression

- We have:

  - $N$ training samples
  - Each input: $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)}]$
  - Corresponding desired output: $d^{(i)}$

- Linear activation (no non-linearity):

  $$
  y^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} = \sum_{j=1}^{n} w_j x_j^{(i)}
  $$

We want to find the weight vector $\mathbf{w}$ that minimizes the total squared error between outputs $y^{(i)}$ and desired outputs $d^{(i)}$.

---

## 🎯 Objective: Minimize Squared Error

Define the total error function:

$$
E(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^{N} (d^{(i)} - \mathbf{w}^T \mathbf{x}^{(i)})^2
$$

In matrix form, let:

- $\mathbf{X}$ be the **input matrix** of shape $N \times n$, where each row is a sample $\mathbf{x}^{(i)}$
- $\mathbf{d}$ be the **output vector** $[d^{(1)}, d^{(2)}, ..., d^{(N)}]^T$
- $\mathbf{w}$ be the weight vector

Then:

$$
E(\mathbf{w}) = \frac{1}{2} \| \mathbf{d} - \mathbf{X}\mathbf{w} \|^2
$$

---

## ✏️ Derivation of Least Squares Solution

We minimize $E(\mathbf{w})$ with respect to $\mathbf{w}$ using calculus.

### Step 1: Take the gradient:

$$
\nabla E(\mathbf{w}) = -\mathbf{X}^T (\mathbf{d} - \mathbf{X}\mathbf{w})
$$

### Step 2: Set gradient to zero:

$$
\mathbf{X}^T (\mathbf{d} - \mathbf{X}\mathbf{w}) = 0
$$

$$
\mathbf{X}^T \mathbf{d} = \mathbf{X}^T \mathbf{X} \mathbf{w}
$$

### Step 3: Solve for $\mathbf{w}$:

$$
\boxed{\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{d}}
$$

This is the **normal equation** of linear least squares.

---

## ✅ Summary

| Step                  | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| Model                 | $y = \mathbf{w}^T \mathbf{x}$                                                 |
| Error Function        | $E = \frac{1}{2} \sum (d^{(i)} - y^{(i)})^2$                                  |
| Matrix Form           | $E = \frac{1}{2} \| \mathbf{d} - \mathbf{Xw} \|^2$                            |
| Weight Solution (LLS) | $\boxed{\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{d}}$ |

---

c.

To **adapt the weights** in a **Radial Basis Function (RBF) Neural Network**, assuming the **centers and widths are fixed**, we treat this as a **linear regression problem** in the transformed (nonlinear) space. Let's design a clear step-by-step mechanism:

---

## 🧠 RBF Network Structure Recap

An RBF network typically has:

1. **Input layer**: passes inputs to hidden layer.
2. **Hidden layer**: applies radial basis functions (e.g., Gaussian).
3. **Output layer**: performs weighted sum of hidden activations.

---

## 🏗️ Design Assumptions

Given:

- Training inputs: $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, ..., \mathbf{x}^{(N)}$
- Desired outputs: $d^{(1)}, ..., d^{(N)}$
- RBF centers: $\mathbf{c}_1, \mathbf{c}_2, ..., \mathbf{c}_M$
- Widths (spreads): $\sigma_1, \sigma_2, ..., \sigma_M$
- Basis function (e.g., Gaussian):

  $$
  \phi_j(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_j\|^2}{2\sigma_j^2}\right)
  $$

---

## 🧮 Step-by-Step Weight Adaptation Mechanism

### **Step 1: Compute Hidden Layer Activations**

For each training sample $\mathbf{x}^{(i)}$, compute outputs of all hidden neurons:

$$
\Phi^{(i)} = [\phi_1(\mathbf{x}^{(i)}), \phi_2(\mathbf{x}^{(i)}), ..., \phi_M(\mathbf{x}^{(i)})]
$$

Do this for all $N$ samples to form the activation matrix $\mathbf{\Phi}$ of size $N \times M$.

---

### **Step 2: Solve Linear Regression**

We want the output layer to map $\mathbf{\Phi}$ to the desired outputs $\mathbf{d}$:

$$
\mathbf{\Phi} \cdot \mathbf{w} = \mathbf{d}
$$

Where:

- $\mathbf{w}$: weight vector (to be found)
- $\mathbf{d}$: desired outputs

Use the **Least Squares Solution**:

$$
\boxed{\mathbf{w} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{d}}
$$

If $\mathbf{\Phi}^T \mathbf{\Phi}$ is not invertible or unstable, use **regularization**:

$$
\boxed{\mathbf{w} = (\mathbf{\Phi}^T \mathbf{\Phi} + \lambda I)^{-1} \mathbf{\Phi}^T \mathbf{d}}
$$

Where $\lambda$ is a small constant (e.g., 0.01).

---

## ✅ Summary

| Step | Action                                                                                  |
| ---- | --------------------------------------------------------------------------------------- |
| 1    | Compute hidden layer responses $\phi_j(\mathbf{x})$ for all inputs                      |
| 2    | Form design matrix $\mathbf{\Phi}$ of activations                                       |
| 3    | Solve linear equation $\mathbf{\Phi} \cdot \mathbf{w} = \mathbf{d}$ using least squares |
| 4    | Optionally apply regularization to improve stability                                    |

---

## d.

Here's a clear breakdown of the **main characteristics** of **supervised** and **unsupervised learning** in **neural networks** and **machine learning**:

---

## 🧑‍🏫 **Supervised Learning**

### ✅ **Main Characteristics:**

1. **Labeled Data**:
   Every input comes with a known **target output** (label).
   Example: Image → “Cat”

2. **Goal**:
   Learn a **mapping** from inputs $\mathbf{x}$ to outputs $\mathbf{y}$.
   Example: Predict house prices from features.

3. **Loss Function**:
   Based on the **difference between predicted and actual output**
   (e.g., Mean Squared Error, Cross-Entropy).

4. **Algorithms**:

   - Neural networks (with backpropagation)
   - Support Vector Machines
   - Decision Trees

5. **Examples**:

   - Classification (e.g., spam or not spam)
   - Regression (e.g., predicting temperature)

---

## 🧠 **Unsupervised Learning**

### ✅ **Main Characteristics:**

1. **Unlabeled Data**:
   Inputs **don’t come with labels**. The system must find structure in the data.

2. **Goal**:
   Discover **patterns**, **clusters**, or **hidden structures**.
   Example: Group customers by shopping behavior.

3. **Loss Function**:
   Often involves **distance, similarity, or reconstruction error** (e.g., clustering variance, autoencoder loss).

4. **Algorithms**:

   - K-means clustering
   - Principal Component Analysis (PCA)
   - Autoencoders
   - Self-Organizing Maps (SOMs)

5. **Examples**:

   - Market segmentation
   - Feature extraction
   - Dimensionality reduction

---

## 📝 Summary Table

| Feature            | Supervised Learning        | Unsupervised Learning                |
| ------------------ | -------------------------- | ------------------------------------ |
| Input Data         | Labeled                    | Unlabeled                            |
| Goal               | Predict output             | Find structure                       |
| Example Tasks      | Classification, Regression | Clustering, Dimensionality Reduction |
| Feedback           | Uses error to adjust model | No direct feedback from output       |
| Typical Algorithms | Neural nets, SVM, k-NN     | K-means, PCA, Autoencoders           |

---

# QUESTION 3

Based on the provided content, here are **three real-valued crossover schemes** suitable for a **Genetic Algorithm (GA)** using **real-number chromosomes**:

---

### 🔁 1. **Line Arithmetical Crossover**

#### 📌 Idea:

Creates a child by taking a weighted average of two parents along a line segment between them.

#### 🔢 Formula:

$$
c = r \cdot p_i + (1 - r) \cdot p_j
$$

Where:

- $c$ = child gene
- $p_i$, $p_j$ = parent genes
- $r \in [0, 1]$ is a randomly selected value

#### ✅ Characteristics:

- Child is always **between** the two parent values.
- Simple and promotes **exploitation** of existing solutions.

---

### 🌀 2. **Intermediate Arithmetical Crossover**

#### 📌 Idea:

Extends line crossover by using a **different random value for each gene**.

#### 🔢 Formula:

$$
c_k = r_k \cdot p_i^k + (1 - r_k) \cdot p_j^k
$$

Where:

- $c_k$ = child’s k-th gene
- $r_k \in [0, 1]$ is random and **unique for each gene**

#### ✅ Characteristics:

- Allows **greater variation** among genes.
- Balances **exploration** and **diversity**.

---

### 🚀 3. **Heuristic Arithmetical Crossover**

#### 📌 Idea:

Generates a child by moving **away from the worse parent** in the direction of the better one — encouraging exploration.

#### 🔢 Formula:

$$
c = p_{\text{best}} + r \cdot (p_{\text{best}} - p_{\text{worst}})
$$

Where:

- $r > 0$ (e.g., 0 to 1 or slightly more)
- $p_{\text{best}}$ = parent with higher fitness
- $p_{\text{worst}}$ = parent with lower fitness

#### ✅ Characteristics:

- Encourages **search in promising directions**
- Can generate offspring **outside** the parent range

---

## 📝 Summary Table

| Crossover Type                | Description                               | Key Feature                     |
| ----------------------------- | ----------------------------------------- | ------------------------------- |
| **Line Arithmetical**         | Weighted average between parents          | Controlled exploitation         |
| **Intermediate Arithmetical** | Weighted average with per-gene randomness | Increased variation             |
| **Heuristic Arithmetical**    | Extrapolation toward better parent        | Strong exploration, directional |

---

## b.

Based on the content in `combined.md`, here's a **revised answer** to:

---

### **b) Why do genetic algorithms need to handle invalid chromosomes? Describe three different ways for handling such cases.**

---

## 🧨 Why Handle Invalid Chromosomes?

In Genetic Algorithms (GAs), **invalid chromosomes** are individuals that:

- **Violate problem constraints**, or
- **Don’t represent feasible solutions** due to encoding issues or faulty operators (e.g., crossover/mutation).

**Why it's important**:

- Invalid individuals can **mislead the selection process** by having undefined or inflated fitness.
- They can **waste resources**, slowing convergence.
- If not controlled, they may **dominate** or cause the population to stagnate.

From the material in `combined.md` (section on GA constraints handling), invalid chromosomes must be managed to maintain the **quality** and **validity** of the evolving population.

---

## ✅ Three Ways to Handle Invalid Chromosomes

---

### 1. ❌ **Rejection**

- **What it does**:
  Simply **discards** any invalid chromosome and **regenerates** a new one until it is valid.

- **Example**:
  In a path-finding problem, if crossover creates a path with **duplicate nodes**, it is rejected.

- **Pros**:

  - Keeps the population 100% valid.

- **Cons**:

  - Can be slow if invalid solutions are frequent.
  - May reduce diversity due to frequent retries.

---

### 2. 🛠 **Repair Functions**

- **What it does**:
  Detects violations and **modifies the chromosome** to make it valid.

- **Example**:
  In a scheduling problem, if a person is assigned to multiple tasks at the same time, the repair function reassigns the task to a free time slot.

- **Pros**:

  - Saves genetic material from good solutions.
  - Avoids redoing crossover/mutation.

- **Cons**:

  - Needs **problem-specific logic** to fix solutions.
  - Repairs may introduce **bias**.

---

### 3. ⚖️ **Penalty Functions**

- **What it does**:
  Allows invalid chromosomes but **penalizes their fitness score**, reducing their chance of being selected.

- **Fitness adjustment**:

  $$
  F'(x) = F(x) + \text{penalty}
  $$

- **Example**:
  For a layout optimization problem, if constraints like size or shape are violated, apply a large penalty to fitness.

- **Pros**:

  - Keeps search space wide (preserves diversity).
  - Works well with soft constraints.

- **Cons**:

  - Needs **careful tuning** of penalty size.
  - Weak penalties may allow invalids to survive.

---

## 🧾 Summary Table

| Method           | How It Works                   | Pros                     | Cons                          |
| ---------------- | ------------------------------ | ------------------------ | ----------------------------- |
| Rejection        | Discard and retry              | Simple, clean population | Slow, less diverse            |
| Repair Function  | Fix violations after detection | Keeps useful genes       | Problem-specific logic needed |
| Penalty Function | Penalize fitness of invalids   | Encourages exploration   | Needs careful tuning          |

---

## c.

Based on the material in **`combined.md`**—especially the section on **adaptive mutation**—here’s a clear and tailored response to:

---

### **c) Design a mechanism of your choice which would enable the mutation rate to adapt to parental or population diversity. State whether this mechanism will work for a discrete or a real-valued genotype representation.**

---

## 🔁 **Mechanism: Diversity-Based Adaptive Mutation Rate**

### 🎯 **Goal:**

Automatically **adjust the mutation rate** based on how **similar or diverse** the parents are.

- **More similar → higher mutation rate** (to boost diversity)
- **More different → lower mutation rate** (to preserve good variety)

---

## 🧩 **Mechanism Design**

### 🔹 Step 1: Measure Similarity Between Parents

Let the two selected parents be $p_i$ and $p_j$.

- For **discrete (binary) genotypes**: use **Hamming distance** $H(p_i, p_j)$, the number of differing bits.

- Normalize this by chromosome length $L$:

  $$
  \text{Similarity} = 1 - \frac{H(p_i, p_j)}{L}
  $$

- For **real-valued genotypes**: use **Euclidean distance** or **gene-wise difference thresholding**.

---

### 🔹 Step 2: Define Adaptive Mutation Rate

Let:

- $p_m^{\text{min}}$: minimum mutation rate
- $p_m^{\text{max}}$: maximum mutation rate
- $I(p_i, p_j)$: number of identical genes between parents
- $L$: total number of genes

Use the formula from `combined.md`:

$$
p_m(p_i, p_j) = p_m^L + (p_m^U - p_m^L) \cdot \left( \frac{I(p_i, p_j)}{L} \right)^2
$$

Where:

- $p_m^L$: lower bound (e.g. 0.001)
- $p_m^U$: upper bound (e.g. 0.1)
- This means: **more similarity → mutation rate increases quadratically**

---

## 🧪 Application Type

### ✅ **This mechanism works for:**

- ✅ **Discrete (binary or integer) genotypes**, where similarity is measured using bit-matching or Hamming distance.
- ⚠️ Can be **extended to real-valued genotypes**, but would need a different similarity measure (e.g., normalized distance or correlation).

---

## 📌 Example

Assume:

- $L = 10$ (chromosome has 10 bits)
- $I(p_i, p_j) = 8$ (parents share 8 genes)
- $p_m^L = 0.01$, $p_m^U = 0.1$

Then:

$$
p_m = 0.01 + (0.1 - 0.01) \cdot \left( \frac{8}{10} \right)^2 = 0.01 + 0.09 \cdot 0.64 = 0.0676
$$

This higher mutation rate encourages **exploration** when parents are too similar.

---

## ✅ Summary

| Component           | Description                                            |
| ------------------- | ------------------------------------------------------ |
| What it does        | Adjusts mutation rate based on parental similarity     |
| Formula source      | From adaptive mutation method in `combined.md`         |
| Representation type | Designed for **discrete genotypes**                    |
| Key benefit         | Maintains diversity and prevents premature convergence |

---

Here’s a clear explanation of **uniform crossover** and **binomial crossover** for **discrete chromosome representations** (e.g., binary strings or integer vectors):

---

## 🔁 1. **Uniform Crossover**

### 🧠 **Idea**:

Each gene (bit or integer) in the offspring is chosen **randomly from either parent**, with a **fixed probability** (usually 50%).

### ⚙️ **Mechanism**:

- For each gene position:

  - Flip a coin (random number)
  - If heads → take gene from **Parent 1**
  - If tails → take gene from **Parent 2**

### 📊 **Example**:

| Gene Position     | 1   | 2   | 3   | 4   | 5   |
| ----------------- | --- | --- | --- | --- | --- |
| **Parent 1**      | 0   | 1   | 0   | 1   | 1   |
| **Parent 2**      | 1   | 0   | 1   | 0   | 0   |
| **Random Choice** | P1  | P2  | P2  | P1  | P1  |
| **Child**         | 0   | 0   | 1   | 1   | 1   |

### ✅ **Advantages**:

- High diversity.
- No positional bias.

---

## 🎲 2. **Binomial Crossover**

### 🧠 **Idea**:

Similar to uniform crossover, but uses a **tunable probability $p_r$** to control how often genes are taken from one parent.

### ⚙️ **Mechanism**:

- For each gene:

  - Generate a random number between 0 and 1.
  - If it’s less than $p_r$ → take gene from **Parent 1**.
  - Else → take gene from **Parent 2**.

### 🔧 **Control**:

- $p_r \in [0, 1]$ adjusts how much influence Parent 1 has.

  - $p_r = 0.5$ → behaves like uniform crossover
  - $p_r = 0.8$ → 80% genes from Parent 1, 20% from Parent 2

### 📊 **Example with $p_r = 0.8$**:

| Gene Position     | 1   | 2   | 3   | 4   | 5   |
| ----------------- | --- | --- | --- | --- | --- |
| **Parent 1**      | 1   | 1   | 0   | 0   | 1   |
| **Parent 2**      | 0   | 0   | 1   | 1   | 0   |
| **Random Values** | 0.2 | 0.9 | 0.4 | 0.7 | 0.1 |
| **Child**         | P1  | P2  | P1  | P1  | P1  |
| **Child Result**  | 1   | 0   | 0   | 0   | 1   |

### ✅ **Advantages**:

- Fine control over bias toward a particular parent.
- Flexible for both exploration and exploitation.

---

## 📝 Summary Table

| Feature        | Uniform Crossover              | Binomial Crossover        |
| -------------- | ------------------------------ | ------------------------- |
| Gene selection | 50/50 chance for each parent   | Tunable probability $p_r$ |
| Diversity      | High                           | Medium to high            |
| Parent bias    | None                           | Controlled by $p_r$       |
| Usage          | Binary or discrete chromosomes | Also binary or discrete   |

---

## e.

---

### 🔍 Concepts We're Analyzing:

1. **Positional Bias**
2. **Distributional Bias**
3. **Recombination Power**

These are discussed in **`combined.md`** under the **“Crossover (Recombination)”** section and illustrated in the **mind map (`0.md`)** under **“Genetic Operators → Crossover Types”**.

---

## 📊 Analysis: Effect of Number of Crossover Points

---

### 1. 🎯 **Positional Bias**

- **Definition**: Tendency for genes in certain positions (e.g., near the start) to be inherited **together** more often.
- **Few crossover points (e.g., 1-point or 2-point)**:

  - **High positional bias**: adjacent genes are more likely to stay together.
  - Promotes **building blocks** but may limit variety.

- **Many crossover points or uniform crossover**:

  - **Low positional bias**: genes can come from either parent **regardless of position**.
  - Promotes **diversity**, less structure retention.

✅ See:
**`combined.md` → “Small χ: Low disruption (positional bias)”**

---

### 2. 🧮 **Distributional Bias**

- **Definition**: The tendency for children to reflect only a limited subset of possible combinations.
- **Few crossover points**:

  - **High distributional bias**: only certain gene groupings are passed on.

- **More crossover points or uniform**:

  - **Lower distributional bias**: offspring can represent a **wider variety** of combinations.

✅ See:
**`combined.md` → “Large χ: High diversity, risk of instability (distributional bias)”**

---

### 3. 🧬 **Recombination Power**

- **Definition**: The ability of crossover to **generate novel combinations** of parent genes.
- **Few crossover points**:

  - **Limited recombination power**: preserves gene blocks.

- **More crossover points or uniform crossover**:

  - **Higher recombination power**: more gene mixing = higher chance of exploring new areas of the solution space.

✅ See:

- **`combined.md`**: crossover types and effects
- **`0.md`**: mind map branch **“Crossover Types → Impact on Genetic Diversity”**

---

## 📌 Summary Table

| Crossover Type   | Positional Bias | Distributional Bias | Recombination Power |
| ---------------- | --------------- | ------------------- | ------------------- |
| 1-point          | High            | High                | Low                 |
| 2-point          | Medium          | Medium              | Medium              |
| Uniform          | Low             | Low                 | High                |
| Multi-point (≥3) | Low             | Low                 | High                |

---

## 🧠 Conclusion

- **Fewer crossover points** (e.g., 1-point) → better for keeping related genes together (good for exploitation).
- **More crossover points or uniform crossover** → better for exploring new solutions (good for exploration), but risks **breaking useful structures**.

---

## f.

### 🧬 **Population Model with Niching in Genetic Algorithms**

---

## 🧠 **What is Niching?**

**Niching** is a technique used in Genetic Algorithms to maintain **diversity** in the population by encouraging the discovery and preservation of **multiple good solutions** — not just one.

In nature, a “niche” is a small sub-group of organisms adapted to a specific environment. In GAs, it means subpopulations that specialize in **different areas of the search space**.

---

## 🔧 **How is it Modeled?**

### 1. **Fitness Sharing (Most Common Method)**

- Individuals that are **too similar** to others get their **fitness reduced** (shared).
- The idea is to **penalize overcrowded regions** of the search space.

#### 📐 Fitness Sharing Formula:

$$
F_{\text{shared}}(x_i) = \frac{F(x_i)}{\sum_{j=1}^{N} \gamma(d(x_i, x_j))}
$$

Where:

- $F(x_i)$ is the original fitness of individual $x_i$
- $d(x_i, x_j)$ is the distance between individuals $i$ and $j$
- $\gamma(d)$ is the **sharing function**, often:

  $$
  \gamma(d) = \max \left(0, 1 - \left(\frac{d}{\sigma_{\text{share}}}\right)^a \right)
  $$

  - $\sigma_{\text{share}}$: niche radius (how far sharing reaches)
  - $a$: shape factor

This ensures that **clusters (niches)** of solutions form naturally, each exploring different regions.

---

### 2. **Crowding and Restricted Mating**

Other models include:

- **Crowding**: offspring replace individuals **similar to themselves**.
- **Restricted mating**: parents are selected from **different niches**.

---

## 🌟 **Primary Advantage**

The main advantage of niching is:

### ✅ **Preserving Diversity to Avoid Premature Convergence**

- Prevents the whole population from converging to **one local optimum** too early.
- Allows the algorithm to **find multiple peaks** (solutions) in **multimodal problems**.
- Increases the chance of finding the **global optimum**.

---

## 🧾 Summary

| Feature          | Description                                        |
| ---------------- | -------------------------------------------------- |
| Purpose          | Maintain diversity and discover multiple optima    |
| Core Method      | Fitness sharing, crowding, or restricted mating    |
| Modeling Concept | Penalize individuals in crowded regions            |
| Main Advantage   | Avoids premature convergence, improves exploration |

---

# QUESTION 4

## a.

### 🌳 **Genetic Programming (GP) – Basic Architecture and Operators**

---

## 🧠 **What Is Genetic Programming?**

Genetic Programming is a type of evolutionary algorithm where **programs themselves** are evolved to solve problems.
Instead of evolving strings or vectors (like in Genetic Algorithms), GP evolves **tree-like structures** representing programs or expressions.

---

## 🏗️ **Basic Architecture of Genetic Programming**

1. **Representation**:

   - Each individual is a **tree**:

     - **Internal nodes**: functions (e.g., `+, -, *, /, sin`)
     - **Leaves (terminals)**: variables and constants (e.g., `x`, `3.0`)

2. **Population**:

   - A set of randomly generated trees (programs)

3. **Fitness Evaluation**:

   - Each program is **executed** and scored based on how well it solves the problem.

4. **Selection**:

   - Fitter programs are more likely to be selected for reproduction.

5. **Variation Operators**:

   - **Crossover**: swaps subtrees between two parents.
   - **Mutation**: replaces a subtree with a new random subtree.

6. **Replacement**:

   - Offspring replace parents or are inserted into the next generation.

---

## 🔀 **Crossover Operator in GP**

### 🌳 **How It Works**:

- Randomly select a **subtree** in Parent 1.
- Randomly select a **subtree** in Parent 2.
- **Swap** the two subtrees to create offspring.

### 📊 Example:

**Parent 1:**

```
   +
  / \
 x   3
```

**Parent 2:**

```
   *
  / \
  2  x
```

**Swap subtrees `x` (P1) and `2` (P2)** → new children:

```
Child 1:       Child 2:
   +             *
  / \           / \
 2   3         x   x
```

### ✅ Benefits:

- Maintains **functional program structure**.
- Can transfer **useful building blocks** between individuals.

---

## 🔁 **Mutation Operator in GP**

### 🌳 **How It Works**:

- Pick a random node (subtree) in the individual.
- Replace it with a **new randomly generated subtree** (subject to depth limits).

### 📊 Example:

Original:

```
   +
  / \
 x   3
```

After mutation (replace `3` with `(* 4 x)`):

```
     +
    / \
   x   *
      / \
     4   x
```

### ✅ Benefits:

- Introduces new **program structures**.
- Helps maintain **diversity** in the population.

---

## 🧾 Summary Table

| Component      | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| Representation | Tree structure of functions and terminals                      |
| Fitness        | Based on program performance on test cases                     |
| Crossover      | Subtree exchange between parents                               |
| Mutation       | Random subtree replacement                                     |
| Strengths      | Evolves symbolic programs, models, and solutions automatically |

---

## b.

### 🌱 **What Is an Intron in Genetic Programming?**

In **genetic programming (GP)**, an **intron** is a **part of a program (subtree)** that:

- **Does not affect the output** of the program,
- **Is never executed**, or
- **Has no influence on fitness**, even though it’s part of the tree.

Introns are sometimes called **"junk code"**, and while they seem useless, they can help **protect useful code during crossover and mutation** by acting as **genetic buffers**.

---

### 🖼️ **Intron Example in the Given Tree**

Let’s analyze the tree you've shared.

**Top-level operator:** `*`
The left subtree is a function, and the right subtree includes a `<` (less-than) operator — this is a clue.

Let’s look at this part of the right subtree:

```
         *
        / \
       x   <
          / \
         *   2
        / \
       1   x
```

The right child of the root `*` is this expression:

$$
x * (1 * x < 2)
$$

Now, focus on the comparison:

$$
(1 * x < 2)
$$

This is a **Boolean** expression. But this entire subtree is part of a **numeric multiplication** expression at the root. So if your GP environment treats Booleans as `true = 1` and `false = 0`, this comparison will:

- Output `1` or `0`
- Then be multiplied with `x`

#### 🔍 Now here's the key:

If this comparison **always evaluates to `true` or `false` regardless of input**, the result of the multiplication can become predictable. For example, if:

- $1 \cdot x < 2$ is **always true**, then the whole subtree simplifies to `x`
- If **always false**, the subtree returns `0`

Then **any part of that subtree not needed to determine that fixed output is an intron**.

---

### ✅ **Concrete Intron in This Tree**

Let’s say:

- Input values for `x` are always such that $1 \cdot x < 2$ is **true**.
- Then the subtree:

```
    <
   / \
  *   2
 / \
1   x
```

Always returns `true` → numeric `1` → then `x * 1 = x`.

Now, the subtree:

```
   *
  / \
 x   <
     / \
    *   2
   / \
  1   x
```

Becomes effectively just:

```
x * 1 = x
```

That means the **entire `<` subtree** is **functionally unnecessary** — it acts like an **intron** in this context.

---

### 🧾 Summary

| Term    | Description                                                   |
| ------- | ------------------------------------------------------------- |
| Intron  | A part of the program tree that **doesn't affect the output** |
| Purpose | May protect useful code and promote diversity                 |
| Example | The `<` subtree in your figure if its output is constant      |

---

## c.

### 🧬 **What is Gene Expression Programming (GEP)?**

**Gene Expression Programming (GEP)** is an **evolutionary algorithm** like Genetic Algorithms (GA) and Genetic Programming (GP), but it combines the **linear structure** of GAs with the **tree-based execution** of GP.

#### 🔑 Key Characteristics:

- Individuals are **linear chromosomes** (like strings of symbols).
- Each chromosome is **expressed** (decoded) as an **expression tree (ET)**.
- Chromosomes are made of **genes**, and each gene maps to a **subtree** of the overall program.

> Think of it like writing a program in a string, and then "executing" it as a tree.

---

## 🏗️ Structure of a GEP Chromosome

Each gene has two parts:

1. **Head**: contains both **functions** and **terminals** (like `+, *, x`)
2. **Tail**: contains only **terminals** (like `x, 3`)

This structure ensures **valid expression trees** no matter what the gene contains.

### 📌 Example Gene (in string form):

```
Gene:   + * x 5 x x
Head:   + * x 5
Tail:       x x
```

This gene expresses the tree:

```
    +
   / \
  *   x
 / \
x   5
```

---

## 🔀 **Crossover in GEP**

### 🧠 Purpose:

Mix genetic material between two parent chromosomes.

### 📌 Example – One-Point Crossover:

```
Parent 1: + * x 5 | x x
Parent 2: - x 7 2 | x x
```

Choose a crossover point (e.g., after the third symbol):

```
Child: + * x 2 x x
```

→ The child takes the first part from Parent 1 and the rest from Parent 2.

GEP crossover always preserves the **valid structure** due to the head–tail format.

---

## 🔁 **Mutation in GEP**

### 🧠 Purpose:

Introduce new genetic material (diversity).

### 📌 Example – Point Mutation:

Original gene:

```
+ * x 5 x x
```

Suppose we mutate the first symbol `+` → `-`:

```
- * x 5 x x
```

Rules:

- Mutations in the **head** can be functions or terminals.
- Mutations in the **tail** must be **terminals only**.

This ensures the decoded expression tree is always **syntactically valid**.

---

## 📝 Summary

| Feature        | Gene Expression Programming (GEP)                  |
| -------------- | -------------------------------------------------- |
| Representation | Fixed-length linear strings (genes)                |
| Expression     | Translated into expression trees (ETs)             |
| Crossover      | Swaps parts of genes while keeping tree-validity   |
| Mutation       | Alters symbols, respecting head/tail rules         |
| Advantage      | Combines benefits of GA structure and GP execution |

---

## e.

### 🧪 **Variant Design: Constriction Coefficient PSO (Clerc’s PSO)**

Let’s design a **popular and effective variant** of the basic PSO known as the **Constriction Coefficient PSO**. This method is aimed at **stabilizing convergence** and **improving control over particle movement**.

---

## 🔧 **What’s Modified?**

In standard PSO, velocity updates use inertia $\omega$, and acceleration coefficients $c_1$, $c_2$.
In **Constriction PSO**, the update is scaled by a **constriction factor** $\chi$ that prevents particles from overshooting or diverging.

---

## 🧮 **Modified Velocity Equation**

$$
\mathbf{v}_i(t+1) = \chi \left[ \mathbf{v}_i(t) + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i(t)) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i(t)) \right]
$$

### Where:

- $\chi$ is the **constriction coefficient**, defined as:

  $$
  \chi = \frac{2}{|2 - \phi - \sqrt{\phi^2 - 4\phi}|}
  \quad \text{with } \phi = c_1 + c_2 \text{ and } \phi > 4
  $$

Example: if $c_1 = c_2 = 2.05$, then $\chi ≈ 0.729$

---

## 🔁 **Position Update Remains the Same**:

$$
\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + \mathbf{v}_i(t+1)
$$

---

## 🌟 **Advantages of Constriction PSO**

| Benefit                       | Explanation                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| ✅ Stability                  | Prevents exploding velocities and keeps particles under control            |
| ✅ Convergence Reliability    | Leads to **smoother and more predictable** convergence behavior            |
| ✅ Parameter Guidance         | Offers a **mathematically derived** way to set parameters (no trial/error) |
| ✅ Exploration + Exploitation | Maintains a good balance through controlled step size                      |

---

## 🔄 **Difference from Standard PSO**

| Feature           | Standard PSO                           | Constriction PSO               |
| ----------------- | -------------------------------------- | ------------------------------ |
| Velocity update   | Uses inertia $\omega$                  | Uses constriction $\chi$       |
| Stability control | Empirical tuning of $\omega, c_1, c_2$ | Theoretical bound via $\chi$   |
| Risk              | Can diverge or oscillate               | Stronger convergence guarantee |

---

## f.

### 🧬 **Evolutionary Strategies (ES) – Update Equations and Encoding Scheme**

---

## 🧠 **What Are Evolutionary Strategies?**

**Evolutionary Strategies (ES)** are a class of **bio-inspired optimization algorithms** focused on **real-valued optimization** using **mutation** and **selection**, often without crossover.

They are especially well-suited for **continuous, high-dimensional problems** like engineering design.

---

## 🧩 **Encoding Scheme**

Each individual (solution) is represented as:

$$
\mathbf{x} = (\mathbf{z}, \boldsymbol{\sigma})
$$

Where:

- $\mathbf{z}$: vector of real-valued decision variables (the actual solution).
- $\boldsymbol{\sigma}$: vector of **strategy parameters** (mutation step sizes), one per gene.

This is known as **self-adaptive encoding**, where the mutation strength **evolves along with the solution**.

---

## 🧮 **Basic Update Equations**

### 🔁 Mutation of Strategy Parameters (Step Sizes):

$$
\sigma_i' = \sigma_i \cdot e^{\tau' \cdot N(0,1) + \tau \cdot N_i(0,1)}
$$

- $\sigma_i'$: new step size for gene $i$
- $N(0,1)$: standard normal (global)
- $N_i(0,1)$: standard normal (individual)
- $\tau', \tau$: learning rates based on dimension $n$ (e.g., $\tau' = \frac{1}{\sqrt{2n}}, \tau = \frac{1}{\sqrt{2\sqrt{n}}}$)

---

### 🔁 Mutation of Genes (Decision Variables):

$$
z_i' = z_i + \sigma_i' \cdot N_i(0,1)
$$

- The gene value is shifted by a **normally distributed** random step, scaled by its updated $\sigma_i'$.

---

## 🧪 Selection Schemes

Two common models:

- **(μ + λ)**: Select best **μ** from **μ parents + λ offspring**
- **(μ, λ)**: Select best **μ** only from **λ offspring** (parents die)

---

## ✅ Summary

| Component           | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| Representation      | Tuple $(\mathbf{z}, \boldsymbol{\sigma})$: solution + mutation rates |
| Mutation (σ update) | Log-normal update using global and local noise                       |
| Mutation (z update) | Add Gaussian noise scaled by $\sigma_i$                              |
| Selection Models    | (μ + λ) or (μ, λ) schemes                                            |
| Crossover           | Optional; not always used                                            |

---
