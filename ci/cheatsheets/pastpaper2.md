# 1

## 1a)

Here are three mechanisms to set the **centre** ($\mathbf{c}_i$) and **width** ($\sigma_i$) parameters of Gaussian activation functions in RBF neural networks:

---

### 1. **Fixed Random Centres and Heuristic Widths**

- **Step 1**: Randomly choose $M$ training samples as centres $\mathbf{c}_i$.

- **Step 2**: Set the same width for all RBFs using a heuristic:

  $$
  \sigma = \frac{d_{\text{max}}}{\sqrt{2M}}
  $$

  where $d_{\text{max}}$ is the maximum distance between any two centres.

- **Pros**: Fast and easy.

- **Cons**: No adaptation to data structure.

---

### 2. **Unsupervised Learning via K-Means Clustering**

- **Step 1**: Apply **K-Means** to group input data into $M$ clusters.

- **Step 2**: Use the cluster **centroids** as RBF centres $\mathbf{c}_i$.

- **Step 3**: Set width $\sigma_i$ based on spread (e.g., average distance to neighbours).

  $$
  \sigma_i = \frac{1}{k} \sum_{j=1}^k \|\mathbf{c}_i - \mathbf{c}_j\|
  $$

- **Pros**: Centres match data distribution.

- **Cons**: Still unsupervised, not optimal for task performance.

---

### 3. **Supervised Gradient Descent Learning**

- Learn $\mathbf{c}_i$ and $\sigma_i$ using gradient descent to minimize the error:

  - **Weight update**:

    $$
    w_i(n+1) = w_i(n) - \eta_1 \cdot \frac{\partial E}{\partial w_i}
    $$

  - **Centre update**:

    $$
    c_{ik}(n+1) = c_{ik}(n) - \eta_2 \cdot \frac{\partial E}{\partial c_{ik}}
    $$

  - **Width update**:

    $$
    \sigma_i(n+1) = \sigma_i(n) - \eta_3 \cdot \frac{\partial E}{\partial \sigma_i}
    $$

- **Pros**: Optimizes all parameters for best performance.

- **Cons**: Slower, risk of overfitting, needs careful tuning.

---

Each method offers a trade-off between simplicity, adaptability, and performance.

## 1b)

We derive the **Linear Least Squares (LLS)** training algorithm for a **single-layer perceptron (SLP)** with **linear activation**.

---

### **Step 1: Problem Setup**

We are given:

- $n$ input-output training pairs $\{(\mathbf{x}(i), d(i))\}$

  - Input vector: $\mathbf{x}(i) \in \mathbb{R}^p$
  - Desired output: $d(i) \in \mathbb{R}$

We want to find weights $\mathbf{w} \in \mathbb{R}^p$ such that:

$$
y(i) = \mathbf{w}^T \mathbf{x}(i)
$$

---

### **Step 2: Matrix Form**

Define matrices:

- Input matrix:

  $$
  \mathbf{X} =
  \begin{bmatrix}
  \mathbf{x}(1)^T \\
  \mathbf{x}(2)^T \\
  \vdots \\
  \mathbf{x}(n)^T
  \end{bmatrix}
  \in \mathbb{R}^{n \times p}
  $$

- Output vector:

  $$
  \mathbf{d} =
  \begin{bmatrix}
  d(1) \\
  d(2) \\
  \vdots \\
  d(n)
  \end{bmatrix}
  \in \mathbb{R}^{n \times 1}
  $$

- Weight vector: $\mathbf{w} \in \mathbb{R}^{p \times 1}$

Predicted output for all samples:

$$
\mathbf{y} = \mathbf{X} \mathbf{w}
$$

---

### **Step 3: Define the Cost Function**

Use the **sum of squared errors**:

$$
E(\mathbf{w}) = \frac{1}{2} \|\mathbf{d} - \mathbf{X} \mathbf{w}\|^2
$$

---

### **Step 4: Minimize the Error**

Take the gradient of $E$ with respect to $\mathbf{w}$ and set it to zero:

$$
\frac{\partial E}{\partial \mathbf{w}} = -\mathbf{X}^T (\mathbf{d} - \mathbf{X} \mathbf{w}) = 0
$$

Solve:

$$
\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{d}
$$

---

### **Step 5: Solve for Weights**

This is a linear system. The solution is:

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{d}
$$

Or using the **Moore–Penrose pseudoinverse**:

$$
\boxed{
\mathbf{w} = \mathbf{X}^+ \mathbf{d}
}
$$

---

### ✅ Summary

- Assumes **all data is available at once** (batch learning).
- Gives **optimal linear weights** in one step.
- Works only when activation is linear and $\mathbf{X}^T \mathbf{X}$ is invertible.

## 1c)

Self-Organizing Maps (SOMs) learn through three key processes: **competitive**, **cooperative**, and **adaptive**. Here's a simple breakdown of each:

---

### 🔹 1. Competitive Process

- **Goal**: Select one “winning” neuron for each input.

- **How**:

  - Each neuron $j$ has a weight vector $\mathbf{w}_j$.

  - Given an input $\mathbf{x}$, compute distance to each weight:

    $$
    d_j = \|\mathbf{x} - \mathbf{w}_j\|
    $$

  - The neuron with the smallest distance is the **winner** (Best Matching Unit, BMU):

    $$
    \text{BMU} = \arg\min_j d_j
    $$

- **Why**: Forces neurons to specialize in representing certain regions of the input space.

---

### 🔹 2. Cooperative Process

- **Goal**: Not only the winner learns—its neighbors learn too.

- **How**:

  - Define a **neighborhood function** around the BMU (usually Gaussian):

    $$
    h_{j, \text{BMU}}(t) = \exp\left(-\frac{\|\mathbf{r}_j - \mathbf{r}_{\text{BMU}}\|^2}{2\sigma(t)^2}\right)
    $$

    - $\mathbf{r}_j$: position of neuron $j$ on the grid.
    - $\sigma(t)$: neighborhood width, shrinks over time.

- **Why**: Promotes smooth learning and topological ordering—similar inputs map to nearby neurons.

---

### 🔹 3. Adaptive Process

- **Goal**: Adjust neuron weights to match the input.

- **How**:

  - Update the weight of each neuron $j$ using:

    $$
    \mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \eta(t) \cdot h_{j,\text{BMU}}(t) \cdot \left[\mathbf{x} - \mathbf{w}_j(t)\right]
    $$

    - $\eta(t)$: learning rate, decays over time.
    - Only neurons close to the BMU are updated significantly.

- **Why**: Allows the SOM to self-organize and represent the input space.

---

### ✅ Summary Table

| Process         | Purpose                          | Mechanism                                   |
| --------------- | -------------------------------- | ------------------------------------------- |
| **Competitive** | Find best-matching neuron        | Minimize distance to input                  |
| **Cooperative** | Involve neighbors in learning    | Use a neighborhood function (e.g. Gaussian) |
| **Adaptive**    | Update weights towards the input | Gradient-style update with decay            |

## 1d)

### 🔹 Supervised Learning

- **Data**: Each input $\mathbf{x}_i$ is paired with a known output label $d_i$.

- **Goal**: Learn a function that maps inputs to correct outputs:

  $$
  f(\mathbf{x}_i) \approx d_i
  $$

- **How it works**:

  - Use error-based learning (e.g. backpropagation).
  - Compare prediction with known target → update weights.

- **Examples**:

  - Classification (e.g. digit recognition)
  - Regression (e.g. predicting house prices)

---

### 🔹 Unsupervised Learning

- **Data**: Only inputs $\mathbf{x}_i$; no labels.

- **Goal**: Discover patterns or structure in the data.

- **How it works**:

  - Uses similarity, distance, or statistical properties.
  - Examples: clustering or dimensionality reduction.

- **Examples**:

  - K-means clustering
  - Self-Organizing Maps (SOMs)
  - PCA (Principal Component Analysis)

---

### ✅ Key Differences

| Feature           | Supervised                 | Unsupervised                         |
| ----------------- | -------------------------- | ------------------------------------ |
| Labeled Data      | Yes                        | No                                   |
| Learning Target   | Input-output mapping       | Hidden structure in data             |
| Typical Tasks     | Classification, Regression | Clustering, Feature Extraction       |
| Example Algorithm | Backpropagation (MLP)      | Competitive Learning (SOMs, K-means) |

# 2

## 2a)

We train a perceptron to learn the **AND** function using:

- Inputs: $x_1, x_2$
- Bias input: $x_0 = -1$
- Initial weights: $w_0 = -2.2$, $w_1 = 0.5$, $w_2 = -1.3$
- Learning rate: $\eta = 0.5$
- Activation: **signum** function (outputs +1 if $v \geq 0$, else –1)
- Target outputs: AND → only (1,1) gives +1; others give –1

---

### 🧾 Perceptron Update Rule

If output $y \neq d$, update:

$$
w_j \leftarrow w_j + \eta \cdot (d - y) \cdot x_j
$$

---

### 📊 Input-Output Table

| Input $\mathbf{x}$ | Desired $d$ |
| ------------------ | ----------- |
| (0, 0)             | –1          |
| (0, 1)             | –1          |
| (1, 0)             | –1          |
| (1, 1)             | +1          |

We'll use **augmented inputs**:

$$
\mathbf{x} = [x_0, x_1, x_2] = [-1, x_1, x_2]
$$

---

### 🔁 Epoch 1

#### 1. Input: (0, 0) → $\mathbf{x} = [-1, 0, 0]$, $d = -1$

$$
v = (-1)(-2.2) + (0)(0.5) + (0)(-1.3) = 2.2 \Rightarrow y = +1
$$

Update (since $y \neq d$):

$$
\Delta w = \eta(d - y)\mathbf{x} = 0.5(-2)[-1, 0, 0] = [1.0, 0, 0]
$$

$$
w_0 = -2.2 + 1.0 = -1.2,\quad w_1 = 0.5,\quad w_2 = -1.3
$$

---

#### 2. Input: (0, 1) → $\mathbf{x} = [-1, 0, 1]$, $d = -1$

$$
v = (-1)(-1.2) + (0)(0.5) + (1)(-1.3) = 1.2 - 1.3 = -0.1 \Rightarrow y = -1
$$

Correct → **no update**

---

#### 3. Input: (1, 0) → $\mathbf{x} = [-1, 1, 0]$, $d = -1$

$$
v = 1.2 + 0.5 = 1.7 \Rightarrow y = +1
$$

Update:

$$
\Delta w = 0.5(-2)[-1, 1, 0] = [1.0, -1.0, 0]
$$

$$
w_0 = -1.2 + 1.0 = -0.2,\quad w_1 = 0.5 - 1.0 = -0.5,\quad w_2 = -1.3
$$

---

#### 4. Input: (1, 1) → $\mathbf{x} = [-1, 1, 1]$, $d = +1$

$$
v = (-1)(-0.2) + (1)(-0.5) + (1)(-1.3) = 0.2 - 0.5 - 1.3 = -1.6 \Rightarrow y = -1
$$

Wrong → update:

$$
\Delta w = 0.5(2)[-1, 1, 1] = [-1.0, 1.0, 1.0]
$$

$$
w_0 = -0.2 - 1.0 = -1.2,\quad w_1 = -0.5 + 1.0 = 0.5,\quad w_2 = -1.3 + 1.0 = -0.3
$$

---

### 🔁 Epoch 2 (run again with updated weights)

Now check all 4 patterns again using new weights:

- All outputs are now **correct** → no updates needed!

---

### ✅ Final Weights After Convergence:

$$
w_0 = -1.2,\quad w_1 = 0.5,\quad w_2 = -0.3
$$

The perceptron successfully learns the **AND** logic after **2 epochs**.

## 2b)

### 🔹 Concepts in Support Vector Machines (SVMs)

SVMs are used for **binary classification**. They aim to:

- Find a **hyperplane** that separates two classes with the **maximum margin**.
- Only some points (support vectors) define this margin.

---

### 🔹 Definitions

- Let training data be:
  $D = \{ (\mathbf{x}_i, d_i) \}$, where $\mathbf{x}_i \in \mathbb{R}^n$, $d_i \in \{+1, -1\}$
- A **linear classifier** has the form:

  $$
  g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
  $$

---

### 🔹 Margin and Separation

To ensure correct classification:

$$
d_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

This creates **two margin boundaries**:

$$
\mathbf{w}^T \mathbf{x} + b = +1 \quad \text{(for class +1)}
$$

$$
\mathbf{w}^T \mathbf{x} + b = -1 \quad \text{(for class –1)}
$$

The **margin** is the perpendicular distance between them:

$$
\text{Margin} = \frac{2}{\|\mathbf{w}\|}
$$

---

### 🔹 Optimization Problem (Primal Form)

We want to **maximize the margin** → minimize $\|\mathbf{w}\|$:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2} \|\mathbf{w}\|^2
$$

subject to:

$$
d_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

---

### 🔹 Support Vectors

- **Support vectors** are the data points that lie **exactly on the margin**.

- For these points:

  $$
  d_i (\mathbf{w}^T \mathbf{x}_i + b) = 1
  $$

- They are **critical**—removing them changes the optimal hyperplane.

---

### ✅ Summary of Key Equations

| Concept           | Equation                                     |
| ----------------- | -------------------------------------------- |
| Hyperplane        | $\mathbf{w}^T \mathbf{x} + b = 0$            |
| Constraints       | $d_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ |
| Margin            | $\frac{2}{\|\mathbf{w}\|}$                   |
| Optimization Goal | Minimize $\frac{1}{2} \|\mathbf{w}\|^2$      |
| Support Vectors   | $d_i (\mathbf{w}^T \mathbf{x}_i + b) = 1$    |

## 2c)

Here’s how the **forward** and **backward** passes work in a **multi-layer perceptron (MLP)** using **backpropagation**.

---

### 🔹 1. Forward Pass

**Goal**: Compute the output of the network.

---

#### **Diagram (simplified)**

```
Input Layer      Hidden Layer        Output Layer
  x₁ ───▶●─┐        h₁ ───▶●─┐
  x₂ ───▶●─┼──▶ σ ─▶ h₂ ───▶●─▶ σ ─▶ ŷ
  x₃ ───▶●─┘                 ...
```

---

#### **Steps (Layer-wise)**

1. **Input**:
   Input vector $\mathbf{x} = [x_1, x_2, ..., x_n]$

2. **Hidden layer**:

   $$
   z^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}
   $$

   $$
   a^{(1)} = \sigma(z^{(1)})
   $$

3. **Output layer**:

   $$
   z^{(2)} = \mathbf{W}^{(2)} a^{(1)} + \mathbf{b}^{(2)}
   $$

   $$
   \hat{y} = \sigma(z^{(2)})
   $$

---

### 🔹 2. Backward Pass

**Goal**: Adjust weights using the error between prediction $\hat{y}$ and target $y$.

---

#### **Steps**

1. **Compute error at output**:

   $$
   e = \hat{y} - y
   $$

2. **Output layer gradient**:

   $$
   \delta^{(2)} = e \cdot \sigma'(z^{(2)})
   $$

3. **Hidden layer gradient**:

   $$
   \delta^{(1)} = \left( \mathbf{W}^{(2)T} \delta^{(2)} \right) \cdot \sigma'(z^{(1)})
   $$

4. **Update weights**:

   $$
   \mathbf{W}^{(2)} \leftarrow \mathbf{W}^{(2)} - \eta \cdot \delta^{(2)} \cdot a^{(1)T}
   $$

   $$
   \mathbf{W}^{(1)} \leftarrow \mathbf{W}^{(1)} - \eta \cdot \delta^{(1)} \cdot \mathbf{x}^T
   $$

---

### ✅ Summary Table

| Phase         | What Happens                                           |
| ------------- | ------------------------------------------------------ |
| Forward Pass  | Compute neuron activations layer by layer              |
| Backward Pass | Compute errors and update weights via gradient descent |

---

# 3

## 3a)

### 🔹 Why Handle Invalid Chromosomes?

In genetic algorithms (GAs), a **chromosome** is a possible solution.
Some chromosomes might be **invalid**, meaning:

- They **violate problem constraints** (e.g. duplicate cities in TSP, exceeding capacity in knapsack).
- They **cannot be evaluated** properly → leads to errors or meaningless fitness.

So, **handling invalid chromosomes** is crucial to ensure the algorithm works and finds valid solutions.

---

### 🔹 Three Methods to Handle Invalid Chromosomes

---

### 1. **Repair Methods**

- Fix the chromosome **after creation** (mutation or crossover).
- Apply a rule-based or algorithmic method to correct it.

**Example**:
In the TSP (Traveling Salesman Problem), if a city appears twice, replace the duplicate with a missing city.

**Pros**: Keeps good genes, keeps chromosome usable
**Cons**: Can be complex and problem-specific

---

### 2. **Penalty Functions**

- Allow invalid chromosomes but **penalize** them in the fitness function.

**Example**:
In knapsack, if weight exceeds the limit, subtract a penalty from the total value.

**Fitness**:

$$
f(x) = \text{value}(x) - \lambda \cdot \text{violation}(x)
$$

**Pros**: Easy to implement
**Cons**: Needs tuning of penalty parameter $\lambda$

---

### 3. **Reject and Regenerate**

- Discard any invalid chromosome during initialization, crossover, or mutation.
- Keep generating until you get a valid one.

**Pros**: Ensures population is always valid
**Cons**: May waste time and slow convergence

---

### ✅ Summary Table

| Method              | Idea                       | Trade-off          |
| ------------------- | -------------------------- | ------------------ |
| Repair              | Fix the invalid chromosome | May be complex     |
| Penalty Function    | Allow, but lower fitness   | Needs tuning       |
| Reject & Regenerate | Skip invalids and retry    | Can be inefficient |

## 3b)

### 🔹 How Baker’s Linear Ranking Works

Baker’s linear ranking is a **selection method** in genetic algorithms.
Instead of using raw fitness, it:

1. **Sorts** individuals by fitness (best to worst).
2. **Assigns a rank** to each individual: best = rank 1, worst = rank $N$
3. **Assigns selection probability** using this formula:

$$
P_i = \frac{1}{N} \left[ \eta_{\text{max}} - (\eta_{\text{max}} - 2)(i - 1)/(N - 1) \right]
$$

Where:

- $P_i$ is the selection probability for rank $i$
- $\eta_{\text{max}}$ controls selection pressure (typically between 1 and 2)
- $N$ is the population size

---

### 🔹 How It Avoids Common Problems

#### ✅ 1. **Close-race problem**

- If all individuals have **very similar fitness**, raw fitness selection treats them almost equally.
- Ranking avoids this because it **spreads probabilities based on rank**, not actual values.

#### ✅ 2. **Super-individual problem**

- If one individual has **extremely high fitness**, it can **dominate selection**.
- Ranking limits this because even the best individual only gets the **highest rank**, not huge extra weight.

---

### ✅ Summary

| Problem              | How Linear Ranking Helps                 |
| -------------------- | ---------------------------------------- |
| Close fitness values | Still ranks and spreads selection fairly |
| One super-individual | Prevents it from dominating              |

## 3c)

### 🔹 Why Parent Selection is Needed

In genetic algorithms (GAs), **parent selection** decides **which individuals get to reproduce**.
This is important because:

- Good parents pass on **useful traits** (good genes).
- It **guides the search** toward better solutions.
- Without selection, evolution becomes random.

---

### 🔹 Three Parent Selection Schemes

---

### 1. **Roulette Wheel Selection (Fitness-Proportional)**

- Each individual gets a slice of a “wheel” sized by its fitness.
- Higher fitness → higher chance of being selected.

**Pros**: Simple
**Cons**: Can fail if fitness values are too close or too extreme

---

### 2. **Tournament Selection**

- Pick $k$ individuals at random.
- Select the **best one** among them as a parent.

**Pros**: Easy to tune selection pressure (via $k$)
**Cons**: Can lose diversity if $k$ is large

---

### 3. **Rank-Based Selection (e.g. Baker’s Ranking)**

- Rank individuals by fitness, not by raw value.
- Assign selection probability based on rank.

**Pros**: Stable, avoids super-individual problem
**Cons**: Slower than basic methods

---

### ✅ Summary Table

| Method         | Based On           | Key Feature               |
| -------------- | ------------------ | ------------------------- |
| Roulette Wheel | Fitness value      | Proportional selection    |
| Tournament     | Best in random set | Easy to control pressure  |
| Rank-Based     | Fitness rank       | Fair and stable selection |

## 3d)

In genetic algorithms, **crossover** combines two parent strings to form offspring.
The number of **crossover points** affects:

---

### 🔹 1. Positional Bias

- **Definition**: Preference for certain gene positions to stay together.

#### Effect:

- **1-point crossover** → strong positional bias: genes near ends stay together.
- **More points** → reduces positional bias (genes mix more evenly).

---

### 🔹 2. Distributional Bias

- **Definition**: Tendency to favor one parent over the other in gene selection.

#### Effect:

- **1-point crossover** → large blocks copied from one parent → higher bias.
- **Uniform crossover** (many points) → each gene has \~50% chance from each parent → lower bias.

---

### 🔹 3. Recombination Power

- **Definition**: The ability to explore new gene combinations.

#### Effect:

- More crossover points → higher recombination power (more mixing).
- But too many → disrupts good gene groups (called **building blocks**).

---

### ✅ Summary Table

| Crossover Points | Positional Bias | Distributional Bias | Recombination Power   |
| ---------------- | --------------- | ------------------- | --------------------- |
| 1-point          | High            | High                | Low–Moderate          |
| 2-point          | Moderate        | Moderate            | Moderate              |
| Uniform (many)   | Low             | Low                 | High (but disruptive) |

In short:
**More crossover points → more mixing**, but **less structure preservation**.

## 3e)

### 🔹 Mechanism: Diversity-Based Adaptive Mutation Rate

We adjust the **mutation rate** based on the **diversity** of the population:

---

### 📌 Step-by-Step Mechanism

1. **Measure population diversity** at each generation:

   For real-valued genotypes:

   $$
   D = \frac{1}{n} \sum_{i=1}^n \text{std}(x_i)
   $$

   where:

   - $x_i$ is gene $i$ across all individuals
   - $\text{std}$ = standard deviation

2. **Adapt mutation rate**:

   $$
   \text{mutation\_rate} =
   \begin{cases}
   \text{high (e.g. 0.1)}, & \text{if } D < D_{\text{min}} \quad (\text{low diversity}) \\
   \text{low (e.g. 0.01)}, & \text{if } D > D_{\text{max}} \quad (\text{high diversity}) \\
   \text{interpolate}, & \text{otherwise}
   \end{cases}
   $$

3. **Apply this mutation rate** during mutation step.

---

### ✅ Why It Works

- If the population is **too similar**, increase mutation to explore.
- If the population is **diverse**, lower mutation to exploit.

---

### 🧬 Applicability

- **Real-valued genotypes**: ✅ Yes (uses std deviation).
- **Discrete genotypes**: ✅ Yes (can use Hamming distance or gene frequency instead).

---

### ✅ Summary

| Feature              | Description                             |
| -------------------- | --------------------------------------- |
| Based on             | Diversity of population                 |
| Adapts mutation rate | High if similar, low if diverse         |
| Works for            | Both discrete and real-valued genotypes |

## 3e)

### 🔹 Mechanism: Diversity-Based Adaptive Mutation Rate

We adjust the **mutation rate** based on the **diversity** of the population:

---

### 📌 Step-by-Step Mechanism

1. **Measure population diversity** at each generation:

   For real-valued genotypes:

   $$
   D = \frac{1}{n} \sum_{i=1}^n \text{std}(x_i)
   $$

   where:

   - $x_i$ is gene $i$ across all individuals
   - $\text{std}$ = standard deviation

2. **Adapt mutation rate**:

   $$
   \text{mutation\_rate} =
   \begin{cases}
   \text{high (e.g. 0.1)}, & \text{if } D < D_{\text{min}} \quad (\text{low diversity}) \\
   \text{low (e.g. 0.01)}, & \text{if } D > D_{\text{max}} \quad (\text{high diversity}) \\
   \text{interpolate}, & \text{otherwise}
   \end{cases}
   $$

3. **Apply this mutation rate** during mutation step.

---

### ✅ Why It Works

- If the population is **too similar**, increase mutation to explore.
- If the population is **diverse**, lower mutation to exploit.

---

### 🧬 Applicability

- **Real-valued genotypes**: ✅ Yes (uses std deviation).
- **Discrete genotypes**: ✅ Yes (can use Hamming distance or gene frequency instead).

---

### ✅ Summary

| Feature              | Description                             |
| -------------------- | --------------------------------------- |
| Based on             | Diversity of population                 |
| Adapts mutation rate | High if similar, low if diverse         |
| Works for            | Both discrete and real-valued genotypes |

## 3f)

Here are two common **crossover schemes** for real-valued chromosomes in genetic algorithms:

---

### 🔹 1. Arithmetic Crossover

- Creates offspring by taking a **weighted average** of two parents.

#### Formula:

If parents are:

- $\mathbf{p}_1 = [x_1^{(1)}, x_2^{(1)}, \dots]$
- $\mathbf{p}_2 = [x_1^{(2)}, x_2^{(2)}, \dots]$

Then child:

$$
\mathbf{c} = \alpha \cdot \mathbf{p}_1 + (1 - \alpha) \cdot \mathbf{p}_2
$$

- $\alpha \in [0, 1]$ is a random number (can vary per gene).

**Pros**: Smooth blending
**Use**: Good for continuous search spaces

---

### 🔹 2. BLX-α (Blend Crossover)

- Creates children in a wider range than the parents' values.

#### Formula:

For each gene $i$:

1. Find the range:

   $$
   d = |x_i^{(1)} - x_i^{(2)}|
   $$

2. Sample child gene:

   $$
   c_i \sim \text{Uniform}\left[\min - \alpha d,\ \max + \alpha d\right]
   $$

- $\alpha \geq 0$ controls exploration (e.g. $\alpha = 0.5$)

**Pros**: Encourages exploration
**Use**: Avoids local optima by creating wider variation

---

### ✅ Summary

| Method     | How it Works                   | Use Case                     |
| ---------- | ------------------------------ | ---------------------------- |
| Arithmetic | Weighted average of parents    | Smooth local search          |
| BLX-α      | Range extension around parents | Broader search / exploration |

# 4

## 4a)

### 🔹 Basic Architecture of Genetic Programming (GP)

Genetic programming evolves **programs** instead of strings or vectors.

Each individual (solution) is usually a **tree structure**, where:

- **Nodes** = functions (e.g. `+`, `*`, `sin`)
- **Leaves** = inputs or constants (e.g. `x`, `2.0`)

---

### 🔹 GP Workflow

1. **Initialize** a population of random trees (programs).
2. **Evaluate** fitness of each tree by running it on problem data.
3. **Select** parents based on fitness.
4. **Apply** crossover and mutation to produce new programs.
5. **Repeat** until convergence or stopping condition.

---

### 🔹 Crossover in GP

- Randomly pick a **subtree** in parent 1.
- Randomly pick a **subtree** in parent 2.
- **Swap the subtrees** to form two new offspring.

✔ Keeps tree structure
✔ Preserves large blocks of logic

---

### 🔹 Mutation in GP

Common types:

1. **Subtree mutation**:

   - Pick a random node → replace its subtree with a **new random subtree**.

2. **Point mutation**:

   - Change a single **function** or **leaf** to another of the same type.

---

### ✅ Summary

| Component      | Description                         |
| -------------- | ----------------------------------- |
| Representation | Tree of functions and terminals     |
| Crossover      | Swap subtrees between parents       |
| Mutation       | Replace or change parts of the tree |

## 4b)

### 🔹 What Is an Intron in Genetic Programming?

In **genetic programming (GP)**, an **intron** is a part of a program tree that:

- **Does not affect** the program’s final output
- Acts like **“junk code”** — it’s executed but irrelevant
- Can help protect useful code from disruption during crossover/mutation

---

### 🔹 Example from the Given Tree

Let’s analyze the **left branch** of the root subtraction node (`-`):

```
      -
     / \
    %   *
   / \  / \
  *  1 x  <
 / \     / \
x  *    *   2
   / \ 2  2
  >  x
 / \
x  1
```

Focus on the subtree rooted at:

$$
* \rightarrow > \ x \ 1 \text{ and } x
$$

This part computes:

$$
> (x, 1) \Rightarrow \text{Boolean (e.g., true/false)}, \quad *(\text{bool}, x)
$$

But this goes into another `*`, then a `%`, and finally:

$$
\%(\text{value}, 1)
$$

Now, **modulo 1** of any number is always **0**.

So this entire **left side of the tree** always becomes **0**, regardless of what’s in it.

Thus, the whole **left branch** is an **intron** — it doesn’t influence the final output of the subtraction.

---

### ✅ Summary

| Term    | Meaning                                       |
| ------- | --------------------------------------------- |
| Intron  | A subtree that doesn't affect the output      |
| Purpose | Acts as neutral code; protects key logic      |
| Example | The entire left branch under `%` in this tree |

## 4c)

### 🔹 What Is Gene Expression Programming (GEP)?

Gene Expression Programming is an evolutionary algorithm that:

- Evolves solutions as **linear chromosomes** (like in genetic algorithms)
- Translates them into **expression trees** (like in genetic programming)

Each **chromosome** has a fixed length and is split into:

- A **head**: can contain functions and terminals
- A **tail**: contains only terminals (ensures valid trees)

---

### 🔹 Example Chromosome (head + tail)

```
Chromosome:   * + x 1 x x x x 1
Tree form:    *
             / \
            +   x
           / \
          x   1
```

---

### 🔹 Crossover in GEP

GEP uses **genetic operators** similar to GAs but tailored for its format.

#### Example: One-point crossover

- Pick a point in the linear chromosome
- Swap segments between two parents

**Parent 1**: `* + x | 1 x x x x 1`
**Parent 2**: `/ x 1 | x + - x 2 2`
**Offspring**: Swap after `|` → new child

---

### 🔹 Mutation in GEP

- Randomly replace a symbol at a gene position
- Must keep functions in the head, terminals in the tail

**Before**: `* + x 1 x x x x 1`
**After mutation at pos 2**: `* + - 1 x x x x 1`

✔ Keeps structure valid
✔ Enables exploration

---

### ✅ Summary

| Feature   | Description                            |
| --------- | -------------------------------------- |
| GEP       | Linear chromosome → expression tree    |
| Crossover | Swaps parts of fixed-length genes      |
| Mutation  | Changes symbols while preserving rules |

## 4d)

Here are **two variants** of Particle Swarm Optimisation (PSO), with their differences from the **standard PSO**:

---

### 🔹 1. Inertia Weight PSO

#### 🛠 How it works:

Adds an **inertia weight** $w$ to control the effect of the previous velocity:

$$
v_i(t+1) = w \cdot v_i(t) + c_1 r_1 (p_{\text{best}} - x_i) + c_2 r_2 (g_{\text{best}} - x_i)
$$

- $w \in [0, 1]$: decays over time (e.g., from 0.9 to 0.4)
- Other terms are the same as standard PSO

#### ✅ Advantages:

- Helps balance **exploration** and **exploitation**
- High $w$ early → explore; low $w$ later → converge

#### 🆚 Difference:

- Standard PSO uses a fixed formula without this adjustable decay

---

### 🔹 2. Local Best PSO (lbest)

#### 🛠 How it works:

Each particle only communicates with its **neighbours**, not the whole swarm.

- Instead of global best $g_{\text{best}}$, use **local best** $l_{\text{best}}$

$$
v_i(t+1) = v_i(t) + c_1 r_1 (p_{\text{best}} - x_i) + c_2 r_2 (l_{\text{best}} - x_i)
$$

#### ✅ Advantages:

- **Better diversity**: reduces swarm convergence to local optima
- **More stable** for multimodal functions

#### 🆚 Difference:

- Standard PSO uses **global best**, which can lead to **premature convergence**

---

### ✅ Summary Table

| Variant            | Key Feature        | Advantage                      | Differs From Standard PSO By...       |
| ------------------ | ------------------ | ------------------------------ | ------------------------------------- |
| Inertia Weight PSO | Weighted velocity  | Smooth transition in behaviour | Adding decay factor $w$               |
| Local Best (lbest) | Neighbourhood best | Higher diversity               | Replacing global best with local best |

## 4e)

---

### **i) Basic Update Equations and Encoding Scheme (4 marks)**

#### 🔹 Encoding Scheme

In **Evolutionary Strategies (ES)**, each individual has:

- A **solution vector**: $\mathbf{x} = [x_1, x_2, ..., x_n]$
- A **strategy vector** (step sizes): $\boldsymbol{\sigma} = [\sigma_1, \sigma_2, ..., \sigma_n]$

Each $\sigma_i$ controls the mutation strength of $x_i$

---

#### 🔹 Update (Mutation) Equations

Each generation, we create offspring using:

1. **Mutate step sizes**:

$$
\sigma_i' = \sigma_i \cdot \exp\left(\tau' \cdot N(0,1) + \tau \cdot N_i(0,1)\right)
$$

- $\tau, \tau'$: learning rates
- $N(0,1)$: standard normal values (shared and individual)

2. **Mutate solution**:

$$
x_i' = x_i + \sigma_i' \cdot N_i(0,1)
$$

This allows **self-adaptive mutation**—step sizes evolve along with solutions.

---

### **ii) One Extension Mechanism (2 marks)**

#### 🔹 Extension: **Covariance Matrix Adaptation (CMA-ES)**

- Instead of using separate step sizes, CMA-ES **learns a full covariance matrix** of the search distribution.

#### ✅ Advantages:

- Captures **variable dependencies**
- Adapts search shape to match the problem landscape
- More efficient on **non-separable** or rotated functions

---

### ✅ Summary Table

| Component | Description                                                 |
| --------- | ----------------------------------------------------------- |
| Encoding  | Real-valued $\mathbf{x}$ + step sizes $\boldsymbol{\sigma}$ |
| Mutation  | Self-adapting normal-based perturbations                    |
| Extension | CMA-ES → learns full covariance                             |
