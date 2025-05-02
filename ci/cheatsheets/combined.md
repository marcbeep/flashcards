# File: 0.md
# 🧠 Computational Intelligence Mind Map

## 📚 Neural Networks & Deep Learning (02-05)

### 1. Neural Network Fundamentals (02)

- Structure of a Neuron
  - Synapses/Connections with weights
  - Adder (Summing Junction)
  - Activation Functions
- Bias Term and Mathematical Foundations
- Activation Functions
  - Heaviside (Threshold)
  - Piecewise Linear
  - Sigmoid (Logistic)
  - Tanh
  - ReLU
- Network Architectures
  - Single-Layer Feedforward
  - Multi-Layer Feedforward (MLP)
  - CNN
  - RNN

### 2. Learning in Neural Networks (03)

- Core Learning Rules
  - Error-Correction Learning
  - Memory-Based Learning
  - Hebbian Learning
  - Competitive Learning
  - Boltzmann Learning
- Learning Paradigms
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Credit Assignment Problem

### 3. Adaptive Filtering & Learning (04)

- System Identification
- Linear Neuron Model
- Linear Least Squares (LLS)
- Least Mean Square (LMS)
- Perceptrons
  - Model & Training
  - Linear Separability
  - Convergence Theorem

### 4. Backpropagation & MLPs (05)

- Backpropagation Algorithm
  - Forward Pass
  - Backward Pass
  - Error Calculation
- Training Modes
  - Sequential
  - Batch
  - Mini-Batch
- Generalization & Complexity Control
- Regularization Techniques

## 📊 Advanced Neural Networks (06-07)

### 1. RBF Networks (06)

- Pattern Separability
- RBF Structure
- Training Strategies
- Comparison with MLPs

### 2. Support Vector Machines (07)

- Margin and Separation
- Support Vectors
- Optimization (Primal & Dual)
- Kernel Methods
- Soft Margin SVM

## 🤖 Reinforcement Learning (08)

### 1. Core Concepts

- Agent-Environment Interaction
- States, Actions, Rewards
- Policies and Value Functions
- Markov Decision Process

### 2. Learning Methods

- Q-Learning
- Deep Q-Learning (DQN)
- Experience Replay
- Target Networks

## 🔍 Optimization Techniques (09-11)

### 1. Mathematical Optimization (09)

- Types of Optimization
  - Unconstrained
  - Constrained
- Optimization Methods
  - Enumerative
  - Gradient-based
  - Stochastic

### 2. Genetic Algorithms (10-11)

- Basic Components
  - Population
  - Selection
  - Crossover
  - Mutation
- Advanced Topics
  - Fitness Transformation
  - Selection Schemes
  - Population Models
  - Parallel Populations
- Practical Implementation
  - Example GA Code
  - Parameter Tuning

## 🌳 Advanced Evolutionary Methods (13-15)

### 1. Genetic Programming (13)

- Tree Representation
- GP Components
  - Terminal Set
  - Function Set
- GP Operators
- Gene Expression Programming (GEP)

### 2. Particle Swarm Optimization (14)

- Basic PSO Algorithm
- PSO Variations
  - Time Varying Inertia Weight
  - Constriction Factor
  - Acceleration Coefficients
  - Social Stereotyping

### 3. Evolution Strategies (15)

- Basic ES Algorithm
- Extensions
  - 1/5 Success Rule
  - (μ+λ)-ES
  - (μ,λ)-ES
  - Covariance Matrix Updates
# File: 02.md
---
## 🧠 Neural Networks Cheat Sheet – COMP 575 (Chapter 2)
---

### 1. 🧩 **Structure of a Neuron**

Each neuron is a processing unit with three main components:

- **Synapses / Connections**:

  - Inputs $x_j$ come through synapses, each with a weight $w_{kj}$.
  - Each synapse applies a scaling: $x_j \cdot w_{kj}$.

- **Adder (Summing Junction)**:

  - All weighted inputs are summed:
    $u_k = \sum_{j=1}^{p} w_{kj} x_j$
    or $u_k = \mathbf{w}_k^T \mathbf{x}$

- **Activation Function**:

  - Applies non-linearity:
    $y_k = \phi(u_k)$

---

### 2. ⚖️ **Bias Term ($\theta_k$)**

- **Purpose**: Shifts activation threshold, improves flexibility.

- **Modified Equation**:
  $y_k = \phi(\mathbf{w}_k^T \mathbf{x} - \theta_k)$

- **Alternative Form** (with fixed input $x_0 = -1$):

  - Let $w_{k0} = \theta_k$ → included in the weight vector.
  - New input vector: $\mathbf{x} = [-1, x_1, ..., x_p]$

- **Effect**:
  $v_k = u_k - \theta_k$: the _induced local field_

---

### 3. 🔀 **Activation Functions**

> I wonder if we have to learn these.

#### 📌 Heaviside (Threshold)

- $\phi(v) = \begin{cases} 1 & v \geq 0 \\ 0 & v < 0 \end{cases}$
- Binary, non-differentiable, good for theoretical models

#### 📌 Piecewise Linear

- $\phi(v) = \begin{cases} 1 & v \geq \frac{1}{2} \\ v + \frac{1}{2} & -\frac{1}{2} < v < \frac{1}{2} \\ 0 & v \leq -\frac{1}{2} \end{cases}$
- Smooth approximation of threshold

#### 📌 Sigmoid (Logistic)

- $\phi(v) = \frac{1}{1 + e^{-\alpha v}}$
- Differentiable, output in (0,1), used for binary classification

#### 📌 Tanh

- $\phi(v) = \tanh(v) = \frac{e^{2v} - 1}{e^{2v} + 1}$
- Output in (–1, 1), zero-centered

#### 📌 ReLU

- $\phi(v) = \max(0, v)$
- Fast to compute, used in modern deep nets (e.g., CNNs)
- **Issue**: Dead neurons (if $v < 0$ permanently)

---

### 4. 🏗️ **Network Architectures**

#### 📘 Single-Layer Feedforward

- One input layer directly connected to one output layer.
- No hidden layers, no cycles.
- Simple computation:
  $y_k = \phi\left( \sum w_{kj} x_j \right)$

#### 📘 Multi-Layer Feedforward (MLP)

- Input → Hidden → Output layers.
- Each layer receives signals from previous.
- Fully connected (dense): all nodes connect to all in next layer.

#### 📘 CNN (Convolutional Neural Network)

- Uses **convolutional filters** for spatial pattern detection.
- Includes:

  - **Convolution layers** (with ReLU)
  - **Pooling** (e.g., max pooling)
  - **Fully connected layers**
  - **Softmax output**

- Great for image classification and recognition

#### 📘 RNN (Recurrent Neural Network)

- Has **feedback loops**, enabling memory of previous outputs.
- Uses **unit delay** elements $z^{-1}$
- Suitable for sequences, e.g., time series, language.

---

### 5. 📐 **Mathematical Expressions by Topology**

| Network Type     | Expression                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Single-Layer** | $y_k(\mathbf{x}) = \phi\left( \sum_{j=0}^{p} w_{kj} x_j \right)$                                                                               |
| **Multi-Layer**  | $y_k(\mathbf{x}) = \phi\left( \sum_{j=0}^{p_{\text{hidden}}} w_{kj} \cdot \phi\left( \sum_{i=0}^{p_{\text{input}}} w_{ji} x_i \right) \right)$ |
| **Recurrent**    | $y_k(n) = \phi\left( \sum_{i=0}^{p_{\text{input}}} w_{ki} x_i(n) + \sum_{j=0}^{p_{\text{output}}} w_{kj} y_j(n-1) \right)$                     |

---

### 6. 🧠 **Knowledge Representation**

- **NNs aim to model knowledge** through learning.

- Knowledge must come from **observations/samples**, which can be:

  - **Noisy / incomplete / redundant**
  - **Labelled** (with output $y$) or **unlabelled**

- **Labelled data**: Used in supervised learning
  e.g. $x =$ features, $y =$ diagnosis

---

### 7. 🧪 **Training and Testing Workflow**

1. **Choose appropriate architecture**:

   - Based on task and input/output size

2. **Prepare Training Dataset**:

   - Subset of data used to adjust weights

3. **Testing Dataset**:

   - New/unseen data to evaluate generalization

4. **Assess model**:

   - Metrics: **accuracy**, **generalization**

---

### 8. 📊 **Applications**

| **Field**            | **x (Input)**      | **y (Output)**   |
| -------------------- | ------------------ | ---------------- |
| **Medical**          | Patient data       | Disease yes/no   |
| **OCR**              | Pixel strokes      | Letter (A–Z)     |
| **Maths/Regression** | $x$                | $f(x)$           |
| **Defence**          | Sensor data        | Threat category  |
| **Weather**          | Current conditions | Future forecast  |
| **Finance**          | Prices now         | Prices tomorrow  |
| **Vehicle Guidance** | Sensors            | Direction, speed |
| **Electronics**      | Specs              | Optimal circuit  |
| **Games**            | Character state    | Enemy reaction   |

---
# File: 03.md
---

# 🧠 Neural Network Learning: Concepts, Mechanisms, and Paradigms

## 🔷 **1. General Concept of Learning in Neural Networks**

Learning in neural networks (NNs) involves **adapting internal parameters** (e.g., synaptic weights) in response to the environment in order to improve performance. The process typically involves:

- **Stimulation** from the environment.
- **Parameter updates** during training (training mode).
- **Adapted responses** in deployment (online mode).

### 🔀 Learning can be classified by:

- **Rules**: How weights are updated (e.g., error correction, Hebbian).
- **Paradigms**: How data is presented (e.g., supervised, unsupervised, reinforcement).

---

## 🔷 **2. Core Learning Rules**

### ✅ **Error-Correction Learning**

- Compares the output $y_k(n)$ to the desired $d_k(n)$, producing an **error signal** $e_k(n) = d_k(n) - y_k(n)$.
- Updates weights to minimize error using the **Delta Rule (Widrow-Hoff)**:

  $$
  \Delta w_{kj}(n) = \eta e_k(n) x_j(n)
  $$

- Learning progresses by **minimizing a cost function** $E(n) = \frac{1}{2} e_k^2(n)$.

---

### ✅ **Memory-Based Learning**

- Stores all training pairs $D_{\text{train}} = \{(x_i, d_i)\}$.
- For new inputs, retrieves closest past example (e.g., **1-Nearest Neighbour**):

  $$
  x^* = \arg\min_{x \in D_{\text{train}}} \|x - x_{\text{test}}\|_2
  $$

- Predicts using the stored output $d^*$.
- Extended to k-NN, weighted k-NN, and Parzen windows.

---

### ✅ **Hebbian Learning**

- Based on biological neurons:

  > “Neurons that fire together, wire together.”

- Weight update:

  $$
  \Delta w_{kj}(n) = \eta y_k(n) x_j(n)
  $$

- Encourages strengthening of connections when neurons activate simultaneously.
- **Limitation**: Unbounded weight growth → **synaptic saturation**.

#### 🛠 **Covariance Hypothesis** (Improved Hebbian Rule)

- Incorporates mean activity:

  $$
  \Delta w_{kj}(n) = \eta (y_k - \bar{y})(x_j - \bar{x})
  $$

- Allows both **synaptic strengthening and weakening**.
- Prevents saturation; leads to more stable learning.

---

### ✅ **Competitive Learning**

- Neurons **compete** to respond to inputs. Only **one wins**:

  $$
  y_k = \begin{cases} 1 & \text{if } v_k > v_j, \forall j \ne k \\ 0 & \text{otherwise} \end{cases}
  $$

- Weight update for winning neuron:

  $$
  \Delta w_{ki} = \eta(x_i - w_{ki})
  $$

- Encourages neurons to specialize in different input regions.
- Used in **clustering** and **Self-Organizing Maps (Kohonen networks)**.

---

### ✅ **Boltzmann Learning**

- Inspired by **statistical mechanics**; neurons are binary ($+1/-1$).
- Defines an **energy function**:

  $$
  E = -\frac{1}{2} \sum_{k \ne j} w_{kj} x_k x_j
  $$

- Neurons stochastically flip states with probability:

  $$
  p(x_k \rightarrow -x_k) = \frac{1}{1 + \exp(-\Delta E_k / T)}
  $$

- Learning updates weights using **correlations**:

  $$
  \Delta w_{kj} = \eta (\rho^+_{kj} - \rho^-_{kj})
  $$

  - $\rho^+_{kj}$: Clamped state correlation.
  - $\rho^-_{kj}$: Free-running state correlation.

---

## 🔷 **3. Credit Assignment Problem**

Refers to identifying **which internal decisions** deserve **credit or blame** for an outcome.

### Two forms:

- **Temporal**: Assigning credit to the right **moment** of action.
- **Structural**: Assigning credit to the **right part** of the system (e.g., a neuron or layer).

### Relevance:

- Crucial in **multi-layer NNs**, where hidden units indirectly influence results.

---

## 🔷 **4. Learning Paradigms**

### 🎓 **Supervised Learning** (Learning with a Teacher)

- Training set: $D_{\text{train}} = \{(x_i, d_i)\}$
- Uses **labeled examples** to guide the network.
- Learns by **minimizing error** between actual and desired output.
- After training, operates without the teacher in **online mode**.

---

### 🔍 **Unsupervised Learning** (Learning without a Teacher)

- No target labels.
- Learns by **identifying structure** (e.g., clusters) in the input data.
- Uses internal measures (e.g., similarity).
- **Competitive Learning** is a classic example.

---

### 🕹️ **Reinforcement Learning**

- Also without a teacher but guided by **rewards**.
- Agent learns through **trial and error**:

  - Observes state.
  - Takes action.
  - Receives reward.

- Objective: Learn a **policy** that **maximizes long-term reward**.
- Components:

  - **Policy**: Decision-making function.
  - **Learning algorithm**: Optimizes the policy.

---

## 🧠 Final Summary Table

| **Aspect**          | **Supervised**              | **Unsupervised**              | **Reinforcement**            |
| ------------------- | --------------------------- | ----------------------------- | ---------------------------- |
| **Teacher Present** | Yes                         | No                            | No                           |
| **Feedback Type**   | Error signal                | Data structure/statistics     | Reward signal                |
| **Goal**            | Learn input-output mapping  | Find hidden patterns/features | Maximize cumulative reward   |
| **Example Methods** | Backpropagation, Delta Rule | Competitive learning, k-NN    | Q-learning, Policy Gradients |

---
# File: 04.md
---
## **Adaptive Filtering & Learning Algorithms**
---

### 🔹 **1. Problem Setup: System Identification**

- We are modeling a dynamic system using discrete-time input-output pairs:

  $$
  D = \left\{(\mathbf{x}(i), d(i))\right\}, \quad \mathbf{x}(i) \in \mathbb{R}^p, \quad d(i) \in \mathbb{R}
  $$

- Input vector $\mathbf{x}(i) = (x_1(i), ..., x_p(i))^T$ is applied across $p$ nodes to produce an output $d(i)$.

#### **Two perspectives on input $\mathbf{x}(i)$:**

- **Spatial:** Inputs from different physical sources.
- **Temporal:** Current and past values of a signal.

---

### 🔹 **2. Linear Neuron Model**

- Uses a weighted sum:

  $$
  y(i) = \mathbf{x}(i)^T \mathbf{w}(i)
  $$

- The **error** is:

  $$
  e(i) = d(i) - y(i)
  $$

- The task is to **adapt the weights $\mathbf{w}$** to minimize this error.

---

## **Linear Least Squares (LLS)**

---

### 🔹 **3. Batch Learning with LLS**

- Minimize total squared error over all $n$ samples:

  $$
  E(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^{n} e(i)^2 = \frac{1}{2} \| \mathbf{d}(n) - \mathbf{X}(n)\mathbf{w} \|^2
  $$

- Optimal weights:

  $$
  \mathbf{w}(n+1) = \arg\min_{\mathbf{w}} E(\mathbf{w})
  $$

#### **Gauss-Newton Method:**

- In the linear case, converges in one step:

  $$
  \mathbf{w}(n+1) = \left(\mathbf{X}(n)^T \mathbf{X}(n)\right)^{-1} \mathbf{X}(n)^T \mathbf{d}(n)
  $$

- Or compactly, using the **Moore-Penrose pseudo-inverse**:

  $$
  \boxed{\mathbf{w}(n+1) = \mathbf{X}(n)^+ \mathbf{d}(n)}
  $$

---

## **Least Mean Square (LMS)**

---

### 🔹 **4. Online Learning with LMS**

- Updates weights using **one sample at a time**:

  $$
  E(n) = \frac{1}{2} e(n)^2
  $$

- Gradient descent update:

  $$
  \mathbf{w}(n+1) = \mathbf{w}(n) + \eta e(n) \mathbf{x}(n)
  $$

- $\eta$: learning rate — balances speed and stability.

  - **Small** $\eta$ → smooth adaptation, more memory.
  - **Large** $\eta$ → faster adaptation, less stability.

---

## **Perceptrons**

---

### 🔹 **5. Perceptron Model**

- Binary classifier based on:

  $$
  v_j = \mathbf{w}_j^T \mathbf{x} - \theta, \quad y_j = \phi(v_j) \in \{-1, +1\}
  $$

- $\phi(v) = \text{signum}(v)$: threshold-based **nonlinear activation**.

#### **Goal:**

- Classify data into two classes $C_1$ and $C_2$ by learning a **linear decision boundary** (hyperplane).

---

### 🔹 **6. Linear Separability**

- Training dataset $D_{\text{train}}$ is split into:

  - $D_1$ (class $C_1$): $\mathbf{w}^T \mathbf{x} > 0$
  - $D_2$ (class $C_2$): $\mathbf{w}^T \mathbf{x} \leq 0$

- If such a $\mathbf{w}$ exists, the data is **linearly separable**.

---

### 🔹 **7. Perceptron Training Algorithm**

1. **Initialize**: $\mathbf{w}(1) = 0$, learning rate $\eta \in (0,1]$
2. **Input**: Present $\mathbf{x}(t)$
3. **Response**:

   $$
   y(t) = \text{signum}(\mathbf{w}(t)^T \mathbf{x}(t))
   $$

4. **Update**:

   $$
   \mathbf{w}(t+1) = \mathbf{w}(t) + \eta \left[d(t) - y(t)\right] \mathbf{x}(t)
   $$

5. **Repeat** until all samples are correctly classified.

---

### 🔹 **8. Perceptron Convergence Theorem**

- **Guarantees** convergence in finite steps **if the data is linearly separable**.
- Each misclassification increases the alignment of $\mathbf{w}$ with the correct classification:

  $$
  \mathbf{w}(t+1)^T \mathbf{x}(t) > \mathbf{w}(t)^T \mathbf{x}(t)
  $$

---

## ✅ **Conclusion**

- **LLS**: Optimal, batch-based learning — closed-form solution.
- **LMS**: Efficient, online learning — sample-by-sample gradient descent.
- **Perceptron**: Nonlinear activation, binary classification — uses error correction for weight update.
# File: 05.md
---
## 🧠 Characteristics & Structure of MLPs

### ✅ Key Features:

1. **Nonlinear Activation Functions**:
  - E.g., Logistic sigmoid, hyperbolic tangent (tanh).
  - Allow approximation of complex, non-linear relationships.
  - Must be smooth and differentiable for gradient-based learning.

2. **Hidden Layers**:
  - Enable MLPs to learn **composite features**.
  - Essential for solving **non-linearly separable problems** (e.g., XOR).
  - Each hidden layer progressively extracts more abstract representations.

3. **Full Connectivity**:
  - Each neuron receives input from **all neurons** in the previous layer.
  - Only forward connections (no feedback loops).
---

## 🔍 Function of Hidden Layers

- Solve complex problems by **combining partial solutions** across neurons.
- Hidden layers let MLPs form **non-linear decision boundaries**.
- The more layers, the more **complex patterns** the network can capture (deep learning).

---

## 🔄 Backpropagation Algorithm (BP)

### 1. **Core Idea**:

- BP trains MLPs using **gradient descent**.
- Minimizes an error function by adjusting weights via **partial derivatives**.

### 2. **Error Definitions**:

- Output neuron error:

  $$
  e_j(n) = d_j(n) - y_j(n)
  $$

- Network error (for one pattern):

  $$
  E(n) = \frac{1}{2} \sum_j e_j^2(n)
  $$

- Average error across data:

  $$
  E_{\text{avg}} = \frac{1}{N} \sum_{n=1}^N E(n)
  $$

### 3. **Forward Pass**:

- Compute each neuron’s output:

  $$
  v_j(n) = \sum_i w_{ji}(n) y_i(n), \quad y_j(n) = \phi_j(v_j(n))
  $$

### 4. **Backward Pass**:

- Compute gradients:

  - **Output layer**:

    $$
    \delta_j(n) = e_j(n) \phi'_j(v_j(n))
    $$

  - **Hidden layer**:

    $$
    \delta_j(n) = \phi'_j(v_j(n)) \sum_k \delta_k(n) w_{kj}(n)
    $$

- Weight update:

  $$
  \Delta w_{ji}(n) = \eta \delta_j(n) y_i(n)
  $$

---

## 🔢 Activation Functions in BP

| Function    | Formula                       | Derivative (used in BP)                 |
| ----------- | ----------------------------- | --------------------------------------- |
| **Sigmoid** | $\frac{1}{1 + e^{-\alpha v}}$ | $\alpha y (1 - y)$                      |
| **Tanh**    | $\alpha \tanh(\beta v)$       | $\frac{\beta}{\alpha} (\alpha^2 - y^2)$ |

---

## 🏋️‍♂️ Training Modes

1. **Sequential Training (STM)**:

   - Updates weights after each pattern.
   - Less memory, can escape local minima.

2. **Batch Training (BTM)**:

   - Updates after all patterns (epoch).
   - Easier to parallelize and analyze.

3. **Mini-Batch Training (MBTM)**:

   - Standard method today.
   - Balance between STM and BTM.

---

## ⚡ Learning Speed

- **Learning Rate $\eta$**:

  - Controls weight update size.
  - Small $\eta$: slow, stable. Large $\eta$: fast, risky.

- **Momentum $\mu$**:

  - Adds previous weight update to current one:

    $$
    \Delta w_{ji}(n) = \eta \delta_j(n) y_i(n) + \mu \Delta w_{ji}(n - 1)
    $$

  - Helps escape shallow minima, reduces oscillations.

---

## 🧩 Generalization & Complexity Control

### ⚠️ Overfitting Risks:

- High-capacity MLPs (many layers) may learn noise in training data.

### ✅ Good Practices:

1. **Data Splitting**:

   - **Training set**: updates weights.
   - **Validation set**: tunes hyperparameters.
   - **Test set**: evaluates final performance.

2. **Early Stopping**:

   - Monitor validation error.
   - Stop when it **increases** (indicates overfitting).

---

## 🛡️ Regularization Techniques

### 1. **General Formula**:

$$
E(\mathbf{w}) = E_{\text{avg}}(\mathbf{w}) + \lambda E_S(\mathbf{w})
$$

### 2. **Types**:

| Technique                       | Description                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------- |
| **Weight Decay**                | Penalize large weights: $E_S = \sum_i w_i^2$                                 |
| **Penalty on Derivatives**      | Penalize lack of smoothness: minimize high-order derivatives of MLP function |
| **Weight Elimination**          | Soft penalty: small weights discouraged more than large ones                 |
| **Dropout**                     | Randomly deactivate neurons during training → train multiple subnetworks     |
| **Optimal Brain Damage (OBD)**  | Uses Hessian of error surface to prune least important weights               |
| **Optimal Brain Surgeon (OBS)** | Full-Hessian version of OBD for better precision in pruning                  |

---

## 🧠 Final Takeaway

Training MLPs using BP involves:

- A **clear learning rule (BP)** with forward and backward passes.
- Selection of **appropriate activation functions**.
- Choices in **training mode** and **learning rate** tuning.
- Avoiding overfitting via:

  - **Early stopping**
  - **Dropout**
  - **Regularization**
  - **Weight pruning (OBD/OBS)**

These concepts are the **foundation of modern neural networks** and remain essential in deep learning frameworks today.

---
# File: 06.md
---
# 🧠 Understanding RBF Networks – A Full Guide
---

## 1. 🎯 **Pattern Separability**

- **Goal**: Separate data points into different classes.
- **Challenge**: Some data cannot be linearly separated in its original space.
- **Solution**: Use a mapping function $\varphi(\mathbf{x})$ to transform the data into a **higher-dimensional space** where linear separation is possible.

### ✅ Cover’s Theorem

> Any complex (non-linearly separable) pattern set can become linearly separable when mapped to a high-enough dimension.

---

## 2. 💡 **φ-Separability Example: XOR Problem**

### XOR input:

- (0,0), (1,1) → Class 0
- (0,1), (1,0) → Class 1

In 2D, this is **not linearly separable**.

### Trick:

Map each point using 2 radial functions:

- $\varphi_1(\mathbf{x}) = \exp(-\|\mathbf{x} - \mathbf{c}_1\|^2), \quad \mathbf{c}_1 = (1,1)$
- $\varphi_2(\mathbf{x}) = \exp(-\|\mathbf{x} - \mathbf{c}_2\|^2), \quad \mathbf{c}_2 = (0,0)$

In the new space $[\varphi_1(x), \varphi_2(x)]$, the classes are now linearly separable!

---

## 3. 📈 **Interpolation with RBFs**

> Fit a smooth surface through given data points exactly.

Given:

- Points $\mathbf{x}_i$
- Target values $y_i$

### Define function:

$$
F(\mathbf{x}) = \sum_{i=1}^{N} w_i \, \varphi(\|\mathbf{x} - \mathbf{x}_i\|)
$$

Each $\varphi$ is an **RBF**, often a Gaussian or similar function.

### Matrix form:

$$
\Phi \mathbf{w} = \mathbf{y} \quad \Rightarrow \quad \mathbf{w} = \Phi^{-1} \mathbf{y}
$$

Where $\Phi$ is an N×N matrix of RBF activations.

---

## 4. 🔍 **Types of RBFs**

### Common choices:

| Type                  | Formula                                                |
| --------------------- | ------------------------------------------------------ |
| Multiquadrics         | $\varphi(r) = \sqrt{r^2 + c^2}$                        |
| Inverse multiquadrics | $\varphi(r) = \frac{1}{\sqrt{r^2 + c^2}}$              |
| Gaussian              | $\varphi(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right)$ |

- **Gaussian and inverse multiquadrics** are **localized** — they respond strongly only near their centers.

---

## 5. 🧮 **RBF Neural Networks**

### Structure:

1. **Input layer**: Inputs go in.
2. **Hidden layer**: Applies RBFs (usually Gaussian) at various centres.
3. **Output layer**: Computes a weighted sum of hidden layer outputs.

### Function:

$$
F(\mathbf{x}) = \sum_{i=1}^{M} w_i \, \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_i\|^2}{2\sigma_i^2}\right)
$$

---

## 6. 🛠️ **Training an RBF Network**

### Objective:

Minimize total error:

$$
E(\mathbf{w}) = \sum \left[F(\mathbf{x}_i) - y_i\right]^2 + \lambda \|D\mathbf{F}\|^2
$$

- First term: fitting error.
- Second term: regularization (helps avoid overfitting).
- Solved using:

$$
\mathbf{w} = (\Phi^T\Phi + \lambda I)^{-1} \Phi^T \mathbf{y}
$$

👉 Uses the **pseudoinverse**.

---

## 7. ⚙️ **RBF Learning Strategies**

### Strategy A: **Fixed Random Centres**

- Randomly choose M data points as RBF centres.
- Set $\sigma = d_{\text{max}} / \sqrt{2M}$
- Solve weights using least squares.

### Strategy B: **Self-Organised Centres (Clustering)**

- Use K-means or similar to find clusters.
- Cluster centers become RBF centres.
- Spread of cluster = RBF width.
- Solve weights as before.

---

## 8. 🎓 **Supervised Centre Selection**

> Learn **all** parameters (weights, centres, widths) using gradient descent.

### Updates:

1. **Weights**:

   $$
   w_i(n+1) = w_i(n) - \eta_1 \cdot \frac{\partial E}{\partial w_i}
   $$

2. **Centres**:

   $$
   c_{ik}(n+1) = c_{ik}(n) - \eta_2 \cdot \frac{\partial E}{\partial c_{ik}}
   $$

3. **Widths**:

   $$
   \sigma_i(n+1) = \sigma_i(n) - \eta_3 \cdot \frac{\partial E}{\partial \sigma_i}
   $$

Each parameter is updated based on its gradient and a learning rate $\eta$.

---

## 9. 🔁 **RBFs vs MLPs (Multilayer Perceptrons)**

| Feature                 | RBF Network                            | MLP Network                               |
| ----------------------- | -------------------------------------- | ----------------------------------------- |
| Purpose                 | Nonlinear mapping                      | Nonlinear mapping                         |
| Approximation Type      | Localized (only responds near centers) | Global (neuron affects large input areas) |
| Structure               | Typically 1 hidden layer               | 1 or more hidden layers                   |
| Hidden Layer Activation | Based on distance to center            | Based on dot product with weights         |
| Hidden Nodes            | Nonlinear (RBFs)                       | Nonlinear                                 |
| Output Nodes            | Linear                                 | Nonlinear (usually)                       |
| Training Focus          | Centres & widths first, then weights   | Weights throughout all layers             |

---

## ✅ Final Summary

- RBFs use **localized** nonlinear activations and are great for **interpretable** and **smooth** mappings.
- MLPs use **global** activation patterns and are better for more **complex or deep architectures**.
- RBF networks can **learn well with fewer layers** and offer good **control over smoothness** via widths and regularization.

---
# File: 07.md
---
# ✅ Support Vector Machines (SVMs) – A Complete Summary
---

## 1. **Goal of SVMs**

SVMs are supervised learning models used for **binary classification**.
They work by finding the **optimal hyperplane** that separates data points of two different classes with the **maximum margin**.

- Dataset:
  $D = \{ (\mathbf{x}_i, d_i) \}$, where:

  - $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector,
  - $d_i \in \{+1, -1\}$ is the class label.

- If the data is **linearly separable**, we aim to find:

  $$
  g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = 0
  $$

  which separates the two classes.

---

## 2. **Margin and Separation**

- The **margin** $\rho$ is the distance between the decision boundary and the closest data points from each class.
- Our goal is to **maximize** this margin to improve generalization.

**Constraints:**

$$
\begin{cases}
\mathbf{w}^T \mathbf{x}_i + b \geq +1 & \text{if } d_i = +1 \\
\mathbf{w}^T \mathbf{x}_i + b \leq -1 & \text{if } d_i = -1
\end{cases}
\Rightarrow d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
$$

---

## 3. **Support Vectors**

- Points that **lie on the margin boundaries** are called **support vectors**.
- These are critical for determining the optimal hyperplane.
- The margin is:

  $$
  \rho = \frac{1}{\|\mathbf{w}\|_2}
  \Rightarrow \text{maximize } \rho \equiv \text{minimize } \frac{1}{2} \|\mathbf{w}\|^2
  $$

---

## 4. **SVM Optimization (Primal and Dual)**

### Primal Problem:

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
$$

### Dual Formulation (using Lagrange multipliers $\lambda_i$):

$$
\max_{\lambda} \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i,j} \lambda_i \lambda_j d_i d_j \mathbf{x}_i^T \mathbf{x}_j
$$

$$
\text{s.t.} \quad \sum \lambda_i d_i = 0,\quad \lambda_i \geq 0
$$

- Once solved, we recover $\mathbf{w} = \sum \lambda_i d_i \mathbf{x}_i$

---

## 5. **Soft-Margin SVM for Non-Separable Data**

Real-world data often can't be separated perfectly. To handle this, we allow for some errors using **slack variables** $\xi_i \geq 0$.

### Modified Primal Problem:

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum \xi_i
\quad \text{s.t.} \quad d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

- $C$ is a **regularization parameter**:

  - Large $C$: penalizes errors heavily (fit the training data).
  - Small $C$: allows more flexibility (generalizes better).

---

## 6. **Kernels for Nonlinear Boundaries**

In many problems, even soft-margin SVMs can’t separate the classes linearly.
We use **Kernels** to map data into a **higher-dimensional feature space** where separation **is** possible.

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \varphi(\mathbf{x}_i)^T \varphi(\mathbf{x}_j)
$$

### Updated Dual Problem:

$$
\max \sum_i \lambda_i - \frac{1}{2} \sum_{i,j} \lambda_i \lambda_j d_i d_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

> We **never compute** $\varphi(\mathbf{x})$ explicitly. We only define the **kernel function** $K$.

---

### Common Kernels:

| Kernel Type    | Expression $K(\mathbf{x}, \mathbf{y})$                                | Comments                   |
| -------------- | --------------------------------------------------------------------- | -------------------------- |
| Polynomial     | $(\mathbf{x}^T \mathbf{y} + 1)^p$                                     | $p$ = degree               |
| RBF (Gaussian) | $\exp\left( -\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2} \right)$ | $\sigma$ = spread width    |
| Tanh           | $\tanh(\alpha \mathbf{x}^T \mathbf{y} + \beta)$                       | $\alpha, \beta$: constants |

---

## 🎯 Final Takeaways

- SVMs aim to find the **widest possible margin** between classes.
- They work great for both **linearly and non-linearly separable data**.
- Soft margins allow for **real-world errors**.
- Kernels allow for **nonlinear boundaries** without complex math.
- Support vectors are the **only points** that determine the final classifier.

---
# File: 08.md
---
# 🧠 Reinforcement Learning (RL) – Combined Explanation
---

## 🌱 **What is Reinforcement Learning?**

Reinforcement Learning is a type of machine learning where an **agent learns by interacting with an environment**, making decisions (actions), and receiving feedback (rewards).

- The goal: **Maximize total rewards over time.**
- Inspired by **trial-and-error learning**, like how humans or animals learn from experiences.

---

## 👨‍🏫 **How RL Compares to Other Learning Types**

| Learning Type     | Description                                          | Example                           |
| ----------------- | ---------------------------------------------------- | --------------------------------- |
| Supervised        | Learn from labeled data (teacher shows right answer) | Recognizing cats in images        |
| Unsupervised      | Find patterns in data without labels                 | Customer clustering               |
| **Reinforcement** | Learn from rewards and penalties over time           | Playing chess, training a chatbot |

**Key Difference:**

- In **supervised learning**, the label is always correct and fixed.
- In **RL**, the reward changes and depends on the agent's actions over time.

---

## ⚙️ **Key Components of RL**

1. **Agent** – The learner or decision-maker.
2. **Environment** – Everything the agent interacts with.
3. **State (s)** – The current situation.
4. **Action (a)** – What the agent can do.
5. **Reward (r)** – Feedback from the environment.
6. **Policy (π)** – Strategy: which action to take in each state.
7. **Value Function (V or Q)** – Predicts future rewards.
8. **Model (optional)** – Simulates environment behavior.

---

## 🧩 **Markov Decision Process (MDP)**

An RL problem is usually described as an **MDP**, which has:

- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probabilities (what happens next)
- **R**: Reward function (feedback)
- **ρ**: Initial state distribution

**Markov Property**: The future depends **only** on the current state and action, not the full history.

---

## 🔁 **How RL Works (Step-by-Step)**

1. Agent sees a **state**.
2. Picks an **action** using its policy.
3. Action changes the environment → new **state** + **reward**.
4. Agent **updates** its knowledge based on reward.
5. Repeat.

This feedback loop continues until the agent learns the best actions.

---

## ⚖️ **Exploration vs Exploitation**

- **Exploration** = Try new actions to learn more.
- **Exploitation** = Use what you already know to get high reward.
- Good RL agents balance both.

**Example:**

- Choosing a new restaurant (exploration) vs. going to your favorite (exploitation).

---

## 🎲 **Action Selection Methods**

- **Greedy**: Always choose the action with the highest reward so far.
- **ε-greedy**: Choose the best most of the time, but explore randomly with small chance ε.
- **Softmax**: Choose actions randomly based on their estimated value (better actions have higher chance).
- **Prioritized Experience Replay**: Sample training data with higher learning potential (based on error).

---

## 💡 **Q-Learning: Core Algorithm in RL**

Q-learning learns the **action-value function** Q(s, a):

- This estimates the **total expected reward** if you take action `a` in state `s` and act optimally after.

**Update rule (Bellman Equation):**

```
Q(s, a) = r + γ * max Q(s', a')
```

- `r` = reward
- `γ` = discount factor (importance of future rewards)

Use:

```
π(s) = argmax Q(s, a)
```

→ Choose the action with the highest Q-value in that state.

---

## 🧠 **Problems with Traditional Q-Learning**

- Needs a **big Q-table** → infeasible with many states/actions.
- Solution: use a **neural network** to approximate Q-values → this is called **Deep Q-Learning (DQN).**

---

## 🤖 **Deep Q-Learning (DQN)**

Instead of a table, use a **Deep Neural Network** to learn `Q(s, a)`. But this introduces new challenges:

### 🧱 Challenges:

1. **Unstable targets**: Q-values change as the model learns → leads to instability.
2. **Correlated data**: Consecutive experiences are not independent (unlike supervised learning).
3. **Moving targets**: The model learns from itself, causing feedback loops.

---

## 🛠️ **Solutions for Stable DQN Training**

### ✅ **Experience Replay**

- Store past experiences in a buffer.
- Randomly sample to break correlations and make learning more stable.

### ✅ **Target Network**

- Use **two networks**:
  - One for choosing actions (updated frequently).
  - One for calculating target Q-values (updated slowly).
- Prevents the model from chasing itself too quickly.

---

## 🧪 **Other Enhancements**

### 🎯 **Double DQN**

- Reduces **overestimation** of Q-values by separating action selection from target value calculation.

### 💥 **Dueling DQN**

- Separates Q into:
  - **V(s)**: Value of being in a state.
  - **A(s, a)**: Advantage of taking an action in that state.
- Learns better in states where actions don’t matter much.

---

## 🤖 **Practical Example: RL Chatbot**

- Takes an action (says something).
- Gets feedback from the user (response).
- Converts this to reward.
- Learns to talk in ways that maximize reward.

---

## 🧾 **Summary**

| Concept                    | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| **Reinforcement Learning** | Learn by doing and getting feedback                        |
| **Q-Learning**             | Learn value of actions in each state                       |
| **DQN**                    | Neural network version of Q-learning                       |
| **Key Challenges**         | Moving targets, instability, data correlation              |
| **Stabilization Tricks**   | Experience Replay, Target Network, Double DQN, Dueling DQN |

---
# File: 09.md
---

## 🧠 **Mathematical Optimisation Overview**

**Optimisation** is the process of finding the **best** design or solution for a system or model. It involves:

- Choosing the **right parameters**,
- Following **mathematical relationships**, goals, or constraints,
- And maximising or minimising an **objective function**.

Optimisation is useful in almost every scientific and engineering field.

---

### 📊 **Types of Optimisation Problems**

#### ✅ **Unconstrained Optimisation**

- Find the best value of a function with **no limits** on the parameters.
- Example: Maximise \( f(x, y) = x \cdot y \)
  - You can pick any values for \( x \) and \( y \).
  - Shown using 3D and contour plots.

#### ✅ **Constrained Optimisation**

- Find the best value of a function **under specific conditions**.
- Example:
  - Maximise \( f(x, y) = x \cdot y \)
  - Subject to:
    - \( g(x, y) = 2(x + y) - 8.2 \leq 0 \)
    - \( h(x, y) = 0.5\sqrt{x^2 + y^2} - 1.8 = 0 \)
  - The best point must:
    - Be on the surface defined by \( h \),
    - Not go above the surface defined by \( g \),
    - And still maximise \( f \).

---

## ⚙️ **Optimisation Methods**

### 1. **Enumerative Methods**

- Try **all possible values** of the parameters.
- For continuous parameters, create a **discretised version** (fixed intervals).
- ✅ **Simple**, guarantees finding the best.
- ❌ **Very slow** and **impractical** for high-dimensional problems (many variables).
- Suffers from the **curse of dimensionality**: performance drops as variables increase.

---

### 2. **Gradient-based Methods**

- Use the **gradient** (slope) of the function to guide the search.
- Formula:
  \[
  \vec{x}(t+1) = \vec{x}(t) - \eta \nabla f[\vec{x}(t)]
  \]
  - \( \eta \) = step size (learning rate),
  - \( \nabla f \) = gradient (shows direction to move).
- ✅ **Fast** for smooth, simple problems.
- ❌ Can get stuck in **local minima** and miss the **global minimum** if not started near it.
- Good at solving problems using local information.

---

### 3. **Stochastic Methods**

- Use **randomness** to explore different solutions.
- Don’t rely only on local information—can explore wider space.
- ✅ Better at finding the **global optimum**, avoids getting stuck.
- ❌ May be slower or require tuning.

#### Types:

- **Random walk**: Try completely random directions (simple but weak).
- **Genetic algorithms**: Mimic evolution—select, combine, and mutate solutions over generations.
- Called **directed random search** when randomness is used in a smart, guided way.

---

## 🔚 Summary Comparison

| Method         | Strategy                     | Strengths                     | Weaknesses                        |
| -------------- | ---------------------------- | ----------------------------- | --------------------------------- |
| Enumerative    | Try everything               | Guaranteed best result        | Extremely slow for large problems |
| Gradient-based | Follow the slope             | Fast and efficient            | Can miss global best              |
| Stochastic     | Try guided random directions | Explores better, finds global | May need many tries               |

---
# File: 10.md
---
# 🧬 **Genetic Algorithms (GAs) – Overview and Key Concepts**
---

## 🔍 What Are Genetic Algorithms?

Genetic Algorithms (GAs) are **search and optimization techniques** inspired by the process of **natural evolution** (like Darwin’s natural selection). They simulate how living things evolve to find better solutions over generations.

- They use ideas from **genetics**: selection, crossover, mutation.
- They work with a **population** of candidate solutions.
- Each solution is like an **individual**, with:

  - **Genotype**: encoded string (e.g. binary)
  - **Phenotype**: actual result (e.g. a number or path)

---

## ⚙️ Main Stages in a Genetic Algorithm

1. **Genetic Coding**

   - Convert each possible solution into a "chromosome" (e.g., a binary string).

2. **Population**

   - A group of solutions is maintained. Should be large enough for variety, but small enough for speed.

3. **Fitness Evaluation**

   - Check how good each solution is by computing a **fitness score**.

4. **Selection**

   - Better solutions are more likely to be chosen to create new ones. (Fitness-proportionate selection is common.)

5. **Crossover**

   - Combine parts of two parent solutions to form children. Helps **exploit** good solutions.

6. **Mutation**

   - Randomly flip bits in children to introduce variety. Helps **explore** new possibilities.

7. **Replacement**

   - New children replace the old population, and the process repeats.

---

## 📉 Genetic Encoding (Binary)

- Variables are encoded as binary strings.
- Each variable `xᵢ` in a range `[xᵢ,min, xᵢ,max]` is represented using `l₀` bits.
- Decoding:

  $$
  xᵢ = xᵢ,min + \left( \frac{xᵢ,max - xᵢ,min}{2^{l₀} - 1} \right) × D
  $$

  where `D` is the decimal value of the binary string.

- The **precision** depends on `l₀` — more bits = finer granularity.

---

## 🧱 Chromosome Shapes & Crossover Types

GAs can use **non-string structures** like:

- **(A)** 1D string → 3-point crossover
- **(B)** 2D matrix → row-based crossover
- **(C)** 2D matrix → two-line crossover
- **(D)** Graph → subgraph crossover
- **(E)** Set → subset crossover

This flexibility allows GAs to solve **complex, structured problems**.

---

## 🔀 Crossover – Mixing Genes

- Purpose: **Exploit good solutions** by combining parent traits.
- Random crossover point chosen.
- Produces two children from parts of two parents.

**Example**:

| Parent 1 | 001000‖0010 |
| -------- | ----------- |
| Parent 2 | 001100‖1100 |

→ Children:

- Child 1: `0010001100`
- Child 2: `0011000010`

⚠️ Too much similarity between parents can hurt diversity.

---

## ⚡ Mutation – Adding Randomness

- Purpose: **Explore** new areas, maintain diversity.
- Small chance for each bit to flip (e.g., 1 → 0).
- Prevents the algorithm from getting **stuck** with "almost good" solutions.

**Example**:

Before:

- `0010001100` →
- `0011000010`

After mutation:

- `0010101100`
- `0001000010`

⚠️ Mutation rate must be low — too high makes the search random.

---

## ✅ Advantages of Genetic Algorithms

- Work well with both **continuous and discrete problems**.
- Handle **discontinuous, noisy, and complex search spaces**.
- Can **search many solutions at once** (population-based).
- Work even when calculus-based methods don’t.
- Useful in a wide range of applications:

  - Machine learning
  - Game playing
  - NP-hard problems
  - Adaptive control systems

---

## ⚠️ Drawbacks of Genetic Algorithms

- **No guarantee of convergence** (may not find the best solution).
- Can be **slow** for complex problems.
- Needs **careful tuning** (mutation rate, population size, etc.).
- General GAs are often **less efficient** than problem-specific ones.
- Risk of **premature convergence** if diversity is lost.

---

## 🎯 Final Thoughts

Genetic Algorithms are **powerful and flexible**, especially for problems that are too hard for traditional methods.
But they require **careful design**, a good balance between **exploration (mutation)** and **exploitation (crossover)**, and **parameter tuning** for best performance.

---

# Example GA:

```python
import random

# Target number we want to reach
TARGET = 2001

# Genetic Algorithm parameters
POPULATION_SIZE = 10
MUTATION_RATE = 0.1
GENERATIONS = 50


# Create a random individual (a number between 0 and 4000)
def create_individual():
    return random.randint(0, 4000)


# Create the initial population
def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


# Fitness function: higher is better
def fitness(individual):
    return 1 / (1 + abs(TARGET - individual))


# Select parents using fitness-proportionate selection
def select_parents(population):
    total_fitness = sum(fitness(ind) for ind in population)
    probabilities = [fitness(ind) / total_fitness for ind in population]
    return random.choices(population, weights=probabilities, k=2)


# Crossover: average of two parents
def crossover(parent1, parent2):
    return (parent1 + parent2) // 2


# Mutation: small random change
def mutate(individual):
    if random.random() < MUTATION_RATE:
        change = random.randint(-10, 10)
        return max(0, individual + change)
    return individual


# Run the Genetic Algorithm
def run_ga():
    population = create_population()
    best_solution = None
    best_fitness = 0

    print(f"Starting Genetic Algorithm to find number closest to {TARGET}")
    print("-" * 50)

    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        best_solution = max(population, key=fitness)
        current_fitness = fitness(best_solution)

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            print(
                f"Generation {generation + 1}: New best solution = {best_solution} (fitness: {best_fitness:.4f})"
            )

    print("-" * 50)
    print(f"Final result: {best_solution}")
    print(f"Distance from target: {abs(TARGET - best_solution)}")
    return best_solution


if __name__ == "__main__":
    best = run_ga()
```
# File: 11.md
---
# ✅ **Genetic Algorithms – Full Summary**
---

## 🧬 1. Fitness Evaluation and Transformation

### 🔹 **Basic Idea**

- Each candidate solution (chromosome) is given a **fitness value** $f(c_i)$, which determines how good it is.
- Better fitness → higher chance of selection.

### 🔹 **Fitness Calculation**

$$
p_i = \frac{f(c_i)}{\sum_{k=1}^{N} f(c_k)}
$$

### 🔹 **Why Transform Fitness?**

- Raw values might be negative or too close together.
- Can cause:

  - **Close-race**: all individuals look similar → random walk.
  - **Super-individual**: one dominates too early → premature convergence.

---

## 🔹 **Fitness Transformation Schemes**

| Method                  | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| **Direct Scaling**      | Subtract worst fitness: $f(c_i) = F(c_i) - F_{\text{worst}}$             |
| **Linear Scaling**      | Stretch or shift values: $f(c_i) = aF(c_i) + b$                          |
| **Sigma Truncation**    | Reduce fitness below average: $f(c_i) = \max(0, F(c_i) + \mu - a\sigma)$ |
| **Power Scaling**       | Raise values to power: $f(c_i) = F(c_i)^a$                               |
| **Exponential Scaling** | Use exponential growth: $f(c_i) = \exp\left(\frac{F(c_i)}{a}\right)$     |

---

## 🔹 **Ranking-Based Transformation**

Instead of raw values, use **ranks**.

### 📊 Baker’s Linear Ranking

$$
p_i = \frac{1}{N} \left( \eta_{\text{max}} - (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{i-1}{N-1} \right)
$$

- $\eta_{\text{max}} \in [1,2]$, $\eta_{\text{min}} = 2 - \eta_{\text{max}}$

### Other Ranking Types:

- **Nonlinear ranking**: uses inverse of rank.
- **Geometric**: $p_i = \frac{\eta(1 - \eta)^{i-1}}{c}$
- **Exponential**: $p_i = \frac{1 - e^i}{c}$

---

## 🔁 2. Selection Schemes

| Scheme                  | Description                                      |
| ----------------------- | ------------------------------------------------ |
| **Roulette Wheel**      | Random selection based on fitness (probability)  |
| **Stochastic Sampling** | No or partial replacement reduces repeat picks   |
| **Universal Sampling**  | Picks several at fixed intervals around roulette |
| **k-Tournament**        | Pick k random individuals; keep the best         |

### 📏 Takeover Time

- **Short** → fast domination (high pressure)
- **Long** → slow domination (more exploration)

---

## 🔄 3. Crossover (Recombination)

### 🔹 **Discrete Crossover Types**

| Type         | Description                                     |
| ------------ | ----------------------------------------------- |
| **χ-point**  | Swap at χ crossover points                      |
| **Uniform**  | Pick gene randomly from either parent (50/50)   |
| **Binomial** | Gene from parent1 with prob $p_r$, else parent2 |

- **Small χ**: Low disruption (positional bias)
- **Large χ**: High diversity, risk of instability (distributional bias)

---

### 🔹 **Real-Valued Crossover Types**

| Type                          | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| **Line Arithmetical**         | Offspring between parents: $c = r p_i + (1-r) p_j$  |
| **Intermediate Arithmetical** | Each gene uses a different r                        |
| **Heuristic Arithmetical**    | Extrapolates away from the worse parent             |
| **Simplex Crossover**         | Uses multiple parents, rejects worst, moves from it |

---

## 🧪 4. Mutation

### 🔹 **Discrete (Bit-flipping)**

- Flip bits (0 ↔ 1) with small mutation rate $p_m$
- **Adaptive Mutation**: Adjust based on similarity between parents:

$$
p_m(p_i, p_j) = p_{mL} + (p_{mU} - p_{mL}) \cdot \left( \frac{I(p_i, p_j)}{\text{len}} \right)^2
$$

---

### 🔹 **Real-Valued Mutation**

| Type                     | Description                                        |
| ------------------------ | -------------------------------------------------- |
| **Uniform**              | Random new value within range $[L_k, U_k]$         |
| **Gaussian**             | Small noise from normal distribution               |
| **Adaptive Non-Uniform** | Mutation gets smaller over time (fine-tunes later) |

---

## ⛔ 5. Invalid Chromosomes

### 💣 Causes

- Constraint violation
- Over-expressive encoding or crossover

### ✅ Handling Methods

| Method                | Description                                               |
| --------------------- | --------------------------------------------------------- |
| **Rejection**         | Discard invalid solutions and retry                       |
| **Penalty Functions** | Add penalties to fitness: $F'(x) = F(x) + \text{penalty}$ |
| **Repair**            | Fix invalid solutions directly                            |
| **Special Operators** | Use custom operators to avoid invalid results             |
| **Decoders**          | Chromosome maps to valid solution using external logic    |

---

## 👨‍👩‍👧‍👦 6. Population Models

### 🔹 Generational Model

- Whole population replaced each generation
- Use **elitism** to keep the best

### 🔹 Overlapping / Steady-State Model

- Only a few new individuals replace a few old ones
- Generation gap = $M / N$
- Often includes **no-duplicates rule**

### 🔹 Population Size (N)

- Can be fixed or dynamic
- Often set as $N = 10 \times \text{dimension}$

---

## 🧭 7. Termination Criteria

| Rule                                  | Meaning                           |
| ------------------------------------- | --------------------------------- |
| **Fixed generations**                 | Stop after $t_{max}$ steps        |
| **Best ≈ Average fitness**            | No progress                       |
| **Best hasn’t improved recently**     | Stalled                           |
| **Low diversity**                     | Population is too similar         |
| **Mathematical condition** (e.g. KKT) | Optimization conditions satisfied |

---

## 🌐 8. Parallel Populations

### 🔹 Fine-Grained (Cellular)

- Individuals on a grid
- Interact with neighbors only

### 🔹 Coarse-Grained (Island Model)

- Multiple subpopulations (islands)
- Occasionally exchange individuals (migration)

**Benefits:**

- Prevents premature convergence
- Solves multimodal problems well

---

## 🌱 9. Hybridisation

| Strategy                     | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| **Smart initial population** | Use known good solutions                               |
| **Custom operators**         | Tailored to encoding/problem                           |
| **Post-GA local search**     | Apply gradient descent or similar after GA finishes    |
| **Memetic GA**               | Apply local search during evolution                    |
| **Lamarckian Model**         | Improvements passed to offspring                       |
| **Baldwinian Model**         | Offspring learn faster but don’t inherit direct traits |

---

## 🧭 10. Niching & Diversity

### Purpose:

- **Find multiple good solutions** (multimodal optimization)
- Prevent population collapse into one peak

### Key Idea: **Fitness Sharing**

- Share fitness among similar individuals to avoid crowding

$$
F_{\text{shared}}(p_i) = \frac{F(p_i)^b}{\sum_{j=1}^{N} \gamma(d(p_i, p_j))}
$$

Where:

$$
\gamma(d) = \max \left[0, 1 - \left( \frac{d}{\sigma_{\text{share}}} \right)^a \right]
$$

- $\sigma_{\text{share}}$: niche radius
- $d(p_i, p_j)$: distance between individuals

---
# File: 12.md
---
# **Theoretical Analysis of Genetic Algorithms**
---

This is non-examinable.
# File: 13.md
---

## 🧬 **Genetic Programming (GP)** – Overview

**GP** is a method where computers evolve programs (solutions) automatically using biological ideas like selection, crossover, and mutation.

- Instead of just adjusting values (like Genetic Algorithms), GP changes both:

  - The **structure** (form) of the program,
  - And the **parameters** (numbers/inputs) in it.

---

## 🌳 **GP Representation**

- Programs are represented as **trees**:

  - **Functions** go in the **branches** (e.g., `+`, `*`, `if`, `sin`)
  - **Terminals** (numbers or variables) go in the **leaves**

- Example Tree:
  `(2 + N) × 4`
  Looks like:

  ```
      ×
     / \
    +   4
   / \
  2   N
  ```

---

## 🧱 GP Components

### Terminal Set (T):

- Endpoints of trees.
- Could be constants (like `2`) or variables (like `x`, `y`).

### Function Set (F):

- Internal nodes like `+`, `-`, `*`, `if`, `loop`, `sin`, etc.
- Should be small (for speed) but expressive (for solving problems).

---

## ⚙️ GP Operators

### 1. **Selection**:

- Pick better programs to survive or mate (based on fitness).

### 2. **Crossover**:

- Swap parts (subtrees) of two programs to create new ones.

### 3. **Mutation**:

- Randomly change one function or terminal in a tree.

### GP Flow (Loop):

1. While the population isn't full:

   - Decide to mutate or crossover.
   - If mutate: pick 1, change a part.
   - If crossover: pick 2, swap parts.
   - Add to new generation.

2. Repeat for many generations.

---

## 🌱 Tree Initialization Methods

### 1. **Full Method**:

- All branches go to max depth (`Dmax`).
- Functions inside, terminals at the end.

### 2. **Grow Method**:

- Varying depths.
- Mixed functions and terminals inside.

### 3. **Ramped Half-and-Half**:

- Mix of Full and Grow to get diverse trees.

---

## ❌ Introns in GP

**Introns** are parts of the program that **do nothing**.

- Example: an unused `if` statement.
- They can:

  - ✅ Protect good parts from bad crossover
  - ❌ Make programs longer, slower, and harder to understand

### Controlling Introns:

- Penalize complex programs using:

  - Node count
  - Instruction count
  - CPU load

---

## 🧠 Gene Expression Programming (GEP)

GEP is similar to GP but uses **fixed-length strings (like DNA)** to represent programs.

### Genotype vs Phenotype:

- **Genotype** = the gene string (e.g., `Q * + - a b c d`)
- **Phenotype** = the tree it decodes into

### Decoding:

- Read the string left to right
- Fill the tree level-by-level
- Use each function’s required number of inputs

---

## 💡 GEP Encoding Example:

Math formula:

```
√((a + b) × (c − d))
```

GEP gene string:

```
Q * + - a b c d
```

Turns into tree:

```
       Q
       |
       *
     /   \
    +     -
   / \   / \
  a   b c   d
```

---

## ✨ GEP Advantages over GP

1. **Fixed-length strings** (easier to manage)
2. **Always valid programs** (no broken trees)
3. **Non-coding regions** (help mutation/crossover without damage)

### Head and Tail:

- **Head**: can have functions and terminals
- **Tail**: only terminals
- Example:

  ```
  + Q - / b * a a Q b a  a b b a
  ```

---

## 🔁 GEP Mutation Example

Original gene:

```
+ Q - / b * a a Q **b** a  a b b a
```

Mutate `"b"` → `"+"` (a function from head set)

Result:

```
+ Q - / b * a a Q **+** a  a b b a
```

This mutation changes the structure of the resulting tree and increases its depth.

---

## 🔀 GEP Crossover Example

Parent 1:

```
- b + Q b b a b b / a Q b b b a a b
```

Parent 2:

```
/ - a / a b a b b - b a - a b a a a
```

Cut and swap after a point:

Child 1:

```
- b + / a b a b b - b a - a b a a a
```

Child 2:

```
/ - a Q b b a b b / a Q b b b a a b
```

This introduces **diversity**, but may disrupt useful structures.

---

## 📊 GEP in Symbolic Regression

Goal: Find this function using basic math (no `^` allowed):

```
f(a) = a⁴ + a³ + a² + a
```

GEP finds:

```
* + + / * * a a a a a a a a
```

Which builds this tree:

```
       *
     /   \
    +     +
   / \   / \
  /   * *   a
 a a a a a a
```

This results in:

```
(1 + a²) × (a + a²) = a + a² + a³ + a⁴
```

🎉 Perfect match!

---
# File: 14.md
---

## 🔷 Overview of PSO

**Particle Swarm Optimization (PSO)** is a nature-inspired optimization method invented in 1995 by Kennedy & Eberhart.

### 🐦 Basic Idea (Bird Flock Analogy)

- Particles = Birds
- Each particle:

  - Remembers its **own best position** (pbest)
  - Follows the **best position in the swarm** (gbest)

- Goal: Find the best solution (like birds searching for food)

---

## 🧮 Standard PSO Algorithm

### **1. Initialization**

- Randomly generate:

  - **Position** `xᵢ = [xᵢ₁, ..., xᵢd]`
  - **Velocity** `vᵢ = [vᵢ₁, ..., vᵢd]`

### **2. Velocity Update**

$$
v_{ij} = w \cdot v_{ij} + c_1 \cdot rand \cdot (pbest_{ij} - x_{ij}) + c_2 \cdot rand \cdot (gbest_j - x_{ij})
$$

- `w`: inertia (momentum)
- `c₁`: cognitive (personal memory)
- `c₂`: social (group memory)
- `rand`: random number between 0 and 1

### **3. Position Update**

$$
x_{ij} = x_{ij} + v_{ij}
$$

### **4. Update pbest and gbest**

$$
pbest_i(t+1) = \begin{cases}
pbest_i(t), & \text{if } F(x_i(t+1)) \geq F(pbest_i(t)) \\
x_i(t+1), & \text{otherwise}
\end{cases}
$$

$$
gbest(t+1) = \arg\min F(pbest_i(t+1))
$$

---

## 🔁 PSO Variations

To improve performance, several variations of PSO have been proposed.

---

### **1. Time Varying Inertia Weight (TVIW)**

- Modify `w` over time:

$$
w = (w_{init} - w_{final}) \cdot \frac{t_{final} - t}{t_{final}} + w_{final}
$$

🔹 **Purpose**:

- Early: large `w` → **exploration**
- Late: small `w` → **fine-tuning**

---

### **2. Constriction Factor (CF) PSO**

- New velocity formula:

$$
v_{ij} = \chi \left[v_{ij} + c_1 \cdot rand \cdot (pbest_{ij} - x_{ij}) + c_2 \cdot rand \cdot (gbest_j - x_{ij})\right]
$$

$$
\chi = \frac{2}{|2 - \varphi - \sqrt{\varphi^2 - 4\varphi}|}, \quad \varphi = c_1 + c_2, \quad \varphi > 4
$$

🔹 **Purpose**:

- Controls speed
- Prevents particles from overshooting
- Improves convergence stability

---

### **3. Time Varying Acceleration Coefficients (TVAC)**

$$
c_i = (c_{i,final} - c_{i,init}) \cdot \frac{t}{t_{final}} + c_{i,init}, \quad i = 1,2
$$

🔹 **Example Values**:

- Start: `c₁ = 2.5`, `c₂ = 0.5` (explore individually)
- End: `c₁ = 0.5`, `c₂ = 2.5` (follow the group)

🔹 **Purpose**:

- Balances exploration vs exploitation across generations

---

### **4. Time Varying Acceleration Coefficients with Mutation (TVAC-M)**

- Add mutation when no improvement:

```text
If rate_of_improvement ≤ threshold:
    If rand < pₘ:
        v_kj := v_kj ± v_max / m
```

🔹 **Purpose**:

- Prevents stagnation
- Adds diversity randomly when stuck

---

### **5. Passive Congregation (PC) PSO**

$$
v_{ij} = w \cdot v_{ij} + c_1 \cdot rand \cdot (pbest_{ij} - x_{ij}) + c_2 \cdot rand \cdot (gbest_j - x_{ij}) + c_3 \cdot rand \cdot (q_j - x_{ij})
$$

- `q` = randomly selected particle
- `c₃` = congregation strength

🔹 **Purpose**:

- Adds natural drift toward others (even non-leaders)
- Inspired by passive gathering (e.g., plankton drifting)

---

### **6. LBEST PSO**

- Swarm is divided into **local neighborhoods**
- Each neighborhood has its own **local best**

🔹 **Purpose**:

- Avoids getting stuck in one spot
- Allows parallel search of different regions

---

### **7. Social Stereotyping PSO**

- Particles grouped by **clustering algorithm**
- Each group has a **cluster center `uₖ`**

🔹 **Velocity Update Options**:

$$
v_i = w \cdot v_i + c_1 \cdot rand \cdot (u_k - x_i) + c_2 \cdot rand \cdot (gbest - x_i)
$$

$$
v_i = w \cdot v_i + c_1 \cdot rand \cdot (pbest_i - x_i) + c_2 \cdot rand \cdot (u_k - x_i)
$$

$$
v_i = w \cdot v_i + c_1 \cdot rand \cdot (u_k - x_i) + c_2 \cdot rand \cdot (u_k - x_i)
$$

🔹 **Purpose**:

- Allows smarter subgroup behavior
- Helps divide work among particle subpopulations

---

## ✅ Summary Table

| Variation        | Changes What?           | Key Benefit                           |
| ---------------- | ----------------------- | ------------------------------------- |
| **TVIW**         | `w` over time           | Good start + better finish            |
| **CF**           | Adds factor `χ`         | Smooth convergence                    |
| **TVAC**         | `c₁`, `c₂` over time    | Balance individual vs social learning |
| **TVAC-M**       | Adds mutation           | Escapes stagnation                    |
| **PC**           | Adds passive drift term | Natural grouping                      |
| **LBEST**        | Uses local bests        | Multi-region search                   |
| **Stereotyping** | Uses cluster centers    | Better group behavior                 |

---
# File: 15.md
---

## 🧬 **Evolution Strategies (ES) - Summary**

### 🔹 Overview

- **ES** is one of the earliest evolutionary algorithms designed for **numerical optimization** (finding minimum/maximum values of functions).
- It uses **real-valued solutions** and works by exploring the solution space using **mutation only**—no crossover or selection pressure in the basic version.
- Can operate with a **very small population**, even just **1 parent + 1 child** (called **(1+1)-ES**).
- The search process is:

  1. Start from an initial point $x^{(0)}$
  2. Add a random change: $x^{(t+1)} = x^{(t)} + r$ where $r \sim \mathcal{N}(0, \Sigma)$
  3. Accept the new point if it improves the objective $F$, otherwise stay with the current one

---

### 🔹 Encoding

Each solution is encoded as:

$$
(x, \Sigma) = [x_1, ..., x_d, \sigma_1, ..., \sigma_k] \quad \text{where } k = \frac{d(d+1)}{2}
$$

- $\Sigma$ is the **covariance matrix**:

  - Diagonal = step sizes (variances)
  - Off-diagonal = rotation angles (correlation between variables)

- Both $x$ and $\Sigma$ can **co-evolve** — the algorithm adapts its search strategy over time.

Basic mutation:

$$
x^{(t+1)} = x^{(t)} + \mathcal{N}(0, \Sigma^{(t)}), \quad \Sigma^{(t+1)} = \Sigma^{(t)} = \sigma I
$$

This is a **simple case** where the step size is fixed and the same in all directions.

---

### 🔹 Theoretical Result

$$
\lim_{t \to \infty} F(x^{(t)}) = F(x_{\text{optimal}})
$$

- The algorithm **will eventually reach** the optimal value.
- But this doesn't help estimate how long it will take — hence, extensions are introduced.

---

## 🔧 Extensions to Improve ES

### 1️⃣ **The 1/5 Success Rule**

- **Goal**: Automatically adjust mutation strength based on recent performance.
- **Rule**: If more than 1 in 5 mutations are successful (i.e. improve the solution), the step size is **too small** → **increase it**. Otherwise, **decrease it**.

Update formula:

$$
\sigma^{(t+1)} =
\begin{cases}
c_{\text{dec}} \cdot \sigma^{(t)} & \text{if } R_k < 1/5 \\
c_{\text{inc}} \cdot \sigma^{(t)} & \text{if } R_k > 1/5 \\
\sigma^{(t)} & \text{if } R_k = 1/5
\end{cases}
$$

- Typical constants: $c_{\text{inc}} = 1.22$, $c_{\text{dec}} \in [0.8, 1.0]$

---

### 2️⃣ **(μ+λ)-ES**

- Use **μ parents** to produce **λ offspring**.
- Then select the best **μ individuals** from the **combined pool (μ + λ)** for the next generation.
- Keeps **strong parents alive** longer.

✔️ Helps avoid premature convergence.

---

### 3️⃣ **(μ,λ)-ES**

- Again, μ parents produce λ children.
- But now select the **next generation only from the λ offspring**.
- Each individual lives only **one generation**.

✔️ Better for **dynamic or noisy** environments.

---

### 4️⃣ **ES with Crossover**

- Like Genetic Algorithms, ES can use **crossover** to combine two parents.

Each parent looks like:

$$
[x_1, \Sigma_1] = [x_{11}, ..., x_{1d}, \sigma_{11}, ..., \sigma_{1k}]
$$

$$
[x_2, \Sigma_2] = [x_{21}, ..., x_{2d}, \sigma_{21}, ..., \sigma_{2k}]
$$

- Crossover types:

  - **Discrete crossover** (e.g., uniform or 2-point)
  - **Real-valued crossover** (e.g., average values)

After crossover, **mutation** is applied.

⚠️ Must ensure the **covariance matrix stays valid**.

---

### 5️⃣ **Updating the Covariance Matrix (Full Form)**

To evolve the **strategy itself**, update both diagonal and off-diagonal elements of Σ:

- **Update variances**:

  $$
  \sigma_{ii}^{(t+1)} = \sigma_{ii}^{(t)} \cdot e^{N(0, \Delta\sigma)}
  $$

- **Update rotation angles**:

  $$
  \alpha_{ij}^{(t+1)} = \alpha_{ij}^{(t)} + N(0, \Delta\alpha) \quad \text{for } i \ne j
  $$

- Then:

  $$
  x^{(t+1)} = x^{(t)} + \mathcal{N}(0, \Sigma^{(t+1)})
  $$

- **Covariance matrix**:

  $$
  \Sigma =
  \begin{bmatrix}
  \sigma_{11} & \dots & \sigma_{1d} \\
  \vdots & \ddots & \vdots \\
  \sigma_{d1} & \dots & \sigma_{dd}
  \end{bmatrix}
  $$

- Rotation angle formula:

  $$
  \tan(2\alpha_{ij}) = \frac{2\sigma_{ij}}{\sigma_{ii} - \sigma_{jj}}
  $$

---

## 🧁 Summary Table

| Feature            | Purpose / Behavior                              |
| ------------------ | ----------------------------------------------- |
| (1+1)-ES           | Simple: 1 parent, 1 child, accept if better     |
| 1/5 Rule           | Adjust step size based on success rate          |
| (μ+λ)-ES           | Keep best μ from μ + λ (parents and children)   |
| (μ,λ)-ES           | Keep best μ from children only                  |
| Crossover          | Combine traits from two parents before mutation |
| Covariance Updates | Adapt how mutation behaves, including direction |

---
# File: formulas.md
# 🔢 Essential Formulas in Computational Intelligence

## 🧠 Neural Networks

### Basic Neuron

- **Neuron Output**: Sum of weighted inputs through activation function
  ```
  y = φ(Σ wᵢxᵢ + b)
  ```
  where φ is activation function, w are weights, x are inputs, b is bias

### Activation Functions

- **Sigmoid**: Smooth, differentiable, outputs between 0 and 1

  ```
  φ(v) = 1 / (1 + e⁻ᵛ)
  ```

- **Tanh**: Like sigmoid but centered at 0, outputs between -1 and 1

  ```
  φ(v) = (e²ᵛ - 1) / (e²ᵛ + 1)
  ```

- **ReLU**: Simple, fast, helps with vanishing gradient
  ```
  φ(v) = max(0, v)
  ```

### Backpropagation

- **Output Layer Error**: Difference between target and actual output

  ```
  δⱼ = (dⱼ - yⱼ) × φ'(vⱼ)
  ```

- **Hidden Layer Error**: Error propagated from next layer

  ```
  δⱼ = φ'(vⱼ) × Σ(δₖwₖⱼ)
  ```

- **Weight Update**: Learning from errors
  ```
  Δwᵢⱼ = η × δⱼ × yᵢ
  ```
  where η is learning rate

## 📊 Support Vector Machines

- **Decision Boundary**: Hyperplane separating classes

  ```
  wᵀx + b = 0
  ```

- **Margin**: Distance to closest data points (support vectors)

  ```
  margin = 2 / ||w||
  ```

- **Kernel Trick**: Transform to higher dimension
  ```
  K(x,y) = φ(x)ᵀφ(y)
  ```

## 🎲 Reinforcement Learning

- **Q-Learning Update**: Learn action values

  ```
  Q(s,a) = Q(s,a) + α[r + γ×max(Q(s',a')) - Q(s,a)]
  ```

  where α is learning rate, γ is discount factor

- **Policy**: Probability of taking action in state
  ```
  π(a|s) = e^(Q(s,a)/τ) / Σ(e^(Q(s,a')/τ))
  ```
  where τ is temperature parameter

## 🧬 Genetic Algorithms

- **Selection Probability**: Chance of being selected based on fitness

  ```
  p(i) = f(i) / Σf(j)
  ```

- **Rank-Based Selection**: Selection based on rank instead of raw fitness
  ```
  p(i) = (2-s)/N + 2i(s-1)/(N(N-1))
  ```
  where s is selection pressure

## 🐦 Particle Swarm Optimization

- **Velocity Update**: How particles move in search space

  ```
  v = w×v + c₁r₁(pbest - x) + c₂r₂(gbest - x)
  ```

  where w is inertia, c₁,c₂ are learning factors

- **Position Update**: New position based on velocity
  ```
  x = x + v
  ```

## 📈 Evolution Strategies

- **Mutation**: Gaussian perturbation of solution

  ```
  x' = x + σ×N(0,1)
  ```

  where σ is step size

- **1/5 Success Rule**: Adapt step size
  ```
  σ' = σ × (success_rate > 0.2 ? 1.22 : 0.82)
  ```

## 🌳 Genetic Programming

- **Tree Size**: Number of nodes in program tree

  ```
  size = 1 + Σ(size of children)
  ```

- **Program Fitness**: Usually includes size penalty
  ```
  fitness = accuracy - λ×size
  ```
  where λ controls complexity penalty

## 🔍 Optimization

- **Gradient Descent**: Basic optimization step

  ```
  x' = x - η∇f(x)
  ```

  where η is step size, ∇f is gradient

- **Momentum**: Helps escape local minima
  ```
  v = μv - η∇f(x)
  x' = x + v
  ```
  where μ is momentum coefficient
