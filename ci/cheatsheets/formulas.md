# 🔢 Essential Formulas in Computational Intelligence

## 🧠 Neural Networks

### Basic Neuron

- **Neuron Output**: Sum of weighted inputs through activation function
  $$
  y = \phi\left(\sum_{i=1}^{p} w_i x_i + b\right)
  $$
  where $\phi$ is activation function, $w$ are weights, $x$ are inputs, $b$ is bias

### Activation Functions

- **Heaviside (Threshold)**:

  $$
  \phi(v) = \begin{cases} 1 & v \geq 0 \\ 0 & v < 0 \end{cases}
  $$

- **Piecewise Linear**:

  $$
  \phi(v) = \begin{cases} 1 & v \geq \frac{1}{2} \\ v + \frac{1}{2} & -\frac{1}{2} < v < \frac{1}{2} \\ 0 & v \leq -\frac{1}{2} \end{cases}
  $$

- **Sigmoid**: Smooth, differentiable, outputs between 0 and 1

  $$
  \phi(v) = \frac{1}{1 + e^{-\alpha v}}
  $$

  where $\alpha$ controls steepness

- **Tanh**: Like sigmoid but centered at 0, outputs between -1 and 1

  $$
  \phi(v) = \tanh(v) = \frac{e^{2v} - 1}{e^{2v} + 1}
  $$

- **ReLU**: Simple, fast, helps with vanishing gradient
  $$
  \phi(v) = \max(0, v)
  $$

### Network Architectures

- **Single-Layer Feedforward**:

  $$
  y_k(\mathbf{x}) = \phi\left( \sum_{j=0}^{p} w_{kj} x_j \right)
  $$

- **Multi-Layer Feedforward**:
  $$
  y_k(\mathbf{x}) = \phi\left( \sum_{j=0}^{p_{\text{hidden}}} w_{kj} \cdot \phi\left( \sum_{i=0}^{p_{\text{input}}} w_{ji} x_i \right) \right)
  $$

### Learning Rules

- **Error-Correction (Delta Rule)**:

  $$
  \Delta w_{kj}(n) = \eta \cdot e_k(n) \cdot x_j(n)
  $$

  where $\eta$ is learning rate, $e$ is error

- **Hebbian Learning**:

  $$
  \Delta w_{kj}(n) = \eta \cdot y_k(n) \cdot x_j(n)
  $$

- **Covariance Hypothesis**:

  $$
  \Delta w_{kj}(n) = \eta \cdot (y_k - \bar{y})(x_j - \bar{x})
  $$

- **Competitive Learning**:
  $$
  y_k = \begin{cases} 1 & \text{if } v_k > v_j, \forall j \ne k \\ 0 & \text{otherwise} \end{cases}
  $$

### Adaptive Filtering

- **Linear Least Squares**:

  $$
  \mathbf{w}(n+1) = \mathbf{X}(n)^+ \mathbf{d}(n)
  $$

  where $\mathbf{X}^+$ is Moore-Penrose pseudo-inverse

- **Least Mean Square (LMS)**:
  $$
  \mathbf{w}(n+1) = \mathbf{w}(n) + \eta \cdot e(n) \cdot \mathbf{x}(n)
  $$

### Backpropagation

- **Output Layer Error**: Difference between target and actual output

  $$
  \delta_j = (d_j - y_j) \cdot \phi'(v_j)
  $$

- **Hidden Layer Error**: Error propagated from next layer

  $$
  \delta_j = \phi'(v_j) \cdot \sum_k \delta_k w_{kj}
  $$

- **Weight Update**: Learning from errors

  $$
  \Delta w_{ij} = \eta \cdot \delta_j \cdot y_i
  $$

  where $\eta$ is learning rate

- **Momentum Update**:
  $$
  \Delta w_{ij}(n) = \eta \cdot \delta_j \cdot y_i + \mu \cdot \Delta w_{ij}(n-1)
  $$
  where $\mu$ is momentum coefficient

## 📊 Support Vector Machines

- **Decision Boundary**: Hyperplane separating classes

  $$
  \mathbf{w}^T \mathbf{x} + b = 0
  $$

- **Margin**: Distance to closest data points (support vectors)

  $$
  \rho = \frac{2}{\|\mathbf{w}\|}
  $$

- **Primal Problem**:

  $$
  \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
  $$

- **Dual Problem**:

  $$
  \max_{\lambda} \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i,j} \lambda_i \lambda_j d_i d_j \mathbf{x}_i^T \mathbf{x}_j
  $$

  $$
  \text{s.t.} \quad \sum \lambda_i d_i = 0, \quad \lambda_i \geq 0
  $$

- **Soft Margin**:

  $$
  \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum \xi_i
  $$

  $$
  \text{s.t.} \quad d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
  $$

- **Kernel Trick**: Transform to higher dimension
  $$
  K(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x})^T \phi(\mathbf{y})
  $$

## 🎲 Reinforcement Learning

- **Q-Learning Update**: Learn action values

  $$
  Q(s,a) = Q(s,a) + \alpha[r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a)]
  $$

  where $\alpha$ is learning rate, $\gamma$ is discount factor

- **Policy**: Probability of taking action in state
  $$
  \pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}
  $$
  where $\tau$ is temperature parameter

## 🧬 Genetic Algorithms

- **Selection Probability**: Chance of being selected based on fitness

  $$
  p(i) = \frac{f(i)}{\sum_j f(j)}
  $$

- **Baker's Linear Ranking**:

  $$
  p_i = \frac{1}{N} \left( \eta_{\text{max}} - (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{i-1}{N-1} \right)
  $$

- **Fitness Sharing**:
  $$
  F_{\text{shared}}(p_i) = \frac{F(p_i)^b}{\sum_{j=1}^{N} \gamma(d(p_i, p_j))}
  $$
  where
  $$
  \gamma(d) = \max \left[0, 1 - \left( \frac{d}{\sigma_{\text{share}}} \right)^a \right]
  $$

## 🐦 Particle Swarm Optimization

- **Velocity Update**: How particles move in search space

  $$
  v_{ij} = w \cdot v_{ij} + c_1 \cdot rand \cdot (pbest_{ij} - x_{ij}) + c_2 \cdot rand \cdot (gbest_j - x_{ij})
  $$

- **Position Update**: New position based on velocity

  $$
  x_{ij} = x_{ij} + v_{ij}
  $$

- **Time Varying Inertia Weight**:

  $$
  w = (w_{\text{init}} - w_{\text{final}}) \cdot \frac{t_{\text{final}} - t}{t_{\text{final}}} + w_{\text{final}}
  $$

- **Constriction Factor**:
  $$
  \chi = \frac{2}{|2 - \varphi - \sqrt{\varphi^2 - 4\varphi}|}, \quad \varphi = c_1 + c_2, \quad \varphi > 4
  $$

## 📈 Evolution Strategies

- **Basic Mutation**:

  $$
  x^{(t+1)} = x^{(t)} + \mathcal{N}(0, \Sigma^{(t)})
  $$

- **1/5 Success Rule**: Adapt step size

  $$
  \sigma^{(t+1)} = \begin{cases}
  c_{\text{dec}} \cdot \sigma^{(t)} & \text{if } R_k < 1/5 \\
  c_{\text{inc}} \cdot \sigma^{(t)} & \text{if } R_k > 1/5 \\
  \sigma^{(t)} & \text{if } R_k = 1/5
  \end{cases}
  $$

- **Covariance Matrix Update**:
  $$
  \sigma_{ii}^{(t+1)} = \sigma_{ii}^{(t)} \cdot e^{\mathcal{N}(0,\Delta\sigma)}
  $$
  $$
  \alpha_{ij}^{(t+1)} = \alpha_{ij}^{(t)} + \mathcal{N}(0,\Delta\alpha)
  $$

## 🌳 Genetic Programming

- **Tree Size**: Number of nodes in program tree

  $$
  \text{size} = 1 + \sum_{\text{children}} \text{size}
  $$

- **Program Fitness**: Usually includes size penalty
  $$
  \text{fitness} = \text{accuracy} - \lambda \cdot \text{size}
  $$
  where $\lambda$ controls complexity penalty

## 🔍 Optimization

- **Gradient Descent**: Basic optimization step

  $$
  \mathbf{x}' = \mathbf{x} - \eta \nabla f(\mathbf{x})
  $$

  where $\eta$ is step size, $\nabla f$ is gradient

- **Momentum**: Helps escape local minima
  $$
  \mathbf{v} = \mu \mathbf{v} - \eta \nabla f(\mathbf{x})
  $$
  $$
  \mathbf{x}' = \mathbf{x} + \mathbf{v}
  $$
  where $\mu$ is momentum coefficient
