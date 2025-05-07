# 🔢 Essential Formulas in Computational Intelligence

## 🧠 Neural Networks

### Basic Neuron

- **Neuron Output**: Sum of weighted inputs through activation function
  $$
  y = \phi\left(\sum_{i=1}^{p} w_i x_i + b\right)
  $$
  where:
  - $\phi$ = activation function (e.g., sigmoid, ReLU)
  - $w_i$ = weight for input i
  - $x_i$ = input value i
  - $b$ = bias term
  - $p$ = number of inputs

### Activation Functions

- **Heaviside (Threshold)**:

  $$
  \phi(v) = \begin{cases} 1 & v \geq 0 \\ 0 & v < 0 \end{cases}
  $$

  where $v$ = input value. Outputs 1 if input ≥ 0, 0 otherwise.

- **Piecewise Linear**:

  $$
  \phi(v) = \begin{cases} 1 & v \geq \frac{1}{2} \\ v + \frac{1}{2} & -\frac{1}{2} < v < \frac{1}{2} \\ 0 & v \leq -\frac{1}{2} \end{cases}
  $$

  where $v$ = input value. Smooth approximation of threshold function.

- **Sigmoid**: Smooth, differentiable, outputs between 0 and 1

  $$
  \phi(v) = \frac{1}{1 + e^{-\alpha v}}
  $$

  where:

  - $v$ = input value
  - $\alpha$ = steepness parameter (larger = steeper curve)
  - Output range: (0,1)

- **Tanh**: Like sigmoid but centered at 0, outputs between -1 and 1

  $$
  \phi(v) = \tanh(v) = \frac{e^{2v} - 1}{e^{2v} + 1}
  $$

  where $v$ = input value. Output range: (-1,1)

- **ReLU**: Simple, fast, helps with vanishing gradient
  $$
  \phi(v) = \max(0, v)
  $$
  where $v$ = input value. Outputs input if positive, 0 if negative.

### Network Architectures

- **Single-Layer Feedforward**:

  $$
  y_k(\mathbf{x}) = \phi\left( \sum_{j=0}^{p} w_{kj} x_j \right)
  $$

  where:

  - $y_k$ = output of neuron k
  - $\mathbf{x}$ = input vector
  - $w_{kj}$ = weight from input j to neuron k
  - $p$ = number of inputs

- **Multi-Layer Feedforward**:
  $$
  y_k(\mathbf{x}) = \phi\left( \sum_{j=0}^{p_{\text{hidden}}} w_{kj} \cdot \phi\left( \sum_{i=0}^{p_{\text{input}}} w_{ji} x_i \right) \right)
  $$
  where:
  - $y_k$ = output of neuron k in output layer
  - $w_{kj}$ = weight from hidden neuron j to output k
  - $w_{ji}$ = weight from input i to hidden neuron j
  - $p_{\text{hidden}}$ = number of hidden neurons
  - $p_{\text{input}}$ = number of input neurons

### Learning Rules

- **Error-Correction (Delta Rule)**:

  $$
  \Delta w_{kj}(n) = \eta \cdot e_k(n) \cdot x_j(n)
  $$

  where:

  - $\Delta w_{kj}$ = weight change
  - $\eta$ = learning rate (step size)
  - $e_k$ = error at output k
  - $x_j$ = input j
  - $n$ = time step

- **Hebbian Learning**:

  $$
  \Delta w_{kj}(n) = \eta \cdot y_k(n) \cdot x_j(n)
  $$

  where:

  - $\Delta w_{kj}$ = weight change
  - $\eta$ = learning rate
  - $y_k$ = output of neuron k
  - $x_j$ = input j
  - $n$ = time step

- **Covariance Hypothesis**:

  $$
  \Delta w_{kj}(n) = \eta \cdot (y_k - \bar{y})(x_j - \bar{x})
  $$

  where:

  - $\Delta w_{kj}$ = weight change
  - $\eta$ = learning rate
  - $y_k$ = output of neuron k
  - $\bar{y}$ = mean output
  - $x_j$ = input j
  - $\bar{x}$ = mean input

- **Competitive Learning**:

  $$
  y_k = \begin{cases} 1 & \text{if } v_k > v_j, \forall j \ne k \\ 0 & \text{otherwise} \end{cases}
  $$

  where:

  - $y_k$ = output of neuron k
  - $v_k$ = activation of neuron k
  - Only the neuron with highest activation outputs 1

- **Induced Local Field**:

  $$
  v_k = u_k - \theta_k
  $$

  where:

  - $v_k$ = induced local field
  - $u_k$ = sum of weighted inputs
  - $\theta_k$ = bias term

- **Boltzmann Learning Energy**:

  $$
  E = -\frac{1}{2} \sum_{k \ne j} w_{kj} x_k x_j
  $$

  where:

  - $E$ = energy of the network
  - $w_{kj}$ = weight between neurons k and j
  - $x_k, x_j$ = states of neurons k and j

- **Boltzmann Learning Probability**:
  $$
  p(x_k \rightarrow -x_k) = \frac{1}{1 + \exp(-\Delta E_k / T)}
  $$
  where:
  - $p$ = probability of flipping neuron k's state
  - $\Delta E_k$ = change in energy if neuron k flips
  - $T$ = temperature parameter (controls randomness)

### Adaptive Filtering

- **Linear Least Squares**:

  $$
  \mathbf{w}(n+1) = \mathbf{X}(n)^+ \mathbf{d}(n)
  $$

  where:

  - $\mathbf{w}$ = weight vector
  - $\mathbf{X}^+$ = Moore-Penrose pseudo-inverse of input matrix
  - $\mathbf{d}$ = desired output vector
  - $n$ = time step

- **Least Mean Square (LMS)**:
  $$
  \mathbf{w}(n+1) = \mathbf{w}(n) + \eta \cdot e(n) \cdot \mathbf{x}(n)
  $$
  where:
  - $\mathbf{w}$ = weight vector
  - $\eta$ = learning rate
  - $e$ = error
  - $\mathbf{x}$ = input vector
  - $n$ = time step

### Backpropagation

- **Output Layer Error**: Difference between target and actual output

  $$
  \delta_j = (d_j - y_j) \cdot \phi'(v_j)
  $$

  where:

  - $\delta_j$ = error at output neuron j
  - $d_j$ = desired output
  - $y_j$ = actual output
  - $\phi'$ = derivative of activation function
  - $v_j$ = weighted sum of inputs

- **Hidden Layer Error**: Error propagated from next layer

  $$
  \delta_j = \phi'(v_j) \cdot \sum_k \delta_k w_{kj}
  $$

  where:

  - $\delta_j$ = error at hidden neuron j
  - $\phi'$ = derivative of activation function
  - $v_j$ = weighted sum of inputs
  - $\delta_k$ = error at next layer neuron k
  - $w_{kj}$ = weight from j to k

- **Weight Update**: Learning from errors

  $$
  \Delta w_{ij} = \eta \cdot \delta_j \cdot y_i
  $$

  where:

  - $\Delta w_{ij}$ = weight change
  - $\eta$ = learning rate
  - $\delta_j$ = error at neuron j
  - $y_i$ = output of neuron i

- **Momentum Update**:
  $$
  \Delta w_{ij}(n) = \eta \cdot \delta_j \cdot y_i + \mu \cdot \Delta w_{ij}(n-1)
  $$
  where:
  - $\Delta w_{ij}$ = weight change
  - $\eta$ = learning rate
  - $\delta_j$ = error at neuron j
  - $y_i$ = output of neuron i
  - $\mu$ = momentum coefficient
  - $n$ = time step

## 📊 Support Vector Machines

- **Decision Boundary**: Hyperplane separating classes

  $$
  \mathbf{w}^T \mathbf{x} + b = 0
  $$

  where:

  - $\mathbf{w}$ = weight vector (normal to hyperplane)
  - $\mathbf{x}$ = input vector
  - $b$ = bias term

- **Margin**: Distance to closest data points (support vectors)

  $$
  \rho = \frac{2}{\|\mathbf{w}\|}
  $$

  where:

  - $\rho$ = margin width
  - $\|\mathbf{w}\|$ = magnitude of weight vector

- **Primal Problem**:

  $$
  \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
  $$

  where:

  - $\mathbf{w}$ = weight vector
  - $b$ = bias term
  - $d_i$ = class label (-1 or +1)
  - $\mathbf{x}_i$ = input vector i

- **Dual Problem**:

  $$
  \max_{\lambda} \sum_{i=1}^N \lambda_i - \frac{1}{2} \sum_{i,j} \lambda_i \lambda_j d_i d_j \mathbf{x}_i^T \mathbf{x}_j
  $$

  $$
  \text{s.t.} \quad \sum \lambda_i d_i = 0, \quad \lambda_i \geq 0
  $$

  where:

  - $\lambda_i$ = Lagrange multipliers
  - $d_i$ = class labels
  - $\mathbf{x}_i$ = input vectors
  - $N$ = number of training examples

- **Soft Margin**:

  $$
  \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum \xi_i
  $$

  $$
  \text{s.t.} \quad d_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
  $$

  where:

  - $\mathbf{w}$ = weight vector
  - $b$ = bias term
  - $\xi_i$ = slack variables (allow misclassification)
  - $C$ = regularization parameter

- **Kernel Trick**: Transform to higher dimension
  $$
  K(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x})^T \phi(\mathbf{y})
  $$
  where:
  - $K$ = kernel function
  - $\phi$ = feature mapping
  - $\mathbf{x}, \mathbf{y}$ = input vectors

## 🎲 Reinforcement Learning

- **Q-Learning Update**: Learn action values

  $$
  Q(s,a) = Q(s,a) + \alpha[r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a)]
  $$

  where:

  - $Q(s,a)$ = action-value for state s and action a
  - $\alpha$ = learning rate
  - $r$ = immediate reward
  - $\gamma$ = discount factor
  - $s'$ = next state
  - $a'$ = next action

- **Policy**: Probability of taking action in state
  $$
  \pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}
  $$
  where:
  - $\pi(a|s)$ = probability of action a in state s
  - $Q(s,a)$ = action-value
  - $\tau$ = temperature parameter
  - $a'$ = all possible actions

## 🧬 Genetic Algorithms

- **Selection Probability**: Chance of being selected based on fitness

  $$
  p(i) = \frac{f(i)}{\sum_j f(j)}
  $$

  where:

  - $p(i)$ = probability of selecting individual i
  - $f(i)$ = fitness of individual i
  - $j$ = all individuals

- **Baker's Linear Ranking**:

  $$
  p_i = \frac{1}{N} \left( \eta_{\text{max}} - (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{i-1}{N-1} \right)
  $$

  where:

  - $p_i$ = probability of selecting individual i
  - $N$ = population size
  - $\eta_{\text{max}}$ = maximum selection pressure
  - $\eta_{\text{min}}$ = minimum selection pressure
  - $i$ = rank of individual

- **Fitness Sharing**:
  $$
  F_{\text{shared}}(p_i) = \frac{F(p_i)^b}{\sum_{j=1}^{N} \gamma(d(p_i, p_j))}
  $$
  where:
  - $F_{\text{shared}}$ = shared fitness
  - $F(p_i)$ = raw fitness
  - $b$ = sharing power
  - $d(p_i, p_j)$ = distance between individuals
  - $\gamma$ = sharing function
  - $N$ = population size

## 🐦 Particle Swarm Optimization

- **Velocity Update**: How particles move in search space

  $$
  v_{ij} = w \cdot v_{ij} + c_1 \cdot rand \cdot (pbest_{ij} - x_{ij}) + c_2 \cdot rand \cdot (gbest_j - x_{ij})
  $$

  where:

  - $v_{ij}$ = velocity of particle i in dimension j
  - $w$ = inertia weight
  - $c_1$ = cognitive parameter
  - $c_2$ = social parameter
  - $rand$ = random number in [0,1]
  - $pbest_{ij}$ = best position of particle i
  - $gbest_j$ = best position in swarm
  - $x_{ij}$ = current position

- **Position Update**: New position based on velocity

  $$
  x_{ij} = x_{ij} + v_{ij}
  $$

  where:

  - $x_{ij}$ = position of particle i in dimension j
  - $v_{ij}$ = velocity of particle i in dimension j

- **Time Varying Inertia Weight**:

  $$
  w = (w_{\text{init}} - w_{\text{final}}) \cdot \frac{t_{\text{final}} - t}{t_{\text{final}}} + w_{\text{final}}
  $$

  where:

  - $w$ = current inertia weight
  - $w_{\text{init}}$ = initial inertia weight
  - $w_{\text{final}}$ = final inertia weight
  - $t$ = current iteration
  - $t_{\text{final}}$ = maximum iterations

- **Constriction Factor**:
  $$
  \chi = \frac{2}{|2 - \varphi - \sqrt{\varphi^2 - 4\varphi}|}, \quad \varphi = c_1 + c_2, \quad \varphi > 4
  $$
  where:
  - $\chi$ = constriction factor
  - $\varphi$ = sum of cognitive and social parameters
  - $c_1, c_2$ = cognitive and social parameters

## 📈 Evolution Strategies

- **Basic Mutation**:

  $$
  x^{(t+1)} = x^{(t)} + \mathcal{N}(0, \Sigma^{(t)})
  $$

  where:

  - $x^{(t)}$ = current solution
  - $\mathcal{N}$ = normal distribution
  - $\Sigma$ = covariance matrix
  - $t$ = generation

- **1/5 Success Rule**: Adapt step size

  $$
  \sigma^{(t+1)} = \begin{cases}
  c_{\text{dec}} \cdot \sigma^{(t)} & \text{if } R_k < 1/5 \\
  c_{\text{inc}} \cdot \sigma^{(t)} & \text{if } R_k > 1/5 \\
  \sigma^{(t)} & \text{if } R_k = 1/5
  \end{cases}
  $$

  where:

  - $\sigma$ = step size
  - $R_k$ = success rate
  - $c_{\text{dec}}$ = decrease factor
  - $c_{\text{inc}}$ = increase factor
  - $t$ = generation

- **Covariance Matrix Update**:
  $$
  \sigma_{ii}^{(t+1)} = \sigma_{ii}^{(t)} \cdot e^{\mathcal{N}(0,\Delta\sigma)}
  $$
  $$
  \alpha_{ij}^{(t+1)} = \alpha_{ij}^{(t)} + \mathcal{N}(0,\Delta\alpha)
  $$
  where:
  - $\sigma_{ii}$ = variance in dimension i
  - $\alpha_{ij}$ = rotation angle between dimensions i and j
  - $\Delta\sigma, \Delta\alpha$ = learning rates
  - $t$ = generation

## 🌳 Genetic Programming

- **Tree Size**: Number of nodes in program tree

  $$
  \text{size} = 1 + \sum_{\text{children}} \text{size}
  $$

  where:

  - size = number of nodes
  - children = all child nodes

- **Program Fitness**: Usually includes size penalty

  $$
  \text{fitness} = \text{accuracy} - \lambda \cdot \text{size}
  $$

  where:

  - accuracy = how well program solves problem
  - size = number of nodes
  - $\lambda$ = complexity penalty weight

- **GEP Tree Construction**:

  - Read string left-to-right
  - Fill tree level-by-level
  - Each function uses required number of inputs
    where:
  - string = fixed-length genotype
  - tree = program phenotype
  - function = operation node
  - input = terminal node

- **GEP Mutation**:

  - Random change in head region
  - May increase tree depth
  - Must maintain valid program structure
    where:
  - head = region containing functions and terminals
  - depth = maximum path length from root to leaf

- **GEP Crossover**:
  - Cut and swap parts of parent strings
  - Maintains fixed length
  - May disrupt useful structures
    where:
  - parent strings = two selected genotypes
  - fixed length = constant string size
  - useful structures = effective program parts

## 🔍 RBF Networks

- **RBF Network Function**:

  $$
  F(\mathbf{x}) = \sum_{i=1}^{M} w_i \, \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_i\|^2}{2\sigma_i^2}\right)
  $$

  where:

  - $F(\mathbf{x})$ = network output
  - $w_i$ = weight for RBF i
  - $\mathbf{x}$ = input vector
  - $\mathbf{c}_i$ = center of RBF i
  - $\sigma_i$ = width of RBF i
  - $M$ = number of RBFs

- **RBF Training Objective**:
  $$
  E(\mathbf{w}) = \sum \left[F(\mathbf{x}_i) - y_i\right]^2 + \lambda \|D\mathbf{F}\|^2
  $$
  where:
  - $E$ = error function
  - $F(\mathbf{x}_i)$ = network output
  - $y_i$ = target output
  - $\lambda$ = regularization parameter
  - $D\mathbf{F}$ = derivative of network function

## 🔍 Optimization

- **Gradient Descent**: Basic optimization step

  $$
  \mathbf{x}' = \mathbf{x} - \eta \nabla f(\mathbf{x})
  $$

  where:

  - $\mathbf{x}$ = current position
  - $\mathbf{x}'$ = new position
  - $\eta$ = learning rate
  - $\nabla f$ = gradient of function

- **Momentum**: Helps escape local minima
  $$
  \mathbf{v} = \mu \mathbf{v} - \eta \nabla f(\mathbf{x})
  $$
  $$
  \mathbf{x}' = \mathbf{x} + \mathbf{v}
  $$
  where:
  - $\mathbf{v}$ = velocity
  - $\mu$ = momentum coefficient
  - $\eta$ = learning rate
  - $\nabla f$ = gradient
  - $\mathbf{x}$ = current position
  - $\mathbf{x}'$ = new position
