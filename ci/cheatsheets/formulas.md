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
