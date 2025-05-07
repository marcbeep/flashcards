# 🧮 Essential Formulas in Bio-Inspired Computing

## 📊 Probability and Statistics

### Basic Probability

- **Probability Mass Function (PMF) or Density Function (PDF)**:

  - **Discrete case**: PMF assigns probability \( p(x) \) to each outcome \( x \)
  - **Continuous case**: PDF assigns probability _density_ over intervals

  **Key Properties**:

  - \( p(x) \geq 0 \) for all \( x \)
  - \( \sum_x p(x) = 1 \) (for discrete)
  - \( \int p(x) dx = 1 \) (for continuous)

  In plain English: These formulas define how probabilities must be non-negative and sum to 1, ensuring that all possible outcomes are accounted for.

### Joint Probability

- **Joint Probability Formula**:
  \[
  \mathbb{P}(X = x, Y = y)
  \]

  In plain English: This calculates the probability of two events happening together, like the chance of picking both an orange AND it being from the red basket.

### Sum Rule (Marginal Probability)

- **Formula**:
  \[
  \mathbb{P}(X = x) = \sum_y \mathbb{P}(X = x, Y = y)
  \]

  In plain English: To find the probability of one event regardless of another, sum up all the joint probabilities over the other variable. For example, to find the probability of picking an orange regardless of which basket it's from.

### Conditional Probability

- **Formula**:
  \[
  \mathbb{P}(Y = y | X = x) = \frac{\mathbb{P}(X = x, Y = y)}{\mathbb{P}(X = x)}
  \]

  In plain English: This calculates the probability of one event given that we know another event has occurred. For example, the probability that a fruit came from the red basket, given that we know it's an orange.

### Bayes' Rule

- **Formula**:
  \[
  \mathbb{P}(A|B) = \frac{\mathbb{P}(B|A) \mathbb{P}(A)}{\mathbb{P}(B)}
  \]

  In plain English: This formula allows us to "reverse" conditional probabilities. If we know how likely B is given A, we can calculate how likely A is given B. It's like working backward from evidence to cause.

## 🤖 Reinforcement Learning

### Q-Learning Update

- **Formula**:
  \[
  Q(s,a) = Q(s,a) + \alpha[r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a)]
  \]
  where:

  - \( Q(s,a) \) = action-value for state s and action a
  - \( \alpha \) = learning rate
  - \( r \) = immediate reward
  - \( \gamma \) = discount factor
  - \( s' \) = next state
  - \( a' \) = next action

  In plain English: This formula updates how good we think an action is in a particular state. It combines the immediate reward we get with the best possible future reward, weighted by how much we care about future rewards.

### State-Value Function

- **Formula**:
  \[
  V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r\_{t+1} \mid s_0 = s \right]
  \]

  In plain English: This calculates how good a state is expected to be if we follow a particular policy. It sums up all possible future rewards, but gives less weight to rewards that are further in the future.

### Action-Value Function

- **Formula**:
  \[
  Q^{\pi}(s,a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r\_{t+1} \mid s_0 = s, a_0 = a \right]
  \]

  In plain English: Similar to the state-value function, but this calculates how good it is to take a specific action in a state and then follow our policy afterward.

### Bellman Equation

- **Formula**:
  \[
  V^{\pi}(s) = \mathbb{E}_{\pi} [r_{t+1} + \gamma V^{\pi}(s\_{t+1}) \mid s_t = s]
  \]

  In plain English: This equation shows how the value of a state relates to the value of the next state. It says that a state's value is the immediate reward plus the discounted value of where we end up next.

### Temporal Difference (TD) Learning

- **TD(0) Update Rule**:
  \[
  V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]
  \]
  where:

  - \( \alpha \) = learning rate
  - \( r \) = immediate reward
  - \( \gamma \) = discount factor
  - \( s' \) = next state

  In plain English: Updates the value estimate based on the difference between predicted and actual values.

### Sarsa (On-policy TD Control)

- **Update Rule**:
  \[
  Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]
  \]
  where:

  - \( a' \) = next action (from current policy)
  - Other variables same as TD(0)

  In plain English: Updates Q-values based on the actual next action taken, following the current policy.

## 🐜 Swarm Intelligence

### Particle Swarm Optimization (PSO)

- **Velocity Update Formula**:
  \[
  v_i^{t+1} = w \cdot v_i^t + \phi_1 U_1 (pb_i - x_i^t) + \phi_2 U_2 (gb - x_i^t)
  \]
  where:

  - \( w \) = inertia weight
  - \( v_i \) = particle velocity
  - \( pb_i \) = personal best position
  - \( gb \) = global best position
  - \( \phi_1, \phi_2 \) = acceleration constants
  - \( U_1, U_2 \) = random numbers between 0 and 1

  In plain English: This formula determines how a particle moves in the search space by combining its current momentum, pull toward its personal best position, and pull toward the global best position found by any particle.

- **Position Update Formula**:
  \[
  x_i^{t+1} = x_i^t + v_i^{t+1}
  \]
  where:

  - \( x_i \) = particle position
  - \( v_i \) = particle velocity

  In plain English: Updates the particle's position based on its current position and new velocity.

### Ant Colony Optimization (ACO)

- **Path Selection Probability**:
  \[
  p*{ij} = \frac{[\tau*{ij}]^\alpha [\eta_{ij}]^\beta}{\sum*{l \in N_i} [\tau*{il}]^\alpha [\eta_{il}]^\beta}
  \]
  where:

  - \( \tau\_{ij} \) = pheromone level on path i,j
  - \( \eta\_{ij} \) = heuristic information (like inverse of distance)
  - \( \alpha, \beta \) = parameters controlling influence
  - \( N_i \) = set of available next nodes

  In plain English: This formula determines how likely an ant is to choose a particular path based on both the amount of pheromone on the path and how good the path looks based on distance or other factors.

- **Pheromone Update Rule**:
  \[
  \tau*{ij} \leftarrow (1 - \rho)\tau*{ij} + \rho \Delta\tau\_{ij}
  \]
  where:

  - \( \tau\_{ij} \) = current pheromone level
  - \( \rho \) = evaporation rate
  - \( \Delta\tau\_{ij} \) = new pheromone deposited

  In plain English: This formula updates the pheromone level on each path by first reducing it through evaporation and then adding new pheromone based on how many ants used the path and how good their solutions were.

- **Global Best Update**:
  \[
  \tau*{ij} \leftarrow (1 - \rho)\tau*{ij} + \rho \Delta\tau\_{ij}^{bs}
  \]
  where:

  - \( \Delta\tau*{ij}^{bs} = 1/L*{bs} \) = reward based on best tour length
  - Other variables same as above

  In plain English: This special update rule reinforces the best solution found so far, helping the colony remember and build upon its best discoveries.

## 🧬 Evolutionary Game Theory

### Replicator Dynamics

- **Single Population**:
  \[
  \dot{x}\_i = x_i(f_i(x) - \bar{f}(x))
  \]
  where:

  - \( x_i \) = frequency of strategy i
  - \( f_i(x) \) = fitness of strategy i
  - \( \bar{f}(x) \) = average fitness

  In plain English: Shows how the frequency of a strategy changes based on its relative fitness.

- **Multi-Population**:
  \[
  \dot{x}\_i = x_i((Ay)\_i - x^T Ay)
  \]
  \[
  \dot{y}\_j = y_j((x^T B)\_j - x^T By)
  \]
  where:

  - \( x, y \) = strategy distributions
  - \( A, B \) = payoff matrices

  In plain English: Shows how two populations' strategies evolve based on their interactions and payoffs.

## 🐝 Bee System

### Lévy Flight

- **Step Length Distribution**:
  \[
  p(l) \sim l^{-\mu}
  \]
  where:

  - \( l \) = step length
  - \( \mu \) = power law exponent (typically 1 < \( \mu \) < 3)

  In plain English: This formula describes how bees mix short local moves with occasional long jumps during foraging, creating an efficient search pattern that balances exploration and exploitation.

### Path Integration

- **Return Vector**:
  \[
  \mathbf{v}_{\text{return}} = -\sum_{i=1}^n \mathbf{v}\_i
  \]
  where:

  - \( \mathbf{v}\_i \) = individual movement vectors
  - \( n \) = number of steps taken

  In plain English: This formula calculates the direct path back to the nest by summing up all the movement vectors taken during foraging, allowing bees to return efficiently even after complex paths.

### Dance Recruitment

- **Probability of Following Dance**:
  \[
  p\_{\text{follow}} = \frac{Q}{Q + K}
  \]
  where:

  - \( Q \) = food quality
  - \( K \) = threshold parameter

  In plain English: This formula determines how likely other bees are to follow a dance based on the quality of the food source, with better food sources attracting more followers.
