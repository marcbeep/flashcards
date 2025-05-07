# Important Formulas in AI and Machine Learning

## Probability and Bayes' Rule

### Bayes' Rule

\[
\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A) \mathbb{P}(A)}{\mathbb{P}(B)}
\]

**Variables:**

- \(\mathbb{P}(A|B)\): Posterior probability (probability of A given B)
- \(\mathbb{P}(B|A)\): Likelihood (probability of B given A)
- \(\mathbb{P}(A)\): Prior probability (initial belief about A)
- \(\mathbb{P}(B)\): Marginal probability (total probability of B)

**Plain English:** Updates our belief about A after observing B. The probability of A given B equals the probability of B given A, times our prior belief in A, divided by the total probability of B.

## Reinforcement Learning

### Q-Learning Update Rule

\[
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
\]

**Variables:**

- \(Q(s,a)\): Current estimate of action-value
- \(\alpha\): Learning rate
- \(r\): Immediate reward
- \(\gamma\): Discount factor
- \(s'\): Next state
- \(a'\): Next action

**Plain English:** Updates our estimate of how good an action is by considering the immediate reward plus the best possible future reward, adjusted by how much we trust our current estimate.

### Bellman Equation for State Value

\[
V^{\pi}(s) = \mathbb{E}_{\pi} [r_{t+1} + \gamma V^{\pi}(s\_{t+1}) \mid s_t = s]
\]

**Variables:**

- \(V^{\pi}(s)\): Value of state s under policy π
- \(r\_{t+1}\): Next reward
- \(\gamma\): Discount factor
- \(s\_{t+1}\): Next state

**Plain English:** The value of a state equals the expected immediate reward plus the discounted value of the next state.

### Bellman Optimality Equations

\[
V^_(s) = \max*a \mathbb{E} [r*{t+1} + \gamma V^_(s\_{t+1}) \mid s_t = s, a_t = a]
\]

\[
Q^_(s,a) = \mathbb{E} [r*{t+1} + \gamma \max*{a'} Q^_(s\_{t+1}, a') \mid s_t = s, a_t = a]
\]

**Variables:**

- \(V^\*(s)\): Optimal state value
- \(Q^\*(s,a)\): Optimal action-value
- \(r\_{t+1}\): Next reward
- \(\gamma\): Discount factor
- \(s\_{t+1}\): Next state
- \(a'\): Next action

**Plain English:** The optimal value of a state equals the maximum expected reward plus discounted future value. The optimal action-value equals the expected reward plus discounted maximum future action-value.

### Temporal Difference (TD) Learning Update

\[
V(s*t) \leftarrow V(s_t) + \alpha [r*{t+1} + \gamma V(s\_{t+1}) - V(s_t)]
\]

**Variables:**

- \(V(s_t)\): Current value estimate
- \(\alpha\): Learning rate
- \(r\_{t+1}\): Next reward
- \(\gamma\): Discount factor
- \(V(s\_{t+1})\): Value estimate of next state

**Plain English:** Updates the value estimate by moving it towards the sum of immediate reward and discounted next state value.

## Neural Networks

### Sigmoid Activation Function

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

**Variables:**

- \(x\): Input to the neuron
- \(\sigma(x)\): Output of the neuron

**Plain English:** Squashes any input to a value between 0 and 1, making it useful for binary classification.

### ReLU (Rectified Linear Unit) Activation Function

\[
f(x) = \max(0, x)
\]

**Variables:**

- \(x\): Input to the neuron
- \(f(x)\): Output of the neuron

**Plain English:** Returns the input if it's positive, otherwise returns zero. Helps with vanishing gradient problem and introduces non-linearity.

### Mean Squared Error (MSE)

\[
\text{MSE} = \frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y}\_i)^2
\]

**Variables:**

- \(y_i\): True value
- \(\hat{y}\_i\): Predicted value
- \(n\): Number of samples

**Plain English:** Measures how far off our predictions are from the true values, by averaging the squared differences.

### Cross Entropy Loss

\[
L = -\frac{1}{n} \sum*{i=1}^{n} \sum*{c=1}^{C} y*{ic} \log(\hat{y}*{ic})
\]

**Variables:**

- \(n\): Number of samples
- \(C\): Number of classes
- \(y\_{ic}\): True label (1 if sample i belongs to class c, 0 otherwise)
- \(\hat{y}\_{ic}\): Predicted probability that sample i belongs to class c

**Plain English:** Measures how well the predicted probabilities match the true labels, commonly used in classification tasks.

## Swarm Intelligence

### Ant Colony Optimization Probability

\[
P*{ij}(t) = \frac{[\tau*{ij}(t)]^\alpha \cdot [\eta_{ij}]^\beta}{\sum*{k \in J} [\tau*{ik}(t)]^\alpha \cdot [\eta_{ik}]^\beta}
\]

**Variables:**

- \(P\_{ij}\): Probability of choosing path ij
- \(\tau\_{ij}\): Pheromone level on path ij
- \(\eta\_{ij}\): Heuristic information (e.g., 1/distance)
- \(\alpha, \beta\): Parameters controlling influence
- \(J\): Set of possible next moves

**Plain English:** Probability of an ant choosing a particular path based on pheromone levels and heuristic information (like distance).

### Particle Swarm Optimization Velocity Update

\[
v_i^{t+1} = w v_i^t + \phi_1 U_1 (pb_i - x_i^t) + \phi_2 U_2 (gb - x_i^t)
\]

**Variables:**

- \(v_i\): Velocity of particle i
- \(w\): Inertia weight
- \(\phi_1, \phi_2\): Learning rates
- \(U_1, U_2\): Random numbers between 0 and 1
- \(pb_i\): Personal best position
- \(gb\): Global best position
- \(x_i\): Current position

**Plain English:** Updates a particle's velocity based on its current movement, its best known position, and the best known position of any particle in the swarm.

## Evolutionary Game Theory

### Replicator Dynamics

\[
\dot{x}\_i = x_i (f_i(x) - \bar{f}(x))
\]

**Variables:**

- \(x_i\): Proportion of population using strategy i
- \(f_i(x)\): Fitness of strategy i
- \(\bar{f}(x)\): Average fitness of population

**Plain English:** Rate of change of a strategy's proportion in the population equals its current proportion times how much better (or worse) it is than average.

### Replicator Dynamics for Two Populations

\[
\dot{x}\_i = x_i \left( (A y)\_i - x^T A y \right) \\
\dot{y}\_j = y_j \left( (x^T B)\_j - x^T B y \right)
\]

**Variables:**

- \(x_i\): frequency of strategy \(i\) in first population
- \(y_j\): frequency of strategy \(j\) in second population
- \(A\): payoff matrix for first population
- \(B\): payoff matrix for second population
- \((A y)\_i\): expected payoff of strategy \(i\) in first population
- \((x^T B)\_j\): expected payoff of strategy \(j\) in second population
- \(x^T A y\): average payoff in first population
- \(x^T B y\): average payoff in second population

**Plain English:** Describes how strategy frequencies change over time in two interacting populations, where each population's evolution depends on the current state of the other population.
