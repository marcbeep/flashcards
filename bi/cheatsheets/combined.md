# File: 03.md
## 1. Introduction to Probability

**Probability** is the mathematical study of uncertainty — it describes how likely events are to happen.

- **Random Variable**:

  - A variable \( X \) that takes different values representing possible outcomes of a process.
  - We denote the probability of \( X \) taking value \( x \) as \( p(x) = \mathbb{P}(X = x) \).

- **Sample Space (\( \mathcal{S} \))**:

  - The set of all possible outcomes of an experiment.
  - Examples:
    - Tossing a coin once: \( \mathcal{S} = \{H, T\} \).
    - Tossing a coin twice: \( \mathcal{S} = \{HH, HT, TH, TT\} \).

- **Probability Mass Function (PMF) or Density Function (PDF)**:
  - **Discrete** case: PMF assigns probability \( p(x) \) to each outcome \( x \) (e.g., coin toss).
  - **Continuous** case: PDF assigns probability _density_ over intervals.

**Key Properties**:

- \( p(x) \geq 0 \) for all \( x \).
- \( \sum_x p(x) = 1 \) (for discrete), or \( \int p(x) dx = 1 \) (for continuous).

---

## 2. Basic Probability Concepts

### 2.1 Random Experiment

An action or process that leads to one of several outcomes, even under identical conditions.

Examples:

- Tossing a coin.
- Rolling a dice.
- Measuring rainfall.

---

## 3. Key Probability Rules

Understanding relationships between events is critical. These rules formalize how probabilities combine.

---

### 3.1 **Joint Probability**

- **Definition**: Probability that **two events happen together**.
- Example:

  - Probability that a randomly selected fruit is an **orange** from the **red basket**.

- **Mathematically**:  
  \[
  \mathbb{P}(X = x, Y = y)
  \]
  where \( X \) and \( Y \) are random variables.

**Illustration** (from the lecture example):
| | F = orange | F = apple | Total |
|---------|------------|-----------|-------|
| B = red | 30 | 10 | 40 |
| B = blue| 15 | 45 | 60 |
| Total | 45 | 55 | 100 |

Example:
\[
\mathbb{P}(B = r, F = o) = \frac{30}{100} = 0.3
\]

---

### 3.2 **Sum Rule** (Marginal Probability)

- **Definition**: To find the probability of an event regardless of another variable, **sum over** the other variable.

- **Formula**:
  \[
  \mathbb{P}(X = x) = \sum_y \mathbb{P}(X = x, Y = y)
  \]
- **Example**:
  - What is the probability of picking an orange? (regardless of basket)
    \[
    \mathbb{P}(F = o) = \frac{45}{100} = 0.45
    \]

---

### 3.3 **Conditional Probability**

- **Definition**: The probability that one event happens given that we know another event happened.

- **Formula**:
  \[
  \mathbb{P}(Y = y | X = x) = \frac{\mathbb{P}(X = x, Y = y)}{\mathbb{P}(X = x)}
  \]

- **Example**:
  - Probability that the basket is red given the fruit is orange:
    \[
    \mathbb{P}(B = r | F = o) = \frac{30}{45} = \frac{2}{3}
    \]

---

### 3.4 **Product Rule**

- **Definition**: Relates joint probability to conditional and marginal probability.

- **Formula**:
  \[
  \mathbb{P}(X = x, Y = y) = \mathbb{P}(Y = y | X = x) \mathbb{P}(X = x)
  \]

- **Example**:
  - We can calculate the joint probability if we know conditional and marginal probabilities.

---

### 3.5 **Bayes’ Rule**

- **Definition**: A way to "invert" conditional probabilities.
- **Motivation**: Given \( \mathbb{P}(B|A) \) and \( \mathbb{P}(A) \), how can we find \( \mathbb{P}(A|B) \)?

- **Formula**:
  \[
  \mathbb{P}(A|B) = \frac{\mathbb{P}(B|A) \mathbb{P}(A)}{\mathbb{P}(B)}
  \]

**Where**:

- \( \mathbb{P}(A) \) = prior probability (belief before evidence)
- \( \mathbb{P}(B|A) \) = likelihood (how likely evidence is under assumption)
- \( \mathbb{P}(B) \) = marginal probability of evidence
- \( \mathbb{P}(A|B) \) = posterior probability (updated belief after evidence)

**Intuition**:

- We update our belief about \( A \) after observing \( B \).

---

## 🍊 Problem Setup (from the lecture):

We have two baskets:

- **Red Basket (R)**: 6 oranges, 2 apples
- **Blue Basket (B)**: 1 orange, 3 apples
- Total fruits in Red: 8
- Total fruits in Blue: 4

Also:

- Probability of choosing **Red Basket**: \( P(R) = 0.4 \)
- Probability of choosing **Blue Basket**: \( P(B) = 0.6 \)

---

## ❓ Question:

**If I pick a fruit and it’s an orange, what is the probability that it came from the red basket?**

That’s:
\[
P(R \mid O) = ?
\]

We are trying to **reverse** the probability:

- We know \( P(O \mid R) \) — the chance of orange **if red basket is chosen**
- We want \( P(R \mid O) \) — the chance it was red **given that we saw an orange**

---

## ✅ Step-by-Step with Bayes' Rule:

### Step 1: Write Bayes' Rule

\[
P(R \mid O) = \frac{P(O \mid R) \cdot P(R)}{P(O)}
\]

---

### Step 2: Find the individual terms

1. **Likelihood**:  
   Probability of orange if we picked the red basket:
   \[
   P(O \mid R) = \frac{6}{8} = 0.75
   \]

2. **Prior**:  
   Probability of choosing red basket:
   \[
   P(R) = 0.4
   \]

3. **Marginal (total) probability of orange**:  
   Use total probability:
   \[
   P(O) = P(O \mid R) \cdot P(R) + P(O \mid B) \cdot P(B)
   \]
   Where:
   \[
   P(O \mid B) = \frac{1}{4} = 0.25,\quad P(B) = 0.6
   \]
   So:
   \[
   P(O) = (0.75 \cdot 0.4) + (0.25 \cdot 0.6) = 0.3 + 0.15 = 0.45
   \]

---

### Step 3: Plug into Bayes' Rule

\[
P(R \mid O) = \frac{0.75 \cdot 0.4}{0.45} = \frac{0.3}{0.45} = \frac{2}{3}
\]

---

## 🎉 Final Answer:

\[
\boxed{P(R \mid O) = \frac{2}{3}}
\]

So, **if you picked an orange**, there’s a **2 in 3 chance** it came from the **red basket**.

---
# File: 04.md
---

## 🧠 **1. Roots of Reinforcement Learning**

- **Fields of Origin**:
  - Mathematical Psychology (1910s)
  - Control Theory (1950s) – Richard Bellman (Dynamic Programming, Bellman Equations)
- **Multidisciplinary Influences**:
  - Artificial Intelligence
  - Neuroscience
  - Deep Neural Networks
  - Psychology
  - Control Theory
  - Operations Research

---

## 🧩 **2. Key Concepts of RL**

### 🔄 What is RL?

- Goal-oriented learning from interaction
- Learns how to act to **maximize a reward**
- Learns during, from, and about interactions with the environment

### 🚀 Agent Characteristics

- Temporally situated
- Continual learning and planning
- Acts on a stochastic, uncertain environment

---

## 🌍 **3. Types of Environments**

| Type                                          | Description                           | Example                                              |
| --------------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| **Deterministic vs. Stochastic**              | Next state is (not) fully predictable | Crossword (deterministic), Taxi driving (stochastic) |
| **Fully Observable vs. Partially Observable** | Agent sees full/partial state         | Chess (fully), Taxi driving (partially)              |
| **Episodic vs. Sequential**                   | Has episodes or not                   | Maze running (episodic), Taxi driving (sequential)   |
| **Dynamic vs. Static**                        | Changes during decision-making        | Poker (static), Taxi driving (dynamic)               |
| **Discrete vs. Continuous**                   | States/actions are distinct or smooth | Poker (discrete), Taxi driving (continuous)          |
| **Single vs. Multi-Agent**                    | One or multiple agents                | Crossword (single), Taxi (multi-agent)               |

**Real-world is**: partially observable, stochastic, sequential, dynamic, continuous, multi-agent.

---

## 📐 **4. Elements of Reinforcement Learning**

| Element            | Description                                       |
| ------------------ | ------------------------------------------------- |
| **Policy**         | What to do – agent’s behavior                     |
| **Reward**         | What is good (immediate feedback)                 |
| **Value Function** | What is good in the long run (predicts reward)    |
| **Model**          | What follows what – predicts environment behavior |

---

## 🧪 **5. RL vs Supervised Learning**

| Aspect          | Supervised Learning  | Reinforcement Learning     |
| --------------- | -------------------- | -------------------------- |
| Output          | Desired output given | Reward/feedback only       |
| Goal            | Match output         | Maximize cumulative reward |
| Training Signal | Labels               | Rewards                    |

---

## 🕹️ **6. Tic-Tac-Toe Example (Simple RL)**

- **Value Table**: Map each state to value (e.g., 1 = win, 0 = loss)
- **Learning Rule**:
  \[
  V(s) \leftarrow V(s) + \alpha [V(s') - V(s)]
  \]
  (move value towards next state's value)
- **Greedy vs. Exploration**: Balance choosing best known move vs. trying new ones

### 🧠 Improvement Ideas

- Use symmetries
- Pre-train via self-play
- Learn from random moves
- Use value function approximators

---

## 🧠 **7. Characteristics of RL Learning**

- No direct instruction on what actions to take
- Learns by **trial-and-error**
- **Delayed rewards** make learning harder
- Must **explore and exploit**
- Learns in **uncertain environments**

---

## 🧬 **8. Applications of RL**

- **Games**: AlphaGo, TD-Gammon (backgammon), Tic-Tac-Toe
- **Robotics**: navigation, walking, grasping
- **Industry**: Elevator control, inventory management, dynamic channel assignment
- **Competitions**: RoboCup Soccer (Stone & Veloso)

---

## 📚 **9. Practice Task**

> Identify the four RL elements (policy, reward, value, model) in various tasks:

- Tic-Tac-Toe
- Backgammon
- Poker
- Chess
- Crossword puzzle
- K-armed bandit

---
# File: 05.md
---

## **1. Reinforcement Learning**

### Roots and Key Features

- **Reinforcement Learning (RL)** is inspired by how agents (biological or artificial) learn from interactions.
- Focuses on **learning by trial and error**, with feedback from actions taken.

### Elements of RL

- **Agent**: Learner or decision maker.
- **Environment**: Everything the agent interacts with.
- **Actions and Rewards**: Agent performs actions and receives feedback (rewards).

---

## **2. Problem Solving from Nature**

- RL and other methods take inspiration from biological systems (e.g., learning mechanisms, swarm behavior).

---

## **3. Multi-Armed Bandit Problems**

### Core Scenario

- **You have multiple slot machines (arms)**, each with unknown reward probabilities.
- You must decide which to play to maximize total reward over a finite number of plays.

### Exploration vs. Exploitation

- **Exploration**: Try new actions to discover their potential.
- **Exploitation**: Choose the best-known action so far.
- **Dilemma**: Must balance both to perform well long-term.

  Formula:

  - Greedy choice: $a_t^* = \arg\max Q_t(a)$
  - Explore: pick $a_t \ne a_t^*$
  - $Q_t(a) \approx Q^*(a)$: estimated vs. true value

---

## **4. Evaluative Feedback**

### Feedback Types

- **Evaluative**: Based only on actions taken (no explicit instructions).
- **Instructive**: Tells what to do (ideal actions).

### Learning Styles

- **Associative**: Learn best output for each input.
- **Nonassociative**: Learn a single best action overall.
- Multi-armed bandits use **evaluative feedback** and are **nonassociative**.

---

## **5. Action-Value Methods**

### Sample Average Estimation

- If action $a$ was chosen $k$ times with rewards $r_1, ..., r_k$:

  $$
  Q_t(a) = \frac{r_1 + r_2 + \dots + r_k}{k}
  $$

### Incremental Update

- Update estimate without storing all data:

  $$
  Q_{n+1} = Q_n + \frac{1}{k+1} (r - Q_n)
  $$

- General form:

  $$
  \text{New Estimate} = \text{Old Estimate} + \text{Step Size} \times (\text{Target} - \text{Old Estimate})
  $$

---

## **6. Nonstationary Problems**

- When reward probabilities change over time:

  - Use **constant step-size** $\alpha$ instead of sample average.

  $$
  Q_{n+1} = Q_n + \alpha (r - Q_n)
  $$

- Equivalent to **recency-weighted average**.

---

## **7. Action Selection Methods**

### ε-Greedy Method

- Choose best-known action with probability $1 - \varepsilon$.
- Random action with probability $\varepsilon$.

  $$
  a =
  \begin{cases}
  a^* & \text{with probability } 1 - \varepsilon \\
  \text{random} & \text{with probability } \varepsilon
  \end{cases}
  $$

### Softmax Action Selection

- Uses probability distribution:

  $$
  P(a) = \frac{e^{Q(a)/\tau}}{\sum_b e^{Q(b)/\tau}}
  $$

  Where $\tau$ is the temperature parameter.

---

## **8. Example: 10-Armed Testbed**

- 10 arms with unknown rewards.
- Simulations with:

  - **Greedy**, **ε-Greedy**, and **Softmax** methods.

- Each experiment repeated 1000 times over 1000 plays and averaged.

---

## **9. Summary**

- **Multi-Armed Bandit Problem**
- **Exploration vs. Exploitation Dilemma**
- **Action-Value Methods**
- **Action Selection Strategies**
- **Stationary vs. Nonstationary Settings**

---
# File: 06.md
---
## 🧠 **Reinforcement Learning (RL)**

### 1. **Introduction & Motivation**

- Inspired by how nature solves problems.
- RL is used in games, decision-making tasks, and robotics.
- Multi-armed bandits introduced as a simple RL setting.
---

### 2. **Key Concepts of RL**

#### a. **Agent-Environment Interface**

- Discrete time steps: $t = 0, 1, 2, \dots$
- At each time $t$, the agent:

  - Observes state $s_t$
  - Chooses action $a_t$
  - Receives reward $r_t$
  - Transitions to next state $s_{t+1}$

#### b. **Policy**

- A policy $\pi$ maps states to action probabilities.
- Example: $\pi(s, \text{UP}) = 0.8, \pi(s, \text{DOWN}) = 0.2$
- The agent updates its policy over time to maximize rewards.

---

### 3. **Goals and Rewards**

#### a. **What is a Goal in RL?**

- Defined via rewards (scalar signals).
- The **reward hypothesis**: all goals can be seen as maximizing cumulative reward.
- Rewards should:

  - Be measurable
  - Occur frequently enough
  - Be outside the agent’s direct control

#### b. **Returns**

- **Episodic tasks**: Break into distinct episodes (e.g., games).

  - Return: $R_t = r_{t+1} + r_{t+2} + \dots + r_T$

- **Continuing tasks**: Go on indefinitely.

  - Use discounted return:
    $R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$

  - $\gamma \in [0, 1]$: discount factor (closer to 1 = more farsighted)

---

### 4. **Example: Pole Balancing**

- **Episodic view**: +1 reward per step before failure → return = steps survived
- **Continuing view**: 0 reward normally, -1 on failure → return = $-\gamma^k$
- Objective in both: Avoid failure for as long as possible

---

### 5. **Notation**

- Time steps usually start at 0
- Same symbols used for state, reward, and return across episodes
- General return formula:
  $R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$
  (Works for both episodic and continuing tasks)

---

### 6. **Markov Property & MDPs**

#### a. **Markov Property**

- A state is Markov if it contains all relevant info from the history.
- Future state & reward depend only on current state and action, not full history.

#### b. **Markov Decision Processes (MDPs)**

- If a task satisfies the Markov property, it can be modeled as an MDP.
- **Finite MDP** includes:

  - Finite set of states and actions
  - Transition probabilities: $P(s'|s, a)$
  - Reward expectations: $R(s, a, s') = E[r_{t+1} | s_t = s, a_t = a, s_{t+1} = s']$

#### c. **Example of an MDP**

- Defined by:

  - Set of states
  - Set of actions
  - Transition function
  - Reward function

---
# File: 07.md
# Prev Sections Omitted

## **3. Reinforcement Learning Core Concepts**

### **3.1. Value Functions**

These help the agent decide what is good in the long run.

- **State-Value Function (Vπ)**: Expected return from state _s_ under policy π.

  $$
  V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right]
  $$

- **Action-Value Function (Qπ)**: Expected return from taking action _a_ in state _s_ and following policy π thereafter.

  $$
  Q^{\pi}(s,a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s, a_0 = a \right]
  $$

---

### **3.2. Bellman Equations**

- **For a Policy π**:

  - Expresses recursive relationship:

    $$
    V^{\pi}(s) = \mathbb{E}_{\pi} [r_{t+1} + \gamma V^{\pi}(s_{t+1}) \mid s_t = s]
    $$

- **Intuition**: Averages over all possible next states weighted by their transition probabilities.

---

## **4. Optimality in RL**

### **4.1. Optimal Policies**

- A policy π\* is **optimal** if it achieves the highest value in every state.
- **All optimal policies share the same value functions**:

  - Optimal State-Value:

    $$
    V^*(s) = \max_{\pi} V^{\pi}(s)
    $$

  - Optimal Action-Value:

    $$
    Q^*(s,a) = \max_{\pi} Q^{\pi}(s,a)
    $$

---

### **4.2. Bellman Optimality Equations**

- For **V**\*:

  $$
  V^*(s) = \max_a \mathbb{E} [r_{t+1} + \gamma V^*(s_{t+1}) \mid s_t = s, a_t = a]
  $$

- For **Q**\*:

  $$
  Q^*(s,a) = \mathbb{E} [r_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') \mid s_t = s, a_t = a]
  $$

- These are systems of **nonlinear equations** with unique solutions for V\* and Q\*.

---

## **5. Example: Golf**

- **State**: Ball location
- **Actions**: `putt` (accurate), `driver` (longer distance, less accurate)
- **Reward**: -1 per stroke until the ball is in the hole
- Demonstrates how different actions and strategies affect the value of states and actions.

---

## **6. Why Value Functions Matter**

- **V\***: Helps find optimal policy using **one-step-ahead search**.
- **Q\***: Enables selection of the best action directly:

  $$
  a^* = \arg\max_a Q^*(s, a)
  $$

---
# File: 08.md
---

## **Reinforcement Learning (RL)**

### **1. Solving Bellman Optimality Equation**

Goal: Find the best way to act to maximize future rewards.

Three main approaches:

- **Dynamic Programming (DP)**
- **Monte Carlo (MC)**
- **Temporal Difference (TD)**

---

## **2. Dynamic Programming (DP)**

- Used when you **know the full model of the environment** (i.e., probabilities and rewards).
- Works well if you have **enough memory and time**, and if the **Markov Property** holds (future depends only on current state, not past states).

### **DP Value Update Rule**:

$$
V(s_t) \leftarrow \mathbb{E}[r_{t+1} + \gamma V(s_{t+1})]
$$

- $V(s_t)$: value of current state
- $r_{t+1}$: reward received after taking action
- $\gamma$: discount factor (between 0 and 1), reduces value of future rewards
- $V(s_{t+1})$: estimated value of next state

**Key idea**: Update value of a state by averaging the expected reward plus future value.

**Limitations**: Rarely used directly because full environment info is usually unavailable.

---

## **3. Monte Carlo (MC) Methods**

- Learn directly from **complete episodes** of experience.
- **No need to know environment model**.
- **Only works for episodic tasks** (tasks that end eventually).

### **MC Policy Evaluation** (Estimate $V(s)$):

- **First-visit MC**: Average returns the first time a state is visited in each episode.
- **Every-visit MC**: Average all returns after every visit to a state.
- Both methods **converge over time** with enough episodes.

### **MC Value Update Rule (Simplified)**:

$$
V(s_t) \leftarrow V(s_t) + \alpha (R_t - V(s_t))
$$

- $\alpha$: learning rate
- $R_t$: return after visiting state $s_t$

### **Monte Carlo Estimation for Action-Value $Q(s,a)$**:

- Estimate expected return from a **state-action pair**.
- Needs **exploring starts**: every state-action pair must have a non-zero chance of being tried.

### **MC Control (Policy Improvement)**:

- Alternate between:

  1. **Evaluating current policy** using MC.
  2. **Improving policy** by choosing better actions (greedy).

- **Converges** with enough episodes and exploration.

**Advantages**:

- Great when state space is large but only a few actions matter (e.g., games like Go, Backgammon).

---

## **4. Temporal Difference (TD) Methods**

- Combine ideas of DP and MC.
- **Do not need full model** of the environment.
- Can **learn online, step-by-step**, without waiting for episodes to finish.

### **TD Value Update Rule**:

$$
V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$

- Update current value estimate based on:

  - actual reward received
  - estimated value of next state
  - current estimate

This is called **bootstrapping**—you update based on other estimates.

### **Tic-Tac-Toe Example**:

$$
V(a) \leftarrow V(a) + \alpha [V(c) - V(a)] \\
V(e) \leftarrow V(e) + \alpha [V(g) - V(e)]
$$

Where `a → c` and `e → g` are transitions between board states.

---

## **5. Summary Comparison**

| Method              | Needs Environment Model? | Learns From   | Updates Based On              | Works Online? |
| ------------------- | ------------------------ | ------------- | ----------------------------- | ------------- |
| Dynamic Programming | Yes                      | Full model    | Expected future values        | No            |
| Monte Carlo         | No                       | Full episodes | Actual returns                | No            |
| Temporal Difference | No                       | Step-by-step  | Reward + estimated next value | Yes           |

---
# File: 09.md
---
# 🧠 Temporal Difference (TD) Learning — Core Content
---

## 1. **What is Temporal Difference Learning?**

TD learning is a method in Reinforcement Learning (RL) used to estimate the value of states **while learning** from incomplete episodes. It combines ideas from:

- **Monte Carlo (MC)**: learning from raw experience
- **Dynamic Programming (DP)**: updating estimates using other learned estimates

➡️ It **learns predictions based on other predictions**, without waiting for the final outcome.

---

## 2. **TD Prediction (Policy Evaluation)**

### Goal:

Given a policy $\pi$, estimate the **state-value function** $V^\pi(s)$, which tells you how good a state is if you follow policy $\pi$.

### TD(0) Update Rule:

$$
V(s) \leftarrow V(s) + \alpha \left[ r + \gamma V(s') - V(s) \right]
$$

Where:

- $s$: current state
- $s'$: next state
- $r$: reward received after transition
- $\alpha$: learning rate
- $\gamma$: discount factor (how much future rewards matter)

🧩 **Intuition**: You're adjusting your current guess $V(s)$ towards the observed reward plus your guess of the next state's value.

---

## 3. **TD vs Other Methods**

| Method        | Bootstraps (uses estimates)? | Samples (uses actual outcomes)? |
| ------------- | ---------------------------- | ------------------------------- |
| Monte Carlo   | ❌ No                        | ✅ Yes                          |
| Dynamic Prog. | ✅ Yes                       | ❌ No                           |
| TD Learning   | ✅ Yes                       | ✅ Yes                          |

- **TD** is a hybrid: it updates based on real experience (**like MC**) and uses current estimates (**like DP**).

---

## 4. **Advantages of TD Learning**

- **No environment model needed**: You don't need to know how the world works.
- **Learn online and incrementally**:

  - No need to wait for episode to finish.
  - Works well with partial episodes or incomplete data.

- **Efficient in memory and computation**.

---

## 5. **TD Control Methods**

TD Prediction only estimates values under a fixed policy. TD **Control** updates both the value estimates and the policy, aiming to **find the optimal policy**.

Two main methods:

---

### A. **Sarsa (On-policy TD Control)**

Update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

- Learns from the **actual action** taken in the next state.
- Follows the policy it's trying to improve (on-policy).

📌 **Behavior**: Conservative. Learns from what you actually did.

---

### B. **Q-learning (Off-policy TD Control)**

Update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

- Learns from the **best possible action** in the next state.
- Learns optimal policy **even while following a different one** (off-policy).

📌 **Behavior**: Aggressive. Learns from what you _should have_ done.

---

## 6. **On-policy vs Off-policy**

| Type       | Learns From     | Learns About   |
| ---------- | --------------- | -------------- |
| On-policy  | Current policy  | Current policy |
| Off-policy | Behavior policy | Optimal policy |

- **Sarsa** is on-policy: learns from and about the same policy.
- **Q-learning** is off-policy: behaves one way, learns about another.

---

## 7. **Convergence to Optimality**

- **Sarsa**:

  - Converges if:

    - Every state-action pair is visited infinitely often
    - Policy becomes greedy over time

- **Q-learning**:

  - Easier convergence
  - Needs:

    - Infinite visits
    - Learning rate $\alpha$ decreases over time (but not too fast)

✅ Even if assumptions aren’t fully met, both methods perform well in practice.

---

## 8. **Summary of TD Learning**

- TD methods allow **value prediction and control** without full models.
- They **bootstrap and sample** — blending the strengths of DP and MC.
- **TD(0)** is the simplest form, using one-step lookahead.
- Control methods:

  - **Sarsa** for on-policy learning.
  - **Q-learning** for off-policy learning.

---
# File: 10.md
---
### 🧠 Lecture 10: Reinforcement Learning (RL)
---

### 🔁 Reward Shaping

**What it is:**
Reward shaping means adding extra (artificial) rewards to help guide the agent toward learning a good policy faster. This is based on expert or domain knowledge.

**Why it's useful:**

- Speeds up learning.
- Can guide the agent more efficiently while still preserving the best (optimal) strategy, if done correctly.

**Q-learning update rule (basic):**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

Where:

- $s_t$: current state
- $a_t$: action taken
- $r_{t+1}$: reward received after action
- $\gamma$: discount factor (how much future rewards are considered)
- $\alpha$: learning rate

**With potential-based shaping:**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + F(s, s') + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

- $F(s, s')$: potential-based function, used to shape the reward based on states
- This method helps by adding extra insight into which states are more promising.

---

### 🎭 Actor-Critic Methods

In these methods, learning is split into two parts:

- **Actor**: Learns the policy (decides what action to take).
- **Critic**: Estimates the value function (judges how good a state or action is).

The value function is now represented using parameters (e.g., weights in a neural network). These are updated as learning progresses.

- Advantage = How much better an action is compared to the average at that state.

---

### 🎯 Policy Gradient Methods

Instead of estimating value functions, these methods **directly learn the policy**.

- Objective function:

  $$
  J(\theta) = \text{Expected return based on parameters } \theta
  $$

- Update rule using gradient ascent:

  $$
  \theta_{t+1} = \theta_t + \Delta \theta_t, \quad \Delta \theta_t = G_t \nabla_\theta \log \pi(a_t | s_t)
  $$

  Where:

  - $\pi(a_t | s_t)$: probability of taking action $a_t$ in state $s_t$
  - $G_t$: return (sum of rewards)
  - This approach works well when using neural networks.

---

### 🪜 Multi-Step Temporal Difference (TD)

Instead of updating after every step, **multi-step TD** backs up rewards over several steps:

For 2-step TD:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma r_{t+2} + \gamma^2 \max_a Q(s_{t+2}, a) - Q(s_t, a_t) \right]
$$

This allows the agent to learn from longer-term outcomes.

---

### 🧾 Eligibility Traces

- Combines multiple-step TD learning into a single method.
- Tracks how recently each state-action pair was visited.
- Recently visited pairs get more credit when rewards arrive.
- Eligibility fades over time (decays), hence "trace."

---

### 🧠 Model-Based TD & Planning

- **Planning** means using a model of the environment to improve the policy.
- **Model-based TD** uses both learning and planning:

  - Learns a model of how the environment works.
  - Uses this model to simulate experiences and plan.

- **Dynamic Programming** is a classic planning method.
- **Monte Carlo (MC)** and **standard TD** are model-free.

---

### 🧮 Function Approximation

Why needed:

- Real-world problems have **too many states/actions** to store exact values in a table.

Solution:

- Use function approximators to estimate value functions:

  - Neural networks
  - Gaussian processes
  - Other regression models

This allows generalization to unseen states.

---

### ✅ Wrap-up

This lecture covered many methods and ideas in reinforcement learning:

- **Reward shaping** to speed up learning.
- **Actor-Critic** splits learning into policy and value learning.
- **Policy gradient** directly optimizes the policy.
- **Multi-step TD** and **eligibility traces** improve learning from long-term outcomes.
- **Function approximation** makes learning scalable.
- **Model-based methods** combine learning and planning.

---
# File: 11.md
---

## Deep Learning

### What is Machine Learning?

- Machine Learning (ML) is a subset of Artificial Intelligence where machines learn from data to perform tasks without being explicitly programmed.
- It involves building models that can analyze data, learn patterns, and make predictions or decisions.

**Tom Mitchell’s definition:**

> A computer program learns from experience **E** with respect to tasks **T** and performance measure **P**, if its performance on **T**, measured by **P**, improves with **E**.

### Types of Learning in ML

1. **Supervised Learning** – Uses labelled data.

   - Example: Given images of apples (input) and labels "apple", the model learns to recognize apples.

2. **Unsupervised Learning** – Uses unlabelled data.

   - Example: The model groups similar images together without knowing what they are.

3. **Semi-supervised Learning** – Mix of labelled and unlabelled data.

4. **Reinforcement Learning** – Learns by interacting with an environment and receiving rewards or penalties.

---

### What is Deep Learning?

- Deep Learning (DL) is a part of ML that uses **deep architectures** (many layers of neurons) to learn **high-level features** from data.
- It mimics how the brain processes information through layered structures.

### Types of Deep Learning Architectures

1. **Convolutional Neural Networks (CNNs)** – Used for image classification and vision tasks.
2. **Autoencoders** – Compress data into lower dimensions and then reconstruct it (used for denoising or dimensionality reduction).
3. **Deep Belief Networks (DBNs)** – Layered networks that learn to represent data and generate new data.
4. **Recurrent Neural Networks (RNNs)** – Good for sequential data like time series or speech, using feedback connections as “memory”.

---

### Applications of Deep Learning

- **Computer Vision**: Object recognition, scene segmentation, 3D face reconstruction.
- **Self-driving Cars**: Real-time perception and decision-making.
- **Robotics**: Learning hand-eye coordination and control from data.
- **Other**: Natural language processing, medical diagnosis, game playing (e.g., AlphaGo).

---

### History and Breakthroughs

- **Early Days**:

  - 1943: McCulloch & Pitts – First mathematical model of a neuron.
  - 1949: Hebb – Learning rule (neurons that fire together, wire together).
  - 1958: Rosenblatt – Perceptron, first learning algorithm.
  - 1960s: ADALINE (Widrow & Hoff), early gradient descent models.

- **Setbacks**:

  - 1969: Minsky & Papert showed limitations of single-layer networks.

- **Renewed Interest**:

  - 1980s: Hopfield networks, Kohonen maps, Boltzmann machines.
  - 1986: Backpropagation algorithm (Rumelhart, Hinton, Williams) revived interest in multi-layer networks.

- **Modern Era**:

  - Advancements in computing and large datasets allowed deep networks to be trained effectively.
  - Key figures: LeCun (CNNs), Hinton (DBNs), Bengio, Ng (modern DL techniques).

---

### Neural Networks: From Biology to Artificial Systems

- **Biological Neurons**:

  - Receive signals through dendrites, process in the cell body, and send signals via axons.
  - Signal strength and timing carry information (spike frequency and phase).

- **Artificial Neural Networks (ANNs)**:

  - Built using nodes (neurons) and weights (synapses).
  - Each artificial neuron receives inputs, applies weights, computes an activation, and sends output to other neurons.
  - Simple in structure but capable of very complex behavior.

- **Information Flow**:

  - Inputs → Hidden layers (processing units) → Outputs
  - Recurrent networks allow feedback, enabling memory of previous inputs.

---

### Summary

- **Deep Learning** uses multi-layered neural networks to model complex patterns in data.
- Applications range across vision, robotics, and autonomous systems.
- Inspired by the brain, artificial neural networks have evolved through decades of research.
- The major breakthrough enabling current DL success is the ability to train deep networks effectively using algorithms like backpropagation.

---
# File: 12.md
---
## Artificial Neural Networks (ANNs)

### What Are ANNs?

- Inspired by biological neural networks (NNs), where learning happens by changing connections (synapses) between neurons.
- In ANNs, learning is modeled by adjusting weights of connections between artificial neurons.
---

## Artificial Neuron Model

An artificial neuron receives inputs and produces an output based on a function.

### Structure

- Inputs: $x_1, x_2, ..., x_n$

- Weights: $w_{i,1}, w_{i,2}, ..., w_{i,n}$ for neuron $i$

- Net input:

  $$
  \text{net}_i(t) = \sum_{j=1}^n w_{i,j}(t) \cdot x_j(t)
  $$

  This means: multiply each input by its weight, then sum everything.

- Output:

  $$
  y_i(t) = f_i(\text{net}_i(t))
  $$

  where $f_i$ is the activation function that transforms the net input into output.

---

## Activation Functions

These determine what output a neuron produces given its net input.

### 1. Threshold Function (used in Perceptrons)

$$
f_i(\text{net}_i(t)) =
\begin{cases}
1 & \text{if } \text{net}_i(t) \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

- Output is binary (0 or 1).
- Used in simple classification tasks.

### 2. Linear Function

- Output is a direct linear combination of inputs.
- Not useful alone for complex tasks since it's not nonlinear.

### 3. Sigmoid (Logistic) Function

$$
f_i(\text{net}_i(t)) = \frac{1}{1 + e^{-\frac{\text{net}_i(t) - \theta}{\tau}}}
$$

- Squashes input to range between 0 and 1.

- Parameters:

  - $\theta$: shifts the curve horizontally (like a threshold).
  - $\tau$: controls the steepness (slope) of the curve.

- Smooth and differentiable, making it suitable for learning algorithms like backpropagation.

---

## Neural Network Structure

### Feed-Forward Neural Network

- **Input Layer**: takes input vector, no computation.
- **Hidden Layer**: processes data using neurons with activation functions.
- **Output Layer**: produces final result (output vector).

Information flows from input → hidden → output.

---

## Learning in ANNs: Setting Weights

### Supervised Learning

- Use **training data**: pairs of input vector $x$ and desired output $y$.
- The goal: Adjust weights so the network's output approximates $y$ for each input $x$.

### Generalization vs Memorization

- The model should generalize well (work on new data) rather than just memorizing training data.

### Analogy: Fitting Functions

- Fitting a curve to points:

  - Too simple (e.g. line): underfitting
  - Too complex (e.g. 9th degree polynomial): overfitting

Same with ANNs:

- **Too few neurons**: not enough capacity → underfitting.
- **Too many neurons**: too much capacity → overfitting.

> There's no exact formula for choosing the perfect network size — only heuristics (rules of thumb).

---
# File: 13.md
---

## **1. Evaluation of Neural Networks**

### Error Functions

- **For regression**:
  Error is measured using the formula:
  **E = Σ (dᵢ − yᵢ)²**

  - _dᵢ_ = desired (target) output
  - _yᵢ_ = actual output

- **For classification**:
  Accuracy = (number of correctly classified samples) ÷ (total samples)

---

## **2. Perceptrons**

### Basic Perceptron

- **Net input**: net = Σ (wᵢ \* xᵢ)
- **Output**:

  - +1 if net > threshold
  - -1 otherwise

This models a decision boundary using a linear function.

### Including Bias

- Represent bias as a fixed input _x₀ = 1_ with a weight _w₀ = -threshold_
- Now the threshold is absorbed into the weights:
  net = w₀ \* x₀ + w₁ \* x₁ + ... + wₙ \* xₙ

---

## **3. Perceptron Computation**

- The decision boundary is a hyperplane:
  **w₀ + w₁x₁ + ... + wₙxₙ = 0**
- If result > 0 → output is 1
  If result ≤ 0 → output is -1
- This only works if the data is **linearly separable**.

---

## **4. Perceptron Training**

- If a point is misclassified, update the weights:

  - For a point _x_ with class _y_, and current weight _w_, update as:
    **w ← w + ηy·x**
  - η is the learning rate (controls how big each update is)

This formula works for both classes (+1 and -1) by using _y_ directly in the update.

### Algorithm

1. Start with random weights
2. While misclassified points exist:

   - Pick one misclassified point
   - Update weights using the rule above

### Notes on Learning Rate

- Too **large**: training becomes unstable
- Too **small**: training is slow
- If tweaking η doesn’t help, the data may not be linearly separable

---

## **5. Visualizing and Example**

In 2D, the decision boundary is a line. You can find intercepts by setting x or y to 0 and solving.

### Learning Example (Step-by-Step)

1. Start with weights (2, 1, -2)
2. Use a misclassified point (-2, -1)

   - Add the vector (-1) \* x = (-1, 2, 1)
   - New weights: (1, 3, -1)

3. Next point: (0, 2)

   - Add (1, 0, 2)
   - New weights: (2, 3, 1)

4. All points correctly classified → stop

**Note:** Often more iterations are needed.

---

## **6. Generalization**

- Perfect training classification does **not** guarantee good performance on new data.
- Simpler or smoother decision boundaries often generalize better.

---

## **7. Multiclass Discrimination**

### One-vs-All

- Use one perceptron per class.
- Each perceptron outputs:

  - 1 if input belongs to its class
  - 0 otherwise

- The class is the one with the **highest** output.

### Visual Example

- Input x₁, x₂, ..., xₙ connected to multiple perceptron outputs o₁, o₂, o₃, o₄

---

## **8. Example Code (PyTorch)**

```python
from torch.nn import Linear, Sigmoid, Module

class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 1)
        self.activation = Sigmoid()

    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
```

---

## **9. Multilayer Networks and Backpropagation**

### Problem with Single-Layer Networks

- Can’t handle **non-linearly separable** problems

### Solution

- Use **multiple layers** and a **differentiable activation function** (like sigmoid)
- This enables **gradient descent** learning

---

## **10. Backpropagation**

### Idea

- Adjust weights to minimize error between network output and desired output

### Error Function

- Use Mean Squared Error (MSE):
  **MSE = (1/N) Σ (yᵢ − dᵢ)²**

### Gradient Descent

- Repeatedly adjust weights in the direction that reduces the error:
  **w ← w − η \* ∂Error/∂w**

### Chain Rule

- Needed to compute gradients across layers
- For a composed function f(g(x)), the derivative is:
  **df/dx = (df/dg) \* (dg/dx)**

---

## **11. Sigmoid Neuron and Derivative**

### Sigmoid Function

- **σ(net) = 1 / (1 + e^(−net))**
- Smooth and differentiable
- Derivative:
  **σ’(net) = σ(net) \* (1 − σ(net))**

Used in backpropagation because it’s easy to compute and differentiable.

---

## **12. Feedforward and Backpropagation Steps**

1. **Feedforward**:

   - Pass inputs forward through network

2. **Backpropagate**:

   - Compute error at output
   - Propagate it backward using gradients
   - Adjust weights using gradient descent

---

## **Summary**

- Evaluating neural networks involves computing output errors.
- Perceptrons are linear classifiers trained with simple weight updates.
- Multiclass problems are handled with multiple perceptrons.
- Nonlinear problems require multilayer networks and backpropagation.
- Backpropagation uses gradient descent and the chain rule to compute and apply weight updates.
# File: 14.md
---

## **Backpropagation for One Neuron**

We want to compute how the error changes with respect to each weight in the neuron. This allows us to update the weights to minimize the error.

- **Error Function**:
  $E = \frac{1}{2}(o - d)^2$
  where:

  - $o$: output of the neuron
  - $d$: desired output (target)

- **Neuron Output**:
  $o = f(q)$
  where:

  - $f$ is the sigmoid function
  - $q = \sum_j w_j x_j$: weighted sum of inputs

### Derivatives (Using Chain Rule):

To find how $E$ changes with weight $w_j$:

$$
\frac{\partial E}{\partial w_j} = \frac{\partial E}{\partial o} \cdot \frac{\partial o}{\partial q} \cdot \frac{\partial q}{\partial w_j}
$$

Each component:

- $\frac{\partial E}{\partial o} = o - d$
- $\frac{\partial o}{\partial q} = f(q)(1 - f(q)) = o(1 - o)$ (derivative of sigmoid)
- $\frac{\partial q}{\partial w_j} = x_j$

So:

$$
\frac{\partial E}{\partial w_j} = (o - d) \cdot o(1 - o) \cdot x_j
$$

### Weight Update Rule:

Use gradient descent:

$$
\Delta w_j = -\eta \cdot \frac{\partial E}{\partial w_j} = -\eta (o - d) o (1 - o) x_j
$$

$$
w_j \leftarrow w_j + \Delta w_j
$$

Where $\eta$ is the learning rate (step size).

---

## **Backpropagation Algorithm Overview**

**Steps:**

1. Initialize weights randomly.
2. Repeat until mean squared error (MSE) is low enough or time runs out:

   - For each input pattern:

     - Compute activations for hidden and output layers.
     - Calculate the error.
     - Adjust weights:

       - Output layer weights
       - Hidden layer weights

---

## **Deep Learning Overview**

- Deep learning uses **deep architectures** (many layers) to learn **useful feature representations** from data.
- Often uses **non-linear transformations** to extract high-level features.
- Two types:

  - **Supervised** (e.g., CNNs: Convolutional Neural Networks)
  - **Unsupervised** (e.g., Autoencoders)

Today’s focus: **Convolutional Neural Networks (CNNs)**

---

## **Convolution and Filtering**

Convolution is used to process signals and images by applying filters. Filters are small grids (kernels) that slide over the data to extract patterns.

### Applications:

- **Noise Removal**: Suppress or remove unwanted noise.
- **Edge Detection**: Highlight edges in images.
- **Pattern Detection**: Match templates or features.
- **Deconvolution**: Reverse effects of filters applied earlier.

---

## **1D Convolution**

### Continuous Case:

Output:

$$
h(t) = (f * g)(t) = \int f(\tau) \cdot g(t - \tau) d\tau
$$

### Discrete Case:

Given:

- Input signal $f$ (length m)
- Filter $g$ (length n)
- Output signal $h$

Each output value is a **weighted sum** of a segment of the input:

$$
h[i] = \sum_{k} f[i - k] \cdot g[k]
$$

---

## **2D Convolution (Image Processing)**

Images are treated as 2D arrays. A 2D filter (kernel) is slid across the image.

**Steps:**

1. Multiply each filter element with corresponding image pixel.
2. Sum the results.
3. Assign this sum to the output pixel.

### Formula:

$$
I'(i,j) = \sum_k \sum_l I(i-k, j-l) \cdot H(k, l)
$$

- $I$: input image
- $H$: filter
- $I'$: output image

---

## **Stride**

Stride defines how many pixels the filter moves at each step:

- Stride = 1 → Filter moves one pixel at a time.
- Larger strides → smaller output.

**Output size formula**:

$$
\text{Output size} = \frac{(N - F)}{\text{stride}} + 1
$$

- $N$: input size
- $F$: filter size

E.g., for $N=7, F=3$:

- Stride 1 → Output size = 5
- Stride 2 → Output size = 3
- Stride 3 → Output size = \~2

---

## **Padding**

To avoid shrinking the output size:

- Add zeros around the input border ("zero-padding").

Common padding:
If filter size $F$, use padding $(F - 1) / 2$

- Keeps input and output the same size.

E.g.:

- $F = 3$: pad = 1
- $F = 5$: pad = 2

---
# File: 15.md
---

## 🔍 Convolution Layer

### 📐 Basic Concept

A convolution layer processes input images using **filters** (small grids of numbers) that slide across the image to detect patterns (like edges or textures).

### 🧮 Convolution Operation

Given:

- An image (e.g., 32x32x3: height, width, and depth for RGB)
- A filter (e.g., 5x5x3)

The operation:

- The filter slides over the image spatially.
- At each step, it performs a **dot product** between the filter and the corresponding image patch.
- The result is one number per position (called an **activation**).

Formula:

$$
\text{Activation} = w^T x + b
$$

Where:

- $w$ = filter weights (flattened into a vector)
- $x$ = image patch (also flattened)
- $b$ = bias term

---

### ➕ Padding

To keep output size the same as input:

- Use **zero-padding**, i.e., add zeros around the border.
- Padding size = (Filter size - 1)/2
  e.g., for 3x3 filter → pad with 1 pixel

---

### 🚶 Stride

Stride = how many pixels the filter moves each time.

**Output size** (1D formula):

$$
\text{Output size} = \frac{N - F}{S} + 1
$$

Where:

- $N$ = input size
- $F$ = filter size
- $S$ = stride

---

## 🧱 Structure of a Convolution Layer

### Example:

- Input: 32x32x3 (RGB image)
- Filter: 5x5x3
- Result: 28x28 activation map (if no padding and stride = 1)

Using multiple filters:

- 6 filters → 6 activation maps → stacked to form 28x28x6 output

### With Padding:

- If padding = 2 and stride = 1, then:

  $$
  \frac{32 + 2\times2 - 5}{1} + 1 = 32
  $$

  → Output: 32x32x6

---

## 🔧 Hyperparameters in Convolution Layer

You need to set:

| Parameter | Description                       |
| --------- | --------------------------------- |
| K         | Number of filters → affects depth |
| F         | Filter size (FxF)                 |
| S         | Stride (how much filter moves)    |
| P         | Padding (zero padding size)       |

**Output volume size:**

- Width: $(W1 - F + 2P)/S + 1$
- Height: $(H1 - F + 2P)/S + 1$
- Depth: = K (number of filters)

---

## 🤖 ConvNet (Convolutional Neural Network)

### What is a ConvNet?

A deep learning model that:

- Applies multiple **convolution layers**
- Uses **activation functions** like ReLU
- Optionally includes **pooling** and **fully connected** layers

### Example Stack:

1. Conv (6 filters of 5x5x3) + ReLU → 28x28x6
2. Conv (10 filters of 5x5x6) + ReLU → 24x24x10
3. Repeat...

---

## 👁️ Visualization

- Zeiler & Fergus (ECCV 2014): showed that different filters detect different features like edges, textures, or object parts.

---

## 📉 Pooling Layer

### Purpose:

- Reduces spatial size (height/width), not depth
- Helps make features more manageable and less sensitive to exact position

### 🏊 Types:

- **Max Pooling**: selects the maximum value in each region
- Applied **independently per activation map**

### 🔧 Hyperparameters:

- Filter size (F)
- Stride (S)

**Output size formula:**

$$
\text{Output width} = \frac{W1 - F}{S} + 1 \\
\text{Output height} = \frac{H1 - F}{S} + 1 \\
\text{Depth remains unchanged}
$$

**Common settings:**

- F = 2, S = 2 (halves spatial dimensions)
- F = 3, S = 2

> Zero-padding is rarely used in pooling.

---

## 🧠 Fully Connected (FC) Layer

### Purpose:

- Final decision-making stage (e.g., classification)
- Connects every neuron from the previous layer to each neuron in this layer

### Example:

- Input volume: 32x32x3 = 3072 values
- Flattened to a vector
- Weight matrix: e.g., 10x3072 for 10 output classes
- Output: 10x1 (one value per class)

---
# File: 16.md
Here's a combined, clear summary of the lecture slides on **Deep Learning**, with explanations and relevant missing context added where useful:

---

## **Convolutional Neural Networks (ConvNets) – Core Concepts**

### **1. Convolution Operations**

- A **convolution** slides a filter (small matrix) over an image (large matrix) to extract features like edges.
- Each filter detects specific patterns, e.g., vertical or horizontal edges.

### **2. Key Parameters**

- **Stride**: how far the filter moves at each step. Larger strides shrink the output.

  - Output size formula: **(N - F)/S + 1**

    - N = input size, F = filter size, S = stride.

- **Padding**: adds extra pixels (typically zeros) around the input to control output size.

  - Padding = (F-1)/2 keeps the output the same size as input when stride = 1.

### **3. Convolutional Layers**

- A **Conv layer** applies multiple filters. Each produces one output channel.
- Layers often use ReLU (Rectified Linear Unit) for non-linearity.

---

## **ConvNet Architectures – Case Studies**

### **LeNet-5 (LeCun et al., 1998)**

- Designed for digit classification (e.g., MNIST).
- Structure: **CONV → POOL → CONV → POOL → FC**
- Used 5x5 filters, 2x2 pooling.

---

### **AlexNet (Krizhevsky et al., 2012)**

- Input: 227x227x3 images.
- First Layer: 96 filters of size 11x11, stride 4 → Output: 55x55x96
- Parameters in CONV1: (11×11×3)×96 = 34,848 ≈ 35K
- Pooling: 3x3 filters, stride 2 → Output size = (55-3)/2 + 1 = 27 → 27x27x96
- Full pipeline:

  ```
  INPUT → CONV1 → POOL1 → NORM1 → CONV2 → POOL2 → NORM2
        → CONV3 → CONV4 → CONV5 → POOL3 → FC6 → FC7 → FC8
  ```

- Innovations:

  - First to use **ReLU**
  - **Dropout** to prevent overfitting (0.5)
  - **Data augmentation**
  - **L2 regularization** (weight decay = 5e-4)
  - Optimizer: **SGD + Momentum 0.9**

---

### **VGG-16 (Simonyan & Zisserman, 2014)**

- 16 layers total; only **3x3 conv filters**, stride 1, pad 1.
- **2-3 CONV layers** per pooling layer.
- Uses **2x2 max pooling**, stride 2.
- Stacking three 3x3 filters = 7x7 receptive field but fewer parameters.
- Deeper than AlexNet, improves accuracy:

  - Top-5 error: 7.3% vs 11.7% (ZFNet)

---

### **GoogLeNet (Szegedy et al., 2014)**

- Introduced the **Inception module** – combines different filter sizes in parallel.
- Efficient: 22 layers, **only 5 million parameters** (12x less than AlexNet).
- No fully connected (FC) layers.
- Winner of ILSVRC 2014: Top-5 error = **6.7%**

---

### **ResNet (He et al., 2015)**

- Very deep (e.g., 152 layers for ImageNet).
- Introduced **residual connections**:

  - Instead of learning **output = f(input)**, learn **residual = f(input) – input**.
  - Helps gradients flow better during training.

- State-of-the-art performance at the time.

---

## **Transformers**

- Encoder-decoder architecture, originally for NLP.
- Introduced in: _"Attention is All You Need"_ (Vaswani et al., 2017)

---

## **Training Deep Networks**

### **Strategies to Avoid Underfitting / Overfitting**

- **Underfitting (high bias)**:

  - Use deeper/wider networks
  - Train longer

- **Overfitting (high variance)**:

  - Use more data
  - Apply **regularization**, e.g., L2 norm
  - Use **dropout**, early stopping

### **Key Techniques**

- **Dropout**: randomly "turn off" neurons during training to prevent co-adaptation.
- **Early stopping**: stop training when validation loss stops improving.

---

## **Optimizers**

### **1. SGD with Momentum**

- Combines current gradient with previous step:

  ```
  v_t = β*v_{t-1} + (1-β)*∇L(w)
  w = w - α*v_t
  ```

  - β = momentum factor (usually 0.9)
  - α = learning rate

### **2. AdaGrad**

- Adapts learning rate per parameter based on past gradients.
- Parameters that receive small updates earlier get bigger updates later.
- Drawback: learning rate shrinks too much over time.

### **3. RMSProp**

- Like AdaGrad, but uses a **moving average** of past squared gradients.
- Helps avoid shrinking the learning rate too much.

### **4. Adam (Adaptive Moment Estimation)**

- Combines **Momentum + RMSProp**
- Tracks both:

  - **m_t**: moving average of gradients
  - **v_t**: moving average of squared gradients

- Bias correction is applied to both.
- Common settings:

  - Learning rate = 0.001
  - β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸

---

## **Deep Learning Software & Tools**

- **Libraries**:

  - TensorFlow / Keras
  - PyTorch
  - Caffe / Caffe2
  - Theano

- **Model Zoos**: Pretrained models available online:

  - Keras: [https://github.com/albertomontesg/keras-model-zoo](https://github.com/albertomontesg/keras-model-zoo)
  - TensorFlow: [https://github.com/tensorflow/models](https://github.com/tensorflow/models)
  - Caffe: [https://github.com/BVLC/caffe/wiki/Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

---

## **Transfer Learning**

- Use pretrained models (e.g., on ImageNet) for a new task.

- Options:

  - **Freeze** earlier layers (keep pretrained weights)
  - **Fine-tune** later layers on new data

- Examples:

  - MNIST, SVHN digit datasets
  - TensorFlow image retraining tutorial:
    [https://www.tensorflow.org/hub/tutorials/image_retraining](https://www.tensorflow.org/hub/tutorials/image_retraining)

---

Let me know if you’d like visual summaries or examples for specific architectures like ResNet or AlexNet.
# File: 17.md
---

## Deep Learning in Reinforcement Learning

### Why Deep Learning in RL?

- Traditional RL methods (like Q-tables) fail with large state spaces.
- For example, a small 4x4 grid = 16 states → manageable.
- But an 8x8 image = 64 pixels → 2⁶⁴ combinations → impractical.
- Larger images (like Atari games: 210x160x3 RGB) make state spaces huge.
- **Solution:** Use a **function approximator** (like a neural network) to estimate the Q-values, instead of a table.

---

### Deep Q-Learning (DQN)

#### What is it?

- Combines **Q-learning** (a type of RL algorithm) with a **deep neural network**.
- The neural network estimates the **Q-function**:

  $$
  Q(s, a; \theta) \approx Q^*(s, a)
  $$

  - $s$: state
  - $a$: action
  - $\theta$: neural network parameters (weights)

#### Q-Learning Formula Recap:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

- $\alpha$: learning rate
- $\gamma$: discount factor (future reward importance)
- $r$: reward
- $s'$: next state
- $a'$: next action

---

#### Loss Function in DQN

To train the network:

- Use **Mean Squared Error (MSE)** between the target and predicted Q-value:

  $$
  L(\theta) = \mathbb{E} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
  $$

  where the target $y$ is:

  $$
  y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
  $$

- $\theta^-$: parameters of a target network (a periodically updated copy of the main network)

---

### Case Study: Playing Atari Games

#### Setup

- **State:** last 4 game frames → stacked as a tensor (shape: 84x84x4 after preprocessing)
- **Action:** game controls (e.g., Left, Right, Up, Down)
- **Reward:** score change at each step

#### DQN Architecture

- **Input:** 84x84x4 grayscale image (stack of frames)
- **Network:**

  - 2 Convolutional layers
  - 2 Fully Connected layers
  - Output layer: one Q-value per action

---

### Case Study: AlphaGo

#### Key Achievements

- In 2016, **AlphaGo** (by DeepMind) beat a top human Go player (Fan Hui), and later world champion Lee Sedol.
- Combined:

  - **Convolutional neural networks (CNNs)** – extract features from Go board
  - **Reinforcement learning** – learn by playing itself
  - **Monte Carlo Tree Search (MCTS)** – to plan several moves ahead

#### How It Worked

- Trained first on expert human games (supervised learning).
- Then improved by playing against itself (reinforcement learning).
- Used tree search to evaluate possible moves.

---

### Gym by OpenAI

- **Gym** is an open-source toolkit to develop and compare RL algorithms.
- Provides environments like Atari, classic control tasks, etc.
- URL: [https://gym.openai.com](https://gym.openai.com)

---

### Final Takeaways

- **Deep RL** can tackle tasks with huge state spaces.
- **CNNs** aren't just for images – they help extract useful state features in games.
- **Combining methods** (supervised learning, RL, tree search) leads to strong AI systems.
- But: AI may play games well, but may never appreciate them like humans.

> "Robots will never understand the beauty of the game the same way that we humans do." – Lee Sedol

---
# File: 18.md
---
## 🔁 Reinforcement Learning (RL)

### Traditional ML vs RL

- In traditional machine learning (ML), learning happens from labeled data.
- In reinforcement learning, an agent learns by **interacting with an environment**, receiving **rewards** based on its actions. The goal is to learn a policy to maximize the cumulative reward over time.
---

### 🎰 Multi-Armed Bandits

- A simplified RL problem where an agent must choose between multiple options ("arms").
- **Exploration vs Exploitation**:

  - **Exploration**: trying different actions to discover their rewards.
  - **Exploitation**: choosing the action that has given the best reward so far.

- Strategies:

  - **Greedy**: always choose the best-known action.
  - **ε-greedy**: usually pick the best-known action, but with probability ε choose a random one.
  - **Random**: choose any action with equal chance.

---

### 🔢 Value Functions

- **State-Value Function (Vπ(s))**: Expected total reward starting from state **s**, following policy **π**.
- **Action-Value Function (Qπ(s, a))**: Expected total reward starting from state **s**, taking action **a**, and then following policy **π**.

---

### 🔄 Bellman Equation

- A recursive way to express value functions.
- For state value:

  $$
  V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]
  $$

  - **π(a|s)**: probability of taking action a in state s
  - **P(s'|s,a)**: transition probability to state s′ from s with action a
  - **R(s,a,s′)**: reward for moving from s to s′ via a
  - **γ**: discount factor (between 0 and 1)

---

### 🧮 Solving Bellman Equations

- **Dynamic Programming (DP)**: Requires a model of the environment. Uses iterative updates.
- **Temporal Difference (TD)**: Learns from experience without a full model. Combines ideas from DP and Monte Carlo.
- **Monte Carlo Tree Search (MCTS)**: Used especially in game-playing agents like AlphaGo. Builds a search tree using simulations.

---

### 📘 Q-learning

- A model-free RL algorithm to learn **Q-values**.
- Update rule:

  $$
  Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
  $$

  - **α**: learning rate
  - **γ**: discount factor
  - **r**: reward
  - **s′**: next state

- Faces the exploration/exploitation dilemma just like the bandit problem.

---

### 🤖 Deep Q-Learning

- Uses a **neural network** to approximate Q(s, a) when the state/action space is too large.
- **Loss function**:

  $$
  L = \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)^2
  $$

  - Trains the neural network to minimize this loss.

#### 🧠 Exam Tip:

**Q: What's the main advantage of Deep Q-Learning over traditional Q-learning?**
**A: It can handle high-dimensional input spaces.**
(Correct answer: **C**)

---

## 🧠 Deep Learning (DL)

### ⚡ Basic Concepts

- **Neurons**: Basic units in a neural network that apply a function to inputs.
- **Neural Network**: Layers of neurons connected with weights (edges).

---

### 🏗️ Network Structure

- **Input Layer**: Takes the data (e.g., image pixels).
- **Hidden Layers**: Intermediate layers that learn features.
- **Output Layer**: Gives the final result (e.g., classification).

> A "deep" network has **multiple hidden layers**.

---

### 📈 Activation Functions

These introduce non-linearity.

- **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$
- **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$

---

### 🧱 Specialized Layers

- **Convolutional Layers**: Useful for images; detect patterns like edges.
- **Pooling Layers**: Reduce the size of feature maps (e.g., max pooling).

---

### 📉 Loss Functions

Used to measure how well the model is doing.

- **MSE (Mean Squared Error)**:

  $$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

  - **y_i**: true value
  - **ŷ_i**: predicted value

---

### 🔁 Backpropagation

- The process of training the network by updating weights:

  1. **Forward Propagation**: Compute output.
  2. **Compute Loss**.
  3. **Backward Propagation**: Use gradients to update weights (via **gradient descent**).

---

### 🛠️ Optimizers

- Algorithms like **SGD (Stochastic Gradient Descent)** or **Adam** are used to adjust weights and minimize the loss.

---

## 📚 Exam Practice Examples

- **Explain Q-learning**: It's a method in RL where agents learn action values (Q-values) to maximize rewards using a simple update rule.
- **List 2 activation functions**: ReLU, Sigmoid

---
# File: 19.md
---
### 📘 Multi-Agent Reinforcement Learning (MARL)
---

### 🔍 What is Multi-Agent Learning?

Multi-agent learning (MAL) studies systems where **two or more autonomous agents** interact and learn from experience. Unlike single-agent learning, where theoretical guarantees exist, these guarantees often break down when **multiple learners** are involved.

**Stone and Tuyls Definition**:

> "The study of multiagent systems in which one or more autonomous entities improve automatically through experience."

---

### 🧠 Background Concepts

#### 🔁 Reinforcement Learning (RL)

- A **key technique** for agents to adapt and learn from interacting with the environment.
- Single agent interacts with environment: chooses actions (A), receives rewards (R), transitions to new states (S).

#### 🎲 Classical Game Theory

- Focuses on **strategic decision making** among **rational players**.
- Assumes full information and rationality.
- Key concept: **Nash Equilibrium** – a situation where no player can benefit by changing strategy while others keep theirs fixed.

#### 🧬 Evolutionary Game Theory (EGT)

- Inspired by biological evolution rather than rationality.
- Agents update strategies based on payoff (fitness), not necessarily rational decisions.

  Two key **replicator dynamic** equations:

  1. $$
     \frac{dp_i}{dt} = p_i(e_i^T A q - p^T A q)
     $$

     - $p_i$: proportion of population using strategy $i$
     - $e_i$: unit vector for strategy $i$
     - $A$: payoff matrix
     - $q$: opponent strategy

  2. $$
     \frac{dq_i}{dt} = q_i(p^T B e_i - q^T B p)
     $$

     - $q_i$: strategy distribution for opponent
     - $B$: opponent’s payoff matrix

---

### 🚨 The Multi-Agent Challenge

In a multi-agent setting:

- Agents no longer adapt in isolation.
- The environment becomes **non-stationary** due to others learning.
- Even **fully cooperative agents** may struggle due to communication costs or failures.
- **Markov property** often breaks down because agent actions depend on unknown actions of others.

---

### 🌍 Why Use Multi-Agent Systems?

Multi-agent systems are used in:

- **Teamwork**: sports, exploration robots, sensor networks
- **Scheduling**: job shops, traffic lights
- **Trading**: auctions, stock markets
- **Simulations**: military strategy, economic modelling

---

### 🧩 Learning Paradigms in Multi-Agent Systems

1. **Online RL toward individual utility**
2. **Online RL toward social welfare**
3. **Co-evolutionary learning**
4. **Swarm Intelligence**
5. **Adaptive mechanism design**

---

### 🛠️ Components of a Multi-Agent Learning System

1. **The Environment**

   - Defines state space, action space, and how states change (transition function).

2. **The Agents**

   - Can communicate with the environment and each other.
   - Have utility functions (preferences).
   - Use policies to choose actions.

3. **The Interaction Mechanism**

   - Specifies how and when agents interact.
   - Determines observability and whether interactions are simultaneous or sequential.

4. **The Learning Mechanism**

   - Defines:

     - Who learns (individual or group)
     - What is being learned (e.g., policies)
     - What data is available
     - How behaviour is updated (learning rule)
     - The learning objective (e.g., maximize reward)

---

### ⚖️ Game Theory as a Framework

Game theory provides a structured way to model interactions:

- **Actions**: what each agent can do
- **Strategies**: probability distributions over actions
- **Joint actions**: determine outcomes (payoffs)
- If players are **rational** and have **full knowledge**, Nash equilibrium can predict behaviour.

---

### 🤝 Cooperation vs Competition

In real applications, agents may:

- Have **common goals** (e.g. robots cleaning a house together).
- Be **competitive** (e.g. buyers in an auction).
- Need to **coordinate** or **negotiate** (e.g. traffic flow).

This leads to questions like:

- What does "optimal" mean in multi-agent settings?
- How can agents learn to act optimally?

---

### ⏳ Historical Perspective

#### 📌 Startup Phase (Late 1980s – Early 2000s)

- Inspired by nature: ant colonies, herding, imitation.
- Exploratory and broad.
- Early MARL efforts began here.

#### 📌 Consolidation Phase (2000s – Now)

- Focus shifted to theoretical foundations.
- Game-theoretic RL became central.

---

### ✅ Summary

- Multi-Agent Learning is fundamentally different from single-agent RL.
- Key questions:

  - What should agents learn (objectives)?
  - How should they learn (algorithms)?

- Multiple paradigms exist, depending on context.
- Game Theory and Evolutionary Game Theory provide useful tools and models.
- Practical applications are wide-ranging and highly impactful.

---
# File: 20.md
---

## 🧠 Multi-Agent Reinforcement Learning (MARL)

### 🧩 The Problem: Single-Agent vs. Multi-Agent Learning

#### In **Single-Agent Reinforcement Learning (RL)**:

- The agent learns in a **stationary** environment (nothing changes over time).
- You can use tools like **Dynamic Programming**, **Monte Carlo methods**, or **Q-learning** to learn optimal behavior.

#### In **Multi-Agent Learning (MAL)**:

- Each agent is **learning simultaneously**, making the environment **non-stationary**.
- It’s more complex because other agents’ actions can change.
- Two main settings:

  - **Cooperative**: agents share goals.
  - **Self-interested (competitive)**: agents have conflicting goals.

- Agents may:

  - Not know others’ payoffs.
  - Not know others’ learning strategies.
  - Still need to make good decisions.

#### Goals in MAL:

- Learn a **stationary strategy**.
- Reach a **joint equilibrium** when all agents use the same algorithm (self-play).
- Perform well as a **best response** to others.
- Guarantee a **minimum payoff** even in worst-case scenarios.

---

## 🎲 Game Theory – The Foundation of Multi-Agent Systems

Game theory studies how agents make decisions when outcomes depend on other agents' choices.

### 🔍 Game Elements:

- **Players**: the agents.
- **Actions**: choices each player can make.
- **Payoffs**: rewards players receive based on all players' actions.

#### Types of Games:

- **Normal-form games** (matrix form): all at once, one-shot decisions.
- **Extensive-form games**: turns, sequences.
- **Repeated games**: played over multiple rounds.
- **Stochastic or Markov games**: stateful games with transitions.

### 🎮 Example: Rock-Paper-Scissors (RPS)

- **Zero-sum**: one player’s gain = the other’s loss.
- **Symmetric**: identical rules for both.
- **Mixed strategy equilibrium**: both play randomly with equal probabilities (1/3, 1/3, 1/3).

### 🤝 Cooperative vs. Competitive:

- **Cooperative** if payoff matrices are identical (shared rewards).
- **Competitive** if payoffs conflict (e.g. zero-sum).

---

## 🔁 Repeated Games & Learning Potential

### 🎭 Example: Prisoner's Dilemma

|              | Confess (Defect) | Hold Out (Cooperate) |
| ------------ | ---------------- | -------------------- |
| **Confess**  | (1,1)            | (5,0)                |
| **Hold Out** | (0,5)            | (3,3)                |

- **Dominant Strategy**: each player is best off confessing, no matter what the other does.
- **Equilibrium (Confess, Confess)**: stable but not socially optimal.
- **Opportunity in learning**: repeated play may allow agents to learn to **cooperate** (3,3), reaching better social outcomes over time.

---

## 🧠 Understanding Strategies and Equilibria

- **Pure strategy**: always pick the same action.
- **Mixed strategy**: choose randomly among actions with specific probabilities.

### Key Concepts:

- **Best Response**: best action assuming other players' strategies.
- **Dominant Strategy**: best action no matter what others do.
- **Nash Equilibrium**: set of strategies where no player benefits by changing alone.

  > Always exists but can be **multiple** and **hard to compute**.

---

## 📌 Key Takeaways

- Multi-agent learning is **fundamentally different** from single-agent due to non-stationarity.
- **Game theory** provides the basic tools for analyzing interactions.
- **Repeated interaction** opens the door to learning cooperative behavior.
- Understanding **dominant strategies** and **Nash equilibria** is critical.
- **Practical examples**: games (Poker, WoW), economics (OPEC), markets (auctions), negotiations.

---
# File: 21.md
---

## **1. Game Theory Basics in Multi-Agent Settings**

### **Prisoners’ Dilemma**

- **Setup:** Two players each choose between “Confess” (Defect) or “Hold out” (Cooperate).

  |                          | Prisoner 2: Confess | Prisoner 2: Hold Out |
  | ------------------------ | ------------------- | -------------------- |
  | **Prisoner 1: Confess**  | (1,1)               | (5,0)                |
  | **Prisoner 1: Hold Out** | (0,5)               | (3,3)                |

- **Dominant Strategy:** Always Confess (Defect), because it's the best response no matter what the other does.

- **Dominant Strategy Equilibrium:** (Confess, Confess), payoff = (1,1)

- **Social Conflict:** (3,3) is better for both, but rational players don’t pick it. Multi-agent learning might help reach the better (cooperative) outcome over time.

---

### **Battle of the Sexes**

- **Scenario:** Alice prefers Ballet, Bob prefers Football. They want to do something together.

  |                     | Bob: Ballet | Bob: Football |
  | ------------------- | ----------- | ------------- |
  | **Alice: Ballet**   | (2,1)       | (0,0)         |
  | **Alice: Football** | (0,0)       | (1,2)         |

- **No dominant strategy equilibrium**

- **Two Pure Nash Equilibria:**

  - (Ballet, Ballet): Alice gets 2, Bob gets 1
  - (Football, Football): Alice gets 1, Bob gets 2

- **Mixed Strategy Equilibrium:**
  Players choose their action randomly based on certain probabilities so each is indifferent.

  For Alice to be indifferent:

  $$
  2y_B = y_F \Rightarrow y_B = \frac{1}{3}, y_F = \frac{2}{3}
  $$

  (y = Bob’s strategy)

  Similarly, Bob will:

  $$
  x_F = \frac{1}{3}, x_B = \frac{2}{3}
  $$

  (x = Alice’s strategy)

- **Problem:** Mixed strategies lead to **lower expected payoffs** due to miscoordination risk.

---

## **2. Simple Multi-Agent Learning Approaches**

### **Fictitious Play**

- Each agent observes others’ past actions and assumes they'll keep using the same pattern.
- The agent then plays the best response to the **estimated frequency** of others’ actions.
- Variants include:

  - Recency-weighted observations
  - Softmax-style responses
  - Small adjustments toward best response

### **If all agents use fictitious play:**

- Strict Nash equilibria are **stable** outcomes.
- Sometimes leads to **cycling behaviors** (strategies go in loops).
- Empirical distributions (averaged actions over time) can still converge to Nash equilibrium.

---

## **3. Stochastic or Markov Games**

### **What are they?**

- Like repeated games, **but with states**.
- Each **state** is a separate game.
- The game transitions between states depending on **joint actions** taken by agents.

### **Key Components:**

- $n$: number of agents
- $S$: set of states
- $R$: rewards/payoffs
- $P$: transition probabilities (how actions lead to new states)
- $\gamma$: discount factor (how future rewards are valued)

### **Example:**

- A game with grid-like states and random movement depending on actions.
- Agents move between states with 30% or 50% probability.
- $\gamma = 0.9$ → future rewards matter a lot.

---

## **4. Learning in Stochastic Games**

### **Why Learn?**

- Nash Equilibria are hard to calculate in these games.
- Agents may **not know**:

  - Their own rewards fully
  - Others’ rewards
  - Transition model
  - Other agents' strategies

### **Learning Strategies:**

Adapted from **Single-Agent Reinforcement Learning (RL):**

- **Independent Learning:** Learn without modeling others.
- **Joint Action Learning:** Model other agents’ actions.
- **Minimax-Q Learning:** For zero-sum games (one's gain = other's loss).
- **Nash-Q Learning:** Learn equilibrium strategies using game theory.
- **Correlated Equilibrium (CE) Q-Learning:** More general than Nash; allows coordination based on shared signals.

---

## **5. Special Case: Zero-Sum Stochastic Games**

- Rewards always add up to a constant (e.g. one wins what the other loses).
- **Nice properties:**

  - All equilibria have the same value.
  - You can use **value iteration** (like in single-agent MDPs) with a **minimax operator** to find Nash equilibrium.
  - Has a Bellman-style update rule (for value estimation).

---

## **6. Spectrum of Multi-Agent Learning (MAL)**

From **Independent Q-Learners** (treat others as part of the environment)

TO

**Joint Action Learners** (fully model and respond to other agents)

> **Modern algorithms lie somewhere in between**, balancing complexity and coordination.

---
# File: 22.md
---

## Multi-Agent Reinforcement Learning (MARL)

### Independent Learners

- Each agent learns _as if it were alone_, ignoring others.
- Other agents’ behavior appears like noise in the environment.

**Pros:**

- Simple to implement using single-agent RL techniques.
- Easily scales to many agents.

**Cons:**

- No convergence guarantee.
- No coordination mechanism.

#### Example: Q-Learners in the “Battle of the Sexes”

|       | B    | F    |
| ----- | ---- | ---- |
| **B** | 2, 1 | 0, 0 |
| **F** | 0, 0 | 1, 2 |

- Two Q-learners learn based only on their own rewards.
- They converge to one of the two pure-strategy Nash equilibria (either (B,F) or (F,B)), depending on random initial conditions.
- Over time, each player improves their strategy, but without coordination.

### Joint Action Learners (JAL)

- Agents _observe what others do_.
- Estimate others’ policies and play optimally against those.
- Inspired by **fictitious play** (assume others use stationary strategies).

**Pros:**

- Enables coordination.

**Cons:**

- Requires observation of others.
- Complexity grows fast with more agents.

---

### Minimax Q-Learning

**For zero-sum games** (e.g., Rock-Paper-Scissors):

- One player’s gain is the other’s loss.
- Each agent tries to **maximize its minimum possible reward** (worst-case thinking).

Let:

- π₁, π₂, π₃ be probabilities of choosing Rock, Paper, Scissors.

Expected payoff depends on what the opponent plays.
E.g., if opponent plays Rock, agent’s payoff = π₂ − π₃
(only Paper beats Rock, Scissors loses).

**Goal:**
Maximize this minimum payoff across all possible opponent actions:

```
max π min (π₂ - π₃, -π₁ + π₃, π₁ - π₂)
```

This ensures the agent isn't easily exploited.

**Solution:**

- Solve with **linear programming** to find π that satisfies constraints.
- For Rock-Paper-Scissors, optimal π = (1/3, 1/3, 1/3), V = 0.

#### In Markov Games (multi-state):

- Redefine value function V(s) as expected reward from state s.
- Q(s, a, o) = expected reward if agent does action a, opponent does o.

**Update Rule:**

```
Q(s, a, o) ← (1 - α) Q(s, a, o) + α [r + γ V(s')]
```

Where:

- α = learning rate
- γ = discount factor
- V(s') is computed using a minimax over Q-values in next state.

**Benefits:**

- Converges under certain conditions.
- More robust than naive Q-learning.

**Limitation:**

- Only works for **zero-sum** games.

---

### Nash-Q Learning (for General-Sum Games)

- Extends minimax-Q to **non-zero-sum** games.
- Agents learn Q-values for all joint actions.

Update Rule:

```
Qᵢ(s, a₁, ..., aₙ) ← (1 - α) Qᵢ(s, a₁, ..., aₙ) + α [r + γ NashPayoffᵢ(s')]
```

**Challenges:**

- Need to compute **Nash equilibrium** at every step.
- Computationally hard.
- Multiple equilibria → selection problem (which one to use?).

---

### Gradient Ascent Approaches

Rather than learning Q-values, directly **adjust policy** using gradient of expected reward.

**Basic formula:**

```
Δπᵢ ← ∇Vᵢ(πᵢ)
πᵢ ← πᵢ + Δπᵢ
```

Examples:

- **IGA**: Infinitesimal Gradient Ascent.
- **GIGA**: Generalized IGA.

**Limitation:** Only works well for **2-player matrix games** (static, not dynamic).

---

### WoLF (“Win or Learn Fast”)

Improves gradient methods using **adaptive learning rate**:

- Learn **slowly** when winning (avoid overreacting).
- Learn **quickly** when losing (adapt fast).

Helps stabilize learning and improve convergence.

Variants:

- **WoLF-IGA, GIGA-WoLF, WoLF-PHC (Policy Hill Climbing)**

---

### Fully Cooperative Tasks

In these tasks, all agents **share the same reward**. The aim is to maximise total team performance.

**Q-learning update:**

```
Q(x, u) ← Q(x, u) + α [r + γ max Q(x', u') - Q(x, u)]
```

Where:

- x = current state
- u = joint action (all agents’ actions)
- r = shared reward
- α = learning rate
- γ = discount factor

**Problem:** Without coordination, agents might break ties differently and choose conflicting actions.

#### Example:

Two robots must choose to go Left or Right to avoid a crash while staying in formation.
Best outcomes: (Left, Left) or (Right, Right) → reward 10
Bad coordination: one goes left, one goes right → reward 0 or negative

**Coordination is critical.**

#### Solutions:

- **Team Q-learning**: assumes unique best joint action. Works only if that's true.
- **Distributed Q-learning**: each agent learns local Q-function. Works in deterministic environments only.

---

### Summary of MARL Approaches

**Main Families:**

- **Independent Learners**: simple but uncoordinated.
- **Joint Action Approaches**: coordinate by observing others.
- **Minimax-Q / Nash-Q**: handle competitive & general-sum games using game theory.
- **Gradient-based Methods**: directly tweak policies.
- **WoLF**: adjusts learning rates dynamically.

**Use Cases:**

- Robot teams
- Online auctions
- Self-managing systems
- Video games
- Military simulations

---
# File: 23.md
# Intentionally blank
# File: 24.md
---

### 🧬 Evolutionary Game Theory (EGT)

- **Classic Game Theory**: Based on economic principles. Players are rational and aim to maximize payoffs. The key idea is **Nash Equilibrium**, where no player benefits from changing strategy alone.
- **Evolutionary Game Theory**: Inspired by biology. Players (individuals) are not necessarily rational but evolve over time through **selection** (rewards) and **mutation** (random change).

  - Focuses on **populations** where strategies (or types) compete.
  - Central concepts:

    - **Replicator Dynamics**
    - **Evolutionarily Stable Strategies (ESS)**

---

### 🔄 The Prisoner’s Dilemma in EGT

- Payoff Matrix:

  |       | C   | D   |
  | ----- | --- | --- |
  | **C** | 3,3 | 0,5 |
  | **D** | 5,0 | 1,1 |

- In EGT, the population includes:

  - **Cooperators (C)**
  - **Defectors (D)**

- Fitness (reproductive success) depends on interactions.

- If equal numbers of C and D:

  - **C’s average fitness** = (3 + 0)/2 = 1.5
  - **D’s average fitness** = (5 + 1)/2 = 3
  - → Defectors increase in the population.

---

### 📈 Replicator Dynamics (Single Population)

Explains how the proportion of each type (strategy) changes over time.

**Formula**:

$$
\dot{x_i} = x_i (f_i(x) - \bar{f}(x))
$$

Where:

- $x_i$: fraction of population playing strategy $i$
- $f_i(x)$: fitness of strategy $i$
- $\bar{f}(x)$: average fitness of the population

**Meaning**: If a strategy does better than average, it grows; if worse, it shrinks.

Applied to Prisoner's Dilemma:

$$
\dot{x} = x^\top A x - x^\top A x
$$

- Where $x$ is a vector of proportions and $A$ is the payoff matrix.

- **Fixed Points**:

  - All **Defectors** (x = 0): Stable
  - All **Cooperators** (x = 1): Unstable

---

### ✌️ Rock-Paper-Scissors Example

Payoff matrix:

$$
A =
\begin{bmatrix}
0 & -1 & 1 \\
1 & 0 & -1 \\
-1 & 1 & 0
\end{bmatrix}
$$

- Mixed Nash Equilibrium at $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$
- No strategy wins long term — dynamics cycle endlessly.

---

### 👥 Multi-Population Replicator Dynamics

Used when **two players or groups** interact.

Formulas:

$$
\dot{x_i} = x_i ((A y)_i - x^\top A y)
$$

$$
\dot{y_i} = y_i ((x^\top B)_i - x^\top B y)
$$

Where:

- $x, y$: distributions in each population
- $A, B$: payoff matrices for each player

Each population evolves **independently**, but their fitness depends on the other.

---

### 🔃 Examples of Multi-Agent Dynamics

Visual simulations show strategy evolution for:

- **Prisoner’s Dilemma**: Defection dominates.
- **Stag Hunt**: Two stable strategies (coordination game).
- **Matching Pennies**: Cycles; no stable equilibrium.

---

### 🔗 Connection to Reinforcement Learning

- **Cross Learning** (a simple learning rule for agents) has been shown to converge to replicator dynamics in the continuous time limit.
- This creates a formal link between **multi-agent reinforcement learning** and **evolutionary game theory**.

---

### 📘 Dictionary Comparison

| Reinforcement Learning | Classical Game Theory | Evolutionary Game Theory |
| ---------------------- | --------------------- | ------------------------ |
| environment            | game                  | game                     |
| agent                  | player                | population               |
| action                 | action                | type                     |
| policy                 | strategy              | distribution over types  |
| reward                 | payoff                | fitness                  |

---

### 🛡️ Evolutionarily Stable Strategies (ESS)

- A strategy $x$ is **ESS** if:

  1. $u(x, x) \ge u(y, x)$ for all $y$
  2. If $u(x, x) = u(y, x)$, then $u(x, y) > u(y, y)$

Where $u(a, b)$ is the payoff when strategy $a$ meets $b$.

In Prisoner's Dilemma:

- Defection is an ESS because it performs better against any mix of strategies.

---

### 🔀 Selection-Mutation Dynamics

Adds **mutation** to the replicator dynamics.

Formula:

$$
\dot{x_i} = x_i(f_i(x) - \bar{f}(x)) + \sum_{j \ne i} [\mu_{ji} x_j - \mu_{ij} x_i]
$$

Where:

- $\mu_{ji}$: mutation rate from $j$ to $i$

This allows diversity in strategies and avoids getting stuck in pure strategies.

---

### 🔚 Final Note

Replicator dynamics are more than just theoretical tools — they help us understand how learning works in **multi-agent reinforcement learning** systems, especially when agents are not fully rational or informed.

---
# File: 25.md
---
## 🧠 Core Idea

There is a formal link between **Reinforcement Learning** (how agents learn through rewards) and **Evolutionary Game Theory** (how populations evolve based on success in interactions). This connection helps us understand, design, and analyze multi-agent systems more effectively.
---

## 🔗 Reinforcement Learning & Evolutionary Game Theory (EGT)

### Evolutionary Game Theory Recap

- Studies **populations** of individuals with different "types" (strategies).
- Individuals interact randomly; **fitness** (success) affects how likely a type spreads.
- Evolution involves:

  - **Selection** → like exploitation in RL.
  - **Mutation** → like exploration in RL.

### Replicator Dynamics (EGT Model)

These equations describe how strategy frequencies change over time in a population:

- **Single population:**

  $$
  \dot{x}_i = x_i \left( (A x)_i - x^T A x \right)
  $$

  - $x_i$: frequency of strategy $i$
  - $A$: payoff matrix
  - $(A x)_i$: expected payoff of strategy $i$
  - $x^T A x$: average payoff in the population

- **Two populations (co-evolving):**

  $$
  \dot{x}_i = x_i \left( (A y)_i - x^T A y \right) \\
  \dot{y}_j = y_j \left( (x^T B)_j - x^T B y \right)
  $$

  - $x$, $y$: strategy distributions for each population
  - $A$, $B$: payoff matrices

---

## 📈 Example Dynamics in Games

Using replicator dynamics, we can simulate how strategies evolve in 3 classic games:

1. **Prisoner’s Dilemma** – Both players tend to defect (D,D), which is stable.
2. **Stag Hunt** – Multiple stable outcomes (e.g., cooperate or not), but some are more stable than others.
3. **Matching Pennies** – Players end up randomizing; strategies cycle around a point.

The dynamics explain not just equilibrium, but **how strategies move over time**.

---

## 🤝 Link to Multi-Agent Learning

### Same Concepts, Different Terms:

| Reinforcement Learning | Classical Game Theory | Evolutionary Game Theory |
| ---------------------- | --------------------- | ------------------------ |
| agent                  | player                | population               |
| action                 | action                | type                     |
| policy                 | strategy              | distribution over types  |
| reward                 | payoff                | fitness                  |

### Key Insight

- The **change in policy** of an RL agent behaves like replicator dynamics.
- The formal link: **Cross learning** (a simple form of RL) leads to replicator dynamics in continuous time (Börgers & Sarin, 1997).

### Example: Matching Pennies Game

- 2 players: choose heads or tails.
- Player 1 wins if they choose differently.
- **Best strategy**: randomize with 50/50 chance.
- Replicator dynamics and RL (e.g., learning automata) both evolve toward this strategy.

---

## ⚙️ Why This Link Matters

### Benefits:

- Helps **understand** and **visualize** what RL agents are learning.
- **Design new RL algorithms** by shaping the dynamics first.
- Makes **parameter tuning easier** by interpreting learning as population behavior.
- Useful in **complex, multi-agent environments**.

---

## 🧩 Applications of EGT in Multi-Agent Systems

### 1. Parameter Tuning

- Simulate agent learning using EGT to find good learning rates, exploration settings, etc.

### 2. Algorithm Design

- Reverse approach: choose desired behavior → derive learning rules to get it.

### 3. Complex Strategic Settings

- EGT helps even when a full payoff matrix or model is impossible:

  - Too many actions or states.
  - Payoffs unknown or hard to calculate.

- Use **meta-strategies** like:

  - Poker styles (e.g., “shark” vs “fish”).
  - Stock trading strategies.
  - Robot coordination strategies.

### 4. Real-World Examples

- Trading in stock markets.
- Space debris removal.
- Strategy analysis in Poker.
- Studying alternative market mechanisms.

---

## ✅ Summary

- Reinforcement Learning and Evolutionary Game Theory describe **the same learning process** from different angles.
- Replicator dynamics provide a **bridge** between game theory, evolution, and machine learning.
- This link gives **practical tools** for designing and analyzing multi-agent systems.

---
# File: 26.md
---

### **What is Swarm Intelligence?**

**Swarm Intelligence (SI)** refers to complex global behaviors that emerge from simple local interactions among individuals (agents) in a system. These agents follow basic rules and do not have knowledge of the global system.
→ Example: Ant colonies, bee hives, fish schools, bird flocks.

Key quote:

> “Swarm Intelligence is the complex global behavior shown by a distributed system that arises from the self-organized local interactions between its constituent agents.” – _Marco Dorigo_

---

### **Inspiration from Nature**

**Social Insects (especially ants):**

- Only \~2% of insects are social (e.g., ants, termites, some bees and wasps).
- Ants make up \~50% of all social insects.
- Ants’ total biomass ≈ human biomass.
- Ants exhibit specialized roles: harvesters, fungus growers, army ants, weavers, etc.
- Ants were arguably the **first farmers**.

**Bees:**

- Specialize in division of labor.
- Communicate food source location based on quality and distance.
- Maintain hive temperature through cooperation.

**Other Animals:**

- **Fish schools** and **bird flocks** also exhibit swarm behavior.
- Humans sometimes show similar behaviors (e.g., crowds, traffic).

---

### **Why Do Animals Swarm?**

Swarming behavior has clear benefits:

- **Energy savings**: e.g., geese flying in a V-formation reduce air resistance, increasing range by \~70% and allowing higher speed.
- **Protection**: confusion and safety in numbers against predators.
- **Hunting**: coordinated movement (e.g., tuna forming crescents to trap prey).

**Insight**: Individual animals don’t plan these outcomes. Instead, simple individual rules create useful **emergent group behaviors**.

---

### **How Do They Coordinate?**

**Key Mechanism: Self-Organization**

- A global pattern arises from local interactions.
- No central control.
- Based only on local information.

**Four Ingredients of Self-Organization:**

1. **Multiple interactions** – agents interact frequently.
2. **Randomness** – promotes exploration of new solutions.
3. **Positive feedback** – reinforces useful behavior (e.g., trail-following).
4. **Negative feedback** – prevents over-commitment (e.g., pheromone evaporation).

---

### **Main Features of Swarm Intelligence Systems**

- Parallel and distributed
- Stochastic (includes randomness)
- Adaptive to environment
- Use feedback mechanisms
- **Autocatalytic**: feedback loops that strengthen successful behavior (e.g., more ants → more pheromone → more ants)

---

### **Swarm as One Mind**

A swarm can behave like a **single intelligent organism**:

- Coordinated hunting, foraging, obstacle avoidance.
- Examples from ants, bees, locusts, birds.

---

### **Stigmergy: Indirect Communication**

**Stigmergy** = communication through the environment.
Instead of direct messaging:

- Individuals leave **markers** (e.g., pheromones).
- Others respond to these, adjusting their own behavior.
- The problem gets solved piece by piece by multiple individuals.

**In ants:**

- Lay pheromone trails.
- Stronger trails attract more ants.
- Evaporation balances the system.
- Leads to **autocatalytic reinforcement** of shorter paths (used in algorithms).

---

### **Example of Ant Behavior**

1. Ants follow a path to food.
2. Obstacle appears → they randomly explore.
3. Some find a shorter route.
4. That route is reinforced by pheromones.
5. Eventually, almost all ants take the shorter route.

This is the **biological basis for Ant Colony Optimization** algorithms.

---

### **Definition of Swarm Intelligence (again)**

> “Any attempt to design algorithms or distributed problem-solving devices inspired by the collective behavior of social insect colonies and other animal societies.”
> – _Bonabeau, Dorigo, Theraulaz (1999)_

---

### **Applications of Swarm Intelligence**

- Brood sorting
- Bird flocking simulations
- Foraging models
- Self-assembling transport systems
- Division of labor
- **Data clustering**
- **Ant Colony Optimization (ACO)**
- **Swarm robotics**
- **Adaptive task allocation**

---
# File: 27.md
---

## **Swarm Intelligence (SI)**

Swarm Intelligence is an approach to problem-solving inspired by the collective behavior of decentralized, self-organized systems—such as bird flocks or ant colonies. Key characteristics:

- **Local search**: Each agent (e.g., ant or particle) searches independently.
- **Global information sharing**: Through simple communication (like pheromone trails or shared best-known positions), agents influence one another.

### Examples:

- **Ant Systems**: Ants lay pheromone trails to share good paths.
- **Particle Swarm Optimization (PSO)**: Agents (particles) share their best-found solutions.

---

## **Origins of Particle Swarm Optimization (PSO)**

### **Inspiration: Flocking Behavior**

- Birds and fish move in coordinated groups through simple local rules.
- **Craig Reynolds’ Boids Model (1986)** simulates flocking using 3 key rules:

  1. **Separation** – Avoid crowding neighbors.
  2. **Alignment** – Match the velocity of neighbors.
  3. **Cohesion** – Move toward the average position of neighbors.

### **Kennedy & Eberhart’s Extension (1995)**

- Introduced a “roost” (goal point) that birds remember.
- Each bird:

  - Tries to return to the closest point it ever was to the roost (personal best).
  - Shares its best with others (global best).

- Insight: Replace the “roost” with a **fitness function** → use this to solve optimization problems.

---

## **What is Particle Swarm Optimization (PSO)?**

PSO is a population-based optimization technique. It tries to find the minimum (or maximum) of a function (e.g., minimize error).

### **Core Idea**:

- Multiple **particles** (candidate solutions) explore the search space.
- Each particle:

  - Has a **position** and a **velocity**.
  - Remembers its **personal best** position.
  - Knows the **global best** position found by any particle.
  - Moves based on a mix of its own best and the global best.

---

## **The Canonical PSO Algorithm**

### **Definitions**:

For each particle $i$:

- $x_i$: current position
- $v_i$: current velocity
- $pb_i$: personal best position
- $gb$: global best position
- $f(x_i)$: fitness value at current position

### **Algorithm Steps**:

1. Initialize particles with random positions and velocities.
2. Repeat until stopping criteria met:

   - For each particle:

     - Evaluate fitness.
     - If current fitness is better than personal best, update $pb_i$.
     - If current fitness is better than global best, update $gb$.

   - Update velocities:

     $$
     v_i^{t+1} = v_i^t + \phi_1 U_1 (pb_i - x_i^t) + \phi_2 U_2 (gb - x_i^t)
     $$

     where:

     - $\phi_1, \phi_2$: constants controlling influence
     - $U_1, U_2$: random numbers between 0 and 1

   - Update positions:

     $$
     x_i^{t+1} = x_i^t + v_i^{t+1}
     $$

---

## **Explanation of Velocity Update**

### Three parts:

- **Momentum**: $v_i^t$ — keeps moving in the same direction.
- **Personal influence**: moves toward personal best.
- **Social influence**: moves toward global best.

Graphically, in 2D, the new direction is a mix of all three.

---

## **PSO in Action – Summary of Steps**

1. Initialize particles (positions and velocities).
2. Evaluate fitness of all particles.
3. Update each particle’s personal best.
4. Update the global best.
5. Compute new velocities based on:

   - Current velocity (momentum)
   - Pull toward personal best
   - Pull toward global best

6. Update positions.
7. Repeat from step 2 until done.

---

## **Extensions of PSO**

### **Population Topology**

- Using only global best can lead to premature convergence.
- **Alternative**: Use **local best** (best in a neighborhood).

  - Slows down convergence.
  - Increases exploration and chance of finding global optimum.

#### Velocity Update (Local best version):

$$
v_i^{t+1} = v_i^t + \phi_1 U_1 (pb_i - x_i^t) + \phi_2 U_2 (lb_i - x_i^t)
$$

### **Neighborhood Structures**:

- **Low connectivity**: slow info sharing → better exploration.
- **High connectivity**: fast info sharing → faster convergence, but risk of local optima.

---

## **Inertia / Momentum Weight**

Adds a weight $w$ to the momentum component:

$$
v_i^{t+1} = w v_i^t + \phi_1 U_1 (pb_i - x_i^t) + \phi_2 U_2 (gb - x_i^t)
$$

- $w$: inertia weight

  - High → more exploration (particles roam more)
  - Low → more exploitation (particles fine-tune)

- Often **decreased over time** to first explore, then exploit.

---

## **Other Variants of PSO**

- Dynamic neighborhood structures
- Enhanced diversity control
- Modified update rules
- PSO adapted to **discrete** problems
- Related to other methods like:

  - **Gradient-based optimization**
  - **Genetic algorithms**
  - **Reinforcement learning**

---

## **Summary**

**Particle Swarm Optimization (PSO)**:

- Inspired by natural flocking behavior.
- Uses a population of particles to explore the search space.
- Each particle adjusts based on its own and neighbors' experiences.
- Extensions include local best, inertia weight, and varied topologies.

---
# File: 28.md
---
## 🐜 Swarm Intelligence

Swarm Intelligence refers to the collective behavior of decentralized, self-organized systems, natural or artificial. It's inspired by the way animals like ants, bees, or birds act together without central control to solve problems like:
  - Finding food
  - Organizing colonies
  - Navigating obstacles

Applications include:
  - Data clustering
  - Adaptive task allocation
  - Network routing
  - Robotics
---

## 🧮 Ant Colony Optimization (ACO)

ACO is inspired by how real ants find the shortest path to food using pheromones. Pheromones are chemicals ants leave on their paths—stronger pheromones attract more ants, reinforcing good paths.

### Key Principles:

- **More pheromone = more ants follow the path**
- **Shorter paths are reinforced more quickly**
- This self-reinforcing loop helps converge to optimal or near-optimal paths

---

## 🔁 From Real Ants to Artificial Ants

### Artificial Ants:

- Move on a graph (like cities connected by roads)
- Choose next step based on:

  - Pheromone levels (τᵢⱼ)
  - Heuristic information (ηᵢⱼ), like 1/distance

- Keep track of:

  - Visited nodes
  - Cost (e.g., total distance)

After reaching the destination:

- They **retrace** their path and **update pheromones**
- **Better paths (shorter/cheaper)** get **more pheromone**

### Probability of Choosing a Path:

$P_{ij}(t) = \frac{[\tau_{ij}(t)]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{k \in J} [\tau_{ik}(t)]^\alpha \cdot [\eta_{ik}]^\beta}$

Where:

- $\tau_{ij}$ = pheromone on edge (i, j)
- $\eta_{ij}$ = heuristic info (e.g., 1/distance)
- $\alpha$, $\beta$ = parameters controlling influence of pheromone vs. heuristic
- J = set of unvisited nodes

---

## 🔄 Pheromone Update Rule

After all ants have completed their tours:

1. **Evaporation**: pheromone fades over time
   $\tau_{ij} \leftarrow (1 - \rho) \cdot \tau_{ij}$

   - $\rho$ = evaporation rate (between 0 and 1)

2. **Deposit**: better paths get more pheromone
   $\Delta \tau_{ij} = \sum_{k=1}^{m} \frac{1}{L_k}$

   - $L_k$ = path length of ant k

---

## 🧪 Example: Travelling Salesman Problem (TSP)

### Problem:

- Visit all cities exactly once, return to start
- Minimize total travel distance

### How ACO solves TSP:

- Ants build tours probabilistically
- Remember visited cities to avoid repeats
- After tour, deposit pheromone based on tour quality (shorter = better)
- Pheromone evaporates over time to avoid premature convergence

### Example with 4 Cities (A, B, C, D):

- Ant randomly starts at a city
- Chooses next city based on pheromones and distances
- Completes tour, updates pheromones on edges used
- All pheromones decay a little
- Process repeats with other ants

---

## 🔧 The Ant System Algorithm (Dorigo et al., 1991)

Loop until stopping condition:

1. Place one ant at each city
2. Each ant builds a tour using probability rule
3. After all tours, update pheromone trails

---

## 🛠 Modified ACO Variants

1. **Elitist Ant System (EAS)**: Best solution gets extra pheromone
2. **Rank-Based (ASrank)**: Top-ranked ants deposit pheromones, based on rank
3. **Max-Min AS (MMAS)**:

   - Only best ants update pheromone
   - Set upper/lower pheromone limits
   - Re-initialize trails if algorithm stagnates

4. **Ant Colony System (ACS)**:

   - Updates pheromones during tour construction (local update)
   - Emphasizes both exploration and exploitation
   - Local and global update rules

---

## 🔀 ACS Details

### State Transition Rule:

If random number $q \leq q_0$: **Exploitation**

- Choose best next city (max pheromone \* heuristic)

Else: **Exploration**

- Choose based on probability distribution:

$P_{ij} = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{k \in J} [\tau_{ik}]^\alpha \cdot [\eta_{ik}]^\beta}$

### Local Update Rule:

While building tour:
$\tau_{ij} \leftarrow (1 - \phi)\tau_{ij} + \phi\tau_0$

- $\phi$: decay rate, $\tau_0$: initial pheromone level

### Global Update Rule:

Only **best-so-far tour** gets updated:
$\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \rho \Delta\tau_{ij}^{bs}$

- $\Delta\tau_{ij}^{bs} = 1/L_{bs}$ (best tour length)

---
# File: 29.md
---

## Swarm Intelligence Concepts

Swarm intelligence is the study of how simple agents (like ants, bees, robots) can work together to solve complex tasks. The key features include:

- **Decentralization**: No single controller.
- **Parallel search**: Many agents search at the same time.
- **Adaptivity**: Agents can adjust to changes.
- **Robustness**: The system still works even if some agents fail.

### Real-world Examples

- Network routing
- Dividing tasks and roles
- Transport and patrolling

---

## Foraging Task

This refers to how a group finds and collects items in an unknown area. It has two main parts:

1. **Path construction/planning**: Figuring out how to reach potential targets.
2. **Path exploitation/repair**: Using good paths repeatedly and fixing them if they stop working.

Different species use different techniques:

- **Ants**:

  - Explore randomly.
  - Leave **pheromones** that evaporate over time.
  - Other ants follow stronger pheromone trails.

- **Bees**:

  - Use **Lévy flights** for exploration.
  - Use **path integration** to track their way home.
  - **Dance** to tell others about food locations.

---

## Bee System (Bee-Inspired Foraging)

### Lévy Flight

- A search pattern that mixes **short local moves** with **occasional long jumps**.
- Helps bees (or robots) explore efficiently.

### Path Integration (PI)

- Bees keep track of **direction and distance** from their nest to food.
- This vector helps them return directly.

### Recruitment (Dancing)

- Bees **dance in the nest** to tell others about food.
- Dance direction = direction to food.
- Dance length = quality of food.
- Other bees use this to decide where to go.
- This is **autocatalytic**: more bees dance for better food, attracting even more.

---

## Swarm-Based Optimization Algorithms

### Particle Swarm Optimization (PSO)

- Inspired by bird flocking.
- Each "particle" adjusts its path using:

  - Its own experience.
  - Its neighbors' best-found position.

**Formula (simplified):**
Each particle updates its velocity and position based on:

- **Current velocity**
- **Distance to its best position**
- **Distance to the global best position**

Variables:

- **v** = velocity of the particle
- **x** = position of the particle
- **p_best** = best position found by the particle
- **g_best** = best position found by any particle

### Ant Colony Optimization (ACO)

- Ants leave **pheromones** on paths they travel.
- More pheromones = more attractive the path.
- Pheromones **evaporate** over time to forget bad paths.
- Good paths are reinforced by repeated use.

---

## Swarm Robotics

This applies swarm ideas to groups of robots. The benefits:

- Work without a central control.
- Communicate only locally.
- Scalable and robust.

### Examples:

- **Firefly Synchronization**:

  - Fireflies sync their flashes by adjusting timing based on neighbors.
  - Robots mimic this for synchronized actions.

- **Bee-Inspired Foraging with Robots**:

  - Robots use:

    - Lévy flights for searching.
    - Path integration to track their movement.
    - Visual markers to identify nest and food.
    - Communication of PI vectors when they meet.

  - Used with **TurtleBots** (robots with limited sensors).

### Other Applications:

- Drone swarms (e.g. bridge inspection).
- Wireless sensor networks.
- Morphogenesis (forming shapes).
- Exploration, patrolling, anomaly detection.

Tool Example:

- **ARGoS**: A swarm simulator for testing swarm robot behaviors.

---

## Summary

Swarm intelligence uses natural strategies (from ants, bees, fireflies) to create smart, flexible, and robust systems. These ideas are used in:

- Optimization algorithms (like PSO and ACO).
- Robotics (swarm robots doing tasks together).
- Real-world problem solving (routing, search, inspection, etc.).

---
