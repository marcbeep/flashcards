# Data Mining Formulas

## 1. Foundations & Core Concepts

### Linear Algebra

- **Vector Operations**:
  $$
  \text{Dot Product: } X \cdot Y = \sum x_i y_i
  $$
  Example: For X = [1, 2, 3] and Y = [4, 5, 6], X·Y = 1(4) + 2(5) + 3(6) = 32
  $$
  \text{Vector Length: } \|X\| = \sqrt{\sum x_i^2}
  $$
  Example: For X = [3, 4], ||X|| = √(3² + 4²) = √25 = 5
  $$
  \text{Matrix Trace: } tr(A) = \sum_{i} a_{ii}
  $$
  Example: For A = [[1, 2], [3, 4]], tr(A) = 1 + 4 = 5

### Feature Normalization

- **[0,1]-Scaling (Min-Max)**:

  $$
  \hat{x} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}
  $$

  Example: For x = [1, 2, 3, 4], min = 1, max = 4:
  x̂ = [0, 0.33, 0.67, 1]

- **Gaussian Normalization (Z-score)**:
  $$
  \hat{x} = \frac{x - \mu}{\sigma}
  $$
  Example: For x = [2, 4, 4, 4, 6], μ = 4, σ = 1.41:
  x̂ = [-1.41, 0, 0, 0, 1.41]

### Distance Metrics

- **Euclidean (L2)**:

  $$
  d(X,Y) = \sqrt{\sum(x_i - y_i)^2}
  $$

  Example: For X = [0, 0] and Y = [3, 4]:
  d = √(3² + 4²) = √25 = 5

- **Manhattan (L1)**:

  $$
  d(X,Y) = \sum|x_i - y_i|
  $$

  Example: For X = [1, 1] and Y = [4, 5]:
  d = |4-1| + |5-1| = 3 + 4 = 7

- **Cosine Similarity**:
  $$
  \cos(X,Y) = \frac{X \cdot Y}{\|X\| \|Y\|}
  $$
  Example: For X = [1, 1] and Y = [1, 0]:
  cos = (1·1 + 1·0) / (√2 · 1) = 0.707

## 2. Machine Learning Fundamentals

### Classification Metrics

- **Accuracy**:

  $$
  \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
  $$

  Example: For TP=45, TN=50, FP=5, FN=10:
  Accuracy = (45+50)/(45+50+5+10) = 0.864

- **Precision**:

  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$

  Example: For TP=45, FP=5:
  Precision = 45/(45+5) = 0.9

- **Recall**:

  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$

  Example: For TP=45, FN=10:
  Recall = 45/(45+10) = 0.818

- **F-score**:
  $$
  \text{F-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$
  Example: For Precision=0.9, Recall=0.818:
  F-score = 2(0.9)(0.818)/(0.9+0.818) = 0.857

### Perceptron & Logistic Regression

- **Activation Score**:

  $$
  a = W^T X + b
  $$

  Example: For W=[0.5, -0.2], X=[2, 1], b=0.1:
  a = 0.5(2) + (-0.2)(1) + 0.1 = 0.9

- **Perceptron Prediction**:

  $$
  y = \text{sign}(a)
  $$

  Example: For a=0.9:
  y = sign(0.9) = +1

- **Perceptron Update Rule** (on misclassification):

  $$
  W = W + yX
  $$

  $$
  b = b + y
  $$

- **Sigmoid Function**:

  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$

  Example: For x=0:
  σ(0) = 1/(1 + e⁰) = 0.5

- **Probability Prediction**:

  $$
  P(y = +1 | X) = \sigma(W^T X + b)
  $$

  $$
  P(y = -1 | X) = \sigma(-(W^T X + b))
  $$

- **Likelihood Function**:

  $$
  \mathcal{L}(b, W) = \prod \sigma(y_i(b + W^T X_i))
  $$

- **Gradient Descent Update**:
  $$
  X_{i+1} = X_i - \gamma_i \cdot \nabla f(X_i)
  $$

## 3. Clustering & Pattern Mining

### k-Means

- **Objective Function** (Within-Cluster Sum of Squares):

  $$
  \text{WCSS} = \sum\sum \|X - Y_j\|^2
  $$

  Example: For cluster points [1,1], [2,2] with centroid [1.5,1.5]:
  WCSS = 0.5² + 0.5² + 0.5² + 0.5² = 1

### k-Medians

- **Objective Function** (L1 Distance):

  $$
  \sum\sum \|X - Y_j\|_1
  $$

  Example: For points [1,1], [2,2] with median [1.5,1.5]:
  Sum = |1-1.5| + |1-1.5| + |2-1.5| + |2-1.5| = 2

### Group-Average Linkage

- **Correlation**:
  $$
  \text{Corr}(\Theta) = \frac{1}{2} \cdot \frac{1}{|\Theta|(|\Theta|-1)} \sum_{(u,v)\in\Theta} \text{sim}(u,v)
  $$

### Association Rule Mining

- **Support of Itemset I**:

  $$
  \text{sup}(I) = \frac{|\{T \in D : I \subseteq T\}|}{|D|}
  $$

  Example: If item "bread" appears in 3 out of 10 transactions:
  sup(bread) = 3/10 = 0.3

- **Confidence of Rule X → Y**:
  $$
  \text{conf}(X \rightarrow Y) = \frac{\text{sup}(X \cup Y)}{\text{sup}(X)}
  $$
  Example: If {bread,butter} appears in 2 transactions and bread in 3:
  conf(bread→butter) = 2/3 = 0.67

## 4. Probabilistic Methods

### Probability Basics

- **Joint Probability**:

  $$
  P(A, B) = P(A|B)P(B)
  $$

  Example: P(rain,cloudy) = P(rain|cloudy)(0.5) = 0.8(0.5) = 0.4

- **Conditional Probability**:
  $$
  P(A|B) = \frac{P(A, B)}{P(B)}
  $$
  Example: If P(rain,cloudy)=0.4 and P(cloudy)=0.5:
  P(rain|cloudy) = 0.4/0.5 = 0.8

### Naive Bayes

- **Conditional Independence**:
  $$
  P(x_1, x_2, ..., x_d | C) = \prod_{i=1}^d P(x_i | C)
  $$

### Set Distances

- **Jaccard Similarity**:

  $$
  J(A,B) = \frac{|A \cap B|}{|A \cup B|}
  $$

  Example: For A={1,2,3}, B={2,3,4}:
  J(A,B) = |{2,3}|/|{1,2,3,4}| = 2/4 = 0.5

- **Overlap Coefficient**:
  $$
  \text{Overlap}(A,B) = \frac{|A \cap B|}{\min(|A|, |B|)}
  $$
  Example: For A={1,2,3}, B={2,3,4}:
  Overlap = |{2,3}|/min(3,3) = 2/3 = 0.67

## 5. Graph Mining & Networks

### Centrality Measures

- **Degree Centrality**:

  $$
  C_D(i) = \frac{\text{deg}(i)}{n - 1}
  $$

  Example: For a node with 3 connections in a 5-node network:
  C_D = 3/(5-1) = 0.75

- **Closeness Centrality**:

  $$
  C_C(i) = \frac{1}{\text{AvDist}(i)} = \frac{n - 1}{\sum \text{dist}(i,j)}
  $$

  Example: For node distances [1,1,2,2]:
  C_C = 4/(1+1+2+2) = 0.67

- **Betweenness Centrality**:
  $$
  C_B(i) = \sum \frac{q_{jk}(i)}{q_{jk}} / \frac{n(n-1)}{2}
  $$

### PageRank

- **Basic PageRank**:

  $$
  P(a) = \sum_{(x,a) \in E} \frac{P(x)}{O_x}
  $$

  Example: For a node with two incoming edges from pages with PR=0.5 each and out-degrees 2 and 3:
  P(a) = 0.5/2 + 0.5/3 = 0.417

- **Matrix Form**:

  $$
  P = A^T P
  $$

- **Power Iteration**:
  $$
  P^{(i)} = A P^{(i-1)}
  $$
