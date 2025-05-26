# Data Mining Formulas

## 1. Foundations & Core Concepts

### Linear Algebra

- **Vector Operations**:
  $$
  \text{Dot Product: } X \cdot Y = \sum x_i y_i
  $$
  $$
  \text{Vector Length: } \|X\| = \sqrt{\sum x_i^2}
  $$
  $$
  \text{Matrix Trace: } tr(A) = \sum_{i} a_{ii}
  $$

### Feature Normalization

- **[0,1]-Scaling (Min-Max)**:

  $$
  \hat{x} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}
  $$

- **Gaussian Normalization (Z-score)**:
  $$
  \hat{x} = \frac{x - \mu}{\sigma}
  $$

### Distance Metrics

- **Euclidean (L2)**:

  $$
  d(X,Y) = \sqrt{\sum(x_i - y_i)^2}
  $$

  where:

  - X, Y = Input vectors
  - x_i, y_i = i-th components of vectors X and Y

- **Manhattan (L1)**:

  $$
  d(X,Y) = \sum|x_i - y_i|
  $$

  where:

  - X, Y = Input vectors
  - x_i, y_i = i-th components of vectors X and Y

- **Cosine Similarity**:
  $$
  \cos(X,Y) = \frac{X \cdot Y}{\|X\| \|Y\|}
  $$
  where:
  - X·Y = Dot product of vectors X and Y
  - ||X|| = Length (magnitude) of vector X

## 2. Machine Learning Fundamentals

### Classification Metrics

- **Accuracy**:

  $$
  \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
  $$

  where:

  - TP = True Positives
  - TN = True Negatives
  - FP = False Positives
  - FN = False Negatives

- **Precision**:

  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$

- **Recall**:

  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$

- **F-score**:
  $$
  \text{F-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

### Perceptron & Logistic Regression

- **Activation Score**:

  $$
  a = W^T X + b
  $$

  where:

  - W = Weight vector
  - X = Input feature vector
  - b = Bias term

- **Perceptron Prediction**:

  $$
  y = \text{sign}(a)
  $$

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

  where:

  - X = Data point
  - Y_j = Centroid of cluster j

- **Centroid Update**:
  $$
  Y_j = \frac{1}{|C_j|} \sum_{X \in C_j} X
  $$

### k-Medians

- **Objective Function** (L1 Distance):

  $$
  \sum\sum \|X - Y_j\|_1
  $$

- **Median Update**:
  $$
  Y_j^{(i)} = \text{median}(X_1^{(i)}, X_2^{(i)}, ..., X_s^{(i)})
  $$

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

- **Confidence of Rule X → Y**:
  $$
  \text{conf}(X \rightarrow Y) = \frac{\text{sup}(X \cup Y)}{\text{sup}(X)}
  $$

## 4. Probabilistic Methods

### Probability Basics

- **Joint Probability**:

  $$
  P(A, B) = P(A|B)P(B)
  $$

- **Conditional Probability**:
  $$
  P(A|B) = \frac{P(A, B)}{P(B)}
  $$

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

- **Overlap Coefficient**:
  $$
  \text{Overlap}(A,B) = \frac{|A \cap B|}{\min(|A|, |B|)}
  $$

## 5. Graph Mining & Networks

### Centrality Measures

- **Degree Centrality**:

  $$
  C_D(i) = \frac{\text{deg}(i)}{n - 1}
  $$

- **Closeness Centrality**:

  $$
  C_C(i) = \frac{1}{\text{AvDist}(i)} = \frac{n - 1}{\sum \text{dist}(i,j)}
  $$

- **Betweenness Centrality**:
  $$
  C_B(i) = \sum \frac{q_{jk}(i)}{q_{jk}} / \frac{n(n-1)}{2}
  $$

### PageRank

- **Basic PageRank**:

  $$
  P(a) = \sum_{(x,a) \in E} \frac{P(x)}{O_x}
  $$

- **Matrix Form**:

  $$
  P = A^T P
  $$

- **Power Iteration**:
  $$
  P^{(i)} = A P^{(i-1)}
  $$
