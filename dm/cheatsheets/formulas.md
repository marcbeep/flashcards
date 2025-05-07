# Data Mining Formulas

## 1. Classification Metrics

### Binary Classification

- **Accuracy**:
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```
- **Precision**:
  ```
  Precision = TP / (TP + FP)
  ```
- **Recall**:
  ```
  Recall = TP / (TP + FN)
  ```
- **F-score**:
  ```
  F-score = (2 * Precision * Recall) / (Precision + Recall)
  ```

## 2. Perceptron & Logistic Regression

### Perceptron

- **Activation Score**:
  ```
  a = W^T X + b
  ```
- **Prediction**:
  ```
  y = sign(a)
  ```
- **Update Rule** (on misclassification):
  ```
  W = W + yX
  b = b + y
  ```

### Logistic Regression

- **Sigmoid Function**:
  ```
  σ(x) = 1 / (1 + e^(-x))
  ```
- **Probability Prediction**:
  ```
  P(y = +1 | X) = σ(W^T X + b)
  P(y = -1 | X) = σ(-(W^T X + b))
  ```
- **Loss Function** (Negative Log-Likelihood):
  ```
  L = -∑ log(σ(y_i * (W^T X_i + b)))
  ```

## 3. Clustering

### k-Means

- **Objective Function** (Within-Cluster Sum of Squares):
  ```
  WCSS = ∑∑ ||X - Y_j||²
  ```
- **Centroid Update**:
  ```
  Y_j = (1/|C_j|) * ∑X for X in C_j
  ```

### k-Medians

- **Objective Function** (L1 Distance):
  ```
  ∑∑ ||X - Y_j||₁
  ```
- **Median Update**:
  ```
  Y_j^(i) = median(X_1^(i), X_2^(i), ..., X_s^(i))
  ```

## 4. Association Rule Mining

### Support and Confidence

- **Support of Itemset I**:
  ```
  sup(I) = |{T ∈ D : I ⊆ T}| / |D|
  ```
- **Confidence of Rule X → Y**:
  ```
  conf(X → Y) = sup(X ∪ Y) / sup(X)
  ```

## 5. Graph Analysis

### Centrality Measures

- **Degree Centrality**:
  ```
  CD(i) = deg(i) / (n - 1)
  ```
- **Closeness Centrality**:
  ```
  CC(i) = 1 / AvDist(i) = (n - 1) / ∑ dist(i,j)
  ```
- **Betweenness Centrality**:
  ```
  CB(i) = ∑(f_jk(i)) / (n(n-1)/2)
  where f_jk(i) = q_jk(i)/q_jk
  ```

### PageRank

- **Basic PageRank**:
  ```
  P(a) = ∑(P(x)/O_x) for all (x,a) ∈ E
  ```
- **Matrix Form**:
  ```
  P = A^T P
  ```
- **Power Iteration**:
  ```
  P^(i) = A P^(i-1)
  ```

## 6. Distance Metrics

### Vector Distances

- **Euclidean (L2)**:
  ```
  d(X,Y) = √(∑(x_i - y_i)²)
  ```
- **Manhattan (L1)**:
  ```
  d(X,Y) = ∑|x_i - y_i|
  ```
- **Cosine Similarity**:
  ```
  cos(X,Y) = (X·Y) / (||X|| ||Y||)
  ```

### Set Distances

- **Jaccard Similarity**:
  ```
  J(A,B) = |A ∩ B| / |A ∪ B|
  ```
- **Overlap Coefficient**:
  ```
  Overlap(A,B) = |A ∩ B| / min(|A|, |B|)
  ```
