# Data Mining Formulas

## 1. Classification Metrics

### Binary Classification

- **Accuracy**:
  \[
  \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
  \]
  where:

  - TP = True Positives (correctly predicted positive cases)
  - TN = True Negatives (correctly predicted negative cases)
  - FP = False Positives (incorrectly predicted positive cases)
  - FN = False Negatives (incorrectly predicted negative cases)

  In plain English: The proportion of all predictions (both positive and negative) that were correct.

- **Precision**:
  \[
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  \]
  where:

  - TP = True Positives
  - FP = False Positives

  In plain English: Out of all cases we predicted as positive, what proportion were actually positive.

- **Recall**:
  \[
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \]
  where:

  - TP = True Positives
  - FN = False Negatives

  In plain English: Out of all actual positive cases, what proportion did we correctly identify.

- **F-score**:
  \[
  \text{F-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]
  where:

  - Precision = Precision score as defined above
  - Recall = Recall score as defined above

  In plain English: A balanced measure that combines precision and recall, giving equal weight to both.

## 2. Perceptron & Logistic Regression

### Perceptron

- **Activation Score**:
  \[
  a = W^T X + b
  \]
  where:

  - W = Weight vector
  - X = Input feature vector
  - b = Bias term
  - a = Activation score

  In plain English: Computes a weighted sum of inputs plus a bias term to determine the neuron's activation.

- **Prediction**:
  \[
  y = \text{sign}(a)
  \]
  where:

  - a = Activation score
  - y = Predicted class (+1 or -1)

  In plain English: Converts the activation score into a binary prediction based on its sign.

- **Update Rule** (on misclassification):
  \[
  W = W + yX
  \]
  \[
  b = b + y
  \]
  where:

  - W = Weight vector
  - X = Input feature vector
  - y = True class label
  - b = Bias term

  In plain English: When a mistake is made, adjust weights and bias in the direction of the correct class.

### Logistic Regression

- **Sigmoid Function**:
  \[
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \]
  where:

  - x = Input value
  - e = Euler's number (≈ 2.71828)

  In plain English: Transforms any real number into a value between 0 and 1, useful for probability predictions.

- **Probability Prediction**:
  \[
  P(y = +1 | X) = \sigma(W^T X + b)
  \]
  \[
  P(y = -1 | X) = \sigma(-(W^T X + b))
  \]
  where:

  - W = Weight vector
  - X = Input feature vector
  - b = Bias term
  - σ = Sigmoid function

  In plain English: Predicts the probability of each class by applying the sigmoid function to the activation score.

- **Loss Function** (Negative Log-Likelihood):
  \[
  L = -\sum \log(\sigma(y_i \cdot (W^T X_i + b)))
  \]
  where:

  - y_i = True class label for instance i
  - X_i = Feature vector for instance i
  - W = Weight vector
  - b = Bias term
  - σ = Sigmoid function

  In plain English: Measures how well our predictions match the true labels, with better predictions giving lower loss.

## 3. Clustering

### k-Means

- **Objective Function** (Within-Cluster Sum of Squares):
  \[
  \text{WCSS} = \sum\sum \|X - Y_j\|^2
  \]
  where:

  - X = Data point
  - Y_j = Centroid of cluster j
  - \|\|.\|\|² = Squared Euclidean distance

  In plain English: Measures the total squared distance between each point and its assigned cluster center.

- **Centroid Update**:
  \[
  Y*j = \frac{1}{|C_j|} \sum*{X \in C_j} X
  \]
  where:

  - Y_j = New centroid position for cluster j
  - C_j = Set of points in cluster j
  - |C_j| = Number of points in cluster j

  In plain English: Updates each cluster center to be the mean of all points assigned to that cluster.

### k-Medians

- **Objective Function** (L1 Distance):
  \[
  \sum\sum \|X - Y_j\|\_1
  \]
  where:

  - X = Data point
  - Y_j = Median of cluster j
  - \|\|.\|\|₁ = Manhattan (L1) distance

  In plain English: Measures the total absolute distance between each point and its assigned cluster median.

- **Median Update**:
  \[
  Y_j^{(i)} = \text{median}(X_1^{(i)}, X_2^{(i)}, ..., X_s^{(i)})
  \]
  where:

  - Y_j^(i) = i-th coordinate of cluster j's median
  - X_k^(i) = i-th coordinate of k-th point in cluster j

  In plain English: Updates each cluster representative to be the coordinate-wise median of all points in that cluster.

## 4. Association Rule Mining

### Support and Confidence

- **Support of Itemset I**:
  \[
  \text{sup}(I) = \frac{|\{T \in D : I \subseteq T\}|}{|D|}
  \]
  where:

  - I = Itemset
  - D = Dataset (set of transactions)
  - T = Transaction
  - |D| = Total number of transactions

  In plain English: The proportion of transactions that contain all items in set I.

- **Confidence of Rule X → Y**:
  \[
  \text{conf}(X \rightarrow Y) = \frac{\text{sup}(X \cup Y)}{\text{sup}(X)}
  \]
  where:

  - X = Antecedent itemset
  - Y = Consequent itemset
  - sup() = Support function as defined above

  In plain English: Given that X appears in a transaction, the probability that Y also appears.

## 5. Graph Analysis

### Centrality Measures

- **Degree Centrality**:
  \[
  C_D(i) = \frac{\text{deg}(i)}{n - 1}
  \]
  where:

  - deg(i) = Number of edges connected to node i
  - n = Total number of nodes in the graph

  In plain English: Measures how connected a node is relative to the maximum possible connections.

- **Closeness Centrality**:
  \[
  C_C(i) = \frac{1}{\text{AvDist}(i)} = \frac{n - 1}{\sum \text{dist}(i,j)}
  \]
  where:

  - AvDist(i) = Average distance from node i to all other nodes
  - dist(i,j) = Shortest path distance between nodes i and j
  - n = Total number of nodes

  In plain English: Measures how close a node is to all other nodes in the network.

- **Betweenness Centrality**:
  \[
  C*B(i) = \sum \frac{q*{jk}(i)}{q\_{jk}} / \frac{n(n-1)}{2}
  \]
  where:

  - q_jk = Total number of shortest paths between j and k
  - q_jk(i) = Number of those paths passing through i
  - n = Total number of nodes

  In plain English: Measures how often a node lies on shortest paths between other nodes.

### PageRank

- **Basic PageRank**:
  \[
  P(a) = \sum\_{(x,a) \in E} \frac{P(x)}{O_x}
  \]
  where:

  - P(a) = PageRank of page a
  - O_x = Number of outgoing links from page x
  - E = Set of edges (links) in the graph

  In plain English: A page's importance is the sum of the importance of pages linking to it, weighted by how many links those pages have.

- **Matrix Form**:
  \[
  P = A^T P
  \]
  where:

  - P = PageRank vector
  - A = Adjacency matrix (normalized by outgoing links)

  In plain English: The PageRank vector is an eigenvector of the transposed adjacency matrix.

- **Power Iteration**:
  \[
  P^{(i)} = A P^{(i-1)}
  \]
  where:

  - P^(i) = PageRank vector at iteration i
  - A = Adjacency matrix

  In plain English: Iteratively compute PageRank by repeatedly multiplying by the adjacency matrix until convergence.

## 6. Distance Metrics

### Vector Distances

- **Euclidean (L2)**:
  \[
  d(X,Y) = \sqrt{\sum(x_i - y_i)^2}
  \]
  where:

  - X, Y = Input vectors
  - x_i, y_i = i-th components of vectors X and Y

  In plain English: The straight-line distance between two points in space.

- **Manhattan (L1)**:
  \[
  d(X,Y) = \sum|x_i - y_i|
  \]
  where:

  - X, Y = Input vectors
  - x_i, y_i = i-th components of vectors X and Y

  In plain English: The sum of absolute differences between corresponding components.

- **Cosine Similarity**:
  \[
  \cos(X,Y) = \frac{X \cdot Y}{\|X\| \|Y\|}
  \]
  where:

  - X·Y = Dot product of vectors X and Y
  - \|\|X\|\| = Length (magnitude) of vector X

  In plain English: Measures the cosine of the angle between two vectors, indicating their directional similarity.

### Set Distances

- **Jaccard Similarity**:
  \[
  J(A,B) = \frac{|A \cap B|}{|A \cup B|}
  \]
  where:

  - A, B = Sets
  - |A ∩ B| = Size of intersection
  - |A ∪ B| = Size of union

  In plain English: The size of the intersection divided by the size of the union of two sets.

- **Overlap Coefficient**:
  \[
  \text{Overlap}(A,B) = \frac{|A \cap B|}{\min(|A|, |B|)}
  \]
  where:

  - A, B = Sets
  - |A ∩ B| = Size of intersection
  - min(|A|, |B|) = Size of smaller set

  In plain English: The size of the intersection divided by the size of the smaller set.
