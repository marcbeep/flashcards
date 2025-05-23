---

### **Problem Set 1: Mathematical Preliminaries**

**Exercise 1**
Given $X = (1, 2, 3)^T$ and $Y = (3, 2, 1)^T$, find:

1. $X + Y$
2. $X^T Y$
3. $Y X^T$

**Exercise 2**
Given two matrices
$A = \begin{pmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{pmatrix}$,
$B = \begin{pmatrix}0 & 1 & 0\\ 1 & 2 & 3\\ -1 & 0 & 1\end{pmatrix}$:

1. Compute $A + B$
2. Compute $B + A$. Is it equal to $A + B$? Is it always the case?
3. Compute $A \cdot B$
4. Compute $B \cdot A$. Is it equal to $A \cdot B$?

**Exercise 3**
Compute the inverse of the matrix $A = \begin{pmatrix}1 & 2\\ -2 & 1\end{pmatrix}$, if it exists. Verify that the matrix product of $A$ and its inverse is the 2×2 identity matrix.

**Exercise 4**
Show that the vectors
$A = (1, 2, -3, 4)^T$,
$B = (1, 1, 0, 2)^T$, and
$C = (-1, -2, 1, 1)^T$
are linearly independent.

**Exercise 5**
Find the ranks of the matrices:
$A = \begin{pmatrix}1 & 0 & 1\\ 0 & 1 & 1\\ 0 & 0 & 0\end{pmatrix}$,
$B = \begin{pmatrix}1 & 2 & 1\\ -2 & -3 & 1\\ 3 & 5 & 0\end{pmatrix}$

**Exercise 6**
Find the eigenvalues and eigenvectors of the matrix
$A = \begin{pmatrix}4 & 2\\ 1 & 3\end{pmatrix}$

**Exercise 7**
Given $f(x) = \log(x)$ and $g(x) = 2x + 1$, compute:

1. $f'(x)$
2. $g'(x)$
3. $(f(x) + g(x))'$
4. $(f(x)g(x))'$
5. $\left(\frac{f(x)}{g(x)}\right)'$
6. $(g(f(x)))'$

**Exercise 8**
Given $f(x, y) = (x + 2y^3)^2$, compute:

1. $\partial f / \partial x$
2. $\partial f / \partial y$
3. $\nabla_{(x, y)} f$

---

### **Problem Set 2: Normalisation & Classifier Evaluation**

**Exercise 1**
Given heights: \[170, 160, 155, 165]

1. Use \[0, 1]-scaling to normalize them
2. Use Gaussian normalization to transform them

**Exercise 2**
A binary classifier on 1000 examples (50% negative) has 0.6 recall and 0.7 accuracy. Write the confusion matrix.

**Exercise 3**
Given the confusion matrix:

|       | Car | Train | Cycle |
| ----- | --- | ----- | ----- |
| Car   | 8   | 3     | 6     |
| Train | 2   | 4     | 2     |
| Cycle | 2   | 4     | 12    |

1. Calculate Precision, Recall, and F-score for each class
2. Calculate Macro F-score

---

### **Problem Set 3: Classification Algorithms**

**Exercise 1**
Analyze Perceptron update when it misclassifies a negative instance (i.e., $y = -1$).

**Exercise 2**
Provide a geometric interpretation of the Perceptron update when a negative instance is mistaken for a positive one.

**Exercise 3**
Given loss $(y - W^T X)^2$, for training data $\{(X_i, y_i)\}_{i=1}^N$:

1. Compute the derivative w\.r.t. $W$
2. Write the gradient descent rule with step-size $\gamma$

---

### **Problem Set 4: Probabilistic Classifiers**

**Exercise 1**

1. For $h$ heads and $t$ tails, find the MLE for $\beta$ by maximizing the log-likelihood $h \log \beta + t \log(1 - \beta)$
2. Interpret the estimate

**Exercise 2**
Given:
$n(\text{cat}, c) = 9$,
$n(\text{dog}, c) = 0$,
$n(\text{rabbit}, c) = 1$

1. Estimate $P(x=\text{cat}|c), P(x=\text{dog}|c), P(x=\text{rabbit}|c)$
2. Apply Laplace smoothing

**Exercise 3**
Using Naive Bayes, classify the test example F1=a, F2=c, F3=b using this table:

| F1  | F2  | F3  | Label |
| --- | --- | --- | ----- |
| a   | c   | b   | +     |
| c   | a   | c   | +     |
| a   | a   | c   | -     |
| b   | c   | a   | -     |
| c   | c   | b   | -     |

---

### **Problem Set 5: Probabilistic Classifiers 2**

**Exercise 1**
Let $W^T = (w_1, ..., w_d)$, $P^T = (p_1, ..., p_d)$, $b \in \mathbb{R}$

1. Compute distance from point $P$ to the hyperplane $W^T X + b = 0$
2. Show it's proportional to $|W^T P + b|$

**Exercise 2**
Show for $\sigma(x) = \frac{1}{1 + e^{-x}}$:

1. $\sigma(x) = 1 - \sigma(-x)$
2. $\frac{d\sigma}{dx} = e^{-x} \sigma^2(x)$

---

### **Problem Set 6: Clustering**

**Exercise 1**
Given clustering output, compute the confusion matrix, macro Precision, Recall, and F-score.

**Exercise 2**
Using same clusters, compute B-CUBED Precision, Recall, and F-score.

---

### **Problem Set 6: Hierarchical Clustering**

**Exercise 1**
Apply single-linkage agglomerative clustering on:
$X_1 = (2,10), X_2 = (2,5), X_3 = (8,4), X_4 = (5,8), X_5 = (7,5), X_6 = (6,4), X_7 = (1,2), X_8 = (4,9)$
Draw the dendrogram.

**Exercise 2**
Given distance matrix, cluster the dataset using:

1. Single-linkage
2. Complete-linkage
   Draw corresponding dendrograms.

---

### **Problem Set 7: Association Pattern Mining**

**Exercise 1**
Given transactions, determine support of itemsets {A, E, F} and {D, F}

**Exercise 2**

1. Find all frequent itemsets at thresholds 0.4, 0.6, 0.8
2. Find all maximal frequent itemsets for same thresholds

**Exercise 3**
Compute confidence and support for rules:

1. $A \Rightarrow F$
2. $A,E \Rightarrow F$

**Exercise 4**
Run Apriori algorithm with threshold 0.4 on given database. Show candidate and frequent itemsets at each level.

---

### **Problem Set 8: Social Network Analysis**

**Exercise 1**
For the given graph, compute highest:

- Degree centrality
- Closeness centrality
- Betweenness centrality

**Exercise 2**
For every vertex in the graph, compute degree prestige.

**Exercise 3**
Compute proximity prestige for each vertex.

---

### **Problem Set 9: PageRank Algorithm**

**Exercise 1**
Using damping factor 0.7 and initial PageRank = 0.2, compute PageRank for the graph shown.

---
