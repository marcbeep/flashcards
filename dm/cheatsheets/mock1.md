---
## 📝 **Mock Exam – Data Mining (Total: 100 Marks)**
---

### **Question 1 \[5 marks]**

**Compute the Jaccard Index** between the following two sentences:

- A: "deep learning is fun and powerful"
- B: "machine learning is powerful and fun"

---

**Solution:**

- A = {deep, learning, is, fun, and, powerful}
- B = {machine, learning, is, powerful, and, fun}

Intersection = {learning, is, powerful, and, fun} → 5 items
Union = {deep, learning, is, fun, and, powerful, machine} → 7 items

Jaccard Index = 5 / 7 = **0.714**

---

### **Question 2 \[4 marks]**

Given vectors:
$X = (3, -2, 1)$, $Y = (1, 4, -1)$
Compute the **dot product** $X^T Y$

---

**Solution:**
$3*1 + (-2)*4 + 1*(-1) = 3 - 8 - 1 = -6$

Answer: **-6**

---

### **Question 3 \[4 marks]**

Find the **Euclidean distance** between:
$X = (-1, 2, 2)$ and $Y = (2, 0, 3)$

---

**Solution:**

$$
\sqrt{(2 + 1)^2 + (0 - 2)^2 + (3 - 2)^2} = \sqrt{9 + 4 + 1} = \sqrt{14} ≈ 3.74
$$

Answer: **3.74**

---

### **Question 4 \[6 marks]**

Given matrix:

$$
A = \begin{pmatrix} 2 & 1 \\ -4 & -2 \end{pmatrix}
$$

Check if **A is invertible**, and if so, compute the **inverse**.

---

**Solution:**

Determinant = $(2)(-2) - (-4)(1) = -4 + 4 = 0$

→ Not invertible.

Answer: **Not invertible**

---

### **Question 5 \[5 marks]**

Let $f(t) = \sin^2(2t)$. Find $f'(t)$

---

**Solution:**

$$
f'(t) = 2\sin(2t)\cdot \cos(2t) \cdot 2 = 4\sin(2t)\cos(2t)
$$

Answer: **4 sin(2t) cos(2t)**

---

### **Question 6 \[4 marks]**

Given $X = (1, -3, 0, 4)$, find $||X||_1$, $||X||_\infty$, and $||X||_0$

---

**Solution:**

- $||X||_1 = 1 + 3 + 0 + 4 = 8$
- $||X||_\infty = 4$
- $||X||_0 = 3$

Answer: **8, 4, 3**

---

### **Question 7 \[5 marks]**

Compute the **gradient** of $f(x, y) = (x^2 + 2y)^3$

---

**Solution:**

Let $f = u^3$, where $u = x^2 + 2y$
Then:

- $\frac{∂f}{∂x} = 3u^2 \cdot 2x = 6x(x^2 + 2y)^2$
- $\frac{∂f}{∂y} = 3u^2 \cdot 2 = 6(x^2 + 2y)^2$

Answer: **Gradient = $(6x(x^2 + 2y)^2, 6(x^2 + 2y)^2)$**

---

### **Question 8 \[6 marks]**

A classifier returns the following confusion matrix:

|        | Actual + | Actual – |
| ------ | -------- | -------- |
| Pred + | 40       | 10       |
| Pred – | 20       | 30       |

Compute:
a) Precision
b) Recall
c) F1-score

---

**Solution:**

- Precision = 40 / (40 + 10) = 0.80
- Recall = 40 / (40 + 20) = 0.666
- F1 = 2(0.8 \* 0.666)/(0.8 + 0.666) ≈ **0.727**

Answer: **Precision: 0.80, Recall: 0.666, F1: 0.727**

---

### **Question 9 \[5 marks]**

You are given 4 points:

- A: (1,1), B: (1,2), C: (5,5), D: (6,5)

Using **single linkage** and **Euclidean distance**, which two points merge first?

---

**Solution:**

Distance(A,B) = 1
Distance(C,D) = 1
All other distances > 1

→ A and B **or** C and D merge first.

Accept either pair.

Answer: **A and B** or **C and D**

---

### **Question 10 \[7 marks]**

Run **1 iteration** of K-Means on:

Points: (1,1), (1,2), (5,5), (6,5)
Initial centroids: μ₁ = (1,1), μ₂ = (6,5)

Assign points to clusters and compute new centroids.

---

**Solution:**

Assignments:

- (1,1) → μ₁
- (1,2) → μ₁
- (5,5) → μ₂
- (6,5) → μ₂

New centroids:

- μ₁ = ((1+1)/2, (1+2)/2) = (1, 1.5)
- μ₂ = ((5+6)/2, (5+5)/2) = (5.5, 5)

Answer: **μ₁ = (1, 1.5), μ₂ = (5.5, 5)**

---

### **Question 11 \[6 marks]**

Run **one iteration** of gradient descent for loss $L = \frac{1}{2}(wx + b - y)^2$

Given: $w = 1, x = 2, b = 0, y = 5, \eta = 0.1$

---

**Solution:**

Pred = 1\*2 + 0 = 2
Error = 2 - 5 = -3

$$
w' = w - \eta \cdot \text{error} \cdot x = 1 - 0.1 \cdot (-3) \cdot 2 = 1 + 0.6 = 1.6
b' = b - \eta \cdot \text{error} = 0 + 0.3 = 0.3
$$

Answer: **w = 1.6, b = 0.3**

---

### **Question 12 \[6 marks]**

In a dataset of 1000 emails, “discount” appears 20 times in spam (100 total spam), and 5 times in ham (900 ham emails).
Using Laplace smoothing, estimate:

$$
P(\text{discount} | \text{spam}), \quad P(\text{discount} | \text{ham})
$$

Vocabulary size = 1000

---

**Solution:**

- Spam: (20 + 1)/(100 + 1000) = 21/1100 ≈ 0.019
- Ham: (5 + 1)/(900 + 1000) = 6/1900 ≈ 0.003

Answer: **0.019, 0.003**

---

### **Question 13 \[5 marks]**

Compute **PageRank** after one iteration with damping $d = 0.85$, equal initial score 1/3 each, and graph:

- A → B
- B → C
- C → A

---

**Solution:**

PageRank of A = (1-d)/3 + d \* PR(C)
\= 0.05 + 0.85 \* (1/3) = 0.05 + 0.283 = **0.333**

Same for B and C due to symmetry.

Answer: **All nodes remain at 0.333**

---

### **Question 14 \[5 marks]**

In a graph with 5 nodes, node D is reachable from 3 nodes with avg. distance 2. Compute **proximity prestige**.

---

**Solution:**

Influence fraction = 3/4 = 0.75
AvDist = 2
PP = 0.75 / 2 = **0.375**

---

### **Question 15 \[5 marks]**

Find eigenvalues of:

$$
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
$$

---

**Solution:**

Det(λI - A) = (λ - 2)^2 - 1 = λ^2 - 4λ + 3
Roots = λ = 1, 3

Answer: **1 and 3**

---

### **Question 16 \[5 marks]**

Run 1 round of **Apriori** with f = 0.5 on:

Transactions:
T1: A, B
T2: A, C
T3: A, B
T4: B, C

Find frequent 1-itemsets and 2-itemsets.

---

**Solution:**

Support:
A = 3/4
B = 3/4
C = 2/4

Pairs:
AB = 2/4
AC = 1/4
BC = 1/4

Frequent 1-itemsets: A, B
Frequent 2-itemsets: AB

---

### **Question 17 \[7 marks]**

Use Naive Bayes to classify test: F1=a, F2=c
Data:

| F1  | F2  | Label |
| --- | --- | ----- |
| a   | c   | +     |
| b   | c   | +     |
| a   | b   | -     |
| b   | b   | -     |

Vocabulary size = 2 per feature

---

**Solution:**

Class +:

- P(+) = 2/4
- P(F1=a|+) = 1+1 / (2+2) = 2/4 = 0.5
- P(F2=c|+) = 2+1 / (2+2) = 3/4

Class –:

- P(–) = 2/4
- P(F1=a|–) = 1+1 / (2+2) = 2/4 = 0.5
- P(F2=c|–) = 0+1 / (2+2) = 1/4

Compute proportional scores:

- +: 0.5 × 0.5 × 0.75 = 0.1875
- –: 0.5 × 0.5 × 0.25 = 0.0625
  → Predict **+**

---
