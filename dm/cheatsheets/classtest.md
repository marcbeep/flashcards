---

### 1. Question:

What is the Jaccard index for:

- Sentence A: “I love natural language processing”
- Sentence B: “I enjoy language processing and learning”?

**Solution:**

- A = {I, love, natural, language, processing}
- B = {I, enjoy, language, processing, and, learning}
- Intersection = 3, Union = 8
- Jaccard Index = 3 / 8 = **0.375**

---

### 2. Question:

Given vectors X = (1, 2, −1.5) and Y = (3, 2, −1), find Z = X + Y.

**Solution:**
Z = \[1 + 3, 2 + 2, −1.5 + (−1)] = **\[4, 4, −2.5]**

---

### 3. Question:

Find the dot product of X = (−8, 9, 8) and Y = (9, 10, 3).

**Solution:**
XᵀY = (−8×9) + (9×10) + (8×3) = −72 + 90 + 24 = **42**

---

### 4. Question:

Use KNN (K=4) with Manhattan distance to label (1,3). Training set:

| Object | x1  | x2  | Class |
| ------ | --- | --- | ----- |
| 1      | -10 | -7  | 1     |
| 2      | -4  | -12 | 2     |
| 3      | 5   | 6   | 3     |
| 4      | 7   | 8   | 1     |
| 5      | 8   | 9   | 2     |

**Solution:**
Distances = \[21, 20, 7, 11, 13]; nearest 4 = \[3, 4, 5, 2]; Labels = \[3, 1, 2, 2]; Majority = **2**

---

### 5. Question:

Differentiate f(t) = cos²(7t) with respect to t.

**Solution:**
f'(t) = 2cos(7t)(−sin(7t))×7 = **−14cos(7t)sin(7t)**

---

### 6. Question:

Gradient descent: w₁ = 0, learning rate = 0.01, gradient = −4.2. What is w₁'?

**Solution:**
w₁' = 0 − 0.01×(−4.2) = **0.042**

---

### 7. Question:

Compute Euclidean distance between X = (−8, 9, 8) and Y = (9, 10, 3).

**Solution:**
Distance = √\[289 + 1 + 25] = √315 = **17.75**

---

### 8. Question:

Loss: L = ½(wx + b − y)². What is the weight update rule for w in gradient descent?

**Solution:**
w' = w − η(wx + b − y)x

---

### 9. Question:

Find ||X||₀ and ||X||∞ for X = (1, −2, −3).

**Solution:**
||X||₀ = 3 (non-zero entries), ||X||∞ = 3 (max abs value)

---

### 10. Question:

Find ||X||₁ and ||X||₂ for X = (1, 2, 3).

**Solution:**
||X||₁ = 6; ||X||₂ = √14 = **3.74**

---

### 11. Question:

Hamming distance between X = (1, 1, 0, 1) and Y = (0, 0, 0, 1)?

**Solution:**
Different bits at positions 1 and 2 → Hamming Distance = **2**

---

### 12. Question:

Which figure is overfitted: Left or Right?

**Solution:**
**Right Figure (Figure 2)** is overfitted.

---

### 13. Question:

What kind of algorithm is KNN?

**Solution:**
KNN is a **supervised** algorithm (uses labeled data).

---

### 14. Question:

Given vectors X (6-dim), Y (6-dim), and Z = XᵀYᵀZ, what’s the dimension of Z'?

**Solution:**
YᵀZ = 1×1, so XᵀYᵀZ = 1×6

---

### 15. Question:

Compute Manhattan distance between X = (−8, 9, 8) and Y = (9, 10, 3).

**Solution:**
|−8−9| + |9−10| + |8−3| = 17 + 1 + 5 = **23**

---

### 16. Question:

Best evaluation metric for YouTube recommendation?

**Solution:**
**Precision** (high precision means relevant content is shown)

---

### 17. Question:

Given X and Y (6-dim), and Z = XYᵀ, find Z₂₁ and Z₂₂.

**Solution:**
Z₂₁ = 9×9 = 81; Z₂₂ = 9×10 = **90**

---

### 18. Question:

In breathalyzer tests, what are true positives?

**Solution:**
True positive = test says over limit AND actually over limit.

---

### 19. Question:

Confusion matrix (predicted vs actual), compute **precision**.

|       | A (+) | B (−) |
| ----- | ----- | ----- |
| A (+) | 30    | 7     |
| B (−) | 15    | 10    |

**Solution:**
Precision = 30 / (30 + 7) = **0.810**

---

### 20. Question:

Same confusion matrix, compute **recall**.

**Solution:**
Recall = 30 / (30 + 15) = **0.666**

---
