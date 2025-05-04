# File: 0.md
# 🧠 Data Mining Mind Map

## 📚 Foundations & Core Concepts (01-03)

### 1. Introduction to Data Mining

- Data Explosion & Motivation
- Data Mining Pipeline
  - Data Collection
  - Preprocessing
  - Analytical Processing
  - Output & Feedback
- Feature Engineering & Types

### 2. Mathematical Foundations

- Linear Algebra
  - Vectors & Matrices
  - Operations & Properties
  - Eigenvalues & Eigenvectors
- Differential Calculus
  - Derivatives & Gradients
  - Optimization Techniques
- Probability & Statistics
  - Key Distributions
  - Statistical Concepts

## 🔧 Machine Learning Fundamentals (04-07)

### 1. Perceptron & Neural Networks

- Perceptron Model
- Training Algorithms
- Geometric Interpretation
- Loss Functions
- Gradient Descent

### 2. Model Evaluation & Regularization

- Confusion Matrix
- Evaluation Metrics
  - Accuracy, Precision, Recall
  - F-score
- Regularization Techniques
  - L1 & L2 Regularization
  - Hyperparameter Tuning

### 3. Classification Algorithms

- Binary vs Multiclass
- One-vs-One & One-vs-Rest
- Model Types
  - Perceptron
  - Logistic Regression
  - Support Vector Machines

## 🎯 Clustering & Pattern Mining (12-20)

### 1. Clustering Fundamentals

- Types of Clustering
  - Representative-based
  - Hierarchical
  - Density-based
- K-Means Algorithm
  - Algorithm Steps
  - Initialization Methods
  - K-Means++

### 2. Advanced Clustering

- K-Medoids
- K-Medians
- Hierarchical Clustering
  - Agglomerative (Bottom-up)
  - Divisive (Top-down)
  - Linkage Methods

### 3. Association Pattern Mining

- Frequent Itemsets
- Support & Confidence
- Apriori Algorithm
- Pattern Generation
- Rule Mining

## 📊 Probabilistic Methods (08-11)

### 1. K-Nearest Neighbors

- Distance Metrics
- Parameter Selection
- Algorithm Implementation

### 2. Probabilistic Classification

- Bayes' Rule
- Naive Bayes
- Maximum Likelihood
- Zero Probability Handling
- Laplace Smoothing

## 🌐 Graph Mining & Networks (25-29)

### 1. Graph Theory Basics

- Graph Types & Properties
- Representation Methods
- Basic Algorithms

### 2. Social Network Analysis

- Centrality Measures
  - Degree Centrality
  - Closeness Centrality
  - Betweenness Centrality
- Prestige Measures
  - Degree Prestige
  - Proximity Prestige

### 3. PageRank Algorithm

- Web as a Graph
- Random Walk Model
- Matrix Computations
- Power Iteration
- Practical Considerations

## 📈 Data Visualization (30)

### 1. Visualization Principles

- Basic Components
  - Aesthetics
  - Position
  - Shape & Size
  - Color

### 2. Types of Visualizations

- Amount & Distribution
- Proportions
- X-Y Relationships
- Geospatial Data
- Uncertainty
- Trends

### 3. Tools & Best Practices

- Color Usage
- Common Pitfalls
- Python Libraries
  - Matplotlib
  - Seaborn
  - Bokeh
  - Others

## 🛠️ Implementation & Tools

### 1. Programming Tools

- Python Libraries
- Data Processing
- Visualization Tools

### 2. Best Practices

- Algorithm Selection
- Parameter Tuning
- Performance Optimization
- Error Handling

## 📊 Applications & Case Studies

### 1. Business Applications

- Market Analysis
- Customer Segmentation
- Recommendation Systems

### 2. Scientific Applications

- Bioinformatics
- Text Mining
- Web Mining
- Social Network Analysis
# File: 01.md
## 1. Introduction & Motivation for Data Mining

### 1.1 Why we need data mining

- **Data explosion** – organisations now collect data at petabyte scale; humans cannot inspect it manually.
- **Pervasive sources** – social media, YouTube, sensor networks, web logs, surveys; collection is cheaper than ever.
- Result: we require **automated analysis of massive data** (the very definition of data mining).

### 1.2 The data-mining pipeline

1. **Data collection** (application-specific; quality decisions here propagate downstream).
2. **Data preprocessing**
   - _Feature extraction_ – reshape raw data into algorithm-friendly forms (tables, time-series, etc.).
   - _Data cleaning_ – fix or impute missing/erroneous values.
   - _Feature selection & transformation_ – drop irrelevant variables; rescale or discretise others.
3. **Analytical processing** – apply algorithms (association mining, clustering, classification, outlier detection).
4. _Optional_ feedback loop, then **output to analysts**.

### 1.3 Feature concepts

- **Features** (a.k.a. attributes, variables) describe **objects** (rows, instances).
- Good features enable rules that generalise beyond the training set; crafting them is an “art” (feature engineering) – although modern deep learning often discovers features automatically.
- **Feature pruning** removes low-variance or irrelevant features (e.g., rare words in text, flat numeric columns).

---

## 2. Types of Data

### 2.1 High-level split

| Class                                 | Description                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Non-dependency-oriented (tabular)** | Objects are assumed independent; representable as an _n × d_ matrix of features.               |
| **Dependency-oriented**               | Objects are linked by **implicit** (order, space, time) or **explicit** (edges) relationships. |

### 2.2 Attribute-level data types inside tabular data

- **Numerical** – integer or real, with natural ordering.
- **Categorical** – unordered discrete values (e.g., colour).
- **Binary** – 0/1; treatable as numeric or two-category categorical and handy for set representation.
- **Text** – raw string (dependency-oriented) or vector-space (bag-of-words) representation.

### 2.3 Dependency-oriented data in detail

- **Implicit** dependencies
  - _Time-series_ (sensor readings over time).
  - _Discrete sequences / strings_ (categorical analogue of time-series).
  - _Spatial_ (each record tagged with coordinates).
  - _Spatiotemporal_ (both space & time).
- **Explicit** dependencies – **graphs/networks** with nodes, edges, and optional node/edge attributes (e.g., social networks, web graphs).

---

## 3. Data Representation Choices

### 3.1 Why representation matters

- It is the _first_ hands-on step after data collection.
- The structures you choose dictate which algorithms are valid and which patterns are even discoverable; there is no universally “best” representation.

### 3.2 Encoding categorical variables

- **One-hot / indicator vectors** – allocate one dimension per category; set to 1 if the record contains that category, else 0.
  - Example words: “awesome”, “burger”, “terrible”
    - (1, 1, 0) might represent a sentence containing “awesome” & “burger” but not “terrible”.

### 3.3 Multiple representations for the same text

Sentence: _“The burger I ate was an awesome burger!”_

1. **Word list (sequence)** – keeps order and duplicates.
2. **Word set** – unique tokens only.
3. **Bag-of-words vector** – frequency per word (TF/TF-IDF).
4. **Character-frequency vector** – stylistic signal useful for language ID, spam detection, etc.  
   Each encoding keeps some structure, discards other structure, and therefore fits different model families.

---

## 4. Four Fundamental Data-Mining Problem Types

| #     | Problem                         | Essence                                                                                                              | Typical outputs & use-cases                                                                            |
| ----- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **1** | **Association-pattern mining**  | Find features/items that co-occur above a support threshold. Special case: _frequent-pattern mining_ on binary data. | Market-basket rules (e.g., {Milk, Butter, Bread} appear together in ≥ 65 % of baskets).                |
| **2** | **Classification** (supervised) | Learn mapping from features → **class label** using labelled data; predict labels for unseen objects.                | Targeted marketing, spam/ham filtering, text recognition. Algorithms: Decision Tree, Naïve Bayes, etc. |
| **3** | **Clustering** (unsupervised)   | Partition data into _k_ clusters with high intra-cluster similarity & low inter-cluster similarity.                  | Customer segmentation, prototype-based data summarisation.                                             |
| **4** | **Outlier detection**           | Identify objects markedly different from the bulk—could be noise or critical exceptions.                             | Credit-card fraud, sensor fault detection, medical anomaly spotting, extreme earth-science events.     |
# File: 02.md
# 📚 **Data Preprocessing & Model Performance Study Sheet**

---

## **1. Missing Values**

### 🔍 **Problem**

Some feature values are unknown or not recorded in the dataset.

### ⚙️ **Handling Strategies**

| Method                    | Description                     | Pros                  | Cons                           |
| ------------------------- | ------------------------------- | --------------------- | ------------------------------ |
| **1. Discard**            | Remove rows with missing values | Simple                | Loss of useful data            |
| **2. Fill by Hand**       | Manually re-measure or annotate | Accurate              | Time-consuming, costly         |
| **3. Set "missingValue"** | Use a constant like `"missing"` | Useful for categories | Misleading for numbers         |
| **4. Replace with Mean**  | Use feature’s average value     | Easy, quick           | Inaccurate if outliers present |
| **5. Predict Missing**    | Train model to fill gaps        | Can be accurate       | Adds complexity                |
| **6. Accept Missing**     | Let algorithm handle it         | No extra work         | Risk if missing values are key |

---

## **2. Noisy Data**

### 🔍 **Problem**

Random errors or corruption in data can lead to **overfitting** and inaccurate models.

### ⚙️ **Detection Techniques**

- Obvious errors: wrong data types, extreme outliers
- Subtle errors: typos like `0.52` instead of `0.25`

### 🛠️ **Handling Strategies**

| Method                  | Description                         | Best For              |
| ----------------------- | ----------------------------------- | --------------------- |
| **Manual Inspection**   | Remove by hand                      | Small datasets        |
| **Clustering**          | Remove outliers based on groups     | High-dimensional data |
| **Linear Regression**   | Remove values far from trend line   | Numerical data        |
| **Frequency Threshold** | Drop rare values                    | Text/misspellings     |
| **Treat as Missing**    | Then apply missing value techniques | Seamless integration  |

---

## **3. Overfitting vs. Underfitting**

### ⚖️ **Definitions**

| Term             | Description                            | Symptoms                               |
| ---------------- | -------------------------------------- | -------------------------------------- |
| **Overfitting**  | Model too complex, fits training noise | High train accuracy, low test accuracy |
| **Underfitting** | Model too simple, misses patterns      | Low train & test accuracy              |

---

### 🛠️ **Solutions**

| Problem          | Fix                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------- |
| **Underfitting** | More training, better features, cleaner data, try better algorithm                           |
| **Overfitting**  | Simplify model, regularization, remove features, early stopping, more data, cross-validation |

---

## **4. Feature Normalisation**

### 📏 **Purpose**

Scale features to a common range so no single feature dominates due to scale.

### 🧮 **Methods**

| Method                     | Formula                                                               | Output Scale  | When to Use                           |
| -------------------------- | --------------------------------------------------------------------- | ------------- | ------------------------------------- |
| **[0,1]-Scaling**          | \(\hat{x} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}\) | [0,1]         | Neural networks, bounded-input models |
| **Gaussian Normalisation** | \(\hat{x} = \frac{x - \mu}{\sigma}\)                                  | Mean 0, Std 1 | Linear models, distance-based methods |

---

✅ **Quick Tips**

- Always check for **missing or noisy data** before training.
- Normalize **numerical features** if algorithms are sensitive to scale.
- Monitor for **overfitting** using validation/test performance.
- Use **cross-validation** to detect and prevent model bias.
# File: 03.md
## **1. Linear Algebra**

This section introduces basic linear algebra concepts used to represent and manipulate data.

### Key Concepts:

- **Vectors**: Represent data points as ordered lists of numbers. Denoted by bold or bar-marked uppercase letters (e.g., **X**, **Y**). Vectors are typically column vectors.
- **Matrices**: Collections of vectors organized by rows or columns (e.g., **M ∈ ℝⁿˣᵐ**). Used to represent datasets.
- **Vector Arithmetic**:
  - Addition, dot (inner) product, and outer product.
- **Matrix Arithmetic**:
  - Addition, multiplication (requires matching dimensions).
- **Transpose and Inverse**:
  - Transpose: flips rows and columns.
  - Inverse: only for full-rank square matrices; used to "undo" matrix multiplication.
- **Linear Independence**:
  - Vectors are linearly dependent if one is a combination of others.
- **Rank**:
  - The number of linearly independent rows or columns. Determines invertibility.
- **Trace**:
  - The sum of a matrix’s diagonal elements.
- **Eigenvalues & Eigenvectors**:
  - Eigenvectors remain in the same direction under transformation by the matrix; eigenvalues scale them.

---

## **2. Differential Calculus**

Essential for optimization and understanding function changes.

### Key Concepts:

- **Derivatives of Basic Functions**:
  - Includes polynomials, exponentials, logarithmic, sine, cosine.
- **Differentiation Rules**:
  - Sum, product, quotient, and chain rule.
- **Partial Derivatives**:
  - Used for functions with multiple variables; derivative with respect to one variable while keeping others constant.
- **Gradient**:
  - Vector of all partial derivatives; indicates the direction of steepest ascent.

---

## **3. Optimisation**

Used in model training and parameter tuning.

### Subtopics:

- **Unconstrained Optimisation (Gradient Descent)**:
  - Iterative method for finding a local minimum by moving in the direction of the negative gradient.
  - Update rule:  
    \[
    X\_{i+1} = X_i - \gamma_i \cdot \nabla f(X_i)
    \]
- **Constrained Optimisation (Lagrange Multipliers)**:
  - Handles optimisation with constraints.
  - Uses a Lagrangian function:  
    \[
    \mathcal{L}(X, \lambda) = f(X) - \lambda \cdot g(X)
    \]
  - Solves by finding points where all partial derivatives equal zero.

---

## **4. Probability**

Introduces key probability distributions for modeling randomness.

### Key Distributions:

- **Bernoulli**: Binary outcomes (e.g., coin toss).
- **Generalised Bernoulli**: Multiple discrete outcomes (e.g., dice).
- **Binomial**: Multiple trials of binary outcomes (e.g., number of heads in coin flips).
- **Multinomial**: Multiple trials of multi-outcome experiments (e.g., dice rolls).
# File: 04.md
---

## **1. Introduction to Perceptron**
- **Nature**: Binary classification algorithm.
- **Inspired by Biology**: 
  - Mimics the human nervous system.
  - Neurons and synapses form the basis.
  - Learning occurs by adjusting synaptic strengths based on external stimuli.
  - Perceptron simulates a **single neuron**.

---

## **2. Perceptron Model**

### **Architecture**

- Inputs: \( x_1, x_2, ..., x_d \)
- Weights: \( w_1, w_2, ..., w_d \)
- Activation score:  
  \[
  a = w_1x_1 + w_2x_2 + \cdots + w_dx_d
  \]
- Output:
  - \( 1 \) if \( a > \theta \)
  - \( -1 \) if \( a \leq \theta \)

---

## **3. Mathematical Notation**

- Input vector: \( X^T = (x_1, x_2, ..., x_d) \)
- Weight vector: \( W^T = (w_1, w_2, ..., w_d) \)
- Activation: \( a = W^T X \)

### **Bias Term**

- Makes threshold \( \theta = 0 \)
- Introduced as \( b = -\theta \)
- Activation:  
  \[
  a = W^T X + b
  \]
- Output: \( \text{sign}(W^T X + b) \)

---

## **4. Notational Trick (for Bias)**

- Introduce a constant feature \( x_0 = 1 \)
- Let \( w_0 = b \)
- Rewrite activation as:
  \[
  a = \sum\_{i=0}^{d} w_i x_i = W^T X
  \]
- Makes equations more elegant by embedding the bias in weights.

---

## **5. Training Algorithm (PerceptronTrain)**

### **Steps**

1. Initialize all weights \( w_i = 0 \) and bias \( b = 0 \)
2. For a maximum number of iterations:
   - For each training example \( (X, y) \):
     - Compute \( a = W^T X + b \)
     - If misclassified \( (y \cdot a \leq 0) \):
       - Update weights: \( w_i = w_i + y \cdot x_i \)
       - Update bias: \( b = b + y \)

---

## **6. Test Algorithm (PerceptronTest)**

- Given: weights \( W \), bias \( b \), and test input \( X \)
- Compute activation: \( a = W^T X + b \)
- Output: \( \text{sign}(a) \)

---

## **7. Key Features**

- **Online Learning**: Updates occur one example at a time.
- **Error-Driven**: Updates only on misclassification.

---

## **8. Misclassification Detection**

- An object is misclassified if:
  \[
  y \cdot a \leq 0
  \]

---

## **9. Perceptron Update Rule**

### **Intuition**

- **Misclassified Positive**:
  - Increase activation: add current example to weights and bias.
- **Misclassified Negative**:
  - Decrease activation: subtract current example from weights and bias.

### **Mathematical Update**

- \( W = W + yX \)
- \( b = b + y \)

---

## **10. Mathematical Explanation of Update**

- After update on misclassification:
  \[
  a' = a + \sum\_{i=1}^{d} x_i^2 + 1 > a
  \]
- The adjustment increases activation (for positive misclassification) or decreases it (for negative).

---

## **11. Remarks**

### **Activation Adjustment**

- One update may not fix misclassification.
- More aggressive algorithms (e.g., Passive Aggressive Classifier) might be needed.

### **Training Order**

- Order of data matters.
- Random shuffling in each iteration works best.

### **Hyperparameter and Overfitting**

- MaxIter is a hyperparameter.
- Too many iterations → overfitting.
- Too few iterations → underfitting.

---

## **1. Geometric Interpretation of Perceptron**

### **Decision Boundary**

- The perceptron decides class based on:
  \[
  W^T X + b > 0 \quad \text{(classified as +1)}
  \]
  \[
  W^T X + b \leq 0 \quad \text{(classified as -1)}
  \]
- The **decision boundary** is the set of points where:
  \[
  W^T X + b = 0
  \]
  This forms a **hyperplane**.

---

## **2. Hyperplane and Weight Vector**

- A **hyperplane** is a flat, N-1 dimensional surface in N-dimensional space.
- In **2D**, this is a **line**:  
  \[
  w_1 x_1 + w_2 x_2 = 0
  \]
- In **N dimensions**, it defines an **(N–1)-dimensional hyperplane**.
- The **weight vector \( W \)** is **perpendicular** to the hyperplane—it defines its orientation.

---

## **3. Visualizing Misclassification**

### **Before Update**

- If a **positive instance** \( X \) makes an angle **greater than 90°** with \( W \), then:
  \[
  W^T X < 0
  \]
  → It is misclassified as negative.

### **After Update**

- Perceptron update rule:
  \[
  W' = W + X
  \]
- The new weight vector \( W' \) now lies **between \( W \) and \( X \)**.
- The angle between \( W' \) and \( X \) becomes **less than 90°**.
- So \( W'^T X > 0 \), and the instance is correctly classified.

This is a key geometric idea: the perceptron "rotates" the weight vector towards misclassified examples.

---

## **4. Linearly Separable Data**

- A dataset is **linearly separable** if a hyperplane exists that correctly separates all positive and negative instances.
- There can be **many possible hyperplanes** that do this; the solution is **not unique**.

---

## **5. Non-linearly Separable Data**

- If no straight line (or hyperplane) can divide the classes perfectly, the data is **not linearly separable**.
- Example: Red and blue points mixed in a circular fashion can’t be separated by a straight line.

---

## **6. Important Observations**

- If the data **is** linearly separable:
  - The **perceptron algorithm is guaranteed to find** a separating hyperplane.
- The final result depends on:
  - The **order** of training instances (as discussed earlier).
  - The **last few updates**, which have more influence on the final weights.

---

## **7. Averaged Perceptron (Concept Mentioned)**

- One way to get more stable and general results:
  - **Take the average** of all weight vectors seen during training.
  - This is known as the **Averaged Perceptron algorithm** (not detailed here).

---
# File: 05.md
---

## **1. Loss Function Minimization in Perceptron**

### **Training Dataset and Parameters**

- Dataset:  
  𝒟 = {(X₁, y₁), ..., (Xₙ, yₙ)} where each Xₖ is a vector of features and yₖ is the label.
- Model Parameters:  
  W = (w₀, w₁, ..., w_d) including bias term.

### **Objective**

- Minimize the **loss function** L(W, 𝒟) by adjusting weights W using optimization techniques.

---

## **2. Perceptron Training Algorithm**

### **Initialization**

- Set all weights to zero:  
  wi = 0 for i = 1 to d  
  b = 0

### **Training Loop**

- Repeat for MaxIter iterations:
  - For each training sample (X, y):
    - Compute activation: a = WᵀX + b
    - If misclassified (y · a ≤ 0):
      - Update weights: wi ← wi + y · xi for all i
      - Update bias: b ← b + y

---

## **3. Loss Functions**

### **A. Step Loss Function (Misclassification Count)**

- For a single point:  
  L(b, W, Xₖ, yₖ) = 1 if misclassified, 0 otherwise
- For dataset:  
  L(b, W, 𝒟) = Total number of misclassifications

#### **Drawbacks**

- Piecewise constant → Not differentiable
- Gradient = 0 → Gradient descent not usable

---

### **B. Hinge-like Loss Function: h(t) = max(0, t)**

- Activation score:  
  aₖ = b + Σᵢ wᵢ xₖ⁽ⁱ⁾
- Individual loss:  
  L(b, W, Xₖ, yₖ) = h(−yₖ · aₖ)
- Dataset loss:  
  L(b, W, 𝒟) = Σₖ h(−yₖ · aₖ)

#### **Properties**

- Loss increases with misclassification severity
- Differentiable (almost everywhere)

---

## **4. Gradient Descent**

### **Gradient Computation**

For each (Xₖ, yₖ):

- ∂L/∂b = −yₖ if misclassified, else 0
- ∂L/∂wᵢ = −yₖ · xₖ⁽ⁱ⁾ if misclassified, else 0

Gradient Vector:

- ∇ = −yₖ · (1, xₖ⁽¹⁾, ..., xₖ⁽ᵈ⁾)ᵀ for misclassified samples
- Else, ∇ = 0 vector

### **Update Rule**

- Full batch gradient descent:  
  Update weights using all misclassified points
- Online gradient descent:  
  Update immediately after each misclassification:
  - (b, w₁, ..., w_d) ← (b, w₁, ..., w_d) + μ · y · (1, x₁, ..., x_d)ᵀ
  - Usually μ = 1

---

## **5. Final Training Algorithm (Simplified)**

1. Initialize weights and bias to 0
2. For each iteration:
   - For each sample (X, y):
     - Compute a = WᵀX + b
     - If y · a ≤ 0:
       - wi ← wi + y · xi
       - b ← b + y

---
# File: 06.md
---

### 1. **Purpose of Classifier Evaluation**

- We need to assess how good a classifier or model is.
- **Absolute Goodness**: Measured when the model is used in real-world settings—but we don’t know this before deployment.
- **Relative Goodness**: Measured using a test dataset (also called a gold standard) where we already know the correct labels.

---

### 2. **Gold Standard (Test Data)**

- Used only for evaluation, never training.
- Each item in the test set has a known correct label.
- Allows comparison between predicted and true labels using various metrics.

---

### 3. **Confusion Matrix**

A table layout for binary classification:

|                   | Actual YES (+)       | Actual NO (–)        |
| ----------------- | -------------------- | -------------------- |
| Predicted YES (+) | True Positives (TP)  | False Positives (FP) |
| Predicted NO (–)  | False Negatives (FN) | True Negatives (TN)  |

- Makes it easier to see if the model confuses two classes.

**Definitions**:

- **TP**: Correctly predicted positive.
- **TN**: Correctly predicted negative.
- **FP**: Incorrectly predicted positive.
- **FN**: Incorrectly predicted negative.

---

### 4. **Examples**

- **Car Detection**:
  - Positive: image has a car.
  - Negative: image does not.
- **Cancer Detection**:
  - FP: Healthy person wrongly predicted as having cancer.
  - FN: Person with cancer wrongly predicted as healthy.
  - Importance of FP vs FN depends on the application.

---

### 5. **Evaluation Measures**

- **Accuracy**: Proportion of total correct predictions.

  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision**: Among predicted positives, how many are actually positive.

  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

- **Recall**: Among actual positives, how many did we correctly predict.

  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

- **F-score**: Harmonic mean of precision and recall.

  \[
  \text{F-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

---

### 6. **Precision vs Recall (Trade-off)**

- **High Recall**: Important in applications like cancer detection.
- **High Precision**: Important in applications like product recommendation.
- Changing classification threshold affects this balance.

---

### 7. **Multi-class Evaluation**

- Metrics are calculated per class.
- **Precision for Class A**:

  \[
  \frac{\text{Correctly classified A}}{\text{Total predicted A}}
  \]

- **Recall for Class A**:

  \[
  \frac{\text{Correctly classified A}}{\text{Total actual A}}
  \]

- **F-score for Class A**:

  \[
  \frac{2 \cdot \text{Precision}\_A \cdot \text{Recall}\_A}{\text{Precision}\_A + \text{Recall}\_A}
  \]

- **Macro F-score**:
  - Average of F-scores across all classes.

---

### 1. **Types of Classification Algorithms**

- **Multiclass Classifiers**:

  - **k-Nearest Neighbors (k-NN)**
  - **Naive Bayes**

- **Binary Classifiers**:
  - **Perceptron**
  - **Logistic Regression**

---

### 2. **Converting Binary to Multiclass Classification**

To use a binary classifier for multiclass classification (with _k_ classes), we use **meta-algorithms**.

#### Two Main Strategies:

- **One-vs-One (OvO)**
- **One-vs-Rest (OvR)**

---

### 3. **One-vs-One (OvO) Approach**

1. For each pair of classes (i, j), train a binary classifier \( A\_{i,j} \) using only data from class i and class j.
2. This creates \( \frac{k(k - 1)}{2} \) classifiers.
3. To classify a new object \( X \), each model \( A\_{i,j} \) votes for either class i or j.
4. Count all votes and assign \( X \) to the class with the most votes.

**Drawbacks**:

- If there's a tie (equal number of votes), you need a confidence score to resolve it (if available from the binary classifier).

---

### 4. **One-vs-Rest (OvR) Approach**

1. For each class i:
   - Treat class i as the positive class and all others as negative.
   - Train a binary classifier \( A_i \).
2. You get **k** classifiers: \( A_1, A_2, ..., A_k \).
3. For a new object \( X \), compute the score from each \( A_i \).
4. Predict the class \( y \) where the score is highest:
   \[
   y = \arg\max\_{i \in \{1, 2, ..., k\}} A_i(X)
   \]

**Confidence Scores** (examples depend on algorithm):

- **Perceptron**: Use the activation score \( a = b + W^T X \)
- **Logistic Regression**: Use the sigmoid function \( \sigma(a) = \frac{1}{1 + e^{-a}} \)

**Drawbacks**:

- Classifiers may produce scores on different scales.
- Class imbalance: each classifier is trained on one small positive set vs a large negative set.

---
# File: 07.md

---

## 🌟 Regularisation Overview

**Purpose**: To reduce **overfitting** by controlling model complexity.

* Overfitting = Model fits training data too closely and performs poorly on new data.
* Regularisation adds a penalty to large model parameters (like weights).

### 🔧 Types of Regularisation:

* **L2 Regularisation**: Penalises the **square** of weights (also called **ridge** or **Tikhonov** regularisation).
* **L1 Regularisation**: Penalises the **absolute value** of weights (leads to sparsity).
* **L1 + L2**: A combination (e.g. ElasticNet).

---

## 📈 Example: Polynomial Curve Fitting

We are given noisy data sampled from:

$$
f(x) = x^3 - 4x^2 + 3x - 2
$$

We try to fit a polynomial of degree $d$ using weights $W = [w_0, w_1, ..., w_d]$, where:

$$
\hat{y}(x, W) = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d
$$

### 📉 Without Regularisation

Loss function (residual sum of squares):

$$
L(D, W) = \sum_{i=1}^n (\hat{y}(x_i, W) - y_i)^2
$$

Problem: As $d$ increases, weights grow large → **overfitting**.

---

## ✅ L2 Regularisation

### 🧮 New Loss Function (with regularisation):

$$
J(D, W) = L(D, W) + \lambda \|W\|^2 = \sum_{i=1}^n (\hat{y}(x_i, W) - y_i)^2 + \lambda \sum_{j=1}^d w_j^2
$$

* $\lambda$: regularisation coefficient (controls penalty strength).
* $\|W\|^2$: sum of squares of weights.
* Choose $\lambda$ via **cross-validation**.

---

## ⚙️ L2-Regularised Perceptron

### 🧠 Model

Perceptron learns a weight vector $W = [w_1, ..., w_d]$ and bias $b$. Prediction:

$$
a = b + \sum_{j=1}^d w_j x_j
$$

$$
\text{Predicted label} = \text{sign}(a)
$$

### 🧮 Perceptron Loss Function:

For input $(X_k, y_k)$, loss:

$$
L(b, W, X_k, y_k) = h(-y_k \cdot a_k) \quad \text{where } h(t) = \max(0, t)
$$

Sum over all data points for full loss.

---

## 🔁 Gradient Descent with L2 Regularisation

We compute gradients and update weights as:

$$
\nabla J = \nabla L + 2\lambda (0, w_1, ..., w_d)^T
$$

This means:

* **No penalty on bias** $b$
* **Penalty on weights** to keep them small

### 🧾 Update Rule (SGD version)

If a data point $(X, y)$ is **misclassified**:

$$
\begin{aligned}
w_i &\leftarrow w_i \cdot (1 - 2\lambda) + y \cdot x_i \\
b &\leftarrow b + y
\end{aligned}
$$

If **correctly classified**:

$$
\begin{aligned}
w_i &\leftarrow w_i \cdot (1 - 2\lambda) \\
b &\leftarrow b
\end{aligned}
$$

This update gradually **shrinks the weights** toward zero unless supported by the data.

---

## 🧪 Choosing λ (Hyperparameter Tuning)

1. Split data into **training** and **validation** sets.
2. Try various $\lambda$ values (e.g., $10^{-5}, 10^{-4}, ..., 10^5$)
3. Train models with each and pick the best based on performance on the validation set.

---

## 📌 Summary

* Regularisation helps generalisation by preventing weight explosion.
* L2 adds a **quadratic penalty** on weights.
* Works well with gradient-based methods like SGD.
* Can be naturally integrated into Perceptron and other models.

# File: 08.md
---

## **K-Nearest Neighbours (K-NN)**

### 1. **What is K-NN?**

K-NN is one of the simplest classification algorithms. It works as follows:

- **Training phase:** Just store the training data. There’s no learning or model building.
- **Classification phase:**

  - For a new (test) data point, find the **k** closest points in the training set.
  - Look at their labels.
  - Assign the **most common label** among them to the test point.

If **k = 1**, we just find the closest point and copy its label.

---

### 2. **How Distance is Measured (Similarity/Distance Functions)**

To find “closest” points, we measure how similar or distant two points are. Several methods are used:

#### a. **Numerical Data**

**i. Euclidean Distance (L2 norm):**

$$
\text{Euclidean}(X, Y) = \sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}
$$

- Measures straight-line distance.
- Larger distances = more dissimilar.

**ii. Manhattan Distance (L1 norm):**

$$
\text{Manhattan}(X, Y) = \sum_{i=1}^{d}|x_i - y_i|
$$

- Adds absolute differences in each dimension.
- Like navigating a city grid.

**iii. Cosine Similarity:**

$$
\text{Cosine}(X, Y) = \frac{X \cdot Y}{\|X\|\|Y\|} = \cos(\theta)
$$

- Measures angle between vectors.
- Used often in text analysis.
- Cosine distance = $1 - \text{Cosine similarity}$.

---

#### b. **Set Data**

- **Jaccard Similarity:**

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

- **Jaccard Distance =** $1 - J(A, B)$

- **Overlap Coefficient:**

$$
\text{Overlap}(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}
$$

---

#### c. **Binary Data**

- **Hamming Distance:**

$$
\text{Hamming}(X, Y) = \text{number of differing positions}
$$

E.g., $X = (1,0,1), Y = (0,0,1) \Rightarrow \text{Distance} = 1$

---

#### d. **Categorical Data**

- **Basic similarity:**

$$
S(x_i, y_i) =
\begin{cases}
1 & \text{if } x_i = y_i \\
0 & \text{otherwise}
\end{cases}
$$

- **Frequency-based similarity (accounts for rare categories):**

$$
S(x_i, y_i) =
\begin{cases}
1/p_i(x_i)^2 & \text{if } x_i = y_i \\
0 & \text{otherwise}
\end{cases}
$$

where $p_i(x_i)$ = frequency of value $x_i$ in the dataset.

---

### 3. **Choosing the Value of k**

- **k** is a hyperparameter: you decide its value, it's not learned.
- Rule of thumb:

  - Large datasets → larger k.
  - Small datasets → smaller k (but not too small).

- Avoid picking k based on test performance—this leads to overfitting.

#### ✅ Good practice:

- Use a **validation set** (a subset of the training data not used for training) to choose k.
- Or use **cross-validation**:

  - Split training data into multiple folds.
  - Train and validate on different folds.
  - Average the performance to choose the best k.

---

### 4. **Complexity of K-NN**

- **Training time:** Very fast. Just store the data.
- **Classification time:** Slow. You must compare each test point to every training point.

If the dataset is big, this becomes a problem. To speed up:

- Use **Approximate Nearest Neighbour (ANN)** algorithms (e.g., FLANN).

---

### 5. **Limitations and Enhancements**

#### a. **Distance-Weighted Voting**

In regular k-NN, all neighbours have equal vote. Instead:

- Weight each vote based on distance:

$$
w_i = \frac{1}{\text{distance}(X', Y_i)^2}
$$

- Add up weights for each class, choose the class with the highest total.

#### b. **Feature Importance**

- K-NN assumes **all features are equally important**, which is often not true.
- Irrelevant features can hurt performance.

#### c. **Feature Scaling**

Features must be on the **same scale**.

- Use **Gaussian Normalization** (a.k.a. Z-score normalization):

$$
x_{\text{new}} = \frac{x - \mu}{\sigma}
$$

Where:

- $\mu$ is the mean of the feature.
- $\sigma$ is the standard deviation.

This centers data around 0 with standard deviation 1. Essential before applying distance measures.

---

### 6. **Practical Tips Summary**

- Always normalize your data before using k-NN.
- Choose k using validation or cross-validation.
- Use L1 (Manhattan) or L2 (Euclidean) distances; try both.
- If high-dimensional, consider **dimensionality reduction** before k-NN.
- Beware of irrelevant features; they can mislead distance calculations.

---
# File: 09.md
---

## **Probabilistic Classifiers**

### **1. Ordinary vs Probabilistic Classifiers**

- **Ordinary classifier**: A function $f(X)$ assigns an input $X$ to a class $c \in \{c_1, c_2, ..., c_k\}$.

- **Probabilistic classifier**: Instead of a single class, it gives probabilities $p_i = P(c_i | X)$ for each class. These probabilities form a distribution and must sum to 1:

  $$
  p_1 + p_2 + \dots + p_k = 1
  $$

---

### **2. Discriminative vs Generative Models**

- **Discriminative Models**:

  - Model the conditional probability directly: $P_\theta(C | X)$
  - Learn parameters $\theta$ from data that best fit this form.

- **Generative Models**:

  - Model the joint probability: $P_\theta(X, C)$
  - Learn parameters from data, then use Bayes’ rule to get $P(C | X)$

---

### **3. Generative Models in Detail**

- Assume data is generated from some unknown distribution $P(X, c)$.

- If known, we could compute:

  $$
  f^*(X) = \arg\max_{c \in \mathcal{C}} P(X, c)
  $$

  This is the **Bayes Optimal Classifier** – it makes the least error on average.

- In practice, we don't know $P$, so we approximate it with $\hat{P}$ learned from data.

---

### **4. Model Assumptions and Parameter Estimation**

- Assume $P(X, c)$ belongs to a parametric family (e.g., Normal distribution).
- Use **i.i.d. assumption**: training data are independently and identically distributed.
- Estimate parameters using **Maximum Likelihood Estimation (MLE)**.

---

### **5. MLE – Example with One Parameter (Bernoulli)**

#### **Example 1: Biased Coin (Binary Classification)**

- Suppose we flip a coin and observe: T, H, H, H
- Let $\beta$ be the probability of heads (H).
- The likelihood of observing this sequence:

  $$
  P_\beta(THHH) = (1 - \beta) \cdot \beta^3 = \beta^3 - \beta^4
  $$

- To find the best $\beta$, differentiate and set to 0:

  $$
  \frac{d}{d\beta}(\beta^3 - \beta^4) = 3\beta^2 - 4\beta^3 = 0
  $$

  Solve:

  $$
  \beta = \frac{3}{4}
  $$

#### **General Case (h heads, t tails):**

- Likelihood: $\beta^h (1 - \beta)^t$
- Log-likelihood (easier to work with):

  $$
  \log(\beta^h (1 - \beta)^t) = h \log \beta + t \log(1 - \beta)
  $$

- Maximise this to get MLE of $\beta$.

---

### **6. MLE – Example with Multiple Parameters (Multiclass)**

#### **Example 3: K-sided Die**

- Let $\beta_1, \beta_2, ..., \beta_K$ be the probabilities of sides 1 through K.
- Suppose $x_i$ is the number of times side $i$ appeared.
- Likelihood:

  $$
  \prod_{i=1}^{K} \beta_i^{x_i}
  $$

- Log-likelihood:

  $$
  \sum_{i=1}^{K} x_i \log \beta_i
  $$

- Subject to: $\sum_{i=1}^{K} \beta_i = 1$

#### **Use Lagrange Multipliers:**

- Define:

  $$
  \mathcal{L}(\beta, \lambda) = \sum x_i \log \beta_i - \lambda\left( \sum \beta_i - 1 \right)
  $$

- Solve:

  $$
  \frac{\partial \mathcal{L}}{\partial \beta_i} = \frac{x_i}{\beta_i} - \lambda = 0 \Rightarrow \beta_i = \frac{x_i}{\lambda}
  $$

- Enforce the constraint:

  $$
  \sum \beta_i = 1 \Rightarrow \lambda = \sum x_i
  $$

  So:

  $$
  \beta_i = \frac{x_i}{\sum x_i}
  $$

---

### **7. Summary: Probabilistic Classifiers**

- **Generative Models** (model $P(X, C)$, use Bayes rule to classify):

  - Naive Bayes

- **Discriminative Models** (model $P(C | X)$ directly):

  - Logistic Regression
  - Neural Networks (Multilayer Perceptrons)

---
# File: 10.md
---

### 📘 Bayes’ Rule

Bayes’ Rule helps us calculate the probability of a hypothesis $H$ given evidence $E$:

$$
P(H | E) = \frac{P(E | H) \cdot P(H)}{P(E)}
$$

Where:

- $P(H | E)$: Posterior — probability of the hypothesis after seeing the evidence.
- $P(E | H)$: Likelihood — how likely the evidence is, assuming the hypothesis is true.
- $P(H)$: Prior — the original probability of the hypothesis.
- $P(E)$: Marginal — total probability of the evidence.

This is helpful when direct estimation of $P(H | E)$ is hard but the other values are easier to calculate.

---

### 📊 Example: Medical Diagnosis

Suppose:

- Meningitis causes a stiff neck 50% of the time.
- Probability of meningitis: 1 in 50,000 = 0.00002
- Probability of stiff neck: 1 in 20 = 0.05

We use Bayes’ Rule to find the chance of having meningitis if a patient has a stiff neck:

$$
P(H | E) = \frac{0.5 \cdot 0.00002}{0.05} = 0.0002
$$

So there's a 0.02% chance the patient has meningitis.

---

### 🧠 High Dimensions and Naive Bayes

When data has multiple features (like symptoms, measurements, etc.), directly calculating $P(H | X)$ for combinations becomes hard. For example, in a car diagnosis:

- $H$ = engine doesn't start
- $A$ = weak battery
- $B$ = no gas

Directly estimating $P(H | A, B)$ could be inaccurate if data is rare.

Instead, Bayes' Rule helps:

$$
P(H | A, B) = \frac{P(A, B | H) \cdot P(H)}{P(A, B)}
$$

---

### 🤖 Naive Bayes Approximation

Assume features are conditionally independent (this is the "naive" part). That means we can say:

$$
P(x_1, x_2, ..., x_d | C) = \prod_{i=1}^d P(x_i | C)
$$

So:

$$
P(C | x_1, x_2, ..., x_d) \propto P(C) \cdot \prod_{i=1}^d P(x_i | C)
$$

Where:

- $P(C)$: proportion of data in class $C$
- $P(x_i | C)$: proportion of class $C$ examples with feature $x_i$

This simplification makes it easy to compute.

---

### 💡 Independence Refresher

If two events A and B are independent:

$$
P(A \text{ and } B) = P(A) \cdot P(B)
$$

If more than 2 events are mutually independent:

$$
P(A_1, A_2, ..., A_n) = P(A_1) \cdot P(A_2) \cdot ... \cdot P(A_n)
$$

Naive Bayes assumes this for features given the class.

---

### 📌 Putting It Together (2-Feature Example)

Using our car diagnosis again:

$$
P(H | A, B) = \frac{P(A | H) \cdot P(B | H) \cdot P(H)}{P(A, B)}
$$

Each conditional probability like $P(A | H)$ is estimated from the data.

---

### 🔍 Classification: Proportional Form

We want the class $C$ that maximizes:

$$
P(C) \cdot \prod_{i=1}^d P(x_i | C)
$$

We don’t need the denominator $P(X)$, since it's the same for all classes.

---

### 🏏 Example: Should We Play?

Given test instance:

- Outlook: Sunny
- Temperature: Cool
- Humidity: High
- Windy: True

We compare:

- **Play = yes**:

$$
2/9 \cdot 3/9 \cdot 3/9 \cdot 3/9 \cdot 9/14 \approx 0.0053
$$

- **Play = no**:

$$
3/5 \cdot 1/5 \cdot 4/5 \cdot 3/5 \cdot 5/14 \approx 0.0206
$$

Since 0.0206 > 0.0053 → predict **Play = no**

---

### ✅ Summary

- **Bayes’ Rule** helps update probabilities with new evidence.
- **Naive Bayes** simplifies calculations by assuming feature independence.
- It's useful in high-dimensional data where direct probability estimates are hard.
- **Classification** is done by choosing the class with the highest posterior probability.
# File: 11.md
---

## Zero Probabilities and Laplace Smoothing

### **Example: Naive Bayes Classification**

We want to predict whether someone will **play** or not based on weather conditions.

Given test instance:
**(Outlook = sunny, Temp = cool, Humidity = high, Windy = true)**

We compute the probability of `play=yes` and `play=no`:

#### Calculation:

**P(Play = yes | X)** =
(2/9) × (3/9) × (3/9) × (9/14) ≈ **0.0053**

**P(Play = no | X)** =
(3/5) × (1/5) × (4/5) × (3/5) × (5/14) ≈ **0.0206**

🡺 Since 0.0206 > 0.0053, we predict **Play = no**.

---

### **How These Probabilities Are Calculated**

Each conditional probability like **P(Outlook = sunny | Play = no)** is calculated using:

$$
P(x_i = a | C = c) = \frac{n(a, c)}{N(c)}
$$

- $n(a, c)$: Number of training examples in class **c** with feature value **a**
- $N(c)$: Total number of training examples in class **c**

---

### **Problem: Zero Probabilities**

Imagine a test instance:
**(Outlook = overcast, Temp = cool, Humidity = high, Windy = true)**

Now, assume:

$$
P(Outlook = overcast | Play = no) = 0
$$

Then:

$$
P(Play = no | X) = 0 × … = 0
$$

No matter what other features say, the result becomes **0** just because of one zero probability.

This is bad for small datasets and **even worse for datasets with many features** (e.g., text classification with 1000s of words).

---

### **Solution: Laplace Smoothing**

To avoid multiplying by **0**, we slightly increase every count—even if it’s currently 0.

#### Formula:

$$
P(x_i = a | C = c) = \frac{n(a, c) + 1}{N(c) + m_i}
$$

Where:

- $n(a, c)$: number of times feature value **a** appears in class **c**
- $N(c)$: total number of examples in class **c**
- $m_i$: number of possible values for feature **x_i**

This ensures:

- No probability is zero
- All probabilities still sum to 1

---

### **Before and After Laplace Smoothing**

Let’s say we have 5 possible values for a feature, and current counts for class **c** are:

| Value | Count (before) | Probability (before) |
| ----- | -------------- | -------------------- |
| a1    | 3              | 3/9                  |
| a2    | 1              | 1/9                  |
| a3    | 0              | 0/9 ← problem!       |
| a4    | 2              | 2/9                  |
| a5    | 3              | 3/9                  |

Apply Laplace smoothing:

New total = 9 + 5 = **14**
Each count increases by 1:

| Value | Count (after) | Probability (after)  |
| ----- | ------------- | -------------------- |
| a1    | 4             | 4/14                 |
| a2    | 2             | 2/14                 |
| a3    | 1             | 1/14 ← no more zero! |
| a4    | 3             | 3/14                 |
| a5    | 4             | 4/14                 |

---

### **Key Takeaway**

- **Zero probabilities** can ruin predictions in Naive Bayes classifiers.
- **Laplace smoothing** fixes this by adding 1 to each count.
- It ensures no probability is zero while keeping total probability valid.
# File: 12.md
---

## 📌 What is Clustering?

**Clustering** is the task of grouping a dataset's objects into **clusters** so that:

- Items in the same cluster are **similar** to each other.
- Items in different clusters are **less similar**.

The key challenge lies in how we define **"similarity"**, which can vary based on the data and use case.

---

## 📌 Why Do We Cluster?

Clustering helps with:

- **Data summarization** – simplifying large datasets.
- **Topic detection** – especially in text or documents.
- **Visualization** – to see structure in data.
- **Outlier detection** – spotting data that doesn’t fit in.
- **Community detection** – often used in social networks or graphs.

---

## 📌 Clustering vs Other Learning Types

| Type of Learning    | Description                                 |
| ------------------- | ------------------------------------------- |
| **Supervised**      | Data has labels (e.g., “cat” or “dog”).     |
| **Unsupervised**    | No labels are given — clustering fits here. |
| **Semi-supervised** | Some data has labels, some does not.        |

In unsupervised learning like clustering, we explore the structure of data using **feature similarity** and **distribution**, even without labels. This can later enhance supervised learning by creating better features.

---

## 📌 Challenges in Clustering

1. **How to cluster a dataset?**
   There is **no single correct answer**. A dataset can be clustered in **many valid ways**.

2. **How many clusters?**
   This is often not known ahead of time and can vary by context.

3. **How to measure quality?**

   - **Extrinsic Evaluation**: Compare clusters with a ground truth or labeled data.
   - **Intrinsic Evaluation**: Judge based only on how well the data is grouped internally (e.g., how compact and separated the clusters are).

---

## 📌 Types of Clustering Algorithms

### 1. **Representative-Based Clustering**

- Select some **central points** (called representatives), then assign data points to the nearest one.
- Iteratively update the clusters.
- Two popular methods:

  - **k-Means**:

    - Choose `k` cluster centers (means).
    - Assign points to the nearest center.
    - Recompute the centers based on assignments.
    - Repeat until convergence.

  - **k-Medoids**:

    - Similar to k-Means, but cluster centers are actual data points (medoids), not averages.

📝 **Explanation**:

- `k` = number of clusters (you must choose this number).
- Works well when clusters are roughly spherical and equally sized.

---

### 2. **Hierarchical Clustering**

- Builds a **tree of clusters** (called a dendrogram).
- Two approaches:

  - **Agglomerative (bottom-up)**:

    - Start with each data point as its own cluster.
    - Merge closest clusters until one big cluster remains.

  - **Divisive (top-down)**:

    - Start with all data in one cluster.
    - Split it recursively into smaller clusters.

📝 Great for discovering nested groupings or when the number of clusters is not known.

---

### 3. **Graph-Based Clustering**

Useful when data is structured as a graph (e.g., social networks).

- **Community Detection (Modularity Optimization)**:

  - Finds densely connected groups in the graph.
  - Modularity measures how well the graph is divided into such groups.

- **Graph-Cut Algorithms (e.g., Spectral Clustering)**:

  - Converts clustering into a graph-cutting problem.
  - Uses the graph’s eigenvalues (spectrum) to find good cuts.

📝 Works well when the data has complex structures not easily captured by distance-based methods.

---

## ✅ Summary

Clustering is a versatile tool in data analysis. It's:

- **Unsupervised**: no labels needed.
- Flexible with many algorithms, each suitable for different data types and goals.
- Evaluated either by comparing to known labels (extrinsic) or by internal structure (intrinsic).
# File: 13.md
---

## Clustering Evaluation

Clustering quality is evaluated using two main types of methods:

### 1. **Extrinsic Methods (Supervised)**

- Use ground truth labels (i.e., known class labels for data).
- Measure how well the clustering output aligns with the known labels.
- Treat clustering as a classification task.

#### Basic Steps:

1. Assign each cluster the label that appears most frequently in it.
2. Merge clusters that are given the same label.
3. For each label (class), compute:

   - **Precision** = Correctly clustered items / Total items in cluster
   - **Recall** = Correctly clustered items / Total items with that label
   - **F1-score** = Harmonic mean of precision and recall.

4. Compute the **macro-average**: average these metrics across all labels.

---

#### **B-CUBED Evaluation Measure**

A widely used extrinsic evaluation method for clustering.

**Definitions**:

- Let **C(x)** = Cluster to which item **x** belongs.
- Let **A(x)** = Actual class/label of item **x**.

**For each item x**:

- **Precision(x)** = Number of items in C(x) that have A(x) / Total items in C(x)
- **Recall(x)** = Number of items in C(x) that have A(x) / Total items with label A(x)

**Overall Metrics**:

- **Average Precision** = Sum of precision(x) for all x / N
- **Average Recall** = Sum of recall(x) for all x / N
- **Average F1-score** = Sum of F(x) for all x / N

Where:

- **F(x)** = (2 × Precision(x) × Recall(x)) / (Precision(x) + Recall(x))
- N = Total number of items in the dataset

---

#### Example: **Community Mining**

Task: Cluster 50 personal names from 5 domains (e.g. actors, politicians) using semantic similarity measures.

**Method**:

- Used **group-average agglomerative hierarchical clustering (GAAC)**.
- Merged clusters based on correlation defined as:

  $$
  \text{Corr}(Θ) = \frac{1}{2} \cdot \frac{1}{|Θ|(|Θ|-1)} \sum_{(u,v)∈Θ} sim(u,v)
  $$

  - Θ: the merged cluster (from clusters A and B)
  - |Θ|: number of items in Θ
  - sim(u,v): similarity between items u and v

**Evaluation**:

- B-CUBED metric applied.
- Best-performing method (proposed) had highest F1 score: **0.7897**.

| Method     | Precision | Recall | F1 Score |
| ---------- | --------- | ------ | -------- |
| Proposed   | 0.7958    | 0.804  | 0.7897   |
| Sahami     | 0.6384    | 0.668  | 0.6426   |
| WebJaccard | 0.5926    | 0.712  | 0.6147   |
| WebDice    | 0.5895    | 0.716  | 0.6179   |
| WebOverlap | 0.5976    | 0.680  | 0.5965   |
| WebPMI     | 0.2649    | 0.428  | 0.2916   |
| Chen       | 0.4763    | 0.624  | 0.4984   |

---

### 2. **Intrinsic Methods (Unsupervised)**

- Do not use ground truth labels.
- Evaluate the quality of clusters based on internal properties like cohesion and separation.

#### **Silhouette Coefficient**

A widely used intrinsic method.

**For a data point x**:

- Let **a(x)** = Average distance from x to all other points in its own cluster.
- Let **b(x)** = Average distance from x to all points in the next nearest cluster.

Then,

$$
s(x) = \frac{b(x) - a(x)}{\max(a(x), b(x))}
$$

- If cluster has only one point (|C| = 1), then s(x) = 0.
- s(x) ∈ \[-1, 1]

**Interpretation**:

- **s(x) ≈ 1** → x is well clustered.
- **s(x) ≈ 0** → x is on the border of two clusters.
- **s(x) ≈ -1** → x may be in the wrong cluster.

**Overall Silhouette Score** = Average of s(x) over all x in the dataset.

---
# File: 14.md
---

## k-Means Algorithm

### Overview

k-Means is a **representative-based clustering algorithm** used to partition data into **k clusters**. The goal is to find **k representative points (centroids)** such that the **sum of distances between each data point and its closest representative is minimized**.

Let:

- **k** = number of clusters
- **D = {X₁, X₂, ..., Xₙ}** = dataset of n points
- **Y₁, ..., Yₖ** = cluster representatives (centroids)

The **objective function** to minimize is:

$$
\sum_{i=1}^{n} \min_{j} d(X_i, Y_j)
$$

This is the total distance from each point to the nearest representative.

### Representative-Based Algorithms

To define a specific algorithm, we need to specify:

1. **How to choose representatives** (e.g., random initialization)
2. **The distance function** (e.g., Euclidean)

Note: Representatives **do not need to be actual data points**.

---

## General k-Representatives Approach

1. **Initialise**: Pick initial k representatives
2. **Iteratively refine**:

   - **Assign step**: Assign each data point to the **nearest representative** using the distance function
     → This creates clusters **C₁, ..., Cₖ**
   - **Optimise step**: For each cluster Cⱼ, update its representative Yⱼ to **minimize local distances**

$$
\sum_{X_i \in C_j} d(X_i, Y_j)
$$

---

## k-Means Specifics

- Uses **squared Euclidean distance**:

  $$
  \|X - Y\|^2 = (X₁ - Y₁)^2 + (X₂ - Y₂)^2 + ...
  $$

- Goal: **Minimize total within-cluster sum of squares (WCSS)**:

  $$
  \sum_{j=1}^{k} \sum_{X \in C_j} \|X - Y_j\|^2
  $$

- The optimal representative (Yⱼ) for each cluster is the **mean (centroid)** of its points:

  $$
  Y_j = \frac{1}{|C_j|} \sum_{X \in C_j} X
  $$

---

## k-Means Algorithm (Step-by-Step)

Given:

- Dataset D = {X₁, ..., Xₙ}
- Number of clusters k

### Steps:

1. **Initialisation Phase**:

   - Randomly select k representatives Y₁, ..., Yₖ

2. **Assignment Phase**:

   - Assign each data point to the **closest representative** (based on squared Euclidean distance)
   - Forms clusters C₁, ..., Cₖ

3. **Optimisation Phase**:

   - Update each representative Yⱼ to be the **mean of points in cluster Cⱼ**

Repeat Steps 2 and 3 until convergence:

- No data points change clusters
- OR a fixed number of iterations is reached

---

## Issues with k-Means

- **Random initialisation** can lead to **different results** (local minima)
- **Sensitive to outliers**: outliers can heavily shift the mean
- **Means are not actual data points**
- **Euclidean distance** is not suitable for **categorical features**
- **No spatial awareness**: pixel location isn't considered in image segmentation

Tip: Run k-Means **multiple times with different initialisations**, choose the run with the **lowest WCSS**.

---

## Example: Image Segmentation

- Goal: Group similar pixels in an image
- Each **pixel is a point in 3D RGB space**
- Use k-Means to cluster pixels into **k color clusters**
- Result: image segmented into k regions of similar color
- **Note**: Ignores pixel location, purely based on color similarity

---
# File: 15.md
---
### 📌 **Goal of Lecture**

Improve k-means clustering results by **carefully choosing initial centroids** (starting cluster centers).
---

### 🟡 **Method 1: Random Initialization**

- Choose **k** points from the data at random to start.
- Run k-means.
- May get **lucky** (good clustering), or **unlucky** (poor results).
- Example shown: 4 iterations each of lucky and unlucky choices.

💡 **Problem**: Random chance can lead to bad initial centroids → poor clustering.

---

### 🟡 **Method 2: Random Initialization + Repetition**

Steps:

1. Randomly choose initial centroids.
2. Run k-means.
3. Repeat multiple times.
4. Pick the result with the **best performance** (e.g. lowest error).

📝 Pros:

- Improves over pure random selection.

👎 Cons:

- Still not guaranteed to be optimal.
- Can be computationally expensive.

---

### 🟡 **Method 3: Sampling + Hierarchical Clustering**

Steps:

1. Take a **small sample** of the dataset.
2. Run **hierarchical clustering** on this sample.
3. Use **means of resulting clusters** as initial centroids.
4. Run standard k-means on the full dataset.

✅ Good if:

- Sample is **small** (hundreds or a few thousand points).
- Number of clusters **k** is small.

👎 Hierarchical clustering is slow on large data, so sample must be small.

---

### 🟡 **Method 4: Furthest-Point Heuristic**

Steps:

1. Pick one point randomly from dataset.
2. Pick next point that is **furthest away** from any chosen point.
3. Repeat until k points are selected.

📌 Let:

- $D$: dataset
- $R(X)$: distance from point $X$ to nearest chosen center

✅ Encourages spread-out centroids.

👎 Risk: May choose **outliers**, which hurt clustering.

---

### 🟡 **Method 5: k-means++ (Smart Seeding)**

**Key Idea**: Choose next centroid with **probability proportional to the square of its distance** from existing centroids.

Steps:

1. Choose one point at random.
2. For each point $X$, compute $R(X)^2$ = distance to closest chosen centroid squared.
3. Choose next centroid with probability:

   $$
   P(X) = \frac{R(X)^2}{\sum R(X)^2}
   $$

4. Repeat until k points are chosen.
5. Run k-means.

✅ **Avoids poor initializations.**
✅ **Theoretically proven** to give better results on average.

---

### 🧪 **Experimental Results: k-means++ vs. k-means**

**Synthetic Data:**

- Datasets called **Norm-10** and **Norm-25** were created with well-separated clusters.
- k-means often merges clusters wrongly due to bad seeds.
- k-means++ consistently finds better results (closer to real centers).

**Real Data:**

- **Cloud dataset**: k-means++ was \~2x faster and gave \~20% better clustering.
- **Intrusion dataset**: k-means++ gave **10x–1000x better clustering** and was up to **70% faster**.

✅ k-means++ offers **speed and accuracy** benefits in both synthetic and real scenarios.

---

### ✅ Summary of Methods

| Method                      | Pros                                        | Cons                         |
| --------------------------- | ------------------------------------------- | ---------------------------- |
| Random                      | Simple                                      | Often poor results           |
| Random + Repeat             | Better than single random                   | Still based on luck, slower  |
| Sampling + Hierarchical     | Effective with small data/sample            | Expensive for large datasets |
| Furthest-Point              | Ensures spread out centroids                | May pick outliers            |
| **k-means++ (recommended)** | Best mix of accuracy, speed, and simplicity | More complex to implement    |

---
# File: 16.md
---
## **k-Medians Algorithm Overview**

The **k-Medians algorithm** is a _representative-based clustering method_, where the goal is to split a dataset into **k clusters** such that the total **Manhattan (L1) distance** between the data points and their assigned cluster representative is minimized.
---

## **1. Representative-Based Clustering**

Let:

- **k** = number of clusters
- **D = {X₁, X₂, ..., Xₙ}** be the dataset
- We want to find **k representatives Y₁, ..., Yₖ** (not necessarily in the dataset)

**Objective:**
Minimize the total distance of each point to its closest representative:

$$
\sum_{i=1}^n \min_{j} d(X_i, Y_j)
$$

Where **d(X, Y)** is a distance function.

---

## **2. General k-Representative Clustering Approach**

Steps:

1. **Initialisation**: Choose k initial representatives.
2. **Iterative Refinement**:

   - **Assign step**: Assign each data point to its nearest representative using the distance function.
   - **Optimize step**: Update each representative to minimize the total distance within its cluster.

---

## **3. k-Means (for Comparison)**

In **k-Means**, the distance function used is **squared Euclidean distance**:

$$
\sum_{i=1}^{k} \sum_{X \in C_i} \|X - Y_i\|^2
$$

- $C_i$ is the i-th cluster.
- The optimal representative for each cluster is the **mean** (centroid) of the points in that cluster.

**How to find it:**

$$
Y_i = \frac{1}{|C_i|} \sum_{X \in C_i} X
$$

---

## **4. k-Medians (Main Focus)**

Instead of Euclidean distance, **k-Medians uses the Manhattan distance (L1 distance)**:

$$
\|X - Y\|_1 = \sum_{i=1}^d |x_i - y_i|
$$

The goal becomes:

$$
\sum_{i=1}^{k} \sum_{X \in C_i} \|X - Y_i\|_1
$$

---

## **5. Finding the Optimal Representative in k-Medians**

Assume a fixed cluster **C**, and want to find the best representative **Y** to minimize:

$$
\sum_{X \in C} \|X - Y\|_1
$$

If:

- Each point $X \in \mathbb{R}^d$ has coordinates $X = (X^{(1)}, X^{(2)}, ..., X^{(d)})$
- The optimal Y will be $Y = (Y^{(1)}, ..., Y^{(d)})$ where each $Y^{(i)}$ is the **median** of the i-th coordinates of all points in the cluster.

**Why?**
Because minimizing L1 distance means choosing the **median** in each coordinate:

$$
Y^{(i)} = \text{median}(X_1^{(i)}, X_2^{(i)}, ..., X_s^{(i)})
$$

This ensures the sum of absolute differences is minimized.

---

## **6. Summary of the k-Medians Algorithm**

**Input**:

- Dataset: $D = \{X_1, ..., X_n\}$
- Number of clusters: $k$

**Steps**:

1. **Initialisation**: Choose $k$ representatives randomly from the dataset.
2. **Assignment**: Assign each data point to its closest representative using L1 distance.
3. **Optimisation**: For each cluster, update the representative to be the **median** of all points in the cluster (coordinate-wise).
4. **Repeat steps 2–3** until:

   - No data point changes its cluster, or
   - A fixed number of iterations is reached.

---
# File: 17.md
---

## **k-Medoids Clustering Algorithm**

### ❗ Problems with k-Means (why k-Medoids is useful)

- **Sensitive to initialisation**: Results vary depending on the starting points.
- **Can get stuck in local minima**: May not find the best (global) clustering.
- **Outliers affect results**: Because k-means uses the mean, which is easily skewed.
- **Cluster centers are not actual data points**: In k-means, centers can be "imaginary".
- **Not suitable for categorical data**: Euclidean distance (used by k-means) doesn’t work well with categories.

---

## ✅ Key Features of k-Medoids

- A **representative-based algorithm**: Like k-means, it tries to find 'k' representatives (medoids) to minimize a total distance cost.
- **Medoids are real data points**: Cluster centers are selected from actual data.
- **Can use any distance (dissimilarity) function**: Not limited to Euclidean distance.
- **More robust to noise/outliers** than k-means.

---

## 🎯 Objective Function

The goal is to minimize the sum of distances from each point to the closest medoid:

$$
\sum_{i=1}^{n} \min_{j} d(X_i, Y_j)
$$

- **$X_i$** = i-th data point
- **$Y_j$** = j-th medoid (cluster representative)
- **$d(⋅,⋅)$** = distance (dissimilarity) function (e.g., Euclidean, Manhattan, etc.)

---

## 🧗 Hill-Climbing Strategy

Used during the optimization phase:

1. Start with a random solution (set of medoids).
2. Make a small change (swap a medoid with a non-medoid).
3. If the change improves the result (i.e., reduces the objective), accept it.
4. Repeat until no further improvement is possible.

---

## 🧮 k-Medoids Clustering Algorithm: Step-by-Step

Given:

- Dataset $\mathcal{D} = \{X_1, ..., X_n\}$
- Number of clusters $k$

### **1. Initialization**

Randomly choose $k$ data points as the initial medoids: $Y_1, ..., Y_k$

### **2. Assignment**

Assign each data point to the nearest medoid → creates clusters $C_1, ..., C_k$

### **3. Optimization (Hill-Climbing)**

- For each possible pair $(X, Y)$ where:

  - $X \in \mathcal{D}$ (a data point)
  - $Y \in \{Y_1, ..., Y_k\}$ (a medoid)

- Try replacing $Y$ with $X$
- If the total distance decreases, update the medoid and repeat the assignment step.
- Stop when no better swap exists.

---

## ✔️ Pros of k-Medoids

- Medoids are actual data → easy to interpret.
- Robust to **noise and outliers**.
- Works with any **distance function** → suitable for **categorical, mixed, time-series** data.

---

## ❌ Cons of k-Medoids

- **Still sensitive to initialisation**: May need multiple runs.
- **Can get trapped in local optima**.
- **Slower than k-means**: Due to complex medoid-swapping step.

---

## ⏱️ Time Complexity Issue

- At each optimization step, checking **every possible swap** requires:

  $$
  k \cdot n \text{ computations}
  $$

  (where $n$ = number of data points, $k$ = number of medoids)

- This is expensive, especially for large datasets.

### 🔧 Solution:

Instead of checking all possible $k \cdot n$ swaps:

- Randomly sample $r$ pairs $(X, Y)$, where:

  - $X$ is a data point
  - $Y$ is a current medoid

- Only compute improvements for these $r$ pairs → saves time.

---
# File: 18.md
---

## **Hierarchical Clustering**

### **Key Features**

- **No need to specify the number of clusters** in advance.
- **Builds a hierarchy of clusters**, giving clusterings at all levels of granularity.
- Useful for **visualising data structures** and organizing data into nested groups or concepts.

---

### **Hierarchy of Clusters**

Clusters can be nested like this:

- Start with: `{1, 2, 3, 4, 5}`
- Then: `{1, 2, 3}, {4, 5}`
- Then: `{1, 2}, {3}, {4, 5}`
- Finally: `{1}, {2}, {3}, {4}, {5}`

This shows how clusters split up step-by-step.

---

### **Dendrogram**

A **dendrogram** is a tree-like diagram used to visualize the hierarchy.

- The **bottom** shows individual items.
- As you go **up**, items get merged into clusters.
- The **height** at which two items merge shows how similar (or distant) they are — lower height means more similar.

---

### **Real-World Applications**

#### **1. Phylogenetic Trees (Evolutionary Biology)**

Steps to build a phylogenetic tree:

1. Generate DNA sequences for different species.
2. Calculate **edit distances** between sequences (i.e. how many changes it takes to turn one into another).
3. Use these distances to compute **similarities**.
4. Build a tree showing evolutionary relationships.

This helps chart how species evolved over time.

#### **2. Tracking Viruses**

- Viruses like **HIV** mutate quickly.
- DNA similarities between viral samples can be used to **trace transmission paths**.
- Example: Used in a court case to show the victim's virus was more similar to a suspect’s than to a control group.

**Visual example:**

- V1–V3: Victim’s strands
- P1–P3: Patient’s (accused) strands
- LA1–LA12: Control group strands
  Dendrograms can show that V and P group more closely than LA, suggesting a link.

#### **3. SARS-CoV-2 (COVID-19) Phylogeny**

- Hierarchical clustering was used to track the **Omicron variant's spread and evolution**.
- Helped scientists understand how the variant differed from others.

---

### **Two Main Types of Hierarchical Clustering**

#### **1. Agglomerative Clustering (Bottom-Up)**

- **Start with each data point as its own cluster.**
- At each step, **merge the two closest clusters**.
- Repeat until all points are merged into one cluster.
- Most commonly used.

#### **2. Divisive Clustering (Top-Down)**

- **Start with one big cluster** containing all points.
- At each step, **split the cluster** into smaller groups.
- Continue until each item is in its own cluster.

---
# File: 19.md
---

## Divisive Clustering (Top-Down Clustering)

### Core Idea

- **Divisive clustering** starts with all data points in a single large cluster (the root) and **repeatedly splits** them into smaller clusters.
- This creates a **tree-like structure** (a hierarchy), going from general to specific.

### Method Used for Splitting

- Any **flat clustering algorithm** can be used to do the splitting at each step.

  - Example: **k-means**.
  - It does **not** have to be a distance-based method.

---

## Trade-Off Strategies for Splitting

You can control how the tree grows based on your goal:

### Strategy 1: Balance number of objects

- Always split the **cluster with the most points**.
- Result: Clusters (leaf nodes) will tend to have **similar sizes**, but the tree might not be balanced.

### Strategy 2: Balance tree structure

- Always split **every cluster into the same number of subclusters**.
- Result: The **tree is balanced** (same number of children per node), but some clusters will have **more data points** than others.

---

## Generic Divisive Clustering Algorithm

**Input:**

- `𝒟`: Dataset
- `𝒜`: Flat clustering algorithm

**Steps:**

1. Start with a **tree `𝒯`** containing just one node (the root) with all the data (`𝒟`).
2. Repeat:

   - Select a **leaf node `L`** to split (based on a rule, like the biggest cluster).
   - Use `𝒜` to split `L` into smaller clusters: `L₁, L₂, ..., Lₖ`.
   - Add these `L₁...Lₖ` as **children of `L`** in the tree.

3. Stop when a **termination condition** is met (e.g., enough clusters made, or clusters are small enough).
4. Return the clustering or the full hierarchy (tree).

---

## Bisecting k-Means Algorithm (A Special Case)

This is a popular way to do divisive clustering.

**Input:**

- `𝒟`: Dataset
- `s`: Desired number of clusters

**Steps:**

1. Start with a **tree `𝒯`** containing one node with all data.
2. Repeat:

   - Choose the **leaf node (cluster) `L`** with the **largest total squared distance** between its points.

     - Mathematically:

       $$
       \sum_{\text{all pairs } X, Y \in L} \text{dist}(X, Y)^2
       $$

       This measures **how spread out** the cluster is.

   - Use **k-means with k=2** to split `L` into `L₁` and `L₂`.
   - Add `L₁` and `L₂` as children of `L` in the tree.

3. Stop when there are `s` leaf clusters.
4. Return the leaf clusters (the final groups).

---

## Visual Example (from slides)

Multiple slides visually show how the **bisecting k-means** process works in steps:

- Start with one big cluster.
- Repeatedly choose the widest-spread cluster to split.
- Use k-means to divide it.
- Grow the tree and stop when you hit the target number of clusters.

---
# File: 20.md
---
### **Agglomerative Clustering: Overview**

Agglomerative clustering is a **bottom-up** hierarchical clustering method. It starts with each object in its **own cluster** and **merges** them step by step into larger clusters.
---

### **Algorithm Steps**

Given a dataset **𝒟**:

1. **Initialize**: Each object is its own cluster.
2. **Repeat**:

   - Find the **closest pair** of clusters, say **i** and **j**.
   - **Merge** them into a new cluster.

3. **Stop** when a **termination criterion** is met (e.g. desired number of clusters).
4. **Output**: A clustering, a hierarchy (dendrogram), or a set of nested clusterings.

---

### **Key Concept: Proximity Between Clusters**

To decide which clusters to merge, we need a way to **measure distance (proximity)** between clusters. Different strategies lead to different results.

---

### **Linkage Methods (Ways to Measure Distance Between Clusters)**

#### 1. **Single-Linkage (Nearest Neighbor)**

- Distance between two clusters = **minimum distance** between any two points, one from each cluster.
- Formula:

  $$
  \text{dist}(P, Q) = \min_{X \in P, Y \in Q} d(X, Y)
  $$

- This can create long, chain-like clusters.

#### 2. **Complete-Linkage (Furthest Neighbor)**

- Distance between two clusters = **maximum distance** between any two points, one from each cluster.
- Formula:

  $$
  \text{dist}(P, Q) = \max_{X \in P, Y \in Q} d(X, Y)
  $$

- Tends to create **compact, spherical** clusters by trying to keep the diameter (largest distance within a cluster) small.

#### 3. **Group-Average Linkage**

- Distance = **average distance** between all pairs of points (one from each cluster).
- Let:

  - **p** = number of points in cluster P
  - **q** = number of points in cluster Q

- Formula:

  $$
  \text{dist}(P, Q) = \frac{1}{p \cdot q} \sum_{X \in P, Y \in Q} d(X, Y)
  $$

- Balances between single- and complete-linkage methods.

---
# File: 21.md
---

## **Logistic Regression Lecture Summary**

### **1. Classifier Types**

- **Ordinary classifier:** Predicts a fixed class label (e.g., cat/dog).
- **Probabilistic classifier:** Predicts a probability distribution over all class labels. For input $X$, it gives:

  $$
  p_i = P(c_i \mid X), \quad \sum p_i = 1
  $$

### **2. Discriminative vs. Generative Models**

- **Discriminative models** learn $P(C \mid X)$: directly model the probability of a class given input.
  Examples: Logistic Regression, Neural Networks.
- **Generative models** learn $P(X, C)$: model how data is generated.
  Examples: Naive Bayes.

Bayes' Rule connects both:

$$
P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}
$$

---

### **3. Problem Setup**

- **Binary classification problem**: Class labels are $y \in \{-1, +1\}$.
- Goal: Build a **probabilistic model** that predicts the probability $P(y = +1 \mid X)$.

---

### **4. Main Idea of Logistic Regression**

- Define a **hyperplane**:

  $$
  b + W^T X = 0
  $$

  where:

  - $W = (w_1, w_2, ..., w_d)$: feature weights
  - $b$: bias
  - $X = (x_1, ..., x_d)$: input features

- In a **Perceptron**, classification depends on the **sign** of $b + W^T X$.

- In **Logistic Regression**, this value (called the "score") also indicates **confidence** in the prediction — the larger its absolute value, the further the point is from the decision boundary.

---

### **5. Mapping Scores to Probabilities**

- Use the **logistic sigmoid function**:

  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}, \quad \text{outputs in } (0,1)
  $$

#### Properties:

- $\sigma(-x) = 1 - \sigma(x)$
- Derivative:

  $$
  \frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
  $$

---

### **6. Logistic Regression Model**

- For input $X$, with score $a = b + W^T X$:

  $$
  P(y = +1 \mid X) = \sigma(a) = \frac{1}{1 + e^{-a}}, \quad
  P(y = -1 \mid X) = \sigma(-a) = \frac{1}{1 + e^a}
  $$

- Compactly:

  $$
  P(y = t \mid X) = \sigma(t \cdot a), \quad t \in \{-1, +1\}
  $$

---

### **7. Training via Maximum Likelihood**

- **Training data**: $\mathcal{D} = \{(X_i, y_i)\}_{i=1}^n$

- **Likelihood function**:

  $$
  \mathcal{L}(b, W) = \prod_{i=1}^n \sigma(y_i (b + W^T X_i))
  $$

- **Negative log-likelihood (loss function)**:

  $$
  -\ell = - \sum_{i=1}^n \log \sigma(y_i (b + W^T X_i))
  $$

---

### **8. Gradient Computation**

Let $a_i = b + W^T X_i$. The gradients are:

- With respect to bias $b$:

  $$
  \frac{\partial \ell}{\partial b} = \sum_{i=1}^n y_i \cdot \sigma(-y_i \cdot a_i)
  $$

- With respect to each weight $w_k$:

  $$
  \frac{\partial \ell}{\partial w_k} = \sum_{i=1}^n y_i \cdot \sigma(-y_i \cdot a_i) \cdot x_k^{(i)}
  $$

These help update the weights and bias using gradient descent.

---

### **9. Gradient Descent Update Rule**

- Basic formula:

  $$
  Z_{i+1} = Z_i - \gamma_i \cdot \nabla_Z f(Z_i)
  $$

- Logistic Regression updates:

  $$
  W \leftarrow W + \mu \sum_{i=1}^n y_i \cdot \sigma(-y_i a_i) X_i
  $$

  $$
  b \leftarrow b + \mu \sum_{i=1}^n y_i \cdot \sigma(-y_i a_i)
  $$

---

### **10. Online vs Batch Learning**

- **Batch**: Use the entire dataset for each weight update.

  - More accurate but slower.
  - Common optimizer: **L-BFGS**.

- **Online**: Use one data point at a time.

  - Faster but less stable.
  - Uses **Stochastic Gradient Descent (SGD)**.

---

### **11. Prediction (After Training)**

Given trained $W$, $b$, and new input $X$:

- Compute:

  $$
  a = b + W^T X
  $$

- If $a > 0$, predict $+1$; else $-1$
- Probability/confidence:

  $$
  P(y = +1 \mid X) = \sigma(a), \quad P(y = -1 \mid X) = 1 - \sigma(a)
  $$

---

### **12. Interpretation as a Neuron**

- Model behaves like a **neuron**:

  $$
  a = b + \sum_{i=1}^d w_i x_i
  $$

  Apply sigmoid to get probability. Output depends on sign of $a$.

---

### **13. L2 Regularisation**

- Penalises large weights to prevent overfitting.

- Objective becomes:

  $$
  J(\mathcal{D}, W) = L(\mathcal{D}, W) + \lambda \|W\|^2 = L + \lambda \sum w_i^2
  $$

  where $\lambda$ is a regularisation parameter.

- Gradient becomes:

  $$
  \nabla_W J = \nabla_W L + 2\lambda W
  $$

- Update rule with regularisation:

  $$
  W \leftarrow (1 - 2\mu \lambda)W + \mu y \cdot \sigma(-y \cdot a) X
  $$

---
# File: 22.md
---

## **Association Pattern Mining**

### **Applications**

Association mining is useful in many areas:

- **Supermarket data**: Find which items are often bought together (e.g., eggs and milk).
- **Marketing**: Place items close together or promote combos (e.g., offer yogurt to people who buy eggs + milk).
- **Text mining**: Spot terms that appear together often.
- **Web log analysis**, **software bug detection**, **spatio-temporal event detection**, etc.

---

### **Terminology**

- **Transaction**: A single data record (like a shopping basket).
- **Itemset**: A set of items (e.g., {milk, bread}).
- **Frequent itemset**: An itemset that appears often in the data.
- **Support**: How common an itemset is—measured as a fraction of transactions that include it.

---

### **Frequent Pattern Mining Model**

Let:

- **U** be the set of all items.
- **𝒟** be the dataset = a list of transactions = {T₁, ..., Tₙ}.
- Each **Tᵢ** is a subset of items from **U**.
- Each transaction can be shown as a binary vector: 1 = item is present, 0 = item is absent.
- **Support of itemset I**, written `sup(I)`, is the fraction of transactions that include I.

**Problem**:
Given a dataset 𝒟 and a minimum support threshold `f`, find all itemsets whose support ≥ `f`.

**Notes**:

- Lower `f` → more frequent itemsets (but more computation).
- Higher `f` → fewer frequent itemsets (may miss useful ones).

---

### **Example**

Dataset:

| Transaction | Milk | Butter | Bread | Mushrooms | Onion | Carrot |
| ----------- | ---- | ------ | ----- | --------- | ----- | ------ |
| 1234        | 1    | 1      | 1     | 0         | 1     | 0      |
| 324         | 0    | 0      | 0     | 1         | 1     | 1      |
| 234         | 1    | 1      | 1     | 0         | 1     | 0      |
| 2125        | 1    | 1      | 1     | 1         | 0     | 1      |
| 113         | 1    | 0      | 0     | 1         | 1     | 0      |
| 5653        | 1    | 1      | 1     | 1         | 1     | 0      |

Let `f = 0.65` (i.e., itemsets must appear in ≥ 65% of the 6 transactions = 4 or more times).

- **{Mushrooms, Onion, Carrot}**: Not frequent (appears in fewer than 4).
- **{Milk, Butter, Bread}**: Frequent (appears ≥ 4 times) → large itemset.

---

### **Monotonicity and Downward Closure**

- If an itemset **I** is frequent, all its subsets are also frequent.
- If a subset is not frequent, its supersets can’t be frequent either.
- A **maximal frequent itemset** is a frequent itemset that has no frequent supersets.

**Example**:

- `{Butter, Bread}` is frequent.
- `{Milk, Butter, Bread}` is **maximal** at `f = 0.65`.

---

### **Frequent vs Maximal Itemsets**

From the same example:

- **Maximal frequent itemsets**:

  - `{Milk, Butter, Bread}`
  - `{Milk, Onion}`
  - `{Mushrooms}`

- **All frequent itemsets** (10 total):

  - Single items: `{Milk}`, `{Butter}`, `{Bread}`, `{Onion}`, `{Mushrooms}`
  - Pairs: `{Milk, Butter}`, `{Milk, Bread}`, `{Butter, Bread}`, `{Milk, Onion}`
  - One triple: `{Milk, Butter, Bread}`

**Note**:
Maximal frequent itemsets are a compact summary—but they don’t tell us the support of all subsets.

---

### **Association Rules**

We form rules like:

**X ⇒ Y**
Means: If a transaction contains **X**, it's likely to contain **Y**.

**Confidence** measures the probability of **Y** occurring given **X**.

Formula:

$$
\text{conf}(X ⇒ Y) = \frac{\text{sup}(X ∪ Y)}{\text{sup}(X)}
$$

Where:

- `sup(X ∪ Y)`: Support of both X and Y together.
- `sup(X)`: Support of just X.

**Example**:

- **conf({Milk} ⇒ {Butter, Bread})**

  - sup({Milk}) = 5/6
  - sup({Milk, Butter, Bread}) = 2/3
  - So conf = (2/3) ÷ (5/6) = 4/5

---

### **Defining Valid Rules**

A rule **X ⇒ Y** is valid if:

1. Support of `X ∪ Y` ≥ frequency threshold `f`
2. Confidence of the rule ≥ confidence threshold `c`

---

### **Association Rule Generation Framework**

**Two Phases**:

**Phase 1**: Find all frequent itemsets (≥ support `f`)

- Use algorithms like:

  - **Bruteforce**
  - **Apriori** (uses downward closure to reduce computation)

**Phase 2**: Generate association rules (≥ confidence `c`)

- For each frequent itemset **I**, split it into all **(X, Y)** pairs where:

  - **X ∪ Y = I**
  - **X** and **Y** are disjoint

- Compute confidence of each rule **X ⇒ Y**

  - Keep only those with conf ≥ `c`

---

### **Optimizing Phase 2**

**Confidence Monotonicity**:
If **X₁ ⊂ X₂**, then:

$$
\text{conf}(X₂ ⇒ I − X₂) ≥ \text{conf}(X₁ ⇒ I − X₁)
$$

**Example**:

- `{Butter} ⇒ {Milk, Bread}`
- `{Butter, Bread} ⇒ {Milk}`
  → The second rule has higher or equal confidence, so the first may be skipped as redundant.

---
# File: 23.md
---

### **Frequent Itemset and Association Rule Generation Framework**

Association rule mining happens in two main steps:

#### **Phase 1: Frequent Itemset Generation**

We try to find **all sets of items** (called **itemsets**) that occur **frequently** in a dataset, based on a **minimum frequency threshold (f)**.

Two methods:

- **Brute Force Algorithm**
- **Apriori Algorithm** (covered separately, not part of this file)

#### **Phase 2: Rule Generation**

From the frequent itemsets, generate **association rules** that meet a **minimum confidence threshold (c)**.

For each frequent itemset `I`:

- Generate **all pairs of subsets (X, Y)** such that:

  - `X ∪ Y = I`
  - `X` and `Y` are disjoint (don’t overlap)

- For each rule `X ⇒ Y`, compute **confidence**:

  - If `confidence ≥ c`, store the rule.

---

### **Brute Force Algorithm**

#### **Key Concepts**

- Let `U` be the **set of all items**.
- The number of possible **non-empty itemsets** is `2^|U| - 1`.

Every one of these is a **candidate** for being frequent.

#### **Basic Brute Force Algorithm**

Input:

- `U`: the universe of items
- `𝒟`: the dataset (a list of transactions)
- `f`: frequency threshold

Steps:

1. For **every non-empty subset `I` of `U`**:
2. Compute **support** of `I`:

   - Support means **how many transactions contain `I`**.

3. If `support(I) ≥ f`, then `I` is **frequent**, so add it to the list.

✅ **Problem:** If `|U| = 1000`, then the number of subsets = `2^1000`, which is more than `10^300` — too large to handle!

---

### **Downward Closure Property**

Very important for improving performance.

**Key idea:**

> If an itemset is frequent, then **all its subsets are also frequent**.

Therefore:

- If a certain size of itemset is **not frequent**, then **larger itemsets** containing it also **can’t be frequent**.
- This helps us **prune the search space** and avoid unnecessary checks.

---

### **Improved Brute Force Algorithm**

This version uses the **Downward Closure Property**.

Steps:

1. For `k` from 1 to `|U|` (i.e. increasing itemset sizes):
2. For each **k-itemset** `I`:

   - Compute support of `I`
   - If `support(I) ≥ f`, then `I` is frequent

3. If **no k-itemsets** are frequent at any step, **stop** early (no point going further).

#### **Why it’s better:**

- For **sparse datasets** (each transaction has only a few items), this method is much faster.
- If the largest transaction contains `l` items, then the total number of candidate itemsets is:

  - ![Formula](https://latex.codecogs.com/png.image?\dpi{110}\sum_{i=1}^l%20\binom{|U|}{i})
  - Much smaller than `2^|U|` when `l << |U|`.

**Example:**

- `|U| = 1000`, `l = 10`
- Then only need to consider roughly `1023` subsets instead of `10^300` — a huge improvement.

---

### **Summary**

- **Brute Force** checks **all item combinations**, which is simple but very slow for large `U`.
- The **Improved Brute Force** algorithm uses the **downward closure** trick to cut out many combinations early.
- Still, even improved methods struggle if the dataset has long transactions or low thresholds.
# File: 24.md
---
## **The Apriori Algorithm – Summary**

### **1. Goal**

To find **frequent itemsets** in a dataset of transactions, based on a minimum **frequency threshold**.
---

### **2. Key Concepts**

#### **Itemset**

A group of items that might appear together in a transaction.

#### **Support of an itemset (sup(I))**

The number of transactions in which itemset **I** appears.

#### **Frequent itemset**

An itemset whose support is **greater than or equal** to a threshold **f**.

---

### **3. Brute Force Approach (Improved)**

Given:

- **U**: Universe of items
- **𝒟**: Dataset (a list of transactions)
- **f**: Frequency threshold

Steps:

1. For each **k** from 1 to |U|:
2. For every possible itemset **I** of size **k**:

   - Calculate **sup(I)** (support).
   - If **sup(I) ≥ f**, add **I** to frequent itemsets.

3. If no itemsets of size **k** are frequent, STOP.

🧠 **Problem**: Checking support for every possible combination is expensive.

---

### **4. Apriori Algorithm – Core Idea**

Avoid checking all combinations by using the **Downward Closure Property**:

> If an itemset is frequent, then **all its subsets** are also frequent.

So:

- Skip generating itemsets that have infrequent subsets.
- Only consider itemsets built from **frequent smaller itemsets**.

---

### **5. Definitions Used**

- **ℱₖ**: Set of frequent itemsets of size **k**
- **𝒞ₖ**: Set of candidate itemsets of size **k**

---

### **6. Apriori Algorithm – Steps**

Given:

- **U**: Universe of items
- **𝒟**: Dataset of transactions
- **f**: Frequency threshold

1. Compute **ℱ₁** (frequent 1-itemsets)
2. For **k = 2 to d** (max itemset size):

   - If **ℱₖ₋₁** is empty → break
   - Generate candidate itemsets **𝒞ₖ** from **ℱₖ₋₁**
   - For each **I ∈ 𝒞ₖ**, if **sup(I) ≥ f**, add **I** to **ℱₖ**

3. Return all **frequent itemsets** (the union of all **ℱᵢ**)

---

### **7. Assumptions**

- **U = {1, 2, ..., d}** (items are numbered)
- Itemsets are ordered subsets of **U**
- Transactions are ordered lexicographically (e.g. {1, 2, 3})

#### **Example Dataset (ordered)**

```
{1, 2, 4}
{1, 3, 4}
{1, 3, 5}
{1, 2, 5}
{2, 2, 3}
```

---

### **8. Downward Closure Property – Details**

Let **I = {j₁, j₂, ..., jₖ} ∈ ℱₖ** (a frequent k-itemset). Then:

1. Every subset of size **(k−1)** is also frequent.
2. **I** can be **created by joining** two itemsets from **ℱₖ₋₁**:

   - `{j₁, j₂, ..., jₖ₋₁}` and
   - `{j₁, j₂, ..., jₖ₋₂, jₖ}`

So:
👉 We only keep **I** in candidates if it can be formed by joining two valid (k−1)-itemsets.

---

### **9. Candidate Generation (generate-candidates)**

**Input**:

- **ℱ**: Set of frequent itemsets of size **k**
- **k**: Current itemset size

#### **Join Phase**

1. Assume items in **ℱ** are ordered
2. For each **I ∈ ℱ**, where **I = {j₁, ..., jₖ₋₁}**:

   - For each **j = jₖ₋₁ + 1 to d**:

     - Create **I′ = {j₁, ..., jₖ₋₂, j}**
     - If **I′ ∈ ℱ**, then form candidate **I ∪ {j}** and add to **𝒞**

#### **Prune Phase**

1. For each **I ∈ 𝒞**:

   - For each **j ∈ I**:

     - If **I - {j} ∉ ℱ**, remove **I** from 𝒞 (since one subset is not frequent)

---

### **10. Example Summary**

- **Frequent itemsets**: itemsets that appear frequently in transactions.
- **Non-frequent itemsets**: do not meet the frequency threshold.
- **Generated candidates**: possible frequent itemsets from joining.
- **Pruned candidates**: removed because some subset was not frequent.

---
# File: 25.md
---

## 📊 Introduction to Graph Mining

Graph mining is a field that focuses on analyzing and understanding data that naturally forms networks or graph structures. These graphs appear in many real-world systems, such as:

- **Air Transportation Networks**: Airports as nodes, flights as edges.
- **Social Networks**: People as nodes, friendships/interactions as edges.
- **Cattle Movement Networks**: Farms as nodes, animal movements as edges.
- **Email Exchange Networks**: People as nodes, email communications as edges.
- **Program Flow Graphs**: Program components as nodes, control flow as edges.
- **Chemical Reaction Networks**: Molecules/reactions as nodes, interactions as edges.
- **Power Networks (Electrical Grids)**: Power stations and substations as nodes, transmission lines as edges.
- **Molecular Graphs (Chemical Graphs)**: Atoms as nodes, chemical bonds as edges.

Each of these examples involves a set of **entities** (nodes) and **relationships or interactions** (edges).

---

## 📌 Key Topics in Graph Mining

1. **Graph Classification**

   - Assign a label or category to an entire graph.
   - Example: Classifying molecules as toxic or non-toxic based on their structure.

2. **Graph Clustering**

   - Group similar graphs or nodes within a graph.
   - Useful for finding similar users, communities, or similar molecules.

3. **Graph Pattern Mining**

   - Discover frequently occurring subgraphs or structures.
   - Example: Identifying common interaction patterns in proteins.

4. **Graph Compression**

   - Reduce the size of the graph while retaining its structure.
   - Helps with storage and speeding up analysis.

5. **Graph Dynamics**

   - Study how a graph changes over time.
   - Important for tracking evolving networks like social media or financial transactions.

6. **Social Network Analysis**

   - Study relationships and influence in social graphs.
   - Tasks include finding influential people, communities, or viral spread patterns.

7. **Graph Visualization**

   - Creating visual representations of graphs to understand structure and patterns.

8. **Link Analysis**

   - Study relationships and predict new connections.
   - Example: Recommending friends or products.

---

## 🧪 Two Common Settings in Graph Mining

### 1. **Database of Many Small Graphs**

- **Use case examples**:

  - Chemical compounds (each compound is a separate graph).
  - Program flow graphs from software.

- **Tasks**:

  - Graph classification: e.g., classifying compounds by activity.
  - Pattern mining: finding substructures that appear frequently.
  - Clustering: grouping similar graphs.

### 2. **A Single Large Graph**

- **Use case examples**:

  - The web (pages as nodes, links as edges).
  - Social networks (people and relationships).
  - Transportation systems.

- **Tasks**:

  - Community detection: identifying tightly connected groups.
  - Influential node identification: finding key individuals (e.g., influencers).
  - Node ranking: like Google's PageRank, ranking web pages.
  - Link prediction: predicting which nodes might connect in the future.

---
# File: 26.md
---
## Graph Theory Basics

A **graph** consists of:
  - A set of **nodes (vertices)**: Represent entities (like airports in a network).
  - A set of **edges (links)**: Represent relationships or connections between nodes.

  ### Terms:

  - Two nodes are **adjacent** if there's an edge between them.
  - An edge is **incident** on its two endpoint nodes.
  - Graphs are usually drawn with:
      - Circles for nodes
      - Lines for edges
---

## Types of Graphs

### 1. **Undirected Graph**:

- Edges have no direction.
- (u, v) = (v, u): Edge from u to v is the same as from v to u.

### 2. **Directed Graph (Digraph)**:

- Edges have direction (called **arcs**).
- (u, v) ≠ (v, u): Edge from u to v is different from v to u.

### 3. **Multigraph**:

- Multiple edges are allowed between the same pair of nodes.

### 4. **Loop**:

- An edge that connects a node to itself (u, u).

---

## Weighted Graphs

- **Edge-weighted**: Each edge has a number (weight), e.g., cost, distance, strength of connection.
- **Vertex-weighted**: Each node has a weight, representing something like importance or size.

---

## Adjacency Matrix

A matrix representation of a graph:

- Let the graph have **n** vertices.
- Then, the adjacency matrix **A** is an n×n matrix.

### For an Undirected Graph:

- **A\[i]\[j] = 1** if there is an edge between vertex i and j.
- Otherwise, **A\[i]\[j] = 0**.
- The matrix is **symmetric** (A\[i]\[j] = A\[j]\[i]).

### For a Directed Graph:

- **A\[i]\[j] = 1** if there is an arc from vertex i to j.
- Otherwise, **A\[i]\[j] = 0**.
- The matrix is **not necessarily symmetric**.

### For a Weighted Graph:

- **A\[i]\[j] = w_ij** where **w_ij** is the weight of the edge or arc from i to j.
- **A\[i]\[j] = 0** if there is no edge/arc.

---

## Neighbours and Degree

### In Undirected Graphs:

- Node **v** has a **neighbour** u if (u, v) is an edge.
- **Neighbourhood N(v)**: Set of all neighbours of v.
- **Degree deg(v)**: Number of neighbours (edges incident on v).

### In Directed Graphs:

- **In-neighbour**: u is an in-neighbour of v if (u, v) is an arc.
- **Out-neighbour**: u is an out-neighbour if (v, u) is an arc.
- **In-degree deg⁻(v)**: Number of arcs coming into v.
- **Out-degree deg⁺(v)**: Number of arcs going out from v.

---

## Paths and Distance

### In Undirected Graphs:

- A **path**: A sequence of distinct nodes where each consecutive pair is connected by an edge.
- **Length of path**: Number of edges in it.
- **Distance between nodes u and v**: Length of the shortest path. If no path, distance is ∞.

### In Directed Graphs:

- A **directed path**: Sequence of nodes where each is connected by a directed arc to the next.
- **Length** and **distance** are defined the same way as above.

---

## Connectedness

### In Undirected Graphs:

- A graph is **connected** if there's a path between every pair of nodes.
- A **connected component** is a maximal set of nodes that are connected to each other.

### In Directed Graphs:

- A graph is **strongly connected** if there’s a directed path from every node to every other node.
- A **strongly connected component** is a maximal subset where all nodes are mutually reachable via directed paths.

---
# File: 27.md
---

## Social Network Analysis (SNA): Centrality and Prestige

### Basic Concepts

- **Actors**: The individuals or entities in a network (like people, organizations).
- **Relationships**: Interactions between actors, forming connections or edges in a graph.
- **Graph Representation**: A social network is shown as a graph **G = (V, E)**, where:

  - **V** = set of nodes (actors),
  - **E** = set of edges (relationships).

---

## Why Analyze Networks?

We use networks to:

- Understand structure
- Identify key/central actors
- Find critical connectors
- Detect communities
- Find unusual patterns
- Predict new connections
- Study influence spread

---

## Centrality Measures (for undirected graphs)

Centrality tells us which nodes are most important.

### 1. **Degree Centrality (CD)**

- **Definition**: Number of connections a node has, normalized by the maximum possible (which is _n – 1_ where _n_ is the number of nodes).

  **Formula**:

  $$
  CD(i) = \frac{\text{deg}(i)}{n - 1}
  $$

  - **deg(i)** = number of direct connections (degree) of node _i_
  - Nodes with high degree often act as **hubs**.

- **Limitation**: Only local view – doesn't consider the broader network.

---

### 2. **Closeness Centrality (CC)**

- Measures how close a node is to all other nodes (based on shortest paths).

- Nodes with smaller average distance to others have higher closeness.

  **Formula**:

  $$
  CC(i) = \frac{1}{AvDist(i)} = \frac{1}{\frac{\sum_{j \neq i} \text{dist}(i, j)}{n - 1}}
  $$

  - **dist(i, j)** = shortest path distance between _i_ and _j_

- **High CC** = fast access to all parts of the network.

---

### 3. **Betweenness Centrality (CB)**

- Captures how often a node appears on shortest paths between other pairs of nodes.

- Reflects **control over information flow**.

  **Formula**:

  $$
  CB(i) = \frac{\sum_{j < k} f_{jk}(i)}{\frac{n(n - 1)}{2}}
  $$

  where:

  - $q_{jk}$ = number of shortest paths between _j_ and _k_
  - $q_{jk}(i)$ = number of those paths that pass through _i_
  - $f_{jk}(i) = \frac{q_{jk}(i)}{q_{jk}}$

- High CB = node is a **bridge** between groups.

- **Can be extended to edges**, identifying **bridging edges** in community detection (e.g. Girvan-Newman algorithm).

---

## Prestige Measures (for directed graphs)

Used when relationships have direction (e.g., who follows whom).

### 1. **Degree Prestige (PD)**

- Based on **in-degree**: how many incoming links a node has.

  **Formula**:

  $$
  PD(i) = \frac{\text{deg}^+(i)}{n - 1}
  $$

  - **deg⁺(i)** = number of incoming links to _i_

- High in-degree = more **popular** or **respected**.

---

### 2. **Proximity Prestige (PP)**

- Considers how many nodes can reach a given node, and how far they are.

  Definitions:

  - **Influence(i)** = set of nodes that can reach _i_
  - **AvDist(i)** = average distance from all nodes in **Influence(i)** to node _i_

    $$
    AvDist(i) = \frac{\sum_{j \in Influence(i)} \text{dist}(j, i)}{|Influence(i)|}
    $$

  - **InfluenceFraction(i)** =

    $$
    \frac{|Influence(i)|}{n - 1}
    $$

  **Final formula**:

  $$
  PP(i) = \frac{InfluenceFraction(i)}{AvDist(i)}
  $$

- Fixes the issue of closeness-style measures favoring nodes with small influence sets.

- **Higher PP** = node is reachable by many and from shorter distances.

---

## Summary Table

| Measure                | Graph Type | What it Captures                | Key Concept                         |
| ---------------------- | ---------- | ------------------------------- | ----------------------------------- |
| Degree Centrality      | Undirected | Local popularity (hubs)         | Number of connections               |
| Closeness Centrality   | Undirected | Reachability                    | Average distance to others          |
| Betweenness Centrality | Undirected | Control over flow               | Paths that pass through the node    |
| Degree Prestige        | Directed   | Popularity (based on followers) | Incoming edges                      |
| Proximity Prestige     | Directed   | Influence                       | Reachability + how close others are |

---
# File: 28.md
---

### **PageRank Algorithm (by Google)**

**Goal:**
Determine how _important_ a webpage is — so that more important pages show up earlier in search results.

---

### **1. Web as a Graph**

Think of the Web as a directed graph:

- **Nodes** = webpages
- **Edges (links)** = hyperlinks from one page to another

Each page can:

- **Receive in-links** (other pages link to it)
- **Send out-links** (it links to other pages)

---

### **2. Intuition Behind PageRank**

PageRank treats links like _votes_:

- A link from Page A to Page B is a vote from A for B.
- But not all votes are equal!
  A vote from an _important_ page is worth _more_.

So, a page is important if:

- Many pages link to it **AND**
- Those linking pages are themselves important.

---

### **3. Basic PageRank Formula**

Let’s define:

- $P(a)$: PageRank score of page $a$
- $O_a$: Number of out-links from page $a$
- $E$: Set of all directed links (edges) in the web graph

The formula for PageRank is:

$$
P(a) = \sum_{(x,a) \in E} \frac{P(x)}{O_x}
$$

This means:

- To get the score of page $a$, sum up the scores of all pages linking to it.
- But each of those scores is divided by how many pages they link to (because each vote is shared).

This creates one equation like this for **every page** — together they form a **system of equations**.

---

### **4. Matrix Representation**

Let’s organize everything into matrices:

- Let there be $n$ pages.
- $P$ is a column vector:

  $$
  P = [P(1), P(2), ..., P(n)]^T
  $$

- Define matrix $A$ such that:

  - $A_{ij} = \frac{1}{O_i}$ if there is a link from page $i$ to page $j$
  - Otherwise, $A_{ij} = 0$

The entire system can be written as:

$$
P = A^T P
$$

So, the PageRank vector $P$ is an **eigenvector** of $A^T$, with eigenvalue 1.

---

### **5. Solving with Power Iteration**

To find $P$, we use the **power iteration method**:

1. Start with an initial guess (say, all pages equally important).
2. Repeatedly compute:

   $$
   P^{(i)} = A P^{(i-1)}
   $$

3. Stop when the values stabilize (change is less than a small threshold $\varepsilon$).

---

### **6. Real-World Challenges**

The above method **assumes** certain mathematical conditions (like connectedness and no "dangling nodes").
But the real web graph doesn't satisfy these, so:

👉 **Later refinements** (not covered here) add fixes like "teleporting" (random jumps) to ensure convergence and robustness.

---
# File: 29.md
---

## **PageRank Algorithm – Explained Simply**

### **1. Web Surfing as a Markov Chain**

- Think of the web as a graph: each **webpage is a node** (or "state"), and each **hyperlink is a connection** (or "transition") between nodes.
- A **random web surfer** clicks on links at random — they don’t use the back button or type new URLs.
- At each page, they choose **one of the available links with equal probability**.

### **2. Transition Matrix (A)**

- This is a square matrix that shows the probability of moving from one page to another.
- If page `i` has `Oi` out-links (links to other pages), then the chance of moving from `i` to page `j` is:

  - `A[i][j] = 1/Oi` if there's a link from `i` to `j`
  - `A[i][j] = 0` if there's no link from `i` to `j`

### **3. Initial Probability Vector (P₀)**

- Represents the chance that a surfer starts on each page.
- Usually initialized with equal probability: e.g., for `n` pages, each page gets `1/n`.

### **4. Iterative Update of PageRank**

At each step, we calculate the new probability distribution `P₁` like this:

```
P₁ = Aᵀ * P₀
```

This means:

- Transpose the transition matrix `A`
- Multiply it by the current probability vector

Do this repeatedly:

```
P₂ = Aᵀ * P₁
...

Pₖ = Aᵀ \* Pₖ₋₁

```

### **5. Stationary Distribution**

- Eventually, the vector stops changing much — it **converges** to a steady-state `Π`.
- This `Π` is the **PageRank vector**: it tells you the long-term probability of ending up on each page.
- Mathematically:

```

Π = Aᵀ \* Π

```

So `Π` is the **principal eigenvector** of the matrix `Aᵀ` with eigenvalue 1.

---

## **Making the Matrix Work for the Real Web**

Real web graphs don’t satisfy the needed conditions for convergence (stochastic, irreducible, aperiodic). So, we **fix the matrix** in two ways:

### **Modification I: Handle Dangling Pages**

- **Dangling pages** have no out-links.
- Fix: Add a link from dangling pages to **every page**, each with equal probability `1/n`.

### **Modification II: Ensure Irreducibility and Aperiodicity**

- Add **a small chance** that the surfer jumps to any page from any page.
- Use a **damping factor** `d` (usually `0.85`) to control this.
- Modified matrix:

```

M = (1 − d)/n _ E + d _ Aᵀ

````

- `E` is a matrix of all 1s
- `(1 − d)/n` means there's always some chance to jump anywhere
- `d * Aᵀ` is the original behavior scaled down

---

## **Power Iteration Algorithm (How PageRank is Computed)**

Given:

- A graph `G` with `n` pages and no dangling vertices
- Damping factor `d` (typically 0.85)
- Tolerance `ε` (small number to stop when change is small)

### **Steps**

1. Initialize `P₀(i) = 1/n` for all pages `i`
2. Repeat:

 - For each page `i`, update:

   ```
   Pₖ(i) = (1 − d)/n + d * Σ [Pₖ₋₁(x)/Oₓ] over all (x → i)
   ```

   - `Oₓ` is the number of out-links from page `x`

3. Stop when `|Pₖ − Pₖ₋₁| ≤ ε` (i.e., small change)
4. Return `Pₖ` — the final PageRank vector

---

## **Additional Notes**

- PageRank doesn’t only apply to the web — it can rank any set of nodes in a graph.
- The algorithm is based on **random walks**, a powerful idea in graph theory.
- Often, we stop early (e.g., after a set number of steps), especially if only **relative ranking** is needed.
- In practice, convergence happens fast. Example: \~52 iterations on 322 million links.

---
````
# File: 30.md
---

### 📊 **Basic Components of Visualization**

- **Aesthetics**: Visual features used to represent data, like position, shape, size, and color.
- **Position**: Where an element is placed (e.g., on x or y axis).
- **Shape**: Can represent categories or groups.
- **Size**: Often used to show quantity (e.g., larger size = more).
- **Color**: Used for grouping or showing values.

### 🗂️ **Types of Data**

- **Quantitative (Numerical)**:

  - Continuous: Any value (e.g., 1.5, 2.6)
  - Discrete: Specific values (e.g., 1, 2, 3)

- **Categorical**:

  - Ordered (e.g., good, fair, poor)
  - Unordered (e.g., dog, cat, fish)

- **Date/Time**: e.g., Jan 5, 2018; 8:03am
- **Text**: Sentences or words.

### 📐 **Coordinate Systems**

- **Cartesian Coordinates**: Regular x-y graph.
- **Nonlinear Axes**: Axes that aren’t straight lines.
- **Curved Axes Systems**: Used for circular or map-like visualizations.

---

### 🎨 **Color Scales**

- Use **qualitative color scales** for categories.
- Recommended sources:

  - **ColorBrewer** – offers light/dark scales.
  - **Okabe-Ito** – colorblind-friendly.
  - **ggplot2 hue** – default in ggplot2.

---

### 📏 **Principles of Visualization**

- Map data values to visual features (aesthetics) using **scales**.
- Scales must be **one-to-one** (each value matches one visual form).
- Visual areas (like bars) should be **proportional** to their values.

  - Bad example: y-axis not starting at 0 can exaggerate differences.

Tips:

- Use **large axis labels**.
- Avoid overly simplistic line drawings.
- Choose suitable visualization tools.

---

### 📈 **Types of Visualizations**

#### 1. **Amount**

- **Bar Charts** – compare quantities.
- **Heat Maps** – color shows value density across a grid (e.g., internet use by year/country).

#### 2. **Distribution**

- **Histogram** – bar plot of value ranges.
- **Density Plot** – smooth line showing distribution.

#### 3. **Proportions**

- **Pie Chart** – show parts of a whole.
- **Side-by-Side Bars** – compare multiple groups.

#### 4. **X-Y Relationships**

- **Scatter Plot** – points show relationship between two variables.
- **Correlograms** – grid of scatter plots to show correlations between multiple pairs.

#### 5. **Geospatial Data**

- **Projections** – map the curved Earth onto flat images (e.g., Mercator).
- **Layers** – multiple data types on one map (e.g., turbine locations on a base map).

#### 6. **Uncertainty**

- **Error Bars** – lines showing the margin of error.
- **Confidence Strips/Bands** – shaded regions showing estimated ranges.

#### 7. **Trends**

- **Smoothing** – use moving averages to show patterns.

  - Example: 20-day, 50-day, 100-day averages.

- **Time Series** – show changes over time.

  - Decomposition: trend + seasonality + random noise.

---

### ❌ **Common Pitfalls in Color Usage**

1. **Too Many Categories**

   - Limit to 3–5 groups.
   - Too many colors confuse the reader.

2. **Non-Monotonic Scales**

   - Avoid rainbow scales – lightness changes unevenly and misleads.

3. **Not Colorblind-Friendly**

   - Some color choices (like rainbow) are unreadable to color-deficient viewers.
   - Use color schemes that are distinguishable by everyone.

---

### 🐍 **Python Visualization Libraries**

- **Matplotlib** – foundational plotting library.
- **Seaborn** – statistical plots, built on Matplotlib.
- **Plotting** – general term.
- **Bokeh** – interactive plots for web.
- **Pygal** – simple interactive SVG charts.
- **Geoplotlib** – mapping/geospatial plotting.

---
