# 02 — Types of Machine Learning

Machine Learning problems can be categorized based on **what kind of data is available, what feedback the model receives, and what the model is expected to learn**.

The major learning paradigms covered in this section are:

1. Supervised Learning
2. Unsupervised Learning
3. Semi-Supervised Learning
4. Self-Supervised Learning
5. Reinforcement Learning

---

# 📚 Table of Contents

* [1. Overview](#1-overview)
* [2. Supervised Learning](#2-supervised-learning)

  * [Regression](#21-regression)
  * [Classification](#22-classification)
* [3. Unsupervised Learning](#3-unsupervised-learning)

  * [Clustering](#31-clustering)
  * [Dimensionality Reduction](#32-dimensionality-reduction)
  * [Anomaly Detection](#33-anomaly-detection)
* [4. Semi-Supervised Learning](#4-semi-supervised-learning)
* [5. Self-Supervised Learning](#5-self-supervised-learning)
* [6. Reinforcement Learning](#6-reinforcement-learning)
* [7. Comparison](#7-comparison)
* [8. How to Identify an ML Problem](#8-how-to-identify-an-ml-problem)
* [9. Real-World Examples](#9-real-world-examples)
* [10. Quick Revision](#10-quick-revision)
* [11. Learning Checklist](#11-learning-checklist)

---

# 1. Overview

A simplified view of Machine Learning paradigms:

```text
                         Machine Learning
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Supervised       Unsupervised    Reinforcement
              │                │
        ┌─────┴─────┐     ┌────┼──────────┐
        ▼           ▼     ▼    ▼          ▼
    Regression Classification Clustering PCA  Anomaly
```

Other important learning approaches include:

```text
Machine Learning
│
├── Supervised Learning
├── Unsupervised Learning
├── Semi-Supervised Learning
├── Self-Supervised Learning
└── Reinforcement Learning
```

These categories are not always completely isolated. Modern ML systems can combine multiple approaches.

---

# 2. Supervised Learning

## Definition

> **Supervised Learning is a type of Machine Learning in which a model learns from labeled training data, where each training example contains both input features and a known target/output.**

The model tries to learn a relationship:

$$
X \rightarrow y
$$

where:

* \(X\) = input features
* \(y\) = actual target

After training:

$$
X_{new} \rightarrow \hat{y}
$$

where:

* \(X_{new}\) = new input
* \(\hat{y}\) = predicted output

---

## Basic Structure

```text
                 Labeled Training Data
                         │
                 ┌───────┴───────┐
                 │               │
              Features         Target
                 X               y
                 │               │
                 └───────┬───────┘
                         ▼
                  Learning Algorithm
                         │
                         ▼
                   Trained Model
                         │
                         ▼
                     New Input
                         │
                         ▼
                    Prediction
```

---

## Example 1 — House Price Prediction

Suppose we have:

| Area | Bedrooms | Age | Price |
| ---: | -------: | --: | ----: |
| 1000 |        2 |  10 |  ₹50L |
| 1500 |        3 |   7 |  ₹75L |
| 2000 |        4 |   5 | ₹100L |
| 1200 |        2 |   8 |  ₹60L |

Here:

### Features

```text
Area
Bedrooms
Age
```

### Target

```text
Price
```

The model learns:

$$
f(\text{Area, Bedrooms, Age}) \rightarrow \text{Price}
$$

Then we provide:

```text
Area = 1800
Bedrooms = 3
Age = 4
```

The model might predict:

```text
Price ≈ ₹90L
```

This is **Supervised Learning → Regression**.

---

# 2.1 Regression

## Definition

> **Regression is a supervised learning task where the model predicts a continuous numerical value.**

Examples:

* House price
* Salary
* Temperature
* Revenue
* Stock-related numerical predictions
* Electricity consumption

### Example

```text
Input:
Area = 1800 sq ft
Bedrooms = 3

       ↓

Regression Model

       ↓

Predicted Price = ₹90L
```

The output is a numerical value rather than a class.

### Common Regression Algorithms

* Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression
* Elastic Net
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting Regression

We will study these in **Phase 2**.

---

# 2.2 Classification

## Definition

> **Classification is a supervised learning task where the model predicts a discrete class or category.**

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Pass / Fail
Cat / Dog
Disease / No Disease
```

---

## Example — Spam Detection

Training data:

| Email                       | Label    |
| --------------------------- | -------- |
| "Win $1000 now!"            | Spam     |
| "Meeting at 3 PM"           | Not Spam |
| "Congratulations, you won!" | Spam     |
| "Please review the report"  | Not Spam |

The model learns patterns associated with spam emails.

For a new email:

```text
"Congratulations! You won a prize!"
```

The model may predict:

```text
Spam
```

This is:

```text
Supervised Learning
        ↓
Classification
```

---

# 2.2.1 Binary Classification

Binary classification has **two possible classes**.

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Pass / Fail
Yes / No
0 / 1
```

Example:

```text
Transaction
     ↓
Classification Model
     ↓
Fraud / Not Fraud
```

---

# 2.2.2 Multiclass Classification

Multiclass classification has **more than two possible classes**.

Example:

```text
Image
 ↓
Model
 ↓
Cat / Dog / Horse
```

Another example is handwritten digit recognition:

```text
0
1
2
3
...
9
```

There are 10 possible classes.

---

# 2.2.3 Multilabel Classification

In multilabel classification, a single example can have **multiple labels simultaneously**.

For example, an image might contain:

```text
Person
Car
Road
Building
```

The model can assign multiple labels to the same image.

```text
Image
  ↓
Model
  ↓
[Person, Car, Road, Building]
```

This is different from multiclass classification, where typically one example is assigned one class.

---

# 3. Unsupervised Learning

## Definition

> **Unsupervised Learning is a type of Machine Learning in which the model learns patterns, structures, or relationships from data without predefined target labels.**

Instead of having:

```text
X + y
```

we primarily have:

```text
X
```

The model attempts to discover useful structure in the data.

---

## Basic Structure

```text
Unlabeled Data
      │
      ▼
ML Algorithm
      │
      ▼
Discovered Patterns
      │
      ├── Groups
      ├── Representations
      └── Anomalies
```

---

# 3.1 Clustering

## Definition

> **Clustering is an unsupervised learning technique that groups data points based on their similarity.**

The important point:

> We don't provide the groups beforehand.

---

## Example — Customer Segmentation

Suppose a company has:

| Age | Annual Spending |
| --: | --------------: |
|  20 |         ₹10,000 |
|  22 |         ₹12,000 |
|  25 |         ₹15,000 |
|  45 |         ₹80,000 |
|  48 |         ₹85,000 |
|  50 |         ₹90,000 |

There is no column saying:

```text
Customer Type
```

A clustering algorithm may discover groups such as:

```text
             Customers
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   Group 1                Group 2
   Low Spending            High Spending
   Younger Customers       Older Customers
```

Possible algorithms:

* K-Means
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Models

---

# 3.2 Dimensionality Reduction

## Definition

> **Dimensionality reduction is the process of representing high-dimensional data using fewer dimensions while attempting to preserve important information or structure.**

Suppose a dataset has:

```text
100 features
```

We may want to represent it using:

```text
2 or 3 dimensions
```

This can help with:

* Visualization
* Noise reduction
* Computational efficiency
* Feature representation

---

## Example

Suppose each customer has:

```text
Age
Income
Spending
Visits
Purchases
Products Viewed
...
```

There may be hundreds of features.

A dimensionality-reduction technique such as **PCA** can create a smaller representation.

```text
100 Features
     ↓
    PCA
     ↓
2 Principal Components
```

### Common techniques

* PCA
* Kernel PCA
* t-SNE
* UMAP

We'll study these in detail in **Phase 4**.

---

# 3.3 Anomaly Detection

## Definition

> **Anomaly detection is the process of identifying observations that differ significantly from the normal or expected pattern of data.**

Examples:

* Credit card fraud
* Network attacks
* Manufacturing defects
* Unusual login activity
* Equipment failures

---

## Example — Credit Card Transactions

Suppose most transactions are:

```text
₹500
₹1200
₹800
₹1500
₹2000
```

Suddenly:

```text
₹5,00,000
```

occurs from an unusual location.

An anomaly-detection system may flag it for investigation.

```text
Normal Transactions
       │
       ▼
Anomaly Detection Model
       │
       ├── Normal
       └── Anomaly
```

Common techniques include:

* Isolation Forest
* One-Class SVM
* Local Outlier Factor

---

# 4. Semi-Supervised Learning

## Definition

> **Semi-Supervised Learning uses a combination of a small amount of labeled data and a large amount of unlabeled data for training.**

This is useful when obtaining labels is expensive or time-consuming.

---

## Example — Image Classification

Suppose we have:

```text
50,000 images
```

but only:

```text
2,000 labeled images
48,000 unlabeled images
```

Manually labeling all 50,000 images may be expensive.

Semi-supervised methods attempt to use both:

```text
2,000 Labeled Images
          +
48,000 Unlabeled Images
          ↓
     Learning System
          ↓
        Model
```

---

## When is Semi-Supervised Learning Useful?

It can be useful when:

* Unlabeled data is abundant.
* Labeled data is expensive.
* Human annotation takes considerable time.
* We want to benefit from both labeled and unlabeled examples.

### Real-world examples

* Medical image classification
* Speech recognition
* Web page classification
* Image recognition
* Document classification

---

# 5. Self-Supervised Learning

## Definition

> **Self-Supervised Learning is a learning approach in which the training signal is automatically created from the data itself rather than being manually provided by humans.**

The data creates a learning task.

This is particularly important in modern AI.

---

# 5.1 Example — Language Models

Suppose the dataset contains:

```text
"The capital of France is Paris."
```

We can create a training task:

```text
Input:
"The capital of France is"

Target:
"Paris"
```

The original data provides the target.

The model learns to predict missing or future information.

```text
Raw Text
   ↓
Create Training Task
   ↓
Input + Automatically Generated Target
   ↓
Model
```

---

# 5.2 Masked Prediction

Another example:

```text
"The cat is sitting on the mat."
```

We can hide part of the sentence:

```text
"The cat is sitting on the [MASK]."
```

The model tries to predict:

```text
mat
```

The original text provides the answer.

This kind of learning has played an important role in training language representation models.

---

# 5.3 Why Self-Supervised Learning Matters

One major advantage is that enormous amounts of raw data can be used without manually labeling every example.

For example:

```text
Internet-scale text
        ↓
Self-Supervised Training
        ↓
Learned Representations
        ↓
Language Model
```

It is an important concept behind modern NLP and foundation-model training.

---

# 6. Reinforcement Learning

## Definition

> **Reinforcement Learning (RL) is a machine learning paradigm in which an agent learns how to make decisions by interacting with an environment and receiving rewards or penalties for its actions.**

Unlike supervised learning, we don't necessarily provide the correct action for every situation.

Instead, the agent learns from feedback.

---

# 6.1 Main Components

Reinforcement Learning has several important components:

### Agent

The learner or decision-maker.

### Environment

The world in which the agent operates.

### State

The current situation of the environment.

### Action

A decision made by the agent.

### Reward

Feedback received after an action.

### Policy

The strategy the agent uses to select actions.

### Value Function

An estimate of how useful or valuable a state or action is in terms of future rewards.

---

# 6.2 Basic RL Loop

```text
             ┌──────────────┐
             │     Agent    │
             └──────┬───────┘
                    │
                  Action
                    │
                    ▼
             ┌──────────────┐
             │ Environment  │
             └──────┬───────┘
                    │
              New State
                    │
                  Reward
                    │
                    ▼
             ┌──────────────┐
             │     Agent    │
             └──────────────┘
```

The cycle repeats:

```text
State
  ↓
Action
  ↓
Environment
  ↓
Reward + New State
  ↓
Learning
  ↓
Next Action
```

---

# 6.3 Example — Game Playing

Imagine an agent learning to play a game.

```text
Current State
     ↓
Agent chooses action
     ↓
Game changes
     ↓
Reward
     ↓
Agent updates its strategy
```

For example:

```text
Winning a game     → +100 reward
Losing a game      → -100 reward
Useful move        → +10 reward
Bad move            → -10 reward
```

Over many interactions, the agent learns a strategy that tends to produce higher cumulative rewards.

---

# 6.4 Example — Robot Navigation

Imagine a robot trying to reach a destination.

```text
Robot
 ↓
Moves forward
 ↓
Gets closer → Positive reward
 ↓
Hits obstacle → Negative reward
 ↓
Learns better actions
```

Eventually, the robot can learn a policy for navigating the environment.

---

# 7. Comparing the Learning Paradigms

| Learning Type       | Data / Feedback          | Target Available?                | Main Goal                         | Example              |
| ------------------- | ------------------------ | -------------------------------- | --------------------------------- | -------------------- |
| **Supervised**      | Labeled data             | Yes                              | Predict target                    | House price          |
| **Unsupervised**    | Unlabeled data           | No                               | Discover structure                | Customer clustering  |
| **Semi-Supervised** | Labeled + unlabeled      | Partially                        | Learn using both                  | Image classification |
| **Self-Supervised** | Data-generated signal    | Generated automatically          | Learn representations/predictions | Language models      |
| **Reinforcement**   | Rewards from interaction | No fixed target for every action | Learn a policy                    | Game playing         |

---

# 8. Supervised vs Unsupervised

This is one of the most important distinctions.

## Supervised Learning

We know the desired output during training.

```text
Input → Known Target
```

Example:

```text
House → ₹80L
Email → Spam
Image → Cat
```

The model learns to predict the target.

---

## Unsupervised Learning

We don't provide the desired output.

```text
Input only
```

Example:

```text
Customer data
     ↓
Algorithm
     ↓
Discovered groups
```

### Easy way to remember

> **Supervised = Learn from answers.**

> **Unsupervised = Find patterns without given answers.**

---

# 9. Classification vs Clustering

These two are frequently confused.

## Classification

Classes are known beforehand.

Training:

```text
Email → Spam
Email → Not Spam
```

The model learns to classify new emails.

## Clustering

Groups are not provided beforehand.

```text
Customer Data
      ↓
Clustering Algorithm
      ↓
Group 1
Group 2
Group 3
```

The algorithm discovers groups based on similarity.

### Remember

```text
Classification → Predefined classes
Clustering    → Discovered groups
```

---

# 10. Regression vs Classification

Both are supervised learning tasks.

| Property           | Regression        | Classification      |
| ------------------ | ----------------- | ------------------- |
| Output             | Numerical value   | Class/category      |
| Example            | ₹80L              | Spam                |
| Target             | Continuous        | Discrete            |
| Example algorithms | Linear Regression | Logistic Regression |
| Question           | "How much?"       | "Which class?"      |

### Easy rule

> **Regression → How much?**

> **Classification → Which class?**

---

# 11. How to Identify the Type of ML Problem

When you receive a new ML problem, use this decision process:

```text
                    ML Problem
                        │
                        ▼
              Do we have target labels?
                   /           \
                 Yes            No
                  │              │
                  ▼              ▼
             Supervised     Unsupervised
                  │              │
           ┌──────┴──────┐   ┌───┴──────────┐
           │             │   │              │
           ▼             ▼   ▼              ▼
       Numerical      Category Clustering  Other
        Target         Target
           │             │
           ▼             ▼
       Regression    Classification
```

For reinforcement learning:

```text
Agent interacting with environment
          +
      Rewards
          ↓
Reinforcement Learning
```

---

# 12. Real-World Examples

## Example 1 — House Price Prediction

```text
Input:
Area
Bedrooms
Location

Output:
Price
```

Type:

**Supervised → Regression**

---

## Example 2 — Spam Detection

```text
Input:
Email content

Output:
Spam / Not Spam
```

Type:

**Supervised → Classification**

---

## Example 3 — Customer Segmentation

```text
Input:
Customer behavior

Output:
Discovered customer groups
```

Type:

**Unsupervised → Clustering**

---

## Example 4 — Fraud Detection

Depending on the system, fraud detection can be approached in different ways.

With historical fraud labels:

```text
Transaction → Fraud / Not Fraud
```

→ **Supervised Classification**

Without reliable labels:

```text
Transaction
     ↓
Find unusual behavior
```

→ **Unsupervised / Anomaly Detection**

This is an important reminder:

> The same real-world problem can sometimes be approached using different ML paradigms depending on the available data and objective.

---

## Example 5 — Image Classification with Limited Labels

```text
1,000 labeled images
+
100,000 unlabeled images
```

→ **Semi-Supervised Learning**

---

## Example 6 — Language Model Training

```text
Large text dataset
       ↓
Predict missing/next tokens
       ↓
Learn language representations
```

→ **Self-Supervised Learning**

---

## Example 7 — Game Playing

```text
Agent
 ↓
Action
 ↓
Game
 ↓
Reward
 ↓
Learning
```

→ **Reinforcement Learning**

---

# 13. Important Terminology

### Label

A known target associated with a training example, especially in supervised learning.

### Unlabeled Data

Data for which the desired target is not provided.

### Reward

Feedback provided to an RL agent after taking an action.

### Policy

A strategy that determines what action an RL agent should take in a given state.

### Clustering

Grouping similar data points.

### Representation

A useful numerical or structured form of data that captures important information.

### Continuous Value

A numerical quantity that can take values across a range.

Examples:

```text
₹50,000
₹50,000.50
₹50,000.75
```

### Discrete Class

A category rather than a continuous numerical quantity.

Examples:

```text
Cat
Dog
Horse
```

---

# 14. Common Mistakes

### Mistake 1: "All classification is unsupervised."

❌ Incorrect.

Classification is generally a **supervised learning** task because the classes are known during training.

---

### Mistake 2: "Clustering is classification."

❌ Incorrect.

Classification:

```text
Known classes → Predict class
```

Clustering:

```text
No predefined classes → Discover groups
```

---

### Mistake 3: "Unsupervised learning means the model doesn't learn."

❌ Incorrect.

The model still learns patterns; it simply does not learn from explicitly provided target labels.

---

### Mistake 4: "Reinforcement Learning is just classification."

❌ Incorrect.

RL is based on **sequential interaction, actions, states, and rewards**.

---

### Mistake 5: "Self-supervised means no labels are used."

This needs nuance.

There may be no **human-provided labels**, but the training objective creates targets/signals automatically from the data.

---

# 15. Quick Revision

## Supervised Learning

> Learns from labeled data.

```text
X + y → Model
```

Main tasks:

```text
Regression
Classification
```

---

## Unsupervised Learning

> Finds patterns in unlabeled data.

```text
X → Patterns / Structure
```

Main tasks:

```text
Clustering
Dimensionality Reduction
Anomaly Detection
```

---

## Semi-Supervised Learning

> Uses both labeled and unlabeled data.

```text
Small Labeled Dataset
          +
Large Unlabeled Dataset
          ↓
        Model
```

---

## Self-Supervised Learning

> Creates training signals automatically from the data itself.

```text
Raw Data
   ↓
Automatically Generated Target
   ↓
Model
```

---

## Reinforcement Learning

> Learns by interacting with an environment and receiving rewards.

```text
State
 ↓
Action
 ↓
Environment
 ↓
Reward
 ↓
Learning
```

---

# 🧠 One-Minute Revision Table

| Type                         | Remember This                        |
| ---------------------------- | ------------------------------------ |
| **Supervised**               | Learn from labeled examples          |
| **Regression**               | Predict a number                     |
| **Classification**           | Predict a class                      |
| **Unsupervised**             | Discover patterns                    |
| **Clustering**               | Discover groups                      |
| **Dimensionality Reduction** | Reduce dimensions                    |
| **Anomaly Detection**        | Find unusual data                    |
| **Semi-Supervised**          | Labeled + unlabeled data             |
| **Self-Supervised**          | Data creates its own learning signal |
| **Reinforcement**            | Learn using rewards                  |

---

# 🎯 Key Formulas / Representations

### Supervised Learning

$$
X \rightarrow y
$$

Prediction:

$$
\hat{y}=f(X)
$$

### Regression

$$
X \rightarrow \text{Continuous Value}
$$

### Classification

$$
X \rightarrow \text{Class}
$$

### Unsupervised Learning

$$
X \rightarrow \text{Discovered Structure}
$$

### Reinforcement Learning

$$
State \rightarrow Action \rightarrow Reward
$$

---

# 📝 Practice Questions

### Q1

Predicting the salary of an employee based on experience, education, and job role.

**Which type of ML?**

---

### Q2

Predicting whether a bank transaction is fraudulent using historical transactions labeled as fraud/not fraud.

**Which type?**

---

### Q3

Grouping customers based on purchasing behavior when no customer categories are provided.

**Which type?**

---

### Q4

Reducing 100 features into 10 principal components.

**Which ML task?**

---

### Q5

Using 1,000 labeled images and 100,000 unlabeled images to train an image classifier.

**Which learning paradigm?**

---

### Q6

A language model learns by predicting the next token in a sentence using raw text.

**Which learning paradigm?**

---

### Q7

An agent learns to play chess by receiving rewards based on game outcomes.

**Which learning paradigm?**

---

# ✅ Learning Checklist

* [ ] Understand Supervised Learning
* [ ] Understand Regression
* [ ] Understand Classification
* [ ] Understand Binary Classification
* [ ] Understand Multiclass Classification
* [ ] Understand Multilabel Classification
* [ ] Understand Unsupervised Learning
* [ ] Understand Clustering
* [ ] Understand Dimensionality Reduction
* [ ] Understand Anomaly Detection
* [ ] Understand Semi-Supervised Learning
* [ ] Understand Self-Supervised Learning
* [ ] Understand Reinforcement Learning
* [ ] Understand Agent, Environment, State, Action, Reward
* [ ] Understand Classification vs Clustering
* [ ] Understand Regression vs Classification
* [ ] Know how to identify an ML problem

---

# ⏭️ Next Topic

## 03 — Machine Learning Terminology

We will cover:

* Dataset
* Samples / Instances
* Features
* Labels
* Targets
* Parameters
* Hyperparameters
* Model
* Algorithm
* Training
* Validation
* Testing
* Epoch
* Batch
* Iteration
* Inference
* Predictions
* Ground Truth

These concepts will become the **vocabulary you use throughout the rest of Machine Learning**.

