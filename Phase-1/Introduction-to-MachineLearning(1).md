# 01 — Introduction to Machine Learning

## 📖 Overview

Machine Learning (ML) is a branch of **Artificial Intelligence (AI)** that enables computers to learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed with rules for every situation.

Instead of manually defining every rule, we provide data and allow an ML algorithm to learn a useful relationship from that data.

### Basic ML Process

```text
Training Data
     ↓
ML Algorithm
     ↓
Trained Model
     ↓
New / Unseen Data
     ↓
Prediction
```

---

# 1. Artificial Intelligence (AI)

**Artificial Intelligence** is the broader field of computer science concerned with building systems that can perform tasks that normally require intelligent behavior.

Examples:

* Reasoning
* Problem solving
* Decision making
* Planning
* Understanding language
* Computer vision
* Learning

AI can include approaches that do not necessarily involve machine learning.

---

# 2. Machine Learning (ML)

**Machine Learning** is a subset of AI in which computers learn patterns or relationships from data and use the learned patterns to make predictions or decisions on new data.

### Example

Suppose we have historical house data:

|       Area | Bedrooms | Price |
| ---------: | -------: | ----: |
| 1000 sq ft |        2 |  ₹50L |
| 1500 sq ft |        3 |  ₹75L |
| 2000 sq ft |        4 | ₹100L |

An ML algorithm can learn the relationship between the input information and house prices.

We can then provide a new house:

```text
Area = 1800 sq ft
Bedrooms = 3
```

and the trained model can predict its price.

---

# 3. Deep Learning (DL)

**Deep Learning** is a subset of Machine Learning that primarily uses neural networks with multiple layers to learn complex patterns from large amounts of data.

Deep Learning is widely used in:

* Image recognition
* Speech recognition
* Natural Language Processing
* Object detection
* Generative AI
* Large Language Models

### Relationship

```text
Artificial Intelligence
        │
        └── Machine Learning
                 │
                 └── Deep Learning
```

---

# 4. Traditional Programming vs Machine Learning

## Traditional Programming

In traditional programming, humans explicitly write the rules.

```text
Rules + Data
     ↓
  Program
     ↓
  Output
```

Example:

```python
if marks >= 40:
    result = "Pass"
else:
    result = "Fail"
```

The programmer explicitly defines the rule.

---

## Machine Learning

In ML, instead of manually writing all the rules, we provide examples and allow an algorithm to learn patterns.

```text
Data + Expected Outputs
          ↓
    ML Algorithm
          ↓
        Model
          ↓
     New Data
          ↓
     Prediction
```

---

# 5. Dataset

A **dataset** is a collection of data used for analysis, training, validation, or testing of a machine learning system.

Example:

| Age | Experience | Salary |
| --: | ---------: | -----: |
|  22 |          1 |   3.5L |
|  25 |          3 |     6L |
|  30 |          7 |    12L |

Each row represents an individual example.

---

# 6. Sample / Instance / Observation

A **sample** is one individual example or observation in a dataset.

For example:

```text
25 years old
3 years experience
₹6L salary
```

is one sample.

The terms **sample**, **instance**, and **observation** are commonly used to describe individual data points.

---

# 7. Features

A **feature** is an input variable or measurable property used by a machine learning model to make a prediction.

For the salary example:

```text
Age
Years of Experience
```

are features.

We commonly represent features using:

$$
X
$$

For one observation:

$$
X = [25, 3]
$$

---

# 8. Target / Label

The **target** is the output that a model is trying to predict.

In supervised learning, the target may also be called the **label**.

For the salary example:

```text
Salary
```

is the target.

We commonly represent the target as:

$$
y
$$

So:

```text
Features → X
Target   → y
```

---

# 9. Algorithm

An **ML algorithm** is a mathematical or computational procedure used to learn patterns from data.

Examples:

* Linear Regression
* Logistic Regression
* Decision Tree
* K-Nearest Neighbors
* Support Vector Machine
* K-Means

An algorithm defines **how learning is performed**.

---

# 10. Model

A **model** is the learned representation of patterns or relationships obtained after training an ML algorithm on data.

For example:

```text
Training Data
      ↓
Linear Regression Algorithm
      ↓
Trained Linear Regression Model
```

The trained model can then be used to make predictions.

### Important Difference

```text
Algorithm → Method used to learn
Model     → Result learned from data
```

---

# 11. Training

**Training** is the process of using data to learn the parameters or patterns of a machine learning model.

A simplified training process:

```text
Training Data
     ↓
Model makes prediction
     ↓
Prediction compared with actual value
     ↓
Error calculated
     ↓
Model updated
     ↓
Repeat
```

The exact process depends on the algorithm.

---

# 12. Prediction

A **prediction** is the output produced by a trained machine learning model when given input data.

For example:

```text
Input:
Area = 1800 sq ft
Bedrooms = 3

        ↓

Trained ML Model

        ↓

Prediction:
₹90L
```

The predicted value is commonly represented as:

$$
\hat{y}
$$

where:

* \(y\) = actual value
* \(\hat{y}\) = predicted value

---

# 13. Inference

**Inference** is the process of using a trained model to generate predictions for new data.

```text
New Data
   ↓
Trained Model
   ↓
Prediction
```

Training and inference are different stages:

```text
Training:
Data → Learning → Model

Inference:
New Data + Model → Prediction
```

---

# 14. Learning a Function

At a high level, Machine Learning attempts to learn a function that maps inputs to outputs.

$$
f(X) \approx y
$$

After training, the model can use new input \(X\) to produce a prediction:

$$
\hat{y} = f(X)
$$

Example:

$$
f(\text{Area, Bedrooms, Location}) \rightarrow \text{Price}
$$

---

# 15. Basic Machine Learning Workflow

A typical ML project follows these steps:

```text
1. Define Problem
       ↓
2. Collect Data
       ↓
3. Explore Data
       ↓
4. Clean Data
       ↓
5. Prepare Features
       ↓
6. Split Data
       ↓
7. Select Model
       ↓
8. Train Model
       ↓
9. Evaluate Model
       ↓
10. Tune Model
       ↓
11. Deploy
       ↓
12. Monitor
```

We will study each of these steps in detail throughout the roadmap.

---

# 16. Why Machine Learning?

Machine Learning is useful when:

* Rules are difficult to write manually.
* There are complex patterns in the data.
* Large amounts of data are available.
* Patterns can be learned from examples.
* The system needs to make predictions on new data.

### Common Applications

| Application            | ML Task                            |
| ---------------------- | ---------------------------------- |
| Spam Detection         | Classification                     |
| House Price Prediction | Regression                         |
| Customer Segmentation  | Clustering                         |
| Fraud Detection        | Classification / Anomaly Detection |
| Recommendation Systems | Recommendation                     |
| Image Recognition      | Classification                     |
| Speech Recognition     | Deep Learning                      |
| Language Models        | Deep Learning / NLP                |

---

# 17. Important Terminology

| Term              | Definition                                                      |
| ----------------- | --------------------------------------------------------------- |
| **AI**            | Broad field of creating systems capable of intelligent behavior |
| **ML**            | Subset of AI that learns patterns from data                     |
| **Deep Learning** | ML based primarily on multi-layer neural networks               |
| **Dataset**       | Collection of examples                                          |
| **Sample**        | One individual observation                                      |
| **Feature**       | Input variable used for prediction                              |
| **Target**        | Output the model tries to predict                               |
| **Label**         | Target value, especially in supervised learning                 |
| **Algorithm**     | Procedure used to learn patterns                                |
| **Model**         | Learned representation obtained from training                   |
| **Training**      | Learning from data                                              |
| **Prediction**    | Output generated by a model                                     |
| **Inference**     | Using a trained model to make predictions                       |

---

# 🧠 Quick Revision

### Remember these relationships

```text
AI
└── ML
    └── Deep Learning
```

### Traditional Programming

$$
Rules + Data \rightarrow Output
$$

### Machine Learning

$$
Data + Algorithm \rightarrow Model
$$

Then:

$$
New\ Data + Model \rightarrow Prediction
$$

### Core notation

$$
X = Features
$$

$$
y = Actual\ Target
$$

$$
\hat{y} = Predicted\ Target
$$

### Core idea

> **Machine Learning learns patterns from data instead of requiring humans to explicitly program every rule.**

---

# ✅ Learning Checklist

* [ ] Understand AI
* [ ] Understand Machine Learning
* [ ] Understand Deep Learning
* [ ] Understand AI vs ML vs DL
* [ ] Understand traditional programming vs ML
* [ ] Understand datasets
* [ ] Understand samples
* [ ] Understand features
* [ ] Understand targets / labels
* [ ] Understand algorithms
* [ ] Understand models
* [ ] Understand training
* [ ] Understand prediction
* [ ] Understand inference
* [ ] Understand the basic ML workflow

---

## ⏭️ Next Topic

**02 — Types of Machine Learning**

* Supervised Learning
* Unsupervised Learning
* Semi-Supervised Learning
* Self-Supervised Learning
* Reinforcement Learning

