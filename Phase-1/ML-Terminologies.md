# 03 — Machine Learning Terminology

Machine Learning has a vocabulary that appears repeatedly across algorithms, research papers, documentation, interviews, and real-world projects.

Understanding these terms clearly is essential before moving into model training, loss functions, optimization, and algorithms.

---

# 📚 Table of Contents

* [1. Dataset](#1-dataset)
* [2. Data Point / Sample / Instance](#2-data-point--sample--instance)
* [3. Features](#3-features)
* [4. Target](#4-target)
* [5. Label](#5-label)
* [6. Input and Output](#6-input-and-output)
* [7. Feature Matrix](#7-feature-matrix)
* [8. Target Vector](#8-target-vector)
* [9. Algorithm](#9-algorithm)
* [10. Model](#10-model)
* [11. Parameters](#11-parameters)
* [12. Hyperparameters](#12-hyperparameters)
* [13. Training](#13-training)
* [14. Validation](#14-validation)
* [15. Testing](#15-testing)
* [16. Prediction](#16-prediction)
* [17. Ground Truth](#17-ground-truth)
* [18. Inference](#18-inference)
* [19. Batch](#19-batch)
* [20. Epoch](#20-epoch)
* [21. Iteration](#21-iteration)
* [22. Training Example](#22-training-example)
* [23. Generalization](#23-generalization)
* [24. Model Evaluation](#24-model-evaluation)
* [25. Complete ML Terminology Example](#25-complete-ml-terminology-example)
* [26. Commonly Confused Terms](#26-commonly-confused-terms)
* [27. Quick Revision](#27-quick-revision)
* [28. Learning Checklist](#28-learning-checklist)

---

# 1. Dataset

## Definition

> A **dataset** is a collection of data examples used to train, validate, test, or analyze a machine learning model.

A dataset usually contains multiple rows and columns.

Example:

| Age | Experience | Education | Salary |
| --: | ---------: | --------- | -----: |
|  22 |          1 | B.Tech    |     4L |
|  25 |          3 | B.Tech    |     7L |
|  30 |          7 | M.Tech    |    12L |
|  35 |         10 | MBA       |    18L |

Here, the complete table is the **dataset**.

---

## Dataset Components

A dataset can contain:

```text
Dataset
│
├── Samples / Rows
│
├── Features / Input Columns
│
└── Target / Output Column
```

For supervised learning:

```text
Dataset
   │
   ├── X → Features
   │
   └── y → Target
```

---

# 2. Data Point / Sample / Instance

## Definition

A **sample**, **instance**, or **data point** is one individual observation in a dataset.

Consider:

| Age | Experience | Salary |
| --: | ---------: | -----: |
|  22 |          1 |     4L |
|  25 |          3 |     7L |
|  30 |          7 |    12L |

The first row:

```text
22, 1, 4L
```

is one sample.

The dataset contains **3 samples**.

---

## Different Names

Depending on the context, you may see:

* Sample
* Instance
* Observation
* Data point
* Example

These often refer to an individual example.

---

# 3. Features

## Definition

> A **feature** is an input variable or measurable property used by a machine learning model to make predictions.

Example:

| Age | Experience | Salary |
| --: | ---------: | -----: |
|  22 |          1 |     4L |
|  25 |          3 |     7L |
|  30 |          7 |    12L |

If we're predicting salary:

```text
Age
Experience
```

are the features.

The target is:

```text
Salary
```

---

## Feature Representation

Features are commonly represented using:

$$
X
$$

For one employee:

$$
X = [25,3]
$$

where:

```text
25 → Age
3  → Experience
```

---

# 4. Target

## Definition

> The **target** is the value that a supervised learning model is trained to predict.

It is often represented as:

$$
y
$$

Example:

```text
Features:
Age
Experience
Education

Target:
Salary
```

So:

$$
X = \text{Features}
$$

$$
y = \text{Target}
$$

---

## Examples of Targets

| Problem                | Target               |
| ---------------------- | -------------------- |
| House price prediction | Price                |
| Spam detection         | Spam / Not Spam      |
| Student result         | Pass / Fail          |
| Disease prediction     | Disease / No Disease |
| Sales forecasting      | Sales                |
| Image classification   | Image class          |

---

# 5. Label

## Definition

> A **label** is the known answer associated with a training example, particularly in supervised learning.

For classification:

```text
Image → Cat
```

`Cat` is the label.

For spam detection:

```text
Email → Spam
```

`Spam` is the label.

---

## Target vs Label

The terms are often used interchangeably, but there is a useful distinction:

### Target

A general term for the value being predicted.

### Label

Often used specifically for categorical/classification outputs.

Example:

```text
Regression:
Target = House Price

Classification:
Label = Cat / Dog
```

However, ML libraries and practitioners may use these terms differently depending on context.

---

# 6. Input and Output

Machine Learning can be viewed as a mapping from inputs to outputs.

```text
Input
  ↓
Model
  ↓
Output
```

Mathematically:

$$
X \rightarrow f(X) \rightarrow \hat{y}
$$

where:

* \(X\) = input features
* \(f\) = learned function/model
* \(\hat{y}\) = prediction

Example:

```text
Input:
Area = 1500
Bedrooms = 3

        ↓

Model

        ↓

Output:
₹75L
```

---

# 7. Feature Matrix

When we have multiple samples and multiple features, the input data can be represented as a **feature matrix**.

Suppose:

| Age | Experience |
| --: | ---------: |
|  22 |          1 |
|  25 |          3 |
|  30 |          7 |
|  35 |         10 |

The feature matrix is:

$$
X =
\begin{bmatrix}
22 & 1\\
25 & 3\\
30 & 7\\
35 & 10
\end{bmatrix}
$$

Here:

* 4 rows = 4 samples
* 2 columns = 2 features

Therefore:

$$
X \in \mathbb{R}^{4 \times 2}
$$

---

## General Form

If a dataset contains:

* \(n\) samples
* \(m\) features

then:

$$
X \in \mathbb{R}^{n \times m}
$$

This notation becomes very important later.

---

# 8. Target Vector

The target values are commonly represented as a vector.

For the previous example:

| Age | Experience | Salary |
| --: | ---------: | -----: |
|  22 |          1 |     4L |
|  25 |          3 |     7L |
|  30 |          7 |    12L |
|  35 |         10 |    18L |

The target vector is:

$$
y =
\begin{bmatrix}
4\\
7\\
12\\
18
\end{bmatrix}
$$

Therefore:

$$
y \in \mathbb{R}^{4}
$$

---

# 9. Algorithm

## Definition

> An **ML algorithm** is a procedure or method used to learn patterns or relationships from data.

Examples:

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* K-Means
* Support Vector Machine
* Neural Networks

The algorithm describes **how the learning process happens**.

---

## Example

```text
Training Data
      ↓
Linear Regression Algorithm
      ↓
Learn Parameters
      ↓
Trained Model
```

---

# 10. Model

## Definition

> A **model** is the learned mathematical representation of patterns or relationships obtained after training an algorithm on data.

Example:

```text
Training Data
      ↓
Linear Regression Algorithm
      ↓
Trained Linear Regression Model
```

The model can then make predictions.

---

## Algorithm vs Model

This distinction is extremely important.

```text
Algorithm = How learning happens

Model = What was learned
```

Example:

```text
Algorithm:
Linear Regression

Training:
Historical house data

Model:
Learned relationship between
house features and price
```

---

# 11. Parameters

## Definition

> **Parameters are values learned by the model from the training data.**

For Linear Regression:

$$
\hat{y}=w_1x_1+w_2x_2+b
$$

The model learns:

* \(w_1\)
* \(w_2\)
* \(b\)

These are parameters.

---

## Example

Suppose the model learns:

$$
\hat{y}=2x+5
$$

Here:

```text
w = 2
b = 5
```

are learned parameters.

The algorithm determines these values during training.

---

# 12. Hyperparameters

## Definition

> **Hyperparameters are configuration values chosen before or during the training process that control how a machine learning algorithm learns.**

Unlike parameters, hyperparameters are generally **not learned directly from the training data in the ordinary training process**.

Examples:

### K-Nearest Neighbors

```text
k = 5
```

### Decision Tree

```text
max_depth = 10
```

### Neural Network

```text
learning_rate = 0.001
batch_size = 32
epochs = 20
```

### Random Forest

```text
n_estimators = 100
max_depth = 10
```

---

# 13. Parameters vs Hyperparameters

| Parameters                 | Hyperparameters                                  |
| -------------------------- | ------------------------------------------------ |
| Learned from training data | Set/configured before or during training         |
| Part of the learned model  | Control the learning process/model configuration |
| Example: weights           | Example: learning rate                           |
| Example: bias              | Example: tree depth                              |
| Updated during training    | Usually selected through configuration/tuning    |

### Easy way to remember

> **Parameters are learned.**

> **Hyperparameters control learning.**

---

# 14. Training

## Definition

> **Training is the process of using data to learn the parameters of a machine learning model.**

Simplified:

```text
Training Data
     ↓
Model
     ↓
Prediction
     ↓
Error
     ↓
Update Parameters
     ↓
Repeat
```

The exact process depends on the algorithm.

---

# 15. Validation

## Definition

> **Validation is the process of evaluating a model during development using data that is separate from the training data, typically to help select models or tune hyperparameters.**

Suppose we have:

```text
Training Data
Validation Data
Test Data
```

The model learns from:

```text
Training Data
```

We use validation data to help make development decisions such as:

* Choosing between models
* Selecting hyperparameters
* Comparing configurations
* Detecting overfitting during development

---

# 16. Testing

## Definition

> **Testing is the process of evaluating the final trained model on previously unseen data that was kept separate from training and model-development decisions.**

The test set provides an estimate of how the final model performs on unseen data.

---

# 17. Training vs Validation vs Test

A typical workflow:

```text
Complete Dataset
       │
       ├──────────────┐
       │              │
       ▼              ▼
   Training       Validation
       │              │
       └──────┬───────┘
              ▼
        Final Model
              │
              ▼
           Test Set
              │
              ▼
       Final Evaluation
```

For example:

```text
100,000 samples

70,000 → Training
15,000 → Validation
15,000 → Testing
```

The exact proportions depend on the problem.

---

# 18. Prediction

## Definition

> A **prediction** is the output produced by a trained model for a given input.

Example:

```text
Input:
Experience = 5 years

        ↓

Model

        ↓

Prediction:
₹10L
```

Mathematically:

$$
\hat{y}=f(X)
$$

---

# 19. Ground Truth

## Definition

> **Ground truth is the actual or trusted value against which a model's prediction can be compared.**

Example:

```text
Actual House Price = ₹80L
Model Prediction   = ₹75L
```

Then:

```text
Ground Truth = ₹80L
Prediction   = ₹75L
```

The difference can be used to calculate an error or loss.

---

# 20. Inference

## Definition

> **Inference is the process of using a trained model to generate predictions for new data.**

Training:

```text
Training Data
     ↓
Learning
     ↓
Model
```

Inference:

```text
New Data
     ↓
Trained Model
     ↓
Prediction
```

---

# 21. Batch

## Definition

> A **batch** is a group of training examples processed together during one training step.

Suppose we have:

```text
10,000 training samples
```

and:

```text
batch_size = 100
```

The model processes:

```text
Batch 1 → Samples 1–100
Batch 2 → Samples 101–200
...
Batch 100 → Samples 9901–10000
```

---

## Why Use Batches?

Processing the entire dataset at once may require too much memory.

Processing one sample at a time can be inefficient.

A batch provides a compromise.

---

# 22. Epoch

## Definition

> An **epoch** is one complete pass through the entire training dataset.

Suppose:

```text
Dataset = 10,000 samples
Batch size = 100
```

Then:

$$
\text{Iterations per epoch}=
\frac{10000}{100}=100
$$

Therefore:

```text
1 Epoch = 100 batches/iterations
```

If we train for:

```text
10 epochs
```

the model processes the training dataset 10 times.

---

# 23. Iteration

## Definition

> An **iteration** generally refers to one parameter-update step, often corresponding to processing one batch.

Example:

```text
10,000 samples
Batch size = 100
```

One epoch contains:

```text
100 iterations
```

If training lasts for 5 epochs:

$$
5 \times 100 = 500
$$

iterations.

---

# 24. Epoch vs Batch vs Iteration

Suppose:

```text
Dataset = 10,000 samples
Batch size = 100
Epochs = 5
```

Then:

```text
1 Batch = 100 samples

1 Iteration ≈ processing 1 batch + updating parameters

1 Epoch = 100 iterations

5 Epochs = 500 iterations
```

### Visual

```text
Dataset
│
├── Batch 1
├── Batch 2
├── Batch 3
├── ...
└── Batch 100
        ↓
      1 Epoch

Repeat 5 times
        ↓
      5 Epochs
```

---

# 25. Training Example

A **training example** is an individual example used during model training.

For example:

```text
Features:
Age = 25
Experience = 3

Target:
Salary = ₹7L
```

This complete input-output pair is one training example.

---

# 26. Generalization

## Definition

> **Generalization is the ability of a trained model to perform well on new, unseen data rather than only memorizing the training data.**

This is one of the most important goals of Machine Learning.

A model that performs well only on training data is not necessarily useful.

We want:

```text
Training Data
      ↓
Learn General Patterns
      ↓
Unseen Data
      ↓
Good Predictions
```

We'll study this deeply when we cover:

* Overfitting
* Underfitting
* Bias
* Variance
* Regularization

---

# 27. Model Evaluation

## Definition

> **Model evaluation is the process of measuring how well a machine learning model performs using appropriate evaluation metrics.**

Different tasks require different metrics.

### Regression

Common metrics:

* MAE
* MSE
* RMSE
* R²

### Classification

Common metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

We will study these in detail in **Phase 3**.

---

# 28. Complete ML Terminology Example

Let's put everything together using a house-price problem.

Suppose our dataset is:

| Area | Bedrooms | Age | Price |
| ---: | -------: | --: | ----: |
| 1000 |        2 |  10 |   50L |
| 1500 |        3 |   7 |   75L |
| 2000 |        4 |   5 |  100L |
| 1200 |        2 |   8 |   60L |

### Dataset

The entire table.

### Samples

There are 4 samples.

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

### Feature Matrix

$$
X=
\begin{bmatrix}
1000&2&10\\
1500&3&7\\
2000&4&5\\
1200&2&8
\end{bmatrix}
$$

### Target Vector

$$
y=
\begin{bmatrix}
50\\
75\\
100\\
60
\end{bmatrix}
$$

### Algorithm

Suppose we choose:

```text
Linear Regression
```

### Parameters

The model learns coefficients and an intercept.

### Hyperparameters

Depending on the algorithm, configuration choices may be involved.

### Training

The model learns from the training data.

### Validation

Used during development to compare configurations or tune hyperparameters.

### Testing

Used for final evaluation on held-out data.

### Prediction

For:

```text
Area = 1800
Bedrooms = 3
Age = 4
```

the model might produce:

```text
Predicted Price = ₹90L
```

### Ground Truth

Suppose the actual selling price is:

```text
₹92L
```

Then:

```text
Prediction   = ₹90L
Ground Truth = ₹92L
```

The difference contributes to the model's error.

---

# 29. Complete ML Pipeline Using Terminology

```text
                    DATASET
                       │
                       ▼
              ┌────────────────┐
              │    Samples     │
              └───────┬────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
           Features          Target
              X                y
              │                │
              └───────┬────────┘
                      ▼
                 Split Data
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Training    Validation     Test
          │           │
          ▼           │
      Algorithm       │
          │           │
          ▼           │
       Training       │
          │           │
          ▼           │
     Learned Model ←──┘
          │
          ▼
    Hyperparameter/
    Model Decisions
          │
          ▼
      Final Model
          │
          ▼
       Test Data
          │
          ▼
      Evaluation
          │
          ▼
        Deploy
          │
          ▼
       Inference
          │
          ▼
      Predictions
```

---

# 30. Commonly Confused Terms

## Feature vs Target

```text
Feature → Input
Target  → Output we want to predict
```

Example:

```text
Area + Bedrooms → Price
Features            Target
```

---

## Parameter vs Hyperparameter

```text
Parameter:
Learned by the model

Hyperparameter:
Configuration used to control training/model behavior
```

---

## Training vs Inference

```text
Training:
Learn from data

Inference:
Use learned model to make predictions
```

---

## Validation vs Testing

```text
Validation:
Used during model development

Testing:
Used for final evaluation
```

---

## Batch vs Epoch

```text
Batch:
Group of training samples processed together

Epoch:
One complete pass through the training dataset
```

---

## Algorithm vs Model

```text
Algorithm:
Method for learning

Model:
Learned result
```

---

## Target vs Prediction

```text
Target:
Actual value

Prediction:
Model's estimated value
```

---

# 31. Important Mathematical Notation

Machine Learning uses a small set of symbols repeatedly.

### Input Features

$$
X
$$

### Target

$$
y
$$

### Prediction

$$
\hat{y}
$$

### Model

$$
f
$$

### General Prediction

$$
\hat{y}=f(X)
$$

### Training Dataset

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
$$

where:

* \(D\) = dataset
* \(n\) = number of samples
* \(x_i\) = features of sample \(i\)
* \(y_i\) = target of sample \(i\)

---

# 32. Example of the Mathematical Notation

Suppose:

```text
Age = 25
Experience = 3
Salary = ₹7L
```

Then:

$$
x_i=[25,3]
$$

and:

$$
y_i=7
$$

If the model predicts:

$$
\hat{y}_i=6.5
$$

then:

```text
Actual = 7
Prediction = 6.5
```

The difference between these values will later be used to define an error/loss.

---

# 33. Quick Revision

## Dataset

Collection of examples.

## Sample

One individual observation.

## Feature

Input variable.

## Target

Value the model tries to predict.

## Label

Known target/output, commonly used in classification.

## Algorithm

Procedure used to learn from data.

## Model

Learned representation produced by training.

## Parameter

Value learned by the model.

## Hyperparameter

Configuration value used to control the learning process/model.

## Training

Learning model parameters from training data.

## Validation

Evaluating during development to guide model/hyperparameter decisions.

## Testing

Final evaluation on unseen held-out data.

## Prediction

Output generated by a trained model.

## Ground Truth

Actual/trusted value used for comparison.

## Inference

Using a trained model to make predictions.

## Batch

Group of samples processed together.

## Iteration

Typically one parameter-update step, often based on one batch.

## Epoch

One complete pass through the training dataset.

## Generalization

Ability to perform well on unseen data.

---

# 34. One-Minute Revision Table

| Term           | Simple Meaning                   |
| -------------- | -------------------------------- |
| Dataset        | Collection of data               |
| Sample         | One row/example                  |
| Feature        | Input                            |
| Target         | Desired output                   |
| Label          | Known target, often a class      |
| Algorithm      | Learning procedure               |
| Model          | Learned result                   |
| Parameter      | Learned value                    |
| Hyperparameter | Configuration value              |
| Training       | Learning from data               |
| Validation     | Development-time evaluation      |
| Test           | Final evaluation                 |
| Prediction     | Model output                     |
| Ground Truth   | Actual value                     |
| Inference      | Prediction using trained model   |
| Batch          | Group of samples                 |
| Iteration      | One training/update step         |
| Epoch          | One full pass over training data |
| Generalization | Performance on unseen data       |

---

# 🧠 Must-Remember Relationships

```text
Dataset
   │
   ├── Samples
   │
   ├── Features (X)
   │
   └── Target (y)
```

```text
Algorithm
    ↓
Training
    ↓
Parameters Learned
    ↓
Model
```

```text
Training Data
    ↓
Model Development
    ↓
Validation Data
    ↓
Final Model
    ↓
Test Data
    ↓
Final Evaluation
```

```text
Batch × Iterations = Approximately one Epoch
```

More precisely, if the number of training samples is \(N\) and batch size is \(B\):

$$
\text{Iterations per epoch}=
\left\lceil\frac{N}{B}\right\rceil
$$

---

# ⚠️ Important Notes

### 1. Not every ML problem has a target

Supervised learning requires targets, but unsupervised learning generally works without predefined targets.

### 2. Parameters and hyperparameters are different

Don't say:

> "The learning rate is a model parameter."

The learning rate is generally a **hyperparameter**.

### 3. Test data should remain untouched during model development

If you repeatedly use the test set to make modeling decisions, it no longer provides a clean final estimate of generalization.

### 4. One epoch does not mean one training update

An epoch can contain many batches and therefore many parameter updates.

### 5. More epochs do not automatically mean a better model

Training too long can contribute to overfitting, depending on the model and training setup.

---

# 📝 Practice Questions

### Basic

**Q1.** What is a dataset?

**Q2.** What is the difference between a sample and a feature?

**Q3.** What is the target in a house-price prediction problem?

**Q4.** What is the difference between a model and an algorithm?

**Q5.** What is a parameter?

**Q6.** What is a hyperparameter?

---

### Intermediate

**Q7.**

A dataset contains:

```text
50,000 samples
20 features
1 target
```

What are the dimensions of the feature matrix \(X\)?

---

**Q8.**

You have:

```text
10,000 training samples
batch_size = 100
```

Approximately how many iterations are there in one epoch?

---

**Q9.**

If you train for 20 epochs, approximately how many iterations will occur?

---

**Q10.**

Explain the difference between:

```text
Training Data
Validation Data
Test Data
```

---

### Conceptual

**Q11.**

Why should we not repeatedly tune our model using the test dataset?

**Q12.**

Explain:

```text
Parameter vs Hyperparameter
```

using an example.

**Q13.**

Explain the difference between:

```text
Ground Truth
Prediction
```

**Q14.**

Explain why generalization is important in Machine Learning.

---

# ✅ Learning Checklist

* [ ] Dataset
* [ ] Sample / Instance
* [ ] Features
* [ ] Target
* [ ] Labels
* [ ] Input / Output
* [ ] Feature Matrix
* [ ] Target Vector
* [ ] Algorithm
* [ ] Model
* [ ] Parameters
* [ ] Hyperparameters
* [ ] Training
* [ ] Validation
* [ ] Testing
* [ ] Prediction
* [ ] Ground Truth
* [ ] Inference
* [ ] Batch
* [ ] Iteration
* [ ] Epoch
* [ ] Generalization
* [ ] Model Evaluation
* [ ] Common ML notation

---

# ⏭️ Next Topic

## 04 — Machine Learning Workflow

Next we will take everything learned so far and understand how a **real Machine Learning project is built from beginning to end**:

```text
Problem Definition
       ↓
Data Collection
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train / Validation / Test Split
       ↓
Model Selection
       ↓
Training
       ↓
Evaluation
       ↓
Hyperparameter Tuning
       ↓
Final Model
       ↓
Deployment
       ↓
Monitoring
```

We will study **why every step exists, what can go wrong at each step, and how the steps connect together**.

