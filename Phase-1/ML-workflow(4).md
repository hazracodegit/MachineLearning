# 04 — Machine Learning Workflow

A Machine Learning model is not created simply by choosing an algorithm and calling `fit()`.

A real Machine Learning project follows a series of steps, starting with understanding the problem and ending with deploying and monitoring the model.

A typical workflow looks like this:

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Validation / Test Split
        ↓
Preprocessing
        ↓
Model Selection
        ↓
Training
        ↓
Evaluation
        ↓
Hyperparameter Tuning
        ↓
Final Evaluation
        ↓
Deployment
        ↓
Monitoring & Maintenance
```

This chapter explains each step in detail.

---

# 📚 Table of Contents

* [1. What Is an ML Workflow?](#1-what-is-an-ml-workflow)
* [2. Step 1 — Problem Definition](#2-step-1--problem-definition)
* [3. Step 2 — Data Collection](#3-step-2--data-collection)
* [4. Step 3 — Data Understanding](#4-step-3--data-understanding)
* [5. Step 4 — Data Cleaning](#5-step-4--data-cleaning)
* [6. Step 5 — Exploratory Data Analysis](#6-step-5--exploratory-data-analysis)
* [7. Step 6 — Feature Engineering](#7-step-6--feature-engineering)
* [8. Step 7 — Train/Validation/Test Split](#8-step-7--trainvalidationtest-split)
* [9. Step 8 — Data Preprocessing](#9-step-8--data-preprocessing)
* [10. Step 9 — Model Selection](#10-step-9--model-selection)
* [11. Step 10 — Model Training](#11-step-10--model-training)
* [12. Step 11 — Model Evaluation](#12-step-11--model-evaluation)
* [13. Step 12 — Hyperparameter Tuning](#13-step-12--hyperparameter-tuning)
* [14. Step 13 — Final Evaluation](#14-step-13--final-evaluation)
* [15. Step 14 — Deployment](#15-step-14--deployment)
* [16. Step 15 — Monitoring](#16-step-15--monitoring)
* [17. Complete Example](#17-complete-example)
* [18. Data Leakage](#18-data-leakage)
* [19. Common Mistakes](#19-common-mistakes)
* [20. Training Pipeline vs Production Pipeline](#20-training-pipeline-vs-production-pipeline)
* [21. Quick Revision](#21-quick-revision)
* [22. Learning Checklist](#22-learning-checklist)

---

# 1. What Is an ML Workflow?

## Definition

> **An ML workflow is the sequence of processes used to transform a real-world problem into a trained, evaluated, deployed, and maintained Machine Learning system.**

The workflow helps us answer:

* What problem are we solving?
* What data do we need?
* What should the model predict?
* How should the data be prepared?
* Which model should we use?
* How do we know whether it works?
* How do we deploy it?
* How do we monitor it after deployment?

---

# 2. Step 1 — Problem Definition

Before touching the dataset, clearly define the problem.

This is one of the most important steps because a technically good model can still be useless if it solves the wrong problem.

---

## 2.1 Define the Objective

Suppose a company wants to predict house prices.

A vague objective:

```text
"Build a house price model."
```

A better objective:

```text
"Predict the selling price of a house
using its location, area, number of bedrooms,
age, and other available information."
```

---

## 2.2 Identify the ML Task

Ask:

> What exactly should the model predict?

Examples:

### House price

```text
Predict ₹ amount
        ↓
Regression
```

### Spam detection

```text
Spam / Not Spam
        ↓
Classification
```

### Customer segmentation

```text
Discover customer groups
        ↓
Clustering
```

---

## 2.3 Define Inputs and Target

For house-price prediction:

```text
Inputs:
    Area
    Bedrooms
    Location
    Age
    Parking

Target:
    House Price
```

Mathematically:

$$
X \rightarrow y
$$

---

## 2.4 Define Success Criteria

We also need to determine what "good" means.

For example:

```text
Goal:
Mean Absolute Error < ₹5 lakh
```

or:

```text
Goal:
Recall > 90% for fraud detection
```

The metric should reflect the actual business or application objective.

---

# 3. Step 2 — Data Collection

## Definition

> **Data collection is the process of obtaining the data required to train and evaluate the ML system.**

Possible sources include:

* Databases
* APIs
* Sensors
* Application logs
* Surveys
* Public datasets
* Transaction systems
* Web data
* Existing company records

---

## Example

For house-price prediction, we might collect:

```text
House ID
Location
Area
Bedrooms
Bathrooms
Age
Parking
Price
```

---

## Data Quality Matters

More data does not automatically mean better data.

Bad data can contain:

* Missing values
* Incorrect values
* Duplicates
* Inconsistent formats
* Outliers
* Incorrect labels
* Data leakage

Example:

```text
Age = -10
```

This is probably invalid.

---

# 4. Step 3 — Data Understanding

Once we have data, we need to understand what it contains.

Typical questions:

* How many rows?
* How many columns?
* What are the data types?
* Which columns are numerical?
* Which are categorical?
* Are values missing?
* Are there duplicate rows?
* What is the distribution of important variables?
* Are there suspicious values?

---

## Example Dataset

| Area | Bedrooms | Location  | Price |
| ---: | -------: | --------- | ----: |
| 1000 |        2 | Hyderabad |   50L |
| 1500 |        3 | Hyderabad |   75L |
| 2000 |        4 | Bengaluru |  100L |
|    ? |        3 | Chennai   |   70L |

We immediately notice:

```text
Area = Missing
```

This needs investigation.

---

# 5. Step 4 — Data Cleaning

## Definition

> **Data cleaning is the process of detecting and correcting or appropriately handling incorrect, incomplete, duplicated, inconsistent, or otherwise problematic data.**

Common problems include:

```text
Missing Values
Duplicates
Invalid Values
Inconsistent Formats
Outliers
Incorrect Labels
```

---

# 5.1 Missing Values

Example:

| Age | Salary |
| --: | -----: |
|  25 |     5L |
|  30 |      ? |
|  35 |    10L |

Possible approaches:

### Remove rows

If only a small number of rows are affected and removing them is appropriate.

### Fill with a statistic

For example:

```text
Mean
Median
Mode
```

### Model-based imputation

Use other information to estimate the missing value.

The appropriate approach depends on the data and problem.

---

# 5.2 Duplicate Data

Example:

```text
ID   Age   Salary
101  25    5L
101  25    5L
```

If these represent the same observation duplicated accidentally, keeping both may distort the dataset.

---

# 5.3 Invalid Values

Examples:

```text
Age = -5
Bedrooms = 1000
Salary = -₹50,000
```

Such values need investigation.

---

# 5.4 Inconsistent Formats

Example:

```text
Hyderabad
hyderabad
HYDERABAD
```

These may represent the same category but appear differently.

Similarly:

```text
01/02/2025
2025-02-01
Feb 1, 2025
```

may represent dates in different formats.

---

# 6. Step 5 — Exploratory Data Analysis

## Definition

> **Exploratory Data Analysis (EDA) is the process of examining and visualizing data to understand its structure, distributions, relationships, patterns, and potential problems.**

EDA helps us answer questions before training a model.

---

## 6.1 Univariate Analysis

Study one variable at a time.

Example:

```text
Distribution of House Prices
```

We might examine:

* Mean
* Median
* Minimum
* Maximum
* Standard deviation
* Distribution

---

## 6.2 Bivariate Analysis

Study the relationship between two variables.

Example:

```text
Area ↔ House Price
```

We might discover:

```text
Larger houses
      ↓
Generally higher prices
```

---

## 6.3 Multivariate Analysis

Study relationships among multiple variables.

Example:

```text
Area
Bedrooms
Location
Age
       ↓
   House Price
```

---

## Why EDA Matters

EDA can reveal:

```text
Missing values
Outliers
Skewed distributions
Relationships
Correlations
Unexpected patterns
Potential leakage
```

---

# 7. Step 6 — Feature Engineering

## Definition

> **Feature engineering is the process of creating, transforming, selecting, or combining input features to produce representations that are useful for Machine Learning.**

This can have a major effect on model performance.

---

# 7.1 Creating a New Feature

Suppose we have:

```text
Total Income
Number of Family Members
```

We could create:

$$
\text{Income Per Person}
=
\frac{\text{Total Income}}
{\text{Family Members}}
$$

This new feature may provide useful information.

---

# 7.2 Extracting Information

Suppose we have:

```text
Date = 2026-09-21
```

We can extract:

```text
Year = 2026
Month = 9
Day = 21
Day of Week = Monday
```

---

# 7.3 Encoding Categorical Features

Machine learning algorithms often require numerical representations.

Suppose:

```text
Color:
Red
Blue
Green
```

We may use techniques such as:

```text
One-Hot Encoding
```

to represent categories numerically.

---

# 7.4 Scaling Features

Suppose:

```text
Age       = 25
Salary    = 800000
```

The numerical scales are very different.

Some algorithms benefit from scaling.

Common techniques include:

* Standardization
* Min-Max scaling
* Robust scaling

We will study these in detail later.

---

# 8. Step 7 — Train/Validation/Test Split

We usually separate data so that we can evaluate generalization properly.

A common structure:

```text
Complete Dataset
       │
       ├───────────────┐
       │               │
       ▼               ▼
   Training        Validation
       │               │
       └───────┬───────┘
               ▼
          Final Model
               │
               ▼
            Test Set
               │
               ▼
        Final Evaluation
```

---

## Example

Suppose we have:

```text
100,000 samples
```

A possible split:

```text
70,000 → Training
15,000 → Validation
15,000 → Test
```

The exact split depends on the problem.

---

## Why Split the Data?

If we train and evaluate on exactly the same data:

```text
Training Data
      ↓
Model
      ↓
Same Data
      ↓
Evaluation
```

the model may appear to perform very well even if it does not generalize to new data.

We therefore evaluate on data that was not used for learning.

---

# 9. Step 8 — Data Preprocessing

## Definition

> **Data preprocessing transforms raw data into a representation suitable for a machine learning algorithm.**

Examples:

```text
Raw Data
   ↓
Missing Value Handling
   ↓
Encoding
   ↓
Scaling
   ↓
Feature Selection
   ↓
Model-Ready Data
```

---

## Common Preprocessing Operations

### Numerical Data

* Imputation
* Scaling
* Transformation

### Categorical Data

* Encoding
* Rare-category handling
* Missing-category handling

### Text

* Tokenization
* Vectorization
* Embeddings

### Images

* Resizing
* Normalization
* Augmentation

---

# 10. Step 9 — Model Selection

## Definition

> **Model selection is the process of choosing suitable algorithms or model families for a given problem.**

Suppose we're solving a regression problem.

Possible choices:

```text
Linear Regression
Decision Tree
Random Forest
Gradient Boosting
Neural Network
```

We should consider:

* Problem type
* Dataset size
* Feature types
* Interpretability requirements
* Computational resources
* Latency requirements
* Evaluation metric

---

# 11. Step 10 — Model Training

Once the data and model are ready, we train the model.

```text
Training Data
      ↓
Algorithm
      ↓
Model learns parameters
      ↓
Trained Model
```

For many algorithms, training involves optimizing some objective function.

A simplified view:

$$
\text{Data}
\rightarrow
\text{Prediction}
\rightarrow
\text{Error/Loss}
\rightarrow
\text{Parameter Update}
$$

This process repeats until the training procedure completes.

---

# 12. Step 11 — Model Evaluation

## Definition

> **Model evaluation is the process of measuring how well a trained model performs using appropriate metrics on data that was not used to fit the model.**

The metric depends on the task.

---

## Regression Metrics

Common metrics include:

### MAE

Mean Absolute Error:

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

### MSE

Mean Squared Error:

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

### RMSE

$$
RMSE = \sqrt{MSE}
$$

### R²

Measures explained variance relative to a baseline.

We will study these properly in the evaluation section.

---

## Classification Metrics

Common metrics:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
```

Different metrics answer different questions.

For example:

* Accuracy → How many predictions were correct overall?
* Precision → Of predicted positives, how many were actually positive?
* Recall → Of actual positives, how many did we detect?
* F1 → Harmonic mean of precision and recall

---

# 13. Step 12 — Hyperparameter Tuning

## Definition

> **Hyperparameter tuning is the process of searching for hyperparameter configurations that produce good validation performance.**

Suppose we're training a Random Forest.

We might need to choose:

```text
n_estimators
max_depth
min_samples_split
```

We can test different configurations.

Example:

```text
Configuration A
max_depth = 5

Validation Score = 0.82
```

```text
Configuration B
max_depth = 10

Validation Score = 0.87
```

```text
Configuration C
max_depth = 20

Validation Score = 0.85
```

We use appropriate validation procedures to select a configuration.

---

# 14. Step 13 — Final Evaluation

After selecting the model and configuration, evaluate the final system on the held-out test set.

Important principle:

> The test set should represent data that was not used to make modeling decisions.

A simplified workflow:

```text
Training Data
     ↓
Train Models
     ↓
Validation Data
     ↓
Choose Model + Hyperparameters
     ↓
Final Model
     ↓
Test Data
     ↓
Final Performance Estimate
```

---

# 15. Step 14 — Deployment

## Definition

> **Deployment is the process of making a trained ML model available for use in a real application or production environment.**

Before deployment:

```text
Training
   ↓
Validation
   ↓
Testing
```

After deployment:

```text
Application
    ↓
New Input
    ↓
Model
    ↓
Prediction
    ↓
Application Response
```

---

# 15.1 Example — House Price API

A user sends:

```json
{
  "area": 1800,
  "bedrooms": 3,
  "age": 5
}
```

The deployed model returns something like:

```json
{
  "predicted_price": 9000000
}
```

The model is now being used for inference.

---

# 15.2 Common Deployment Forms

A model can be deployed as:

* REST API
* Web application
* Mobile application
* Batch prediction system
* Embedded system
* Cloud service
* Edge device

---

# 16. Step 15 — Monitoring

Deployment is not the end of an ML project.

A model can degrade over time.

Why?

Because the real world changes.

Examples:

```text
Customer behavior changes
Economic conditions change
Products change
Data distributions change
User behavior changes
```

Therefore, production ML systems often require monitoring.

---

# 16.1 Data Drift

## Definition

> **Data drift refers to changes in the distribution of input data over time.**

Example:

Training data:

```text
Average customer age = 30
```

Production months later:

```text
Average customer age = 45
```

The input distribution has changed.

---

# 16.2 Concept Drift

## Definition

> **Concept drift occurs when the relationship between input variables and the target changes over time.**

Example:

A model learns:

```text
Customer behavior → Purchase
```

But customer behavior changes due to a new market environment.

The old relationship may no longer hold.

---

# 16.3 Production Monitoring

Things that may be monitored include:

```text
Prediction quality
Input distributions
Latency
Errors
Missing values
Model failures
Data drift
Concept drift
Business metrics
```

---

# 17. Complete Example

Let's build a conceptual ML project for **house-price prediction**.

---

## Step 1 — Problem Definition

Goal:

```text
Predict house selling price.
```

Task:

```text
Regression
```

---

## Step 2 — Data Collection

Collect:

```text
Area
Bedrooms
Bathrooms
Location
Age
Parking
Price
```

---

## Step 3 — Data Understanding

Check:

```text
Number of samples
Number of features
Data types
Missing values
Duplicates
Distributions
```

---

## Step 4 — Data Cleaning

Handle:

```text
Missing area values
Duplicate records
Invalid ages
Inconsistent locations
```

---

## Step 5 — EDA

Investigate:

```text
Area vs Price
Location vs Price
Bedrooms vs Price
Age vs Price
```

Visualizations may reveal useful relationships and unusual observations.

---

## Step 6 — Feature Engineering

Create:

```text
Price per square foot
Age groups
Location encoding
Total rooms
```

depending on the available data and modeling goal.

---

## Step 7 — Split Data

For example:

```text
70% → Training
15% → Validation
15% → Test
```

---

## Step 8 — Preprocessing

For example:

```text
Missing-value imputation
Categorical encoding
Feature scaling where appropriate
```

---

## Step 9 — Select Models

Try:

```text
Linear Regression
Random Forest
Gradient Boosting
```

---

## Step 10 — Train

Train each candidate model on the training data.

---

## Step 11 — Validate

Compare their performance on validation data using a chosen metric such as MAE or RMSE.

---

## Step 12 — Tune

Tune important hyperparameters.

```text
max_depth
learning_rate
number_of_estimators
```

depending on the model.

---

## Step 13 — Final Evaluation

Evaluate the selected final model once on the test set.

Example:

```text
Test MAE = ₹4.8L
```

This number is only an example; actual performance depends on the dataset and model.

---

## Step 14 — Deploy

Create an API:

```text
POST /predict
```

Input:

```json
{
  "area": 1800,
  "bedrooms": 3,
  "age": 5
}
```

Output:

```json
{
  "price": 9000000
}
```

---

## Step 15 — Monitor

Track:

```text
Prediction performance
Input data
Latency
Errors
Data drift
```

If performance degrades significantly, investigate and potentially retrain the model.

---

# 18. Data Leakage

## Definition

> **Data leakage occurs when information that would not legitimately be available at prediction time influences model training or evaluation.**

Data leakage can make a model appear much better than it really is.

---

## Example

Suppose we're predicting whether a customer will default on a loan.

We accidentally include:

```text
Loan Recovery Status
```

as a feature.

But this information becomes available only **after** the customer defaults.

The model now has access to future information.

That's leakage.

---

# 18.1 Another Common Example

Suppose we normalize the entire dataset before splitting:

```text
Complete Dataset
      ↓
Calculate mean/std
      ↓
Normalize
      ↓
Train/Test Split
```

Information from the test set has influenced preprocessing.

A safer conceptual workflow is:

```text
Dataset
   ↓
Split
   ├── Training
   └── Test
        ↓
Learn preprocessing parameters from Training
        ↓
Apply same transformation to Test
```

In practice, preprocessing pipelines help enforce this separation.

---

# 18.2 Why Leakage Is Dangerous

Leakage can produce:

```text
Very high validation score
```

but:

```text
Poor real-world performance
```

Therefore:

> Always ensure that information unavailable at prediction time does not enter the model through features or preprocessing.

---

# 19. Common Mistakes

## Mistake 1 — Starting with the algorithm

Beginners often think:

```text
"I have data → I'll use Random Forest."
```

A better approach:

```text
Problem
 ↓
Data
 ↓
Understand Data
 ↓
Prepare Data
 ↓
Choose Models
```

---

## Mistake 2 — Training before understanding the data

You should inspect:

* Missing values
* Data types
* Distributions
* Duplicates
* Invalid values
* Target distribution

before blindly training models.

---

## Mistake 3 — Evaluating on training data

Example:

```text
Train:
95% accuracy

Test:
70% accuracy
```

The training score alone is not enough.

---

## Mistake 4 — Tuning on the test set

If you repeatedly change your model based on test performance, you gradually make decisions based on the test data.

That compromises its role as an unbiased final evaluation set.

---

## Mistake 5 — Ignoring the business objective

A model can have good statistical metrics but still fail to solve the actual business problem.

Always ask:

> What decision will this prediction support?

---

## Mistake 6 — Treating deployment as the end

A deployed model can become outdated.

Production ML often requires:

```text
Monitoring
↓
Evaluation
↓
Retraining
↓
Redeployment
```

---

# 20. Training Pipeline vs Production Pipeline

It is useful to distinguish the two.

## Training Pipeline

```text
Raw Data
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Training
   ↓
Evaluation
   ↓
Model Artifact
```

---

## Production Inference Pipeline

```text
New Input
    ↓
Validation
    ↓
Same Preprocessing
    ↓
Trained Model
    ↓
Prediction
    ↓
Response
```

The preprocessing applied during inference must be consistent with the preprocessing used during training.

---

# 21. End-to-End ML Workflow

The entire process can be visualized as:

```text
                    REAL-WORLD PROBLEM
                            │
                            ▼
                    Problem Definition
                            │
                            ▼
                     Data Collection
                            │
                            ▼
                    Data Understanding
                            │
                            ▼
                      Data Cleaning
                            │
                            ▼
                           EDA
                            │
                            ▼
                   Feature Engineering
                            │
                            ▼
                Train / Validation / Test
                            │
                            ▼
                     Preprocessing
                            │
                            ▼
                     Model Selection
                            │
                            ▼
                        Training
                            │
                            ▼
                       Evaluation
                            │
                            ▼
                  Hyperparameter Tuning
                            │
                            ▼
                     Final Evaluation
                            │
                            ▼
                       Deployment
                            │
                            ▼
                       Monitoring
                            │
                            ▼
                    Retraining / Update
                            │
                            └───────────┐
                                        │
                                        ▼
                                  New Model
```

The workflow is often iterative rather than strictly linear.

---

# 22. The ML Workflow in One Example

Consider a **spam detection system**.

### Problem

Predict whether an incoming email is spam.

### Data

Historical emails labeled:

```text
Spam
Not Spam
```

### Features

Potential inputs:

```text
Email text
Sender information
Message metadata
Other legitimate features
```

### Target

```text
Spam / Not Spam
```

### Data Cleaning

Handle:

```text
Duplicate emails
Missing information
Malformed records
```

### Feature Engineering

Convert text into a representation usable by the model.

### Split

```text
Training
Validation
Test
```

### Model

Try appropriate classification models.

### Training

Learn from historical labeled emails.

### Evaluation

Use appropriate classification metrics.

### Deployment

New emails are passed to the deployed model.

### Inference

```text
New Email
   ↓
Model
   ↓
Spam probability/class
```

### Monitoring

Monitor:

```text
Model quality
False positives
False negatives
Changes in incoming email patterns
```

---

# 23. Why the Workflow Is Iterative

Real ML projects rarely follow this process only once.

For example:

```text
Problem
   ↓
Data
   ↓
Model
   ↓
Evaluation
   ↓
Poor Performance
   ↓
Return to:
   ├── Data
   ├── Features
   ├── Model
   └── Hyperparameters
```

You may discover that:

* More data is needed.
* Some features are useless.
* Labels contain errors.
* The chosen metric is inappropriate.
* The model is overfitting.
* The production data differs from training data.

Therefore:

> Machine Learning is an iterative engineering process, not a single training operation.

---

# 24. Quick Revision

## Problem Definition

Clearly define what needs to be predicted and why.

## Data Collection

Obtain relevant data.

## Data Understanding

Understand structure, types, distributions, and quality.

## Data Cleaning

Handle missing, invalid, duplicated, and inconsistent data.

## EDA

Explore patterns and relationships.

## Feature Engineering

Create or transform useful features.

## Data Splitting

Separate training, validation, and test data.

## Preprocessing

Transform data into a suitable representation.

## Model Selection

Choose candidate algorithms.

## Training

Learn model parameters from training data.

## Evaluation

Measure performance using appropriate metrics.

## Hyperparameter Tuning

Search for useful model configurations.

## Final Testing

Evaluate the selected system on held-out test data.

## Deployment

Make the model available for real-world inference.

## Monitoring

Track performance, data quality, drift, and system behavior.

---

# 25. One-Minute Revision Table

| Step                | Main Question                                      |
| ------------------- | -------------------------------------------------- |
| Problem Definition  | What are we trying to solve?                       |
| Data Collection     | Where do we get the data?                          |
| Data Understanding  | What does the data contain?                        |
| Data Cleaning       | Is the data reliable?                              |
| EDA                 | What patterns exist?                               |
| Feature Engineering | Can we create better inputs?                       |
| Data Split          | How will we evaluate generalization?               |
| Preprocessing       | How should data be transformed?                    |
| Model Selection     | Which algorithms are appropriate?                  |
| Training            | What does the model learn?                         |
| Evaluation          | How well does it perform?                          |
| Tuning              | Can we improve configuration?                      |
| Final Testing       | How does the final model perform on held-out data? |
| Deployment          | How will users/applications use it?                |
| Monitoring          | Does it continue working in production?            |

---

# 🧠 Must-Remember Pipeline

Memorize this:

```text
Problem
   ↓
Data
   ↓
Clean
   ↓
Explore
   ↓
Engineer Features
   ↓
Split
   ↓
Preprocess
   ↓
Train
   ↓
Validate
   ↓
Tune
   ↓
Test
   ↓
Deploy
   ↓
Monitor
```

---

# ⚠️ Most Important Concepts

### 1. Don't start with the model

Start with the problem.

### 2. Don't trust raw data blindly

Understand and clean it.

### 3. Don't evaluate only on training data

Use appropriate held-out data.

### 4. Don't use the test set for repeated tuning

Keep it for final evaluation.

### 5. Watch for data leakage

Future or inappropriate information can make results misleading.

### 6. Deployment isn't the end

Production models need monitoring and maintenance.

---

# 📝 Practice Questions

## Basic

### Q1

What is an ML workflow?

### Q2

Why is problem definition important?

### Q3

What is EDA?

### Q4

What is feature engineering?

### Q5

Why do we split data into training and test sets?

### Q6

What is deployment?

---

## Intermediate

### Q7

Explain the difference between:

```text
Training
Validation
Testing
```

### Q8

Why shouldn't we tune our model using the test dataset?

### Q9

Give three examples of data-cleaning operations.

### Q10

What is the purpose of feature engineering?

### Q11

What is hyperparameter tuning?

---

## Advanced Thinking

### Q12

A model achieves:

```text
Training Accuracy = 99%
Test Accuracy = 70%
```

What might this indicate?

### Q13

A model performed extremely well during development but poorly after deployment.

What possible causes would you investigate?

### Q14

A feature contains information that becomes available only after the prediction should have been made.

What problem might this create?

### Q15

Why is Machine Learning workflow considered iterative rather than strictly linear?

---

# ✅ Learning Checklist

* [ ] Understand the complete ML workflow
* [ ] Problem definition
* [ ] Data collection
* [ ] Data understanding
* [ ] Data cleaning
* [ ] Exploratory Data Analysis
* [ ] Feature engineering
* [ ] Train/validation/test split
* [ ] Data preprocessing
* [ ] Model selection
* [ ] Model training
* [ ] Model evaluation
* [ ] Hyperparameter tuning
* [ ] Final evaluation
* [ ] Deployment
* [ ] Monitoring
* [ ] Data drift
* [ ] Concept drift
* [ ] Data leakage
* [ ] Training pipeline
* [ ] Production inference pipeline
* [ ] Understand why ML workflows are iterative

---

# ⏭️ Next Topic

## 05 — Linear Algebra for Machine Learning

Before diving deeply into ML algorithms, we need the mathematical foundation used to represent and manipulate data.

We will cover:

```text
Scalars
Vectors
Matrices
Tensors
Vector Operations
Dot Product
Matrix Multiplication
Transpose
Norms
Distance
Linear Transformations
Eigenvalues
Eigenvectors
```

These concepts will later appear everywhere in:

* Linear Regression
* PCA
* Logistic Regression
* Neural Networks
* Deep Learning
* Optimization
* Computer Vision
* NLP

