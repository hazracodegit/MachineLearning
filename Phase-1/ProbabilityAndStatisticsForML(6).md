# 06 — Probability & Statistics for Machine Learning

Probability and Statistics are fundamental mathematical tools for Machine Learning.

Machine Learning deals with data that contains:

* Uncertainty
* Noise
* Variation
* Patterns
* Randomness
* Missing information

Probability helps us reason about **uncertainty**, while Statistics helps us **understand, summarize, analyze, and make conclusions from data**.

For example, suppose we have the exam scores of 1,000 students:

```text
72, 85, 64, 91, 78, 55, ...
```

We may want to know:

* What is the average score?
* How spread out are the scores?
* What score is most common?
* How likely is a student to score above 90?
* Are study hours related to marks?
* What distribution do the scores follow?
* Can we make predictions from a sample?

These are statistical and probabilistic questions.

---

# 📚 Table of Contents

* [1. Probability vs Statistics](#1-probability-vs-statistics)
* [2. Population and Sample](#2-population-and-sample)
* [3. Descriptive vs Inferential Statistics](#3-descriptive-vs-inferential-statistics)
* [4. Random Experiment](#4-random-experiment)
* [5. Sample Space](#5-sample-space)
* [6. Events](#6-events)
* [7. Probability](#7-probability)
* [8. Basic Probability Rules](#8-basic-probability-rules)
* [9. Conditional Probability](#9-conditional-probability)
* [10. Independence](#10-independence)
* [11. Bayes' Theorem](#11-bayes-theorem)
* [12. Random Variables](#12-random-variables)
* [13. Discrete and Continuous Variables](#13-discrete-and-continuous-variables)
* [14. Probability Distribution](#14-probability-distribution)
* [15. Bernoulli Distribution](#15-bernoulli-distribution)
* [16. Binomial Distribution](#16-binomial-distribution)
* [17. Normal Distribution](#17-normal-distribution)
* [18. Standard Normal Distribution](#18-standard-normal-distribution)
* [19. Mean](#19-mean)
* [20. Median](#20-median)
* [21. Mode](#21-mode)
* [22. Range](#22-range)
* [23. Variance](#23-variance)
* [24. Standard Deviation](#24-standard-deviation)
* [25. Percentiles](#25-percentiles)
* [26. Quartiles](#26-quartiles)
* [27. IQR](#27-iqr)
* [28. Outliers](#28-outliers)
* [29. Covariance](#29-covariance)
* [30. Correlation](#30-correlation)
* [31. Covariance vs Correlation](#31-covariance-vs-correlation)
* [32. Expectation](#32-expectation)
* [33. Law of Large Numbers](#33-law-of-large-numbers)
* [34. Central Limit Theorem](#34-central-limit-theorem)
* [35. Sampling](#35-sampling)
* [36. Sampling Bias](#36-sampling-bias)
* [37. Confidence Intervals](#37-confidence-intervals)
* [38. Hypothesis Testing](#38-hypothesis-testing)
* [39. p-value](#39-p-value)
* [40. NumPy and Statistics](#40-numpy-and-statistics)
* [41. Pandas and Statistics](#41-pandas-and-statistics)
* [42. ML Applications](#42-ml-applications)
* [43. Common Mistakes](#43-common-mistakes)
* [44. Quick Revision](#44-quick-revision)
* [45. Formula Sheet](#45-formula-sheet)
* [46. Practice Questions](#46-practice-questions)
* [47. Learning Checklist](#47-learning-checklist)

---

# 1. Probability vs Statistics

Although Probability and Statistics are closely related, they are not the same.

## Probability

Probability generally starts with a model or assumptions and asks:

> What could happen?

Example:

```text
A fair coin is flipped.
What is the probability of getting heads?
```

Answer:

$$
P(H)=0.5
$$

---

## Statistics

Statistics generally starts with observed data and asks:

> What can we learn from what happened?

Example:

```text
We flip a coin 1,000 times.
We observe 523 heads.

What can we infer about the coin?
```

This is a statistical problem.

---

# 2. Population and Sample

## Population

A **population** is the complete group we are interested in studying.

Example:

```text
All students in a university
```

---

## Sample

A **sample** is a subset of that population.

Example:

```text
500 students selected from the university
```

---

## Machine Learning Connection

Suppose we want to predict house prices in India.

Population:

```text
All houses in the target market
```

Sample:

```text
100,000 houses in our dataset
```

Usually, we don't have access to every possible observation.

We use a sample to learn patterns.

---

# 3. Descriptive vs Inferential Statistics

## Descriptive Statistics

Descriptive statistics summarize existing data.

Examples:

* Mean
* Median
* Mode
* Variance
* Standard deviation
* Minimum
* Maximum
* Percentiles

Example:

```text
Average salary = ₹8 lakh
```

This describes the observed data.

---

## Inferential Statistics

Inferential statistics use sample data to make conclusions about a larger population.

For example:

```text
Sample of 10,000 customers
          ↓
Analyze sample
          ↓
Infer something about customers in the population
```

---

# 4. Random Experiment

A **random experiment** is an experiment whose exact outcome cannot be known beforehand.

Examples:

* Tossing a coin
* Rolling a die
* Selecting a random customer
* Measuring a randomly selected person's height

---

# 5. Sample Space

The **sample space** is the set of all possible outcomes of an experiment.

For a coin:

$$
S=\{H,T\}
$$

For a six-sided die:

$$
S=\{1,2,3,4,5,6\}
$$

---

# 6. Events

An **event** is a collection of one or more possible outcomes.

For a die:

```text
Sample space:
{1,2,3,4,5,6}
```

Event A = getting an even number:

$$
A=\{2,4,6\}
$$

---

# 7. Probability

Probability measures how likely an event is.

For an event \(A\):

$$
0\le P(A)\le1
$$

where:

```text
0   → impossible
1   → certain
0.5 → 50% probability
```

---

## Equally Likely Outcomes

If all outcomes are equally likely:

$$
P(A)=
\frac{\text{Number of favorable outcomes}}
{\text{Total number of outcomes}}
$$

Example:

Probability of rolling a 4:

$$
P(4)=\frac{1}{6}
$$

---

# 8. Basic Probability Rules

## Rule 1 — Complement

The probability that event \(A\) does not happen is:

$$
P(A^c)=1-P(A)
$$

Example:

If:

$$
P(A)=0.7
$$

then:

$$
P(A^c)=0.3
$$

---

## Rule 2 — Addition Rule

For two events:

$$
P(A\cup B)
=
P(A)+P(B)-P(A\cap B)
$$

where:

```text
∪ → OR
∩ → AND
```

---

## If Events Are Mutually Exclusive

If A and B cannot happen together:

$$
P(A\cap B)=0
$$

Therefore:

$$
P(A\cup B)=P(A)+P(B)
$$

---

# 9. Conditional Probability

Conditional probability asks:

> What is the probability of A given that B has already happened?

It is written:

$$
P(A|B)
$$

and calculated as:

$$
P(A|B)=
\frac{P(A\cap B)}
{P(B)}
$$

provided:

$$
P(B)>0
$$

---

## Example

Suppose:

```text
100 students
60 study Mathematics
40 study Computer Science
20 study both
```

Probability that a student studies Computer Science given that they study Mathematics:

$$
P(CS|Math)
=
\frac{20}{60}
=
\frac13
$$

---

# 10. Independence

Two events are independent if the occurrence of one does not change the probability of the other.

For independent events:

$$
P(A|B)=P(A)
$$

Another equivalent condition is:

$$
P(A\cap B)=P(A)P(B)
$$

---

## Example

Repeated flips of a fair coin are commonly modeled as independent.

The result of one flip does not determine the next flip.

---

# 11. Bayes' Theorem

Bayes' theorem is one of the most important probability concepts in Machine Learning.

It allows us to update a probability when new evidence becomes available.

The formula is:

$$
P(A|B)=
\frac{P(B|A)P(A)}
{P(B)}
$$

Where:

```text
P(A|B) → Posterior
P(B|A) → Likelihood
P(A)   → Prior
P(B)   → Evidence
```

---

## Intuition

Think of Bayes' theorem as:

```text
Prior belief
     ↓
New evidence
     ↓
Updated belief
```

---

## Example

Suppose a disease affects a small percentage of people.

A medical test returns positive.

We want:

$$
P(Disease|Positive)
$$

We need to consider:

* How common the disease is
* How often the test detects the disease
* How often the test produces false positives

This is why a positive test does not automatically mean the probability of disease is 100%.

---

# 12. Random Variables

A **random variable** is a variable whose value depends on the outcome of a random experiment.

Example:

Roll a die.

Let:

$$
X=\text{number shown on the die}
$$

Then:

$$
X\in\{1,2,3,4,5,6\}
$$

---

# 13. Discrete and Continuous Variables

## Discrete Random Variable

A discrete random variable takes countable values.

Examples:

* Number of customers
* Number of defective products
* Number of heads
* Number of children

Example:

$$
X\in\{0,1,2,3,...\}
$$

---

## Continuous Random Variable

A continuous random variable can take values across a continuous range.

Examples:

* Height
* Weight
* Temperature
* Time
* Distance

For example:

$$
Height=172.54\text{ cm}
$$

---

# 14. Probability Distribution

A probability distribution describes how probability is assigned across possible values of a random variable.

Example:

For a fair die:

|  X | Probability |
| -: | ----------: |
|  1 |         1/6 |
|  2 |         1/6 |
|  3 |         1/6 |
|  4 |         1/6 |
|  5 |         1/6 |
|  6 |         1/6 |

The probabilities sum to 1:

$$
\sum_xP(X=x)=1
$$

---

# 15. Bernoulli Distribution

The Bernoulli distribution models a single trial with two possible outcomes.

Usually:

```text
1 → Success
0 → Failure
```

Example:

```text
Customer purchases product?
```

Possible outcomes:

$$
X\in\{0,1\}
$$

If:

$$
P(X=1)=p
$$

then:

$$
P(X=0)=1-p
$$

---

## Examples

Bernoulli situations include:

* Spam / Not Spam
* Fraud / Not Fraud
* Pass / Fail
* Click / No Click
* Disease / No Disease

This is closely related to binary classification.

---

# 16. Binomial Distribution

The Binomial distribution counts the number of successes in a fixed number of independent Bernoulli trials with the same success probability.

If:

$$
X\sim Binomial(n,p)
$$

then:

$$
P(X=k)
=
\binom nk p^k(1-p)^{n-k}
$$

where:

* \(n\) = number of trials
* \(k\) = number of successes
* \(p\) = probability of success

---

## Example

Suppose a customer has a 20% probability of purchasing.

For 10 independent customers:

$$
X\sim Binomial(10,0.2)
$$

We can calculate the probability of exactly 3 purchases.

---

# 17. Normal Distribution

The Normal distribution is one of the most important distributions in statistics.

It is commonly represented as:

$$
X\sim N(\mu,\sigma^2)
$$

where:

```text
μ     → Mean
σ²    → Variance
σ     → Standard deviation
```

---

## Shape

A normal distribution has a bell-shaped curve:

```text
                    *
                 *     *
              *           *
            *               *
          *                   *
        *                       *
-------*-------------------------*-------
                 μ
```

It is symmetric around the mean.

---

# 18. Standard Normal Distribution

A standard normal distribution has:

$$
\mu=0
$$

and:

$$
\sigma=1
$$

It is commonly denoted by:

$$
Z\sim N(0,1)
$$

---

## Z-Score

A z-score tells us how many standard deviations a value is from the mean.

$$
z=
\frac{x-\mu}{\sigma}
$$

Example:

Suppose:

$$
\mu=70
$$

$$
\sigma=10
$$

and:

$$
x=90
$$

Then:

$$
z=
\frac{90-70}{10}
=
2
$$

The score is 2 standard deviations above the mean.

---

# 19. Mean

The arithmetic mean is commonly called the average.

For values:

$$
x_1,x_2,...,x_n
$$

the mean is:

$$
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i
$$

---

## Example

Data:

```text
10, 20, 30
```

Mean:

$$
\frac{10+20+30}{3}
=
20
$$

---

## ML Connection

The mean is used extensively in:

* Data analysis
* Feature scaling
* Standardization
* Loss functions
* Statistics
* Model evaluation

---

# 20. Median

The median is the middle value after sorting the data.

Example:

```text
10, 20, 30
```

Median:

```text
20
```

---

## Even Number of Values

Example:

```text
10, 20, 30, 40
```

Median:

$$
\frac{20+30}{2}=25
$$

---

## Mean vs Median

Consider:

```text
10, 20, 30, 40, 1000
```

Mean:

$$
220
$$

Median:

$$
30
$$

The mean is heavily affected by the extreme value 1000.

The median is much less affected.

---

# 21. Mode

The mode is the most frequently occurring value.

Example:

```text
2, 3, 3, 4, 5
```

Mode:

$$
3
$$

---

# 22. Range

The range is:

$$
Range=Maximum-Minimum
$$

Example:

```text
10, 20, 30, 40
```

$$
Range=40-10=30
$$

Range gives a basic measure of spread.

---

# 23. Variance

Variance measures how far values tend to spread around their mean.

For a population:

$$
\sigma^2=
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\mu)^2
$$

For a sample, the commonly used unbiased sample variance is:

$$
s^2=
\frac{1}{n-1}
\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

---

## Example

Data:

```text
2, 4, 6
```

Mean:

$$
\mu=4
$$

Differences:

```text
2 - 4 = -2
4 - 4 =  0
6 - 4 =  2
```

Squared differences:

```text
4, 0, 4
```

Population variance:

$$
\frac{4+0+4}{3}
=
\frac83
$$

---

# 24. Standard Deviation

Standard deviation is the square root of variance.

Population:

$$
\sigma=\sqrt{\sigma^2}
$$

Sample:

$$
s=\sqrt{s^2}
$$

---

## Why Use Standard Deviation?

Variance is expressed in squared units.

For example:

```text
Data → kilograms
Variance → kilograms²
Standard deviation → kilograms
```

Standard deviation is therefore easier to interpret in the original units.

---

# 25. Percentiles

A percentile tells us where a value stands relative to a dataset.

For example:

> 90th percentile

means that approximately 90% of observations are at or below that value, subject to the specific percentile convention used.

---

## Example

Suppose your score is at the 95th percentile.

This means your score is higher than or equal to approximately 95% of the observations in the reference dataset.

---

# 26. Quartiles

Quartiles divide ordered data into four parts.

```text
Minimum
   ↓
Q1
   ↓
Median (Q2)
   ↓
Q3
   ↓
Maximum
```

Where:

$$
Q_1=25th\ percentile
$$

$$
Q_2=50th\ percentile
$$

$$
Q_3=75th\ percentile
$$

---

# 27. IQR

IQR means **Interquartile Range**.

It is:

$$
IQR=Q_3-Q_1
$$

The IQR represents the spread of the middle 50% of observations.

---

# 28. Outliers

An outlier is an observation that is unusually far from the rest of the data.

One common IQR-based rule identifies potential outliers below:

$$
Q_1-1.5(IQR)
$$

or above:

$$
Q_3+1.5(IQR)
$$

These are often called **potential outliers** rather than automatically being treated as errors.

---

## Example

Suppose:

$$
Q_1=10
$$

$$
Q_3=20
$$

Then:

$$
IQR=10
$$

Lower boundary:

$$
10-1.5(10)=-5
$$

Upper boundary:

$$
20+1.5(10)=35
$$

Values outside these boundaries may be flagged for investigation.

---

# 29. Covariance

Covariance measures how two variables vary together.

For a sample:

$$
Cov(X,Y)
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})(y_i-\bar{y})
$$

---

## Interpretation

### Positive covariance

When X increases, Y tends to increase.

### Negative covariance

When X increases, Y tends to decrease.

### Near-zero covariance

There may be little linear co-movement.

However:

> Zero covariance does not generally mean that two variables are completely independent.

---

# 30. Correlation

Correlation measures the strength and direction of a linear relationship between two variables.

A common

