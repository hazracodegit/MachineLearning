# 05 — Linear Algebra for Machine Learning

Linear Algebra is one of the most important mathematical foundations of Machine Learning.

Machine Learning works with large amounts of numerical data, and linear algebra gives us the language and operations needed to represent and manipulate that data.

For example:

```text
A dataset
     ↓
Matrix

One observation
     ↓
Vector

One numerical value
     ↓
Scalar
```

Later, concepts such as:

* Linear Regression
* Logistic Regression
* Principal Component Analysis
* Neural Networks
* Computer Vision
* Natural Language Processing
* Deep Learning
* Optimization

will heavily use linear algebra.

---

# 📚 Table of Contents

* [1. Why Linear Algebra?](#1-why-linear-algebra)
* [2. Scalar](#2-scalar)
* [3. Vector](#3-vector)
* [4. Vector Dimensions](#4-vector-dimensions)
* [5. Row Vector and Column Vector](#5-row-vector-and-column-vector)
* [6. Vector Representation of ML Data](#6-vector-representation-of-ml-data)
* [7. Vector Operations](#7-vector-operations)
* [8. Vector Addition](#8-vector-addition)
* [9. Scalar Multiplication](#9-scalar-multiplication)
* [10. Dot Product](#10-dot-product)
* [11. Geometric Meaning of Dot Product](#11-geometric-meaning-of-dot-product)
* [12. Matrix](#12-matrix)
* [13. Matrix Dimensions](#13-matrix-dimensions)
* [14. Matrix Elements](#14-matrix-elements)
* [15. Dataset as a Matrix](#15-dataset-as-a-matrix)
* [16. Matrix Addition](#16-matrix-addition)
* [17. Scalar Multiplication of Matrices](#17-scalar-multiplication-of-matrices)
* [18. Matrix Transpose](#18-matrix-transpose)
* [19. Matrix Multiplication](#19-matrix-multiplication)
* [20. Matrix-Vector Multiplication](#20-matrix-vector-multiplication)
* [21. Identity Matrix](#21-identity-matrix)
* [22. Inverse Matrix](#22-inverse-matrix)
* [23. Norms](#23-norms)
* [24. Distance](#24-distance)
* [25. Linear Independence](#25-linear-independence)
* [26. Basis](#26-basis)
* [27. Linear Transformations](#27-linear-transformations)
* [28. Eigenvalues](#28-eigenvalues)
* [29. Eigenvectors](#29-eigenvectors)
* [30. Why Eigenvalues Matter in ML](#30-why-eigenvalues-matter-in-ml)
* [31. Tensors](#31-tensors)
* [32. NumPy Representation](#32-numpy-representation)
* [33. Complete ML Example](#33-complete-ml-example)
* [34. Common Mistakes](#34-common-mistakes)
* [35. Quick Revision](#35-quick-revision)
* [36. Formula Sheet](#36-formula-sheet)
* [37. Practice Questions](#37-practice-questions)
* [38. Learning Checklist](#38-learning-checklist)

---

# 1. Why Linear Algebra?

Suppose we have information about a student:

```text
Age = 20
Hours studied = 5
Attendance = 90%
Previous score = 80
```

We can represent this as a vector:

$$
x =
\begin{bmatrix}
20\\
5\\
90\\
80
\end{bmatrix}
$$

A dataset containing thousands of students can then be represented as a matrix:

$$
X =
\begin{bmatrix}
20 & 5 & 90 & 80\\
21 & 7 & 95 & 85\\
19 & 3 & 80 & 65\\
22 & 8 & 98 & 92
\end{bmatrix}
$$

Machine Learning algorithms perform mathematical operations on these vectors and matrices.

For example:

$$
Xw
$$

can represent a weighted combination of features.

This is why linear algebra is so important.

---

# 2. Scalar

## Definition

> A **scalar** is a single numerical value.

Examples:

$$
5
$$

$$
-3
$$

$$
2.5
$$

$$
100
$$

In Machine Learning:

```text
Learning rate = 0.01
Age = 25
Temperature = 32
```

Each individual value is a scalar.

---

## Simple Representation

```text
Scalar
  ↓
One value
```

Example:

$$
x=5
$$

---

# 3. Vector

## Definition

> A **vector** is an ordered collection of numbers.

Example:

$$
x=
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

This vector contains three values.

We can think of a vector as a list of numbers with a specific order.

---

## ML Example

Suppose we have:

```text
Area = 1500
Bedrooms = 3
Bathrooms = 2
```

We can represent the house as:

$$
x=
\begin{bmatrix}
1500\\
3\\
2
\end{bmatrix}
$$

This is a feature vector.

---

# 4. Vector Dimensions

The **dimension** of a vector is the number of values it contains.

Example:

$$
x=
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

has 3 values.

Therefore:

$$
x \in \mathbb{R}^{3}
$$

We say:

> `x` is a 3-dimensional vector.

---

## Examples

### One-dimensional

$$
x=[5]
$$

### Two-dimensional

$$
x=
\begin{bmatrix}
2\\
3
\end{bmatrix}
$$

### Three-dimensional

$$
x=
\begin{bmatrix}
2\\
3\\
4
\end{bmatrix}
$$

### Ten-dimensional

A vector containing 10 numerical values.

---

# 5. Row Vector and Column Vector

A vector can be written horizontally or vertically.

## Row Vector

$$
x=
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

Shape:

$$
1\times3
$$

---

## Column Vector

$$
x=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

Shape:

$$
3\times1
$$

Both contain the same three values, but their shapes are different.

This distinction becomes extremely important during matrix multiplication.

---

# 6. Vector Representation of ML Data

Suppose we have:

| Feature    |  Value |
| ---------- | -----: |
| Age        |     25 |
| Experience |      3 |
| Salary     | 700000 |

If salary is the target, the input vector could be:

$$
x=
\begin{bmatrix}
25\\
3
\end{bmatrix}
$$

and:

$$
y=700000
$$

So:

```text
x → Input features
y → Target
```

---

# 7. Vector Operations

Common vector operations include:

* Addition
* Subtraction
* Scalar multiplication
* Dot product
* Norm
* Distance

These operations appear frequently in Machine Learning.

---

# 8. Vector Addition

Two vectors of the same dimension can be added element by element.

Suppose:

$$
a=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

and:

$$
b=
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

Then:

$$
a+b=
\begin{bmatrix}
1+4\\
2+5\\
3+6
\end{bmatrix}
$$

Therefore:

$$
a+b=
\begin{bmatrix}
5\\
7\\
9
\end{bmatrix}
$$

---

## Important Rule

Vectors must have compatible dimensions.

You cannot directly add:

$$
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

to:

$$
\begin{bmatrix}
3\\
4\\
5
\end{bmatrix}
$$

because their dimensions differ.

---

# 9. Scalar Multiplication

A scalar can multiply every element of a vector.

Suppose:

$$
x=
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

and:

$$
c=3
$$

Then:

$$
cx=
3
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
6\\
12\\
18
\end{bmatrix}
$$

---

## ML Connection

Model parameters often multiply input features.

For example:

$$
w_1x_1+w_2x_2
$$

contains scalar-vector-like operations that eventually lead to the dot product.

---

# 10. Dot Product

The **dot product** is one of the most important linear algebra operations in Machine Learning.

Suppose:

$$
a=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

and:

$$
b=
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

The dot product is:

$$
a\cdot b
=
1(4)+2(5)+3(6)
$$

$$
=4+10+18
$$

$$
=32
$$

The result is a **scalar**.

---

## General Formula

For:

$$
a=
\begin{bmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{bmatrix}
$$

and:

$$
b=
\begin{bmatrix}
b_1\\
b_2\\
\vdots\\
b_n
\end{bmatrix}
$$

the dot product is:

$$
a\cdot b=
\sum_{i=1}^{n}a_ib_i
$$

---

# 11. Geometric Meaning of Dot Product

The dot product can also be written as:

$$
a\cdot b
=
\|a\|\|b\|\cos\theta
$$

where:

* \(\|a\|\) = magnitude of \(a\)
* \(\|b\|\) = magnitude of \(b\)
* \(\theta\) = angle between the vectors

This gives useful intuition.

### Same direction

$$
\cos(0)=1
$$

Dot product is positive.

### Perpendicular

$$
\cos(90^\circ)=0
$$

Dot product is zero.

### Opposite direction

$$
\cos(180^\circ)=-1
$$

Dot product is negative.

This geometric interpretation becomes useful later when studying cosine similarity.

---

# 12. Matrix

## Definition

> A **matrix** is a rectangular arrangement of numbers organized into rows and columns.

Example:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

This matrix contains:

```text
2 rows
3 columns
```

Therefore its shape is:

$$
2\times3
$$

---

# 13. Matrix Dimensions

A matrix is described as:

$$
\text{rows}\times\text{columns}
$$

For:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

we have:

$$
A\in\mathbb{R}^{2\times3}
$$

---

## Important

Do not confuse:

```text
2 × 3
```

with:

```text
3 × 2
```

They represent different shapes.

---

# 14. Matrix Elements

Consider:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

The element in row 1, column 2 is:

$$
A_{1,2}=2
$$

The element in row 2, column 3 is:

$$
A_{2,3}=6
$$

Generally:

$$
A_{ij}
$$

means:

> element at row \(i\), column \(j\).

---

# 15. Dataset as a Matrix

This is one of the most important connections to Machine Learning.

Suppose:

| Age | Experience | Salary |
| --: | ---------: | -----: |
|  22 |          1 |      4 |
|  25 |          3 |      7 |
|  30 |          7 |     12 |
|  35 |         10 |     18 |

If salary is the target, the feature matrix is:

$$
X=
\begin{bmatrix}
22&1\\
25&3\\
30&7\\
35&10
\end{bmatrix}
$$

Shape:

$$
4\times2
$$

Meaning:

```text
4 samples
2 features
```

This is how a typical tabular ML dataset can be represented mathematically.

---

# 16. Matrix Addition

Two matrices can be added when they have the same shape.

Suppose:

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

and:

$$
B=
\begin{bmatrix}
5&6\\
7&8
\end{bmatrix}
$$

Then:

$$
A+B=
\begin{bmatrix}
6&8\\
10&12
\end{bmatrix}
$$

Each corresponding element is added.

---

# 17. Scalar Multiplication of Matrices

A scalar multiplies every matrix element.

Suppose:

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

Then:

$$
3A=
\begin{bmatrix}
3&6\\
9&12
\end{bmatrix}
$$

---

# 18. Matrix Transpose

## Definition

> The **transpose** of a matrix is obtained by converting its rows into columns and its columns into rows.

Suppose:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

Then:

$$
A^T=
\begin{bmatrix}
1&4\\
2&5\\
3&6
\end{bmatrix}
$$

Original shape:

$$
2\times3
$$

Transposed shape:

$$
3\times2
$$

---

## Important Property

$$
(A^T)^T=A
$$

---

# 19. Matrix Multiplication

Matrix multiplication is extremely important in Machine Learning.

Suppose:

$$
A
$$

has shape:

$$
m\times n
$$

and:

$$
B
$$

has shape:

$$
n\times p
$$

Then:

$$
AB
$$

is possible and produces a matrix with shape:

$$
m\times p
$$

---

## Example

Let:

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

and:

$$
B=
\begin{bmatrix}
5&6\\
7&8
\end{bmatrix}
$$

Then:

$$
AB=
\begin{bmatrix}
1(5)+2(7)&1(6)+2(8)\\
3(5)+4(7)&3(6)+4(8)
\end{bmatrix}
$$

Therefore:

$$
AB=
\begin{bmatrix}
19&22\\
43&50
\end{bmatrix}
$$

---

# 20. Matrix Multiplication Rule

This rule is extremely important:

> **The inner dimensions must match.**

Suppose:

$$
A_{m\times n}
$$

and:

$$
B_{n\times p}
$$

Then:

$$
AB
$$

is valid.

The result has shape:

$$
m\times p
$$

---

## Example

### Valid

$$
(3\times2)(2\times4)
$$

Result:

$$
3\times4
$$

because the inner dimensions:

```text
3 × 2
    2 × 4
      ↑
      Match
```

---

### Invalid

$$
(3\times2)(4\times5)
$$

because:

```text
3 × 2
    4 × 5
    ↑
    2 ≠ 4
```

---

# 21. Matrix-Vector Multiplication

This operation appears constantly in Machine Learning.

Suppose:

$$
X=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
$$

and:

$$
w=
\begin{bmatrix}
10\\
20
\end{bmatrix}
$$

Shapes:

$$
X:3\times2
$$

$$
w:2\times1
$$

Therefore:

$$
Xw
$$

is valid.

Result:

$$
Xw=
\begin{bmatrix}
1(10)+2(20)\\
3(10)+4(20)\\
5(10)+6(20)
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
50\\
110\\
170
\end{bmatrix}
$$

---

## ML Connection

Suppose:

$$
X
$$

contains training samples and:

$$
w
$$

contains model weights.

Then:

$$
Xw
$$

can produce a vector of model outputs.

This is fundamental to linear models and neural networks.

---

# 22. Identity Matrix

## Definition

> An **identity matrix** is a square matrix with 1s on the main diagonal and 0s everywhere else.

Example:

$$
I=
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1
\end{bmatrix}
$$

It behaves similarly to the number 1 in multiplication:

$$
AI=A
$$

and:

$$
IA=A
$$

when dimensions are compatible.

---

# 23. Inverse Matrix

For certain square matrices, an inverse exists.

The inverse of \(A\) is written:

$$
A^{-1}
$$

and satisfies:

$$
AA^{-1}=I
$$

where \(I\) is the identity matrix.

---

## Simple Analogy

For numbers:

$$
5\times\frac{1}{5}=1
$$

For matrices:

$$
A A^{-1}=I
$$

---

## Important

Not every matrix has an inverse.

A matrix must satisfy specific conditions to be invertible.

One key concept is its determinant.

---

# 24. Norms

## Definition

> A **norm** measures the size or magnitude of a vector.

One common norm is the **L2 norm**.

For:

$$
x=
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

the L2 norm is:

$$
\|x\|_2=
\sqrt{3^2+4^2}
$$

$$
=\sqrt{9+16}
$$

$$
=5
$$

This is essentially the Pythagorean theorem.

---

# 24.1 L1 Norm

The L1 norm is:

$$
\|x\|_1=
\sum_i|x_i|
$$

For:

$$
x=
\begin{bmatrix}
-3\\
4
\end{bmatrix}
$$

we get:

$$
\|x\|_1=|-3|+|4|
$$

$$
=7
$$

---

# 24.2 L2 Norm

The L2 norm is:

$$
\|x\|_2=
\sqrt{\sum_i x_i^2}
$$

It is the usual Euclidean magnitude.

---

# 24.3 Why Norms Matter in ML

Norms are used in:

* Regularization
* Distance calculations
* Optimization
* Similarity calculations
* Geometry of feature spaces

Later we will encounter:

```text
L1 Regularization
L2 Regularization
```

---

# 25. Distance

Distance measures how far apart two points or vectors are.

---

## Euclidean Distance

Suppose:

$$
a=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

and:

$$
b=
\begin{bmatrix}
4\\
6
\end{bmatrix}
$$

Then:

$$
a-b=
\begin{bmatrix}
-3\\
-4
\end{bmatrix}
$$

Euclidean distance is:

$$
\|a-b\|_2
$$

$$
=\sqrt{(-3)^2+(-4)^2}
$$

$$
=5
$$

---

## ML Connection

Distance is important in algorithms such as:

* K-Nearest Neighbors
* K-Means
* Clustering
* Similarity search

---

# 26. Linear Independence

## Definition

> A set of vectors is **linearly independent** if none of the vectors can be represented as a linear combination of the others.

Consider:

$$
v_1=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

and:

$$
v_2=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

These are linearly independent.

Neither can be created by multiplying the other by a scalar.

---

## Dependent Example

Consider:

$$
v_1=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

and:

$$
v_2=
\begin{bmatrix}
2\\
4
\end{bmatrix}
$$

Here:

$$
v_2=2v_1
$$

Therefore they are linearly dependent.

---

# 27. Basis

## Definition

> A **basis** is a set of linearly independent vectors that can be combined to represent every vector in a particular vector space.

For 2D space, a common basis is:

$$
e_1=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

$$
e_2=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

Any 2D vector:

$$
x=
\begin{bmatrix}
a\\
b
\end{bmatrix}
$$

can be written as:

$$
x=ae_1+be_2
$$

---

# 28. Linear Transformations

## Definition

> A **linear transformation** is a function that preserves vector addition and scalar multiplication.

A transformation \(T\) is linear if:

$$
T(a+b)=T(a)+T(b)
$$

and:

$$
T(ca)=cT(a)
$$

---

## Example

Suppose:

$$
T(x)=2x
$$

For:

$$
x=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

we get:

$$
T(x)=
\begin{bmatrix}
2\\
4
\end{bmatrix}
$$

---

## ML Connection

Matrix multiplication can represent linear transformations.

For example:

$$
y=Ax
$$

transforms vector \(x\) into vector \(y\).

This concept becomes fundamental in neural networks.

---

# 29. Eigenvalues

Eigenvalues are more advanced but important for later ML topics.

Suppose we have:

$$
Av=\lambda v
$$

where:

* \(A\) = matrix
* \(v\) = eigenvector
* \(\lambda\) = eigenvalue

The equation says:

> When matrix \(A\) operates on eigenvector \(v\), the direction of \(v\) remains unchanged; it is only scaled by \(\lambda\).

---

# 30. Eigenvectors

An **eigenvector** of a matrix is a non-zero vector whose direction remains unchanged when the matrix is applied to it.

Example conceptually:

```text
Before transformation:

       ↗
      /
     /
    →

After transformation:

       ↗
      /
     /
    →

Same direction,
different magnitude
```

The amount of scaling is the eigenvalue.

---

# 31. Why Eigenvalues Matter in ML

Eigenvalues and eigenvectors appear in important techniques such as:

* Principal Component Analysis (PCA)
* Covariance analysis
* Dimensionality reduction
* Spectral methods
* Some optimization analyses

For example, PCA uses directions associated with important variance in the data.

You don't need to master eigenvalue calculations immediately.

At this stage, remember the concept:

```text
Matrix
   ↓
Eigenvector
   ↓
Direction remains the same
   ↓
Eigenvalue determines scaling
```

---

# 32. Tensors

## Definition

> A **tensor** is a general multidimensional numerical array.

You can think of tensors as a generalization of scalars, vectors, and matrices.

```text
0D → Scalar
1D → Vector
2D → Matrix
3D → 3D Tensor
4D → 4D Tensor
...
```

---

## Example

### Scalar

```text
5
```

Shape:

```text
()
```

### Vector

```text
[1, 2, 3]
```

Shape:

```text
(3,)
```

### Matrix

```text
[
 [1, 2],
 [3, 4]
]
```

Shape:

```text
(2, 2)
```

### 3D Tensor

Imagine multiple matrices stacked together:

```text
[
  [
    [1, 2],
    [3, 4]
  ],

  [
    [5, 6],
    [7, 8]
  ]
]
```

Shape:

```text
(2, 2, 2)
```

---

# 33. Tensors in Machine Learning

Tensors are particularly important in Deep Learning.

## Image

A color image can be represented as:

```text
Height × Width × Channels
```

For example:

```text
224 × 224 × 3
```

where:

```text
224 → Height
224 → Width
3   → RGB channels
```

A batch of images could have:

```text
Batch × Height × Width × Channels
```

For example:

```text
32 × 224 × 224 × 3
```

This means:

```text
32 images
224 pixels high
224 pixels wide
3 color channels
```

---

# 34. NumPy Representation

Python's NumPy library is commonly used for numerical computing.

Import NumPy:

```python
import numpy as np
```

---

## Scalar

```python
x = 5
```

---

## Vector

```python
x = np.array([1, 2, 3])
```

---

## Matrix

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
```

Check its shape:

```python
X.shape
```

Output:

```text
(3, 2)
```

---

# 35. Vector Addition in NumPy

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = a + b

print(result)
```

Output:

```text
[5 7 9]
```

---

# 36. Scalar Multiplication in NumPy

```python
x = np.array([1, 2, 3])

result = 3 * x

print(result)
```

Output:

```text
[3 6 9]
```

---

# 37. Dot Product in NumPy

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)

print(result)
```

Output:

```text
32
```

Modern NumPy also supports:

```python
result = a @ b
```

---

# 38. Matrix Multiplication in NumPy

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

C = A @ B

print(C)
```

Output:

```text
[[19 22]
 [43 50]]
```

---

# 39. Element-wise Multiplication vs Matrix Multiplication

This is a very important distinction.

Suppose:

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])
```

### Element-wise multiplication

```python
A * B
```

gives:

```text
[[ 5 12]
 [21 32]]
```

because:

$$
A_{ij}B_{ij}
$$

is calculated element by element.

---

### Matrix multiplication

```python
A @ B
```

gives:

```text
[[19 22]
 [43 50]]
```

These are completely different operations.

---

# 40. Transpose in NumPy

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.T)
```

Output:

```text
[[1 4]
 [2 5]
 [3 6]]
```

---

# 41. Norm in NumPy

```python
x = np.array([3, 4])

np.linalg.norm(x)
```

Output:

```text
5.0
```

---

# 42. Complete ML Example

Suppose we have three houses.

Features:

```text
Area
Bedrooms
```

Dataset:

| Area | Bedrooms |
| ---: | -------: |
| 1000 |        2 |
| 1500 |        3 |
| 2000 |        4 |

Feature matrix:

$$
X=
\begin{bmatrix}
1000&2\\
1500&3\\
2000&4
\end{bmatrix}
$$

Suppose our model weights are:

$$
w=
\begin{bmatrix}
0.05\\
2
\end{bmatrix}
$$

and:

$$
b=10
$$

The model is:

$$
\hat{y}=Xw+b
$$

First calculate:

$$
Xw=
\begin{bmatrix}
1000(0.05)+2(2)\\
1500(0.05)+3(2)\\
2000(0.05)+4(2)
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
54\\
81\\
108
\end{bmatrix}
$$

Then add bias:

$$
\hat{y}=
\begin{bmatrix}
64\\
91\\
118
\end{bmatrix}
$$

This simple example demonstrates why matrix multiplication is so important in Machine Learning.

---

# 43. Linear Regression Connection

Linear Regression can be written compactly as:

$$
\hat{y}=Xw+b
$$

where:

```text
X → Feature matrix
w → Weight vector
b → Bias/intercept
ŷ → Predictions
```

Instead of calculating every prediction separately, matrix multiplication calculates them together.

This is one of the most important mathematical connections to understand before studying Linear Regression.

---

# 44. Neural Network Connection

A basic neural-network layer can be represented as:

$$
z=XW+b
$$

followed by an activation function:

$$
a=f(z)
$$

So:

```text
Input
  ↓
Matrix Multiplication
  ↓
Add Bias
  ↓
Activation Function
  ↓
Output
```

This is why linear algebra becomes even more important when we reach Deep Learning.

---

# 45. Shape Tracking

One of the most useful skills in ML mathematics is learning to track shapes.

Suppose:

$$
X=(1000\times20)
$$

This means:

```text
1000 samples
20 features
```

Suppose:

$$
w=(20\times1)
$$

Then:

$$
Xw
$$

has shape:

$$
(1000\times20)(20\times1)
$$

Therefore:

$$
Xw=(1000\times1)
$$

So the model produces:

```text
1000 predictions
```

---

# 46. Shape Rules You Must Remember

### Vector

```text
n values → dimension n
```

### Matrix

```text
rows × columns
```

### Matrix multiplication

$$
(m\times n)(n\times p)
$$

produces:

$$
m\times p
$$

### Dataset

If:

```text
n = number of samples
m = number of features
```

then:

$$
X\in\mathbb{R}^{n\times m}
$$

---

# 47. Common Mistakes

## Mistake 1 — Confusing shape

Don't confuse:

```text
(3, 2)
```

with:

```text
(2, 3)
```

They are different.

---

## Mistake 2 — Confusing `*` and `@` in NumPy

```python
A * B
```

means element-wise multiplication.

```python
A @ B
```

means matrix multiplication.

---

## Mistake 3 — Forgetting matrix multiplication dimensions

Remember:

```text
(m × n) × (n × p)
```

is valid.

The middle dimensions must match.

---

## Mistake 4 — Thinking a vector is always a column

In mathematics, vectors can be represented as rows or columns.

In Python/NumPy, a 1D array:

```python
np.array([1, 2, 3])
```

has shape:

```text
(3,)
```

It is neither explicitly `(3,1)` nor `(1,3)`.

This distinction becomes important when working with ML libraries.

---

## Mistake 5 — Treating matrix multiplication as element-wise multiplication

These are different operations.

---

# 48. Quick Revision

## Scalar

One numerical value.

$$
5
$$

## Vector

Ordered collection of values.

$$
[1,2,3]
$$

## Matrix

2D arrangement of values.

$$
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

## Tensor

General multidimensional numerical array.

---

## Dot Product

$$
a\cdot b=\sum_i a_ib_i
$$

Produces a scalar.

---

## Matrix Multiplication

$$
(m\times n)(n\times p)
\rightarrow
(m\times p)
$$

---

## Transpose

Rows become columns.

$$
A\rightarrow A^T
$$

---

## L2 Norm

$$
\|x\|_2=\sqrt{\sum_i x_i^2}
$$

---

## Euclidean Distance

$$
d(a,b)=\|a-b\|_2
$$

---

## Linear Regression

$$
\hat{y}=Xw+b
$$

---

## Neural Network Layer

$$
z=XW+b
$$

---

# 49. Formula Sheet

## Vector Addition

$$
a+b=
\begin{bmatrix}
a_1+b_1\\
a_2+b_2\\
\vdots\\
a_n+b_n
\end{bmatrix}
$$

---

## Scalar Multiplication

$$
ca=
\begin{bmatrix}
ca_1\\
ca_2\\
\vdots\\
ca_n
\end{bmatrix}
$$

---

## Dot Product

$$
a\cdot b=
\sum_{i=1}^{n}a_ib_i
$$

---

## Dot Product Geometry

$$
a\cdot b=
\|a\|\|b\|\cos\theta
$$

---

## L1 Norm

$$
\|x\|_1=
\sum_i|x_i|
$$

---

## L2 Norm

$$
\|x\|_2=
\sqrt{\sum_i x_i^2}
$$

---

## Euclidean Distance

$$
d(a,b)=
\sqrt{\sum_i(a_i-b_i)^2}
$$

---

## Matrix Multiplication

$$
C=AB
$$

where:

$$
C_{ij}=\sum_k A_{ik}B_{kj}
$$

---

## Transpose

$$
(A^T)_{ij}=A_{ji}
$$

---

## Identity

$$
AI=IA=A
$$

---

## Inverse

$$
AA^{-1}=I
$$

---

## Linear Regression

$$
\hat{y}=Xw+b
$$

---

# 50. One-Minute Revision Table

| Concept               | Meaning                                       |
| --------------------- | --------------------------------------------- |
| Scalar                | Single number                                 |
| Vector                | Ordered list of numbers                       |
| Matrix                | Rows and columns of numbers                   |
| Tensor                | Multidimensional array                        |
| Dimension             | Number of values in a vector                  |
| Shape                 | Size along each axis                          |
| Dot Product           | Weighted sum of corresponding elements        |
| Transpose             | Rows ↔ columns                                |
| Matrix Multiplication | Combines matrices using row-column products   |
| Identity Matrix       | Matrix equivalent of 1                        |
| Inverse               | Matrix that produces identity when multiplied |
| Norm                  | Measures vector magnitude                     |
| Distance              | Measures separation between vectors           |
| Linear Independence   | No vector can be formed from the others       |
| Basis                 | Independent vectors spanning a space          |
| Eigenvector           | Direction preserved by a transformation       |
| Eigenvalue            | Scaling factor of an eigenvector              |
| Tensor                | General multidimensional array                |

---

# 🧠 Must-Remember Concepts

If you remember only the following, you have the foundation:

```text
Scalar
   ↓
One number

Vector
   ↓
List of numbers

Matrix
   ↓
Rows + Columns

Tensor
   ↓
Multiple dimensions
```

And:

```text
Dataset
   ↓
Matrix X

One sample
   ↓
Vector x

Model weights
   ↓
Vector w

Prediction
   ↓
Xw + b
```

The most important matrix rule:

```text
(m × n) × (n × p)
          ↓
       (m × p)
```

---

# 📝 Practice Questions

## Basic

### Q1

What is a scalar?

### Q2

What is a vector?

### Q3

What is a matrix?

### Q4

What is the difference between a row vector and a column vector?

### Q5

What does the shape `(1000, 20)` mean for an ML dataset?

---

## Intermediate

### Q6

Calculate:

$$
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
+
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

---

### Q7

Calculate:

$$
3
\begin{bmatrix}
2\\
4\\
5
\end{bmatrix}
$$

---

### Q8

Calculate the dot product:

$$
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
\cdot
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

---

### Q9

Calculate the L2 norm:

$$
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

---

### Q10

Can these matrices be multiplied?

$$
(3\times4)(4\times2)
$$

If yes, what is the output shape?

---

### Q11

Can these matrices be multiplied?

$$
(3\times4)(3\times2)
$$

Explain why or why not.

---

## ML Questions

### Q12

If:

$$
X\in\mathbb{R}^{5000\times20}
$$

what do 5000 and 20 represent in a typical tabular dataset?

---

### Q13

If:

$$
X=(1000\times10)
$$

and:

$$
w=(10\times1)
$$

what is the shape of:

$$
Xw
$$

?

---

### Q14

Explain:

$$
\hat{y}=Xw+b
$$

in terms of Machine Learning.

---

### Q15

Why is matrix multiplication important in neural networks?

---

# 🎯 Mini Coding Practice

Try these without looking at the solution first.

```python
import numpy as np

# 1. Create a vector
x = np.array([1, 2, 3])

# 2. Create a matrix
A = np.array([
    [1, 2],
    [3, 4]
])

# 3. Calculate 2x
# 4. Calculate A transpose
# 5. Calculate x dot [4, 5, 6]
# 6. Calculate A @ A
# 7. Calculate the norm of x
```

Expected concepts:

```python
2 * x
A.T
np.dot(x, np.array([4, 5, 6]))
A @ A
np.linalg.norm(x)
```

---

# ✅ Learning Checklist

* [ ] Understand scalars
* [ ] Understand vectors
* [ ] Understand vector dimensions
* [ ] Understand row vs column vectors
* [ ] Understand matrices
* [ ] Understand matrix shapes
* [ ] Understand feature matrices
* [ ] Perform vector addition
* [ ] Perform scalar multiplication
* [ ] Calculate dot products
* [ ] Understand the geometric meaning of dot product
* [ ] Perform matrix addition
* [ ] Perform matrix multiplication
* [ ] Understand matrix multiplication shape rules
* [ ] Calculate matrix transpose
* [ ] Understand identity matrices
* [ ] Understand matrix inverse conceptually
* [ ] Understand L1 norm
* [ ] Understand L2 norm
* [ ] Calculate Euclidean distance
* [ ] Understand linear independence
* [ ] Understand basis
* [ ] Understand linear transformations
* [ ] Understand eigenvectors conceptually
* [ ] Understand eigenvalues conceptually
* [ ] Understand tensors
* [ ] Use NumPy for basic linear algebra
* [ ] Track matrix/vector shapes
* [ ] Understand \(Xw+b\)

---

# ⏭️ Next Topic

## 06 — Probability & Statistics for Machine Learning

Next we will build the second major mathematical foundation for ML:

```text
Probability
    ↓
Random Variables
    ↓
Probability Distributions
    ↓
Mean
Median
Mode
Variance
Standard Deviation
    ↓
Covariance
Correlation
    ↓
Conditional Probability
    ↓
Bayes' Theorem
    ↓
Expectation
    ↓
Sampling
    ↓
Statistical Thinking
```

These concepts will become especially important when we study:

* Linear Regression
* Logistic Regression
* Classification
* Naive Bayes
* Model evaluation
* Probability distributions
* Hypothesis testing
* Statistical learning
* Deep Learning

