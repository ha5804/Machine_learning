## 1. Problem definition

### 1.1. Polynomial regression by least square problems

- data

$$
\{ (x_i, y_i) \}_{i=1}^n = \{ (x_1, y_1), (x_2, y_2), \cdots, (x_n, y_n) \}
$$

- model

$$
\hat{f}(\theta ; x) = \theta_0 x^0 + \theta_1 x^1 + \cdots + \theta_{p-1} x^{p-1}
$$

- model parameters

$$
\theta = (\theta_0, \theta_1, \cdots, \theta_{p-1})
$$

- residual

$$
\gamma_{i}(\theta) = y_i - \hat{f}(\theta ; x_i)
$$

- objective function

$$
\mathcal{L}(\theta) = \frac{1}{2 n} \sum_{i=1}^n \gamma_{i}^2(\theta)
$$

- solution

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
$$

### 1.2. Matrix representation for the polynomial regression

- problem definition in matrix representation

$$
A \, \theta = y
$$

$$
A =
\begin{bmatrix}
x_{1}^{0} & x_{1}^{1} & \cdots & x_{1}^{p-1} \\
x_{2}^{0} & x_{2}^{1} & \cdots & x_{2}^{p-1} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n}^{0} & x_{n}^{1} & \cdots & x_{n}^{p-1}
\end{bmatrix},
\quad
\theta =
\begin{bmatrix}
\theta_{0} \\
\theta_{1} \\
\vdots \\
\theta_{p-1}
\end{bmatrix},
\quad
y =
\begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{n}
\end{bmatrix}
$$

where $p \in \mathbb{N}$ denotes the power of the polynomial function

- objective function

$$
\begin{align*}
\mathcal{L}(\theta) &= \frac{1}{2 n} \| A \, \theta - y \|_2^2\\
&= \frac{1}{2 n} ( A \, \theta - y )^T ( A \, \theta - y )\\
&= \frac{1}{2 n} ( \theta^T A^T - y^T ) ( A \, \theta - y )
\end{align*}
$$

- solution

$$
\begin{align*}
\nabla \mathcal{L}(\theta^*) = \frac{1}{n} ( A^T A \theta^* - A^T y) = 0\\
( A^T A \theta^* - A^T y) = 0\\
A^T A \theta^* = A^T y\\
\theta^* = (A^T A)^{-1} (A^T y)
\end{align*}
$$