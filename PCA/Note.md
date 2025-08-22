# Statistics
## Mean of dataset
$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$
這是一個常用的遞迴更新公式，用於在加入新資料點 $x_n$ 時更新平均值：
$$ \bar{x}_{n}=\bar{x}_{n-1}+\frac{1}{n}(x_n-\bar{x}_{n-1}) $$

$D = \{x_1, x_2, \dots, x_n\}, \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $
```python
import numpy as np

def reshape(x):
    """return x_reshaped as a flattened vector of the multi-dimensional array x"""
    # 或者用 x.flatten()。
    # 注意：reshape(-1) 在可能的情況下會回傳一個視圖 (view)，不佔用新記憶體，
    # 而 flatten() 總是回傳一個副本 (copy)。
    x_reshaped = x.reshape(-1)
    return x_reshaped

# 範例測試
img = np.arange(28*28).reshape(28, 28)  # 建立一個 28x28 的測試影像
print(reshape(img).shape)  # (784,)
```
### Effect on the mean
D: dataset
- $D+2$ then $E+2$
- $D*2$ then $E*2$

## Variance of dataset
$$ \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 $$
其中 $\mu$ 是數據集的平均值。

對於樣本方差，我們通常使用 $n-1$ 作為分母，以提供無偏估計：
$$ s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$
其中 $\bar{x}$ 是樣本平均值。

$\sigma_{n}^2 = \frac{n-1}{n}\sigma_{n-1}^2 + \frac{1}{n}(x_i - \mu_{n-1})(x_i - \mu_{n})$


$Var[D] = E[(D - \mu)(D-\mu)^T]$

### Effect on the Variance
D: dataset 
- $D*2$ then Variance $is 4*Var(D)$
- $D+2$ then Variance is $Var(D)$

where $A$ is a matrix.

$Var[AD] = E[(AD-A\mu)(AD-\mu)^T] = E[A(D-\mu)A^T(D-\mu)^T]$

$ = E[A(D-\mu)(D-\mu)^TA^T] = AVar[D]A^T$

## Covariance of dataset
$$ Cov(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) $$
其中 $\bar{x}$ 和 $\bar{y}$ 分別是 $X$ 和 $Y$ 的平均值。

對於樣本協方差，我們使用 $n-1$ 作為分母：
$$ s_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) $$

- $Cov(X, Y) = E[(X - E[X])(Y - E[Y])]$
- $Cov(X, Y) = E[XY] - E[X]E[Y]$
- $Cov(X, X) = Var(X)$
- $Cov(aX + b, cY + d) = ac \ Cov(X, Y)$
- 如果 $X$ 和 $Y$ 獨立，則 $Cov(X, Y) = 0$ (反之不一定成立)

$\begin{bmatrix} Cov(x,x) & Cov(x,y) \\ Cov(y,x) & Cov(y,y) \end{bmatrix} = \begin{bmatrix} Var(x) & Cov(x,y) \\ Cov(y,x) & Var(y) \end{bmatrix}$

### Effect on the Covariance
D: dataset
- $D+2$ then Cov(D)
- $D*2$ then Cov(D)

## Correlation
$$ \rho_{XY} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y} $$
其中 $\sigma_X$ 和 $\sigma_Y$ 分別是 $X$ 和 $Y$ 的標準差。

- 相關係數 $\rho_{XY}$ 的值介於 -1 和 1 之間。
- $\rho_{XY} = 1$ 表示完全正線性相關。
- $\rho_{XY} = -1$ 表示完全負線性相關。
- $\rho_{XY} = 0$ 表示沒有線性相關（但可能存在非線性相關）。

# Inner Product
## Dot product
$x^T y = \sum_{i=1}^{n} x_i y_i $ where $x , y \in R^N$

$x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, y = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

$||x|| = \sqrt{x^Tx} = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{1^2 + 2^2} = \sqrt{5}$

$||y|| = \sqrt{\sum_{i=1}^{n} y_i^2} = \sqrt{2^2 + 1^2} = \sqrt{5}$

$d(x,y) = ||x-y|| = \sqrt{(x-y)^T(x-y)} = \sqrt{\sum_{i=1}^{n} (x_i-y_i)^2} = \sqrt{(1-2)^2 + (2-1)^2} = \sqrt{1+1} = \sqrt{2}$

$cos(\theta) = \frac{x^T y}{||x|| ||y||} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}} = \frac{1 \times 2 + 2 \times 1}{\sqrt{5} \times \sqrt{5}} = \frac{4}{5} = 0.8$

$\theta = \arccos(cos(\theta)) = \arccos(0.8) = 0.65 rad$

```python
import numpy as np

def length(x):
  """Compute the length of a vector"""
  length_x = np.linalg.norm(x) # <--- compute the length of a vector x here.
  
  return length_x
  
print(length(np.array([1,0])))
```

### Definition
點積（Dot Product），也稱為純量積（Scalar Product），是兩個向量的乘積，其結果是一個純量。

對於兩個 $n$ 維向量 $\mathbf{a} = [a_1, a_2, \dots, a_n]$ 和 $\mathbf{b} = [b_1, b_2, \dots, b_n]$，它們的點積定義為對應分量的乘積之和：
$$ \mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} $$
或者寫成：$$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i $$這也可以寫成：
$$ \mathbf{a} \cdot \mathbf{b} = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} $$

- Symmetric 交換律：$\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$
- Positive definite : $\mathbf{a} \cdot \mathbf{a} \ge 0$，且 $\mathbf{a} \cdot \mathbf{a} = 0$ 若且唯若 $\mathbf{a} = \mathbf{0}$
- Bilinear 對於兩個參數都是線性的。

#### Example
Let’s check each property of

$$\beta(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T\begin{bmatrix} 2 & -1 \\ -1 & 1 \end{bmatrix} \mathbf{y}$$
1. Bilinearity

    This expression is of the form $x^T A y$ with a constant matrix $A$. Such a function is **bilinear**:

    * Linear in $\mathbf{x}$ when $\mathbf{y}$ is fixed.
    * Linear in $\mathbf{y}$ when $\mathbf{x}$ is fixed.
  
2. Symmetry

    Symmetric means $\beta(x,y) = \beta(y,x)$ for all $x, y$.
    The matrix $$A = \begin{bmatrix} 2 & -1 \\ -1 & 1 \end{bmatrix}$$
    is symmetric ($A = A^T$), so $\beta(x,y) = x^T A y = y^T A x = \beta(y,x).$

3. Positive Definiteness

    An inner product must satisfy $\beta(x,x) > 0$ for all $x \neq 0$.

    Check $ \beta(x,x) = x^T A x$:

    $$x^T A x = x_1^2(2) + x_1x_2(-1) + x_2x_1(-1) + x_2^2(1) = 2x_1^2 - 2x_1x_2 + x_2^2.$$

    Test $x = (1,1)^T$:

    $$= 2(1)^2 - 2(1)(1) + (1)^2 = 2 - 2 + 1 = 1 > 0.$$

    Test $x = (2,4)^T$:

    $$= 2(4) - 2(8) + 16 = 8 - 16 + 16 = 8 > 0.$$

    Test $x = (1,2)^T$:

    $$= 2(1) - 2(2) + 4 = 2 - 4 + 4 = 2 > 0.$$

    Now check for any zero or negative values: the determinant of $A$ is $ (2)(1) - (-1)^2 = 2 - 1 = 1 > 0$ and the leading principal minors are positive ($2 > 0$, det $= 1 > 0$), so $A$ is positive definite.

4. Inner Product

    Since $\beta$ is bilinear, symmetric, and positive definite, it satisfies all inner product axioms.

---

$$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n $$
在幾何學中，點積的定義為：
$$ \mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}|| \ ||\mathbf{b}|| \cos(\theta) $$
其中 $||\mathbf{a}||$ 和 $||\mathbf{b}||$ 分別是向量 $\mathbf{a}$ 和 $\mathbf{b}$ 的長度（或範數），$\theta$ 是兩個向量之間的夾角。

點積的性質：
- **交換律**：$\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$
- **分配律**：$\mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c}$
- **與純量乘法的結合律**：$(c\mathbf{a}) \cdot \mathbf{b} = \mathbf{a} \cdot (c\mathbf{b}) = c(\mathbf{a} \cdot \mathbf{b})$
- **非負性**：$\mathbf{a} \cdot \mathbf{a} \ge 0$，且 $\mathbf{a} \cdot \mathbf{a} = 0$ 若且唯若 $\mathbf{a} = \mathbf{0}$
- **長度平方**：$\mathbf{a} \cdot \mathbf{a} = ||\mathbf{a}||^2$

```python
import numpy as np

def dot(a, b):
  """Compute dot product between a and b.
  Args:
    a, b: (2,) ndarray as R^2 vectors
  
  Returns:
    a number which is the dot product between a, b
  """
  dot_product = np.dot(a, b)
  return dot_product

# Test your code before you submit.
a = np.array([1,0])
b = np.array([0,1])
print(dot(a,b))
```

### Length of vectors
$||x|| = \sqrt{<x,x>}$ where x>=0 $$<x,y> = x^Ty => ||x|| = \sqrt{2}$$
$$<x,y> = x^T \begin{bmatrix} 1 & \frac{-1}{2} \\ \frac{-1}{2} & 1 \end{bmatrix} y = x_1y_1 - \frac{1}{2}(x_2y_1+x_1y_2)+ x_2y_2$$
$$||x|| = \sqrt{x_1^2 - \frac{1}{2}(x_1x_2+x_2x_1)+ x_2^2} = \sqrt{x_1^2 - (x_1x_2)+ x_2^2}$$
$$||x||^2 = 1+1-1 = 1$$
$$||\lambda x|| = \lambda ||x||$$
三角不等式(Triangle inequality)：
$$||x+y|| \le ||x|| + ||y||$$
Cauchy-Schwart inequality：
$$||<x,y>|| \le ||x|| + ||y||$$

### Distances between vectors
$$d(x,y) = ||x-y||$$
### **已知條件**
$$\mathbf{x} = \begin{bmatrix} 1 \\ -1 \\ 3 \end{bmatrix}$$

內積定義：
$$\langle a, b \rangle = a^T M b$$

其中
$$M = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}$$

向量的長度（norm）定義為：

$\|\mathbf{x}\| = \sqrt{\langle x, x \rangle}$

### **步驟 1：計算 $Mx$**

$$
M \cdot x = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ -1 \\ 3 \end{bmatrix} = \begin{bmatrix} 2(1) + 1(-1) + 0(3) \\ 1(1) + 2(-1) + (-1)(3) \\ 0(1) + (-1)(-1) + 2(3) \end{bmatrix} = \begin{bmatrix} 1 \\ -4 \\ 7 \end{bmatrix}$$

### **步驟 2：計算 $x^T (Mx)$**

$$
x^T (Mx) =
\begin{bmatrix} 1 & -1 & 3 \end{bmatrix}
\begin{bmatrix} 1 \\ -4 \\ 7 \end{bmatrix}
= 1(1) + (-1)(-4) + 3(7)
= 1 + 4 + 21
= 26
$$

### **步驟 3：取平方根得到長度**

$$\|\mathbf{x}\| = \sqrt{\langle x, x \rangle} = \sqrt{26}$$
$$\boxed{\sqrt{26}}$$

## Angles & Orthogonality
$$ \cos(\theta) = \frac{\langle x, y \rangle}{||x|| \ ||y||} $$
其中 $\langle x, y \rangle$ 是向量 $x$ 和 $y$ 的內積，$||x||$ 和 $||y||$ 分別是它們的長度。

**正交性 (Orthogonality)**

如果兩個向量的內積為零，則稱它們是正交的。
$$ \langle x, y \rangle = 0 $$
這表示兩個向量之間的夾角為 $90^\circ$（或 $\frac{\pi}{2}$ 弧度），即它們互相垂直。

**範例**

假設我們有兩個向量 $x = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 和 $y = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$。

1.  **計算內積**：
    $$ \langle x, y \rangle = x^T y = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 0 \cdot 1 = 0 $$

2.  **計算長度**：
    $$ ||x|| = \sqrt{1^2 + 0^2} = 1 $$
    $$ ||y|| = \sqrt{0^2 + 1^2} = 1 $$

3.  **計算夾角餘弦**：
    $$ \cos(\theta) = \frac{0}{1 \cdot 1} = 0 $$

4.  **計算夾角**：
    $$ \theta = \arccos(0) = 90^\circ \text{ 或 } \frac{\pi}{2} \text{ 弧度} $$

由於內積為零，向量 $x$ 和 $y$ 是正交的。

```python
# the matrix A defines the inner product
A = np.array([[1, -1/2],[-1/2,5]])
x = np.array([0,-1])
y = np.array([1,1])

def find_angle(A, x, y):
    """Compute the angle"""
    inner_prod = x.T @ A @ y
    # Fill in the expression for norm_x and norm_y below
    norm_x = np.sqrt(x.T @ A @ x)
    norm_y = np.sqrt(y.T @ A @ y)
    alpha = inner_prod/(norm_x*norm_y)
    angle = np.arccos(alpha)
    return np.round(angle,2) 

find_angle(A, x, y)
```
```python
# Fill in the following arrays and use `find_angle` to aim your calculation.
A = np.array([[1,0,0],[0,2,-1],[0,-1,3]])
x = np.array([1,1,1])
y = np.array([2,-1,0])

find_angle(A, x, y)
```

# Orthogonal Projection
## projection onto 1D
1. $\pi_U(x) \in U => \exist \lambda \in R : \pi_U(x) = \lambda b$ (as $\pi_U(x) \in U$)
2. $\langle b, \pi_U(x)-x \rangle = 0$ (orthogonality)
<=> $$\langle b, \pi_U(x) \rangle - \langle b,x \rangle  =0$$ 
<=> $$\langle b,\lambda b \rangle - \lambda \langle b,x \rangle$$
<=> $$\lambda ||b||^2 - \langle b,x \rangle $$
<=> $$\lambda = \frac{\langle b,x \rangle}{||b||^2}$$

    => $\pi_U(x) = \lambda b = \frac{\langle b,x \rangle b}{||b||^2}$

### Example
$\pi_U(x) = \frac{x^Tbb}{||b||^2}$
where $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $b = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

$\pi_U(x) = \frac{4}{5} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

If a projection matrix = $\frac{bb^T}{||b||^2}$ then 
- It is a square matrix, i.e., the number of columns equals the number of rows.
- It is symmetric.($bb^T = (bb^T)^T$)

### Reconstruction error (Euclidean distance)
$x=\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix},$ projection of $x = \frac{1}{9}\begin{bmatrix} 5  \\ 10 \\ 10 \end{bmatrix}$ then x - projection of x = $\begin{bmatrix} 1-\frac{5}{9} \\ 1-\frac{10}{9} \\ 1-\frac{10}{9} \end{bmatrix} = \begin{bmatrix} \frac{4}{9} \\ -\frac{1}{9} \\ -\frac{1}{9} \end{bmatrix}$
$$\sqrt{(\frac{4}{9})^2 + (-\frac{1}{9})^2 + (-\frac{1}{9})^2} = \sqrt{\frac{16}{81}} = \frac{\sqrt{2}}{3} = 0.47... $$

## Projections onto higher-dimensional subspaces
$$u = [b_1, b_2], \pi_U(x) = \lambda_1b_1+\lambda_2b_2$$
$$\langle x-\pi_U(x), b_1 \rangle = 0, \langle x-\pi_U(x), b_2 \rangle = 0$$
---

$\lambda = \begin{bmatrix} \lambda_1 \\ \lambda_2 \\ ... \\ \lambda_M \end{bmatrix}_{M\times1}, B = \begin{bmatrix} b_1 & b_2 & ... & b_M \end{bmatrix}_{D\times M}$
1. $\pi_U(x) = \sum_{i=1}^M \lambda_i b_i$
2. $\langle \pi_U(x) - x, b_i \rangle = 0, i = 1,...,M$

Therefore
$$\pi_U(x) = B\lambda$$

$$\langle \pi_U(x)-x, b_i \rangle = \langle B\lambda - x, b_i \rangle = 0 ?$$

<=> $ \langle B \lambda, b_i \rangle - \langle B \lambda, x \rangle = 0, i = 1, ..., M$

<=> $ \lambda^T B^T bi \lambda - \lambda^T B^T x = 0$

<=> $\lambda^TB^TB - x^TB = 0$

<=> $\lambda^T = x^TB(B^TB)^{-1}$

<=> $\lambda = (B^TB)^{-1}B^Tx$

=> $\pi_U(x) = B(B^TB)^{-1}B^Tx$ → **Projection Matrix**

## projection onto 2D
### Example
where $x = \begin{bmatrix} 2 \\ 1 \\ 1 \end{bmatrix}, b_1 = \begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}, b_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$
$$u = [b_1, b_2], \pi_U(x) = \lambda_1b_1+\lambda_2b_2$$
$$\langle x-\pi_U(x), b_1 \rangle = 0, \langle x-\pi_U(x), b_2 \rangle = 0$$
$$B = \begin{bmatrix} b_1 & b_2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 2 & 1 \\ 0 & 0 \end{bmatrix}$$
$$\lambda = (B^TB)^{-1}B^Tx$$

$B^T = \begin{bmatrix} 1 & 2 & 0 \\ 1 & 1 & 0 \end{bmatrix}, B^Tx = \begin{bmatrix} 4 \\ 3 \end{bmatrix}, B^TB = \begin{bmatrix} 5 & 3 \\ 3 & 2 \end{bmatrix}, (B^TB)^{-1} = \begin{bmatrix} 2 & -3 \\ -3 & 5 \end{bmatrix}$
$$\lambda = \begin{bmatrix} -1 \\ 3 \end{bmatrix}, \pi_U(x) = B\lambda = -1b_1+3b_2 =  \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}$$


---

## 📊 矩陣三兄弟比較表

| 名稱                                     | 定義                 | 計算方式                                | 公式 (以 2×2 為例)                                                                                               | 用途         | 容易搞混的地方                  |
| -------------------------------------- | ------------------ | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------- | ------------------------ |
| **轉置** (Transpose, $A^T$)              | 把行和列互換             | $(A^T)_{ij} = A_{ji}$               | $$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^T = \begin{bmatrix} a & c \\ b & d \end{bmatrix}$$             | 對稱性檢查、內積計算 | **不涉及行列式**，只是位置互換        |
| **伴隨矩陣** (Adjugate, $\mathrm{adj}(A)$) | 把矩陣的**代數餘子式矩陣**再轉置 | 先算每個元素的代數餘子式，再整個矩陣轉置                | $$\mathrm{adj}\begin{bmatrix} a & b \\ c & d \end{bmatrix} = \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$ | 計算逆矩陣的中間步驟 | 很多人以為這是轉置，**其實是完全不同的東西** |
| **逆矩陣** (Inverse, $A^{-1}$)            | 滿足 $AA^{-1} = I$   | $\frac{1}{\det(A)} \mathrm{adj}(A)$ | $$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$                                   | 解線性方程組     | 要求 $\det(A) \neq 0$ 才能存在 |

---