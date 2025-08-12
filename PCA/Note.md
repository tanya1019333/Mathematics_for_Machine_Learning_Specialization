# Statistics
## Mean of dataset
$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$
$\bar{x}_{n+1}=\bar{x}_{n-1}+\frac{1}{n}(x_i-\bar{x}_{n-1})$

$D = \{x_1, x_2, \dots, x_n\}, E= \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $
```python
import numpy as np

def reshape(x):
    """return x_reshaped as a flattened vector of the multi-dimensional array x"""
    x_reshaped = x.reshape(-1)  # 或者用 x.flatten()
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

