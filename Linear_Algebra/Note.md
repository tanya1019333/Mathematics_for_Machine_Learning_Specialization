## Motivations
1. Simultaneous equations
2. Optimization Problem in Regression
<img src="./optimizeEquation.png" width="200">

# Vectors
### What is a vector?

假設有兩個向量：
  1. 要投影的向量: $\vec{a}$ 
  2. 投影的方向: $\vec{b}$
  - 向量投影的意思是：找出一個跟 $\vec{b}$ 方向相同的向量，長度等於 $\vec{a}$ 在 $\vec{b}$ 上的影子。

1. #### *A list of numbers*
    - Vectors are usually viewed by computers as an ordered list of numbers which they can perform "operations" on - some operations are very natural and, as we will see, very useful!

2. #### *Position in three dimensions of space and in one dimension of time*
    - A vector in space-time can be described using 3 dimensions of space and 1 dimension of time according to some co-ordinate system.

3. #### *Something which moves in a space of fitting parameters*
    - Vectors can be viewed as a list of numbers which describes some optimisation problem.
4. #### *Commutative of Vectors*
    - Example: Let’s consider two vectors:

      $\vec{a} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \vec{b} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$  

      - `Vector Addition:`

        $\vec{a} + \vec{b} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 2+4 \\ 3+1 \end{bmatrix} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}$

      - `Vector Subtraction:`

        $\vec{a} - \vec{b} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} - \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 2-4 \\ 3-1 \end{bmatrix} = \begin{bmatrix} -2 \\ 2 \end{bmatrix}$

      - `Scalar Multiplication:`

        $3\vec{a} = 3 \begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 3 \times 2 \\ 3 \times 3 \end{bmatrix} = \begin{bmatrix} 6 \\ 9 \end{bmatrix}$

      - `Dot Product (Scalar Product):`

        $\vec{a} \cdot \vec{b} = (2)(4) + (3)(1) = 8 + 3 = 11$

      - `Magnitude (Length) of a Vector:`

        $|\vec{a}| = \sqrt{2^2 + 3^2} = \sqrt{4 + 9} = \sqrt{13}$
        
        $|\vec{b}| = \sqrt{4^2 + 1^2} = \sqrt{16 + 1} = \sqrt{17}$

      - `Unit Vector:`

        $\hat{a} = \frac{\vec{a}}{|\vec{a}|} = \frac{1}{\sqrt{13}} \begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 2/\sqrt{13} \\ 3/\sqrt{13} \end{bmatrix}$

      - `Commutativity of Vector Addition:`

        $\vec{a} + \vec{b} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 2+4 \\ 3+1 \end{bmatrix} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}$

      ✅ The result is the same regardless of the order:
      > $\vec{a} + \vec{b} = \vec{b} + \vec{a}$

5. #### *Vectors are "orthogonal" or "perpendicular"* 
    ### $\vec{a} \cdot \vec{b}=0$

---
### *Projection of a Vector Using Cosine* 
  1. #### *The `scalar projection` of $\vec{a}$ onto $\vec{b}$ is: $|\vec{a}| \cos \theta$*

      since $\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos \theta$ , $\frac{\vec{a} \cdot \vec{b}}{|\vec{b}|} = |\vec{a}| \cos \theta$
      - How much of $\vec{a}$ is in the direction of $\vec{b}$ (length only). This represents the **length of the shadow** of $\vec{a}$ onto the direction of $\vec{b}$.

  2. #### *The `vector projection` further scales this shadow in the direction of $\vec{b}$.* 
    
      (The actual vector pointing in the same direction as $\vec{b}$)

      since $\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos \theta => \frac{\vec{a} \cdot \vec{b}}{|\vec{b}|} = |\vec{a}| \cos \theta$
      - $\frac{\vec{a} \cdot \vec{b}}{|\vec{b}|} \times \frac{\vec{b}}{|\vec{b}| } = \frac{\vec{a} \cdot \vec{b}}{|\vec{b}|^2} \vec{b} = \vec{a} \cos \theta \cdot \frac{\vec{b}}{|\vec{b}|}$

      - The projection of $\vec{a}$ onto $\vec{b}$ can be calculated using the following formula:

        ### $\text{proj}_{\vec{b}} \vec{a} = \frac{\vec{a} \cdot \vec{b}}{|\vec{b}|^2} \vec{b}$

      - Using the cosine of the angle between the two vectors, the formula can also be expressed as:
  
        ### $\text{proj}_{\vec{b}} \vec{a} = \frac{|\vec{a}| \cdot |\vec{b}| \cdot \cos \theta}{|\vec{b}|^2} \vec{b} = \frac{|\vec{a}| \cos \theta}{|\vec{b}|} \vec{b}$

        ### = (scaler projection of $\vec{a}$ onto $\vec{b}$) $\cdot \frac{\vec{b}}{|\vec{b}}$ 

        Where $\vec{a} \cdot \vec{b}$ is the dot product, $|\vec{a}|$ and $|\vec{b}|$ are the magnitudes, and $\theta$ is the angle between $\vec{a}$ and $\vec{b}$.
        
---
### *Changing basis*
- Example
    
     $\vec{V} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$
    Suppose we want to express **v** in a new basis: 
    
    $\vec{b}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad \vec{b}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ Find $\vec{V_b}$ ?
  1. #### Find the vector projection of $\vec{V}$ onto $\vec{b}_1$:
    
      $\text{proj}_{\vec{b_1}} \vec{V} = \frac{\vec{V} \cdot \vec{b_1}}{|\vec{b_1}|^2} \vec{b_1}$
        $\vec{V} \cdot \vec{b_1} = (4)(1) + (2)(1) = 6$

      $|\vec{b_1}|^2 = 1^2 + 1^2 = 2$
      
      $\text{proj}_{\vec{b_1}} \vec{V} = \frac{6}{2} \vec{b_1} = 3 \vec{b_1} = \begin{bmatrix} 3 \\ 3 \end{bmatrix}$

  2. #### Find the vector projection of $\vec{V}$ onto $\vec{b}_2$:
      $\text{proj}_{\vec{b_2}} \vec{V} = \frac{\vec{V} \cdot \vec{b_2}}{|\vec{b_2}|^2} \vec{b_2}$
      
      $\vec{V} \cdot \vec{b_2} = (4)(1) + (2)(-1) = 2$
      
      $|\vec{b_2}|^2 = 1^2 + (-1)^2 = 2$
      
      $\text{proj}_{\vec{b_2}} \vec{V} = \frac{2}{2} \vec{b_2} = 1 \vec{b_2} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

  3. #### Express $\vec{V}$ as a linear combination of $\vec{b_1}$ and $\vec{b_2}$:
      $\vec{V} = c_1 \vec{b_1} + c_2 \vec{b_2}$
      We know that $c_1 = \frac{\vec{V} \cdot \vec{b_1}}{|\vec{b_1}|^2}$ and $c_2 = \frac{\vec{V} \cdot \vec{b_2}}{|\vec{b_2}|^2}$ 
      if the basis vectors are orthogonal.
      
      In this case, $\vec{b_1} \cdot \vec{b_2} =(1)(1) + (1)(-1) = 1 - 1 = 0$. Since the dot product is 0, $\vec{b_1}$ and $\vec{b_2}$ are orthogonal. Therefore, $c_1 = 3$ and $c_2 = 1$.
      So, $\vec{V} = 3 \vec{b_1} + 1 \vec{b_2}$.
      
      The coordinates of $\vec{V}$ in the new basis are $\vec{V_b} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$.
---
### *Linear Independent*
- What does it mean for a $\vec{b_3}$  to be linearly independent to the $\vec{b_1}$ and $\vec{b_2}$？
  1. $\vec{b_3}$  does not lie in the plane spanned by $\vec{b_1}$ and $\vec{b_2}$. This is a geometric way of understanding linear independence.
  2. $\vec{b_3}$  is not equal to $a_1 \vec{b_1} + a_2\vec{b_2}$ , for any a1  or a2 . This is an algebraic way of understanding linear independence.
  
  一組向量中，沒有任何一個向量可以用其他向量的線性組合表示。

  ✅ 彼此完全不重複方向。

  ❌ 如果有一個向量是其他向量的延伸（倍數或組合），就不是線性獨立。
  
- Given $\vec{v_1}, \vec{v_2},\vec{v_3},...,\vec{v_n}$ , If these  vectors are Linear Independent then $c_1 \vec{v_1} + c_2 \vec{v_2} + ... + c_n \vec{v_n} = 0$
  
  The Only solution is $c_1 = c_2 = ... = c_n = 0$ `如果有非零解 → 向量線性獨立`
  
  Ways to check if linear independent:
        
  **Example**: $\vec{v_1} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ ,   $\vec{v_2} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$

  1. $a \vec{v_1} + b \vec{v_2} = 0$ (Check if the only solution is $a=0, b=0$)
  
      $a \begin{bmatrix} 1 \\ 2 \end{bmatrix} + b \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} → \begin{cases} a + 3b = 0 \quad (1) \\ 2a + 4b = 0 \quad (2) \end{cases}$
      
      Since the only solution is $a=0$ and $b=0$, the vectors $\vec{v_1}$ and $\vec{v_2}$ are **linearly independent**.
          
  2. 矩陣行列式檢查
      - 把向量寫成矩陣。
      - 計算行列式（determinant）。
      - 如果行列式 ≠ 0 → 向量線性獨立。
      - 如果行列式 = 0 → 向量線性依賴。
          
      -> combine as Matrix $A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$

      -> determinant $|A| = 1 \times 4 - 2 \times 3 = -2 \neq 0$ 
      
      → 向量線性獨立 (Since the determinant is non-zero, the vectors are linearly independent.)

  3.  Row Reduction（高斯消去法）
      - 把向量寫成矩陣。
      - 用 row reduction 化簡成行階梯型。
      - 如果每個向量都產生一個 pivot（主元） → 線性獨立。

| 結果      | 說明   |
| ------- | -------- |
| 行列式 ≠ 0 | 線性獨立     |
| 行列式 = 0 | 線性依賴 |
| 唯一解     | 線性獨立 (for $c_1 \vec{v_1} + ... + c_n \vec{v_n} = 0$) |
| 有無窮多解   | 線性依賴 (for $c_1 \vec{v_1} + ... + c_n \vec{v_n} = 0$) |
---
### *Geometric Transformations and Matrix Multiplication*
1. Why Learn Geometric Transformations?
  - **Any shape change** (like modifying an image) can be built using **combinations** of:
    - Rotations
    - Shears
    - Inverses (reflections)
  - Composition of Transformations
    If I apply $A_1$ to vector $r$, and then apply $A_2$:
    - This is written as $A_2 \times A_1 \times r$
    - The `order matters` (matrix multiplication is not commutative)
2. Example: 90° Rotation and Vertical Reflection
  - 逆時鐘： $A = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
  - 順時鐘： $A = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}$

  -  Basis Vectors: 
    $e_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ , 
    $e_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
  -  First Transformation: 90° Anticlockwise Rotation (A1)

      $e_1 \to e'_1 = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$ , 
      $e_2 \to e'_2 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
      
      Matrix: $A_1 = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$
  - Second Transformation: Vertical Reflection (A2)

      $e_1 \to e'_1 = \begin{bmatrix} -1 \\ 0 \end{bmatrix}$ , 
      $e_2 \to e'_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
      
      Matrix: $\vec{A_2} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$

3. Matrix Composition: 

    - Apply A2 to A1 Calculate: 
    
      $A_2 \times A_1 = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} \times \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ , 

      Result: $A_2 A_1 = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$

    - Non-Commutativity
    
      If we reverse the order: $A_1 \times A_2 = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \times \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$

      Result: $A_1 A_2 = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \times \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$
    - **The two results are not the same.**
    - `Matrix multiplication is not commutative.`

#### Important Notes:
- ✅ **Matrix multiplication is associative.**
    - ### $A_3 (A_2 A_1) = (A_3 A_2) A_1$
- ❌ **Matrix multiplication is not commutative.**
    - ### $A_2 A_1 \neq A_1 A_2$

#### Key Insight:
> Matrix multiplication is the foundation for **geometric transformations** and solving **simultaneous equations.**
>  
> Understanding how matrices transform vectors is the **heart of linear algebra.**

  - #### Example
    - $\vec{v_1} = \begin{bmatrix} 2 \\ 10 \end{bmatrix}$, $\vec{v_2} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$ and $\vec{V} = \begin{bmatrix} 8 \\ 13 \end{bmatrix}$
    - $\begin{bmatrix} 2 & 3 \\ 10 & 1 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} 8 \\ 13 \end{bmatrix} => \begin{cases} 2a + 3b = 8 \quad (1) \\ 10a + b = 13 \quad (2) \end{cases}$
    Therefore, the solution is $a = \frac{31}{28}$ and $b = \frac{27}{14}$.

---
### *Matrix Transform Space*
- A matrix can transform a vector space by changing the basis vectors.
- Example:
  - Given a matrix $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and a vector $\vec{v} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$, the transformed vector is:
  - $A \cdot \vec{v} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$
  
---
### *Matrix Inverse*
1. Gaussian Elimination:

    where $A = \begin{bmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{bmatrix} \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 15 \\ 21 \\ 13 \end{bmatrix}$

    $A^{-1} \cdot A \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} = I \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 5 \\ 4 \\ 2 \end{bmatrix} , therefore$
    `a = 5, b = 4, c = 2`
    
2. The inverse of a matrix $A$, denoted as $A^{-1}$, is a matrix that, when multiplied by $A$, yields the identity matrix $I$.
    ### $A \cdot r = S → A^{-1} \cdot A \cdot r = A^{-1} \cdot S$
    - Only square matrices can have an inverse.
    - A matrix has an inverse if and only if its determinant is non-zero. If the determinant is zero, the matrix is singular and has no inverse.

    - For a 2x2 matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:
      $A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$
      where $ad - bc$ is the determinant of $A$.

    - For larger matrices, methods like Gaussian elimination or cofactor expansion are used to find the inverse.

    - **Properties of Matrix Inverse:**
      - $(A^{-1})^{-1} = A$
      - $(AB)^{-1} = B^{-1}A^{-1}$
      - $(A^T)^{-1} = (A^{-1})^T$
      - $IA = AI = A$ (Identity Matrix)

    - **Solving Systems of Linear Equations using Inverse:**

      - If you have a system of linear equations in the form $Ax = b$, where $A$ is the coefficient matrix, $x$ is the vector of unknowns, and $b$ is the constant vector, you can solve for $x$ by multiplying both sides by $A^{-1}$:
      
      - $A^{-1}Ax = A^{-1}b$

      - $Ix = A^{-1}b$

      - $x = A^{-1}b$

    - Find the Inverse Matrix of $A^{-1}$: where
    $A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 1 \\ 3 & 4 & 2 \end{bmatrix}$

    $\quad(1) \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 1 \\ 3 & 4 & 2 \end{bmatrix} \cdot \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} = I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = A \cdot A^{-1}$

    $\quad(2) \begin{bmatrix} 1 & 1 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & -1 \end{bmatrix} \cdot \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ -1 & 0 & 1 \end{bmatrix}$

    $\quad(3) \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} = \begin{bmatrix} -2 & 0 & 3 \\ -2 & 1 & 1 \\ 1 & 0 & -1 \end{bmatrix}$
    
    $\quad(4) \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} = \begin{bmatrix} 0 & -1 & 2 \\ -2 & 1 & 1 \\ 1 & 0 & -1 \end{bmatrix} = I \cdot A^{-1} = A^{-1}$
```python
import numpy as np
A = [[4, 6, 2],
     [3, 4, 1],
     [2, 8, 13]]
s1 = 9; s2 = 7; s3 = 2
s = [s1, s2, s3]

r = np.linalg.solve(A, s)
print(r)
```
---
### *Determinant*
<img src="./Determinant.png" width="300">

$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, Area = (a+b)(c+d)-ac-bd-2bc = ac+ad+bc+bd -ac-bd-2bc = ad-bc = |A|$
- Find $A^{-1}$:
    $A \cdot A^{-1} = I = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \cdot A^{-1} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$

    $\begin{bmatrix} \frac{ad-bc}{d} & 0 \\ 0 & \frac{ad-bc}{a} \end{bmatrix} \begin{bmatrix} 1 & \frac{-b}{d} \\ \frac{-c}{a} & 1 \end{bmatrix}$

    $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \frac{d}{ad-bc} & \frac{-b}{ad-bc} \\ \frac{-c}{ad-bc} & \frac{a}{ad-bc} \end{bmatrix}$

    ### $A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

- If det = 0, then it's singular matrix（奇異矩陣)

- 這個矩陣描述的空間壓縮了維度，例如：3D 壓扁成 2D，或 2D 壓扁成 1D。

| 條件      | 結果                 |
| ------- | ------------------ |
| det ≠ 0 | 有反矩陣，線性獨立，可解唯一解    |
| det = 0 | 沒有反矩陣，線性依賴，無解或無限多解 |

[Practice of Converting a matrix to echelon form](./IdentifyingSpecialMatrices.ipynb)

---
### *Einstein summation convention*
$A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n} \\ a_{21} & a_{22} & ... & a_{2n} \\ ... & ... & ... & ... \\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix}, B = \begin{bmatrix} b_{11} & b_{12}& ... & b_{1n} \\ b_{21} & b_{22} & ... & b_{2n} \\ ... & ... & ... & ... \\ b_{m1} & b_{m2} & ... & b_{mn} \end{bmatrix}$

$(ab)_{23} = a_{21}b_{13} + a_{22}b_{23} + ... + a_{2n}b_{n3} = \sum_{i=1}^{n} a_{2i}b_{i3}$

With $\vec{v} \cdot \vec{u}  = v_{1} \cdot u_{1} ...v_{i} \cdot u_{i} = $

#### 🌞 陰影投影問題重點
- 陽光方向用單位向量 \( \hat{s} \) 表示。
- 物體上的點 \( \vec{r} \) 投影到地面的點為 \( \vec{r'} \)。

#### 投影公式推導：
地面條件：
$r'_3 = 0$

光線公式：
$\vec{r'} = \vec{r} + \lambda \hat{s}$

求出： $\lambda = - \frac{\vec{r} \cdot \hat{e}_3}{s_3}$, 
since $r'_3 = 0 = \vec{r} \cdot \hat{e}_3 + \lambda \hat{s}_3$

投影公式：
$\boxed{\vec{r'} = \vec{r} - \frac{\hat{s} (\vec{r} \cdot \hat{e}_3)}{s_3}}$

### 📐 矩陣表示法

### Einstein Summation 公式：
### $r'_i = \left( I_{ij} - \frac{s_i [e^3]_j}{s_3} \right) r_j$

這可以寫成矩陣乘法：
$\vec{r'} = A \vec{r}$

- 投影矩陣 A 的形式（只取前兩行）

  $A = \begin{bmatrix} 1 & 0 & -\frac{s_1}{s_3} \\ 0 & 1 & -\frac{s_2}{s_3} \end{bmatrix}$
  
  投影點已經在地面（z = 0），只需要 x, y 兩個座標。

### ✅ 重要觀念
- **非方陣矩陣可以實現降維映射**（如：3D → 2D）。
- 矩陣乘法可以套用到**多個向量的集合**，使用矩陣相乘處理更有效率。

- 常見錯誤提醒
  1. 投影矩陣第三列會是 $[0, 0, 0]$ 因為 z 座標被壓到 0。
  2. 計算單一向量與批量向量的邏輯一致，差別在是否用矩陣一次計算。

### Calculation of Projected Points
```python
import numpy as np
# Given values
s = np.array([4/13, -3/13, -12/13])
r = np.array([6, 2, 3])
# Compute projection matrix A
A = np.array([
    [1, 0, -s[0]/s[2]],
    [0, 1, -s[1]/s[2]]
])
# Apply A to r
rp = A @ r
# Prepare result for Q6
R = np.array([
    [5, -1, -3, 7],
    [4, -4, 1, -2],
    [9, 3, 0, 12]
])
Rp = A @ R
rp, Rp
```
---
### *Matrices Changing basis*
- Bear's basis vector: $\begin{bmatrix} 3 \\ 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ in my frame.
- Vector that want to transfrom: $\begin{bmatrix} 3/2 \\ 1/2 \end{bmatrix}$
   $\begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix} (Bear's \ basis \ in \ my \ coordinates) \cdot \begin{bmatrix} 3/2 \\ 1/2 \end{bmatrix} (Bear's \ vector) = \begin{bmatrix} 5 \\ 2 \end{bmatrix} (My \ vector)$
  
  $B^{-1} \cdot my vector = \frac{1}{3 \cdot 1 - 1 \cdot 1} \cdot \begin{bmatrix} 1 & -1 \\ -1 & 3 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 2 \end{bmatrix} = \begin{bmatrix} 3/2 \\ 1/2 \end{bmatrix}$
  
  Projections: $\begin{bmatrix} 3/2 \\ 1/2 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ 1 \end{bmatrix} = 9/2 + 1/2 = 5 ; \begin{bmatrix} 3/2 \\ 1/2 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 3/2 + 1/2 = 2$
- Transfrom in changed basis:
  $For \ 45 \degree : R = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}$ in my frame.
  $R B = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 2 & 0 \\ 4 & 2 \end{bmatrix} = \sqrt{2} \begin{bmatrix} 1 & 0 \\ 2  & 1 \end{bmatrix}$
  
  $B^{-1} R B = \begin{bmatrix} 1/2 & -1/2 \\ -1/2 & 3/2 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix} = R_{B} = \frac{\sqrt{2}}{2} \begin{bmatrix} -1 & -1 \\ 5 & 3 \end{bmatrix}$
---
### *Orthogonal Matrices*
$A^{T}_{ij} = A_{ji}$

Then $A^{T}_{ij} \cdot A_{ij} = I$ (identity matrix), therefore, $A^{T}_{ij} = A^{-1}$

---
### *The Gram–Schmidt process*
The Gram–Schmidt process is a method for orthonormalizing a set of vectors in an inner product space, typically Euclidean space $\mathbb{R}^n$ with the standard dot product. It takes a finite, linearly independent set of vectors $S = \{v_1, ..., v_k\}$ and generates an orthogonal set $S' = \{u_1, ..., u_k\}$ that spans the same subspace as $S$. If you further normalize each vector in $S'$, you get an orthonormal set.

vector projection of $\vec{v_2}$ onto $\hat{e_1} = \frac{\vec{v_2} \cdot \hat{e_1}}{|\hat{e_1}|} = \vec{v_2} \cdot \hat{e_1}$

**Steps:**

Given a set of linearly independent vectors $\{v_1, v_2, ..., v_k\}$:

1.  **First vector:**
    The first orthogonal vector $u_1$ is simply the first vector $v_1$:
    $u_1 = v_1$

2.  **Second vector:**
    The second orthogonal vector $u_2$ is $v_2$ minus its projection onto $u_1$:
    $u_2 = v_2 - \text{proj}_{u_1} v_2 = v_2 - \frac{v_2 \cdot u_1}{u_1 \cdot u_1} u_1$

3.  **Third vector:**
    The third orthogonal vector $u_3$ is $v_3$ minus its projections onto $u_1$ and $u_2$:
    $u_3 = v_3 - \text{proj}_{u_1} v_3 - \text{proj}_{u_2} v_3 = v_3 - \frac{v_3 \cdot u_1}{u_1 \cdot u_1} u_1 - \frac{v_3 \cdot u_2}{u_2 \cdot u_2} u_2$

4.  **General step (for $j$-th vector):**
    For any $j > 1$, the orthogonal vector $u_j$ is $v_j$ minus its projections onto all previously found orthogonal vectors $u_1, ..., u_{j-1}$:
    $u_j = v_j - \sum_{i=1}^{j-1} \text{proj}_{u_i} v_j = v_j - \sum_{i=1}^{j-1} \frac{v_j \cdot u_i}{u_i \cdot u_i} u_i$

#### Example: Reflecting in a plane in shortnote
A plane can be defined by a point $P_0$ on the plane and a normal vector $\vec{n}$ to the plane. The equation of the plane is $\vec{n} \cdot (\vec{x} - P_0) = 0$.

To reflect a vector $\vec{v}$ across a plane, we can use the following formula:

$\text{Reflect}_{\vec{n}}(\vec{v}) = \vec{v} - 2 \text{proj}_{\vec{n}} \vec{v}$

Where $\text{proj}_{\vec{n}} \vec{v} = \frac{\vec{v} \cdot \vec{n}}{\vec{n} \cdot \vec{n}} \vec{n}$ is the vector projection of $\vec{v}$ onto the normal vector $\vec{n}$.

So, the reflection formula becomes:

$\text{Reflect}_{\vec{n}}(\vec{v}) = \vec{v} - 2 \frac{\vec{v} \cdot \vec{n}}{\|\vec{n}\|^2} \vec{n}$

This formula calculates the reflected vector assuming the plane passes through the origin. If the plane does not pass through the origin, you need to adjust the vector $\vec{v}$ relative to a point on the plane, reflect it, and then translate it back.

Let $P$ be the point to be reflected, $P_0$ be a point on the plane, and $\vec{n}$ be the normal vector to the plane.
1. Translate $P$ so that $P_0$ is the origin: $\vec{v} = P - P_0$.
2. Reflect $\vec{v}$ across the plane through the origin with normal $\vec{n}$:
   $\vec{v}_{\text{reflected}} = \vec{v} - 2 \frac{\vec{v} \cdot \vec{n}}{\|\vec{n}\|^2} \vec{n}$
3. Translate back: $P_{\text{reflected}} = \vec{v}_{\text{reflected}} + P_0$.

### *Eigen values/vectors*
Eigenvalues and eigenvectors are fundamental concepts in linear algebra that describe how a linear transformation stretches or compresses vectors.

→ 零向量永遠不是 eigenvector（因為無法定義方向或比例）。

-   **Eigenvector**: An eigenvector of a linear transformation is a non-zero vector that changes at most by a scalar factor when that linear transformation is applied to it. It only scales, it doesn't change direction.

-   **Eigenvalue**: The scalar factor by which an eigenvector is scaled is called its eigenvalue.

Mathematically, for a square matrix $A$, a non-zero vector $\vec{v}$ is an eigenvector of $A$ if there exists a scalar $\lambda$ (lambda) such that:

### $A\vec{v} = \lambda\vec{v}$

Where:
-   $A$ is an $n \times n$ matrix (the linear transformation).
-   $\vec{v}$ is the eigenvector (a non-zero vector).
-   $\lambda$ is the eigenvalue (a scalar).

#### Finding Eigenvalues and Eigenvectors:

To find the eigenvalues, we rearrange the equation:

$A\vec{v} - \lambda\vec{v} = \vec{0}$
$(A - \lambda I)\vec{v} = \vec{0}$

For a non-zero eigenvector $\vec{v}$ to exist, the matrix $(A - \lambda I)$ must be singular (non-invertible), which means its determinant must be zero:

### $\text{det}(A - \lambda I) = 0$

This equation is called the **characteristic equation**. Solving it for $\lambda$ gives the eigenvalues. Once the eigenvalues are found, they can be substituted back into $(A - \lambda I)\vec{v} = \vec{0}$ to find the corresponding eigenvectors.

#### Geometric Interpretation:
-   Eigenvectors are the "special" directions in space that are simply scaled by the transformation, without being rotated or sheared.
-   Eigenvalues tell us how much the eigenvectors are scaled. A positive eigenvalue means stretching, a negative eigenvalue means stretching and flipping, and an eigenvalue of 1 means no change. An eigenvalue of 0 means the vector is collapsed to the origin.

#### Equations:
- When $C = \begin{bmatrix} x1 & x2 & x3 \\ ... & ... & ... \end{bmatrix}$, $D = \begin{bmatrix} \lambda1 & 0 & 0 \\ 0 & \lambda2 & 0 \\ 0 & 0 & \lambda3 \end{bmatrix}$, $T^n = \begin{bmatrix} a^n & 0 & 0 \\ 0 & b^n & 0 \\ 0 & 0 & c^n \end{bmatrix}$

  then $T = C D C^{-1} → T^2 = C D C^{-1} C D C^{-1} = C D^2 C^{-1}$
  
  $→ T^n = C D^n C^{-1}$

#### Pagerank Algorithm
- PageRank is an algorithm used by Google Search to rank web pages in their search engine results. It works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying idea is that more important websites are likely to receive more links from other websites.

Mathematically, PageRank can be viewed as an eigenvector problem. Each web page is a node in a directed graph, and hyperlinks are edges. The PageRank of a page is calculated iteratively, considering the PageRank of pages linking to it and the number of outgoing links from those pages.

Let $P_i$ be the PageRank of page $i$. The basic PageRank formula is:

$P_i = \sum_{j \in B_i} \frac{P_j}{L_j}$

Where:
- $B_i$ is the set of pages that link to page $i$.
- $L_j$ is the number of outgoing links from page $j$.

To handle issues like pages with no outgoing links (dangling nodes) and to ensure convergence, a damping factor $d$ (typically set to 0.85) is introduced, and a small constant value is added to each page's rank:

$P_i = (1-d) + d \sum_{j \in B_i} \frac{P_j}{L_j}$

In matrix form, this can be expressed as:

$\mathbf{P} = (1-d)\mathbf{1} + d\mathbf{M}\mathbf{P}$

Where:
- $\mathbf{P}$ is the PageRank vector, where $P_i$ is the PageRank of page $i$.
- $\mathbf{1}$ is a vector of all ones.
- $\mathbf{M}$ is the transition matrix of the web graph, where $M_{ij} = 1/L_j$ if page $j$ links to page $i$, and 0 otherwise.

This equation can be rewritten as an eigenvalue problem:

$\mathbf{P} = d\mathbf{M}\mathbf{P} + (1-d)\mathbf{1}$

This is a variation of the standard eigenvalue problem, but it can be solved iteratively until $\mathbf{P}$ converges. The PageRank vector $\mathbf{P}$ is the principal eigenvector of a modified transition matrix.

#### Applications:
-   **Principal Component Analysis (PCA)**: Used in data science for dimensionality reduction. Eigenvectors of the covariance matrix represent the principal components (directions of maximum variance), and eigenvalues represent the amount of variance along those directions.




---
### Applications
  - linear equation between the differences with the data points
  - Machine Learning: Used extensively in algorithms like Linear Regression, Principal Component Analysis (PCA), Support Vector Machines (SVMs), and Neural Networks.
  - Computer Graphics: For transformations (scaling, rotation, translation) of objects in 2D and 3D space.
  - Physics and Engineering: Solving systems of equations, analyzing forces, and modeling dynamic systems.

---
### Project
1. calculator of vectors includes addition/subtraction/multiplication/fraction
2. 
---
### Reference
| 角度 $\theta$ | 弧度 $\theta$       | $\sin \theta$         | $\cos \theta$         | 所在象限 |
| ----------- | ----------------- | --------------------- | --------------------- | ---- |
| 0°          | 0                 | 0                     | 1                     | 第一象限 |
| 30°         | $\frac{\pi}{6}$   | $\frac{1}{2}$         | $\frac{\sqrt{3}}{2}$  | 第一象限 |
| 45°         | $\frac{\pi}{4}$   | $\frac{\sqrt{2}}{2}$  | $\frac{\sqrt{2}}{2}$  | 第一象限 |
| 60°         | $\frac{\pi}{3}$   | $\frac{\sqrt{3}}{2}$  | $\frac{1}{2}$         | 第一象限 |
| 90°         | $\frac{\pi}{2}$   | 1                     | 0                     | 第一象限 |
| 120°        | $\frac{2\pi}{3}$  | $\frac{\sqrt{3}}{2}$  | $-\frac{1}{2}$        | 第二象限 |
| 135°        | $\frac{3\pi}{4}$  | $\frac{\sqrt{2}}{2}$  | $-\frac{\sqrt{2}}{2}$ | 第二象限 |
| 150°        | $\frac{5\pi}{6}$  | $\frac{1}{2}$         | $-\frac{\sqrt{3}}{2}$ | 第二象限 |
| 180°        | $\pi$             | 0                     | -1                    | 第二象限 |
| 210°        | $\frac{7\pi}{6}$  | $-\frac{1}{2}$        | $-\frac{\sqrt{3}}{2}$ | 第三象限 |
| 225°        | $\frac{5\pi}{4}$  | $-\frac{\sqrt{2}}{2}$ | $-\frac{\sqrt{2}}{2}$ | 第三象限 |
| 240°        | $\frac{4\pi}{3}$  | $-\frac{\sqrt{3}}{2}$ | $-\frac{1}{2}$        | 第三象限 |
| 270°        | $\frac{3\pi}{2}$  | -1                    | 0                     | 第三象限 |
| 300°        | $\frac{5\pi}{3}$  | $-\frac{\sqrt{3}}{2}$ | $\frac{1}{2}$         | 第四象限 |
| 315°        | $\frac{7\pi}{4}$  | $-\frac{\sqrt{2}}{2}$ | $\frac{\sqrt{2}}{2}$  | 第四象限 |
| 330°        | $\frac{11\pi}{6}$ | $-\frac{1}{2}$        | $\frac{\sqrt{3}}{2}$  | 第四象限 |
| 360°        | $2\pi$            | 0                     | 1                     | 第一象限 |
