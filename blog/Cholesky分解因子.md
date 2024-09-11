---
authors: [BUAAer-xing]
---

# Cholesky分解因子

## Cholesky分解因子

Cholesky 分解因子（Cholesky factor）是指**将一个正定对称矩阵分解成一个下三角矩阵和其转置的乘积**的过程。
$$A=LL^T$$
这个下三角矩阵被称为 Cholesky 分解因子。

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230811182801.png)

Cholesky 分解因子的计算可以用于求解线性方程组、计算矩阵的逆以及进行随机数生成等各种数值计算任务。

由于 **Cholesky 分解**只需要计算下三角矩阵的元素，相对于 **LU 分解**等其他矩阵分解方法，Cholesky 分解具有更高的计算效率和数值稳定性。

注意：

$$(AB)^{-1} = B^{-1}A^{-1}$$

所以，$A=LL^T,A^{-1}=G^TG$ 其中 $G≈L^{-1}$ 的推导过程如下：

$$
\begin{split}
	A&=LL^T \\
	A^{-1}&=(LL^T)^{-1} \\
	A^{-1}&=(L^{T})^{-1}L^{-1} \\
	A^{-1}&=(L^{-1})^{T}L^{-1} \\
	if \ L^{-1}& = G \ to: \\
	A^{-1}&=(G)^{T}G
\end{split}
$$

## Cholesky 分解

### 分解过程

当进行Cholesky分解的过程中，数学公式可以用LaTeX表示如下：

1. 初始化：$$ L(0,0) = \sqrt{A(0,0)}$$
2. 逐步计算$L(i,j)$，其中$i > j$：
   $$L(i,j) = \frac{A(i,j) - \sum_{k=0}^{j-1} L(i,k) \cdot L(j,k)}{L(j,j)}$$

这些公式描述了Cholesky分解的关键步骤，其中A是原始对称正定矩阵，L是下三角矩阵，i和j是矩阵元素的索引。Cholesky分解通过迭代计算这些公式来逐渐填充L矩阵的元素，直到得到完整的下三角矩阵。

### C++实现

```cpp
#include <iostream>
#include <vector>
#include <cmath>

using namespace std; 

// Cholesky分解函数
bool choleskyDecomposition(const vector<vector<double>>& A, vector<vector<double>>& L) {
    int n = A.size();
    L.resize(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += L[j][k] * L[j][k];
                }
                L[j][j] = sqrt(A[j][j] - sum);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
        if (L[i][i] <= 0) {
            return false;  // 输入矩阵不是正定的
        }
    }

    return true;
}

int main() {
    vector<vector<double>> A = {
        {4.0, 12.0, -16.0},
        {12.0, 37.0, -43.0},
        {-16.0, -43.0, 98.0}
    };

    vector<vector<double>> L;

    if (choleskyDecomposition(A, L)) {
        cout << "Cholesky分解成功：" << endl;
        for (int i = 0; i < L.size(); i++) {
            for (int j = 0; j < L[i].size(); j++) {
                cout << L[i][j] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "Cholesky分解失败：输入矩阵不是正定的。" << endl;
    }

    return 0;
}

```

## 分解的三种形式

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20231011224650.png)


## Cholesky分解与LU分解

正定对称矩阵在使用Cholesky分解时具有更好的数值稳定性，这是因为Cholesky分解是一个特别适用于正定对称矩阵的数值分解方法，可以充分利用矩阵的性质，避免了一些数值计算中的困难情况。

以下是一些原因：

1. 无需主元选取：Cholesky分解不需要像高斯消元法中的主元选取那样复杂，因为**正定对称矩阵的特性使得Cholesky分解的主元总是正的**，而**不需要担心数值稳定性问题**。这消除了部分主元选取（partial pivoting）所涉及的额外计算步骤，简化了算法。

2. 避免了数值误差的累积：在高斯消元法等一些其他数值分解方法中，数值误差可能会在计算过程中逐步累积，导致数值不稳定。Cholesky分解的特性可以帮助避免这种情况。因为Cholesky分解中的所有元素都是实数，并且只有上三角矩阵的元素需要计算，所以它可以更好地控制误差的传播。

3. 更高的计算效率：Cholesky分解通常比LU分解（高斯消元法的一种形式）更快，因为它只需要计算上三角矩阵的元素，而LU分解需要计算整个矩阵。这使得Cholesky分解更适合大规模问题，而且更容易融入数值计算库和软件中。

总之，正定对称矩阵的特性使Cholesky分解成为一种更为稳定和高效的数值分解方法，特别适用于求解线性方程组。因此，当处理正定对称矩阵时，Cholesky分解通常是首选的方法，以提高数值稳定性和计算效率。