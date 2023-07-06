

from numba import njit
import numpy as np

@njit
def calculate_result(a, b):
    result = 0.0
    for i in range(len(a)):
        temp = 0.0
        for j in range(len(a)):
            temp += a[j] * float(b[j][i])  # 将NumPy数组元素转换为float类型
        result += temp * a[i]
    
    return result

# 示例用法
a = np.array([[1.0], [2.0], [3.0]])
b = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

result = calculate_result(a, b)
print(result)