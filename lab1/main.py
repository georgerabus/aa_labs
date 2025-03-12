import time
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

def measure_time(func, n):
    start = time.time()
    func(n)
    end = time.time()
    return (end - start) * 1000  # Convert to milliseconds

# 1. Memoization Fibonacci (O(n))
@lru_cache(None)
def fibonacci_memoization(n):
    if n <= 1:
        return n
    return fibonacci_memoization(n - 1) + fibonacci_memoization(n - 2)

# 2. Modulo Fibonacci (O(log n))
def fibonacci_modulo(n, mod=1000000007):
    if n == 0:
        return 0
    F = np.array([[1, 1], [1, 0]], dtype=object)
    result = matrix_power(F, n - 1, mod)
    return result[0, 0] % mod

def matrix_power(matrix, n, mod):
    result = np.eye(2, dtype=object)
    base = matrix.copy()
    while n:
        if n % 2:
            result = np.dot(result, base) % mod
        base = np.dot(base, base) % mod
        n //= 2
    return result

# 3. Continued Fraction Fibonacci (O(n))
def fibonacci_continued_fraction(n):
    if n <= 1:
        return n
    result = 0
    for _ in range(n):
        result = 1 / (2 + result)
    return int(result + 2)

def generate_graph(ns, times, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(ns, times, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("n (Fibonacci Term)")
    plt.ylabel("Execution Time (ms)")
    plt.grid()
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved as {filename}")

if __name__ == "__main__":
    max_n = 16000
    ns = [1 + i * (max_n - 1) // 39 for i in range(40)]
    
    times_memoization = [measure_time(fibonacci_memoization, n) for n in ns]
    times_modulo = [measure_time(fibonacci_modulo, n) for n in ns]
    times_continued_fraction = [measure_time(fibonacci_continued_fraction, n) for n in ns]
    
    generate_graph(ns, times_memoization, "Memoization Fibonacci Time", "Memoization_Fibonacci_Time.png")
    generate_graph(ns, times_modulo, "Modulo Fibonacci Time", "Modulo_Fibonacci_Time.png")
    generate_graph(ns, times_continued_fraction, "Continued Fraction Fibonacci Time", "ContinuedFraction_Fibonacci_Time.png")
