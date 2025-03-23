import time
import random
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import sys
from tabulate import tabulate

sys.setrecursionlimit(10**6)

class ArrayType(Enum):
    RANDOM = "Random"
    SORTED = "Sorted"
    REVERSE_SORTED = "ReverseSorted"
    NEARLY_SORTED = "NearlySorted"

def quicksort(arr, left, right, stats):
    while left < right:
        pivot = partition(arr, left, right, stats)
        if pivot - left < right - pivot:
            quicksort(arr, left, pivot - 1, stats)
            left = pivot + 1
        else:
            quicksort(arr, pivot + 1, right, stats)
            right = pivot - 1

def partition(arr, left, right, stats):
    pivot = arr[right]
    i = left - 1
    for j in range(left, right):
        stats["comparisons"] += 1
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            stats["swaps"] += 1
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    stats["swaps"] += 1
    return i + 1

def mergesort(arr, left, right, stats):
    if left >= right:
        return
    mid = (left + right) // 2
    mergesort(arr, left, mid, stats)
    mergesort(arr, mid + 1, right, stats)
    merge(arr, left, mid, right, stats)

def merge(arr, left, mid, right, stats):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]
    i = j = 0
    k = left
    while i < len(left_part) and j < len(right_part):
        stats["comparisons"] += 1
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1
    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1
    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1

def heapsort(arr, stats):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, stats)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        stats["swaps"] += 1
        heapify(arr, i, 0, stats)

def heapify(arr, n, i, stats):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n:
        stats["comparisons"] += 1
        if arr[left] > arr[largest]:
            largest = left
    if right < n:
        stats["comparisons"] += 1
        if arr[right] > arr[largest]:
            largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        stats["swaps"] += 1
        heapify(arr, n, largest, stats)

def shell_sort(arr, stats):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap:
                stats["comparisons"] += 1
                if arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                    stats["swaps"] += 1
                else:
                    break
            arr[j] = temp
        gap //= 2

def generate_array(size, array_type):
    rng = np.random.default_rng()
    if array_type == ArrayType.RANDOM:
        return rng.integers(-size * 2, size * 2, size).tolist()
    elif array_type == ArrayType.SORTED:
        return list(range(size))
    elif array_type == ArrayType.REVERSE_SORTED:
        return list(range(size, 0, -1))
    elif array_type == ArrayType.NEARLY_SORTED:
        arr = list(range(size))
        for _ in range(size // 10):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    else:
        raise ValueError("Invalid array type")

def run_analysis():
    sizes = [100, 1000, 5000, 10000]
    results = {}
    table_data = []

    for size in sizes:
        results[size] = {}
        for array_type in ArrayType:
            row = [size, array_type.value]
            array = generate_array(size, array_type)

            for name, func in [("QuickSort", quicksort), ("MergeSort", mergesort), 
                               ("HeapSort", heapsort), ("ShellSort", shell_sort)]:
                time_taken, stats = test_algorithm(name, func, array.copy())
                results[size][name] = time_taken
                row.append(f"{time_taken:.3f} ms")  

            table_data.append(row)

    headers = ["Array Size", "Array Type", "QuickSort", "MergeSort", "HeapSort", "ShellSort"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    plot_results(results)

def plot_results(results):
    plt.ioff()  
    figures = [] 

    for size, data in results.items():
        fig = plt.figure()
        plt.bar(data.keys(), data.values(), color=['blue', 'green', 'red', 'purple'])
        plt.xlabel("Sorting Algorithm")
        plt.ylabel("Time (ms)")
        plt.title(f"Sorting Performance for Array Size {size}")
        figures.append(fig)  

    plt.show(block=True)  


def test_algorithm(name, func, arr):
    stats = {"comparisons": 0, "swaps": 0}
    start_time = time.time()
    if name in ["QuickSort", "MergeSort"]:
        func(arr, 0, len(arr) - 1, stats)
    else:
        func(arr, stats)
    return (time.time() - start_time) * 1000, stats

if __name__ == "__main__":
    run_analysis()
