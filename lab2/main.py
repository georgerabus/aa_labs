import time
import random
import numpy as np
from enum import Enum
import sys

sys.setrecursionlimit(10**6)

class ArrayType(Enum):
    RANDOM = "Random"
    SORTED = "Sorted"
    REVERSE_SORTED = "ReverseSorted"
    NEARLY_SORTED = "NearlySorted"

def quicksort(arr, left, right):
    while left < right:
        pivot = partition(arr, left, right)
        if pivot - left < right - pivot:
            quicksort(arr, left, pivot - 1)
            left = pivot + 1
        else:
            quicksort(arr, pivot + 1, right)
            right = pivot - 1

def partition(arr, left, right):
    pivot = arr[right]
    i = left - 1
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1

def mergesort(arr, left, right):
    if left >= right:
        return
    mid = (left + right) // 2
    mergesort(arr, left, mid)
    mergesort(arr, mid + 1, right)
    merge(arr, left, mid, right)

def merge(arr, left, mid, right):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]
    i = j = 0
    k = left
    while i < len(left_part) and j < len(right_part):
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

def heapsort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
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

def test_algorithm(name, func, arr):
    start_time = time.time()
    if name in ["QuickSort", "MergeSort"]:
        func(arr, 0, len(arr) - 1)
    else:
        func(arr)
    return time.time() - start_time

def run_analysis():
    sizes = [100, 1000, 5000, 10000]
    results = []
    
    for size in sizes:
        for array_type in ArrayType:
            array = generate_array(size, array_type)
            
            results.append(("QuickSort", array_type.value, size, test_algorithm("QuickSort", quicksort, array.copy())))
            results.append(("MergeSort", array_type.value, size, test_algorithm("MergeSort", mergesort, array.copy())))
            results.append(("HeapSort", array_type.value, size, test_algorithm("HeapSort", heapsort, array.copy())))
            results.append(("ShellSort", array_type.value, size, test_algorithm("ShellSort", shell_sort, array.copy())))
    
    export_results(results)

def export_results(results):
    print("Algorithm,ArrayType,Size,TimeMs")
    for result in results:
        print(f"{result[0]},{result[1]},{result[2]},{result[3] * 1000:.3f}")

if __name__ == "__main__":
    run_analysis()