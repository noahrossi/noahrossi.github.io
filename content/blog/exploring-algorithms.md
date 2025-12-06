+++
title = "Exploring Sorting Algorithms"
date = 2024-01-05
description = "A look at different sorting algorithms and their time complexities"
draft = true
+++

Sorting algorithms are fundamental to computer science. Let's explore a few common ones and understand their trade-offs.

## Bubble Sort

The simplest sorting algorithm, but also one of the least efficient. It repeatedly steps through the list, compares adjacent elements, and swaps them if they're in the wrong order.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

**Time Complexity:** $O(n^2)$ in the worst and average cases.

## Quick Sort

A divide-and-conquer algorithm that's much more efficient in practice.

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

**Time Complexity:** $O(n \log n)$ on average, but $O(n^2)$ in the worst case.

## Merge Sort

Another divide-and-conquer algorithm with guaranteed $O(n \log n)$ performance.

The recurrence relation for merge sort is:

$$T(n) = 2T\left(\frac{n}{2}\right) + O(n)$$

Using the Master Theorem, this solves to $T(n) = O(n \log n)$.

## Comparison

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Bubble Sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ |
| Quick Sort | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ |
| Merge Sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ |

Choose your algorithm based on your specific needs!