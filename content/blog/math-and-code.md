+++
title = "Exploring Math and Code"
date = 2024-01-15
description = "A demonstration of mathematical notation and code syntax highlighting"
draft = true
+++

One of the great things about having a technical blog is being able to write about both mathematics and programming. Let me demonstrate both!

## The Beauty of Mathematics

Mathematics is the language of the universe. Let's start with something simple—the quadratic formula. For any equation of the form $ax^2 + bx + c = 0$, the solutions are:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

### Euler's Identity

Perhaps the most beautiful equation in mathematics is Euler's identity:

$$e^{i\pi} + 1 = 0$$

This single equation connects five fundamental mathematical constants: $e$, $i$, $\pi$, $1$, and $0$.

### Calculus

The fundamental theorem of calculus relates differentiation and integration:

$$\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$$

And here's the Gaussian integral, which appears frequently in probability and statistics:

$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

## Code Examples

Now let's look at some code. Here's a Python function that implements the quadratic formula:

```python
import math

def quadratic_formula(a: float, b: float, c: float) -> tuple[float, float]:
    """
    Solve ax^2 + bx + c = 0 using the quadratic formula.
    Returns both solutions as a tuple.
    """
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        raise ValueError("No real solutions exist")
    
    sqrt_discriminant = math.sqrt(discriminant)
    x1 = (-b + sqrt_discriminant) / (2 * a)
    x2 = (-b - sqrt_discriminant) / (2 * a)
    
    return x1, x2

# Example usage
solutions = quadratic_formula(1, -5, 6)
print(f"Solutions: x = {solutions[0]} and x = {solutions[1]}")
```

### Numerical Integration in Rust

Here's a more complex example—numerical integration using Simpson's rule in Rust:

```rust
fn simpsons_rule<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    
    let mut sum = f(a) + f(b);
    
    for i in 1..n {
        let x = a + i as f64 * h;
        let coefficient = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coefficient * f(x);
    }
    
    sum * h / 3.0
}

fn main() {
    // Approximate the Gaussian integral from -5 to 5
    let result = simpsons_rule(|x| (-x * x).exp(), -5.0, 5.0, 1000);
    println!("Approximation of Gaussian integral: {:.10}", result);
    println!("Expected (√π): {:.10}", std::f64::consts::PI.sqrt());
}
```

## Combining Math and Code

The interplay between mathematical theory and implementation is fascinating. When we write $O(n \log n)$ for an algorithm's complexity, we're using mathematical notation to describe computational behavior.

Consider the sum of the first $n$ natural numbers:

$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

We can verify this with code:

```python
def sum_formula(n: int) -> int:
    return n * (n + 1) // 2

def sum_loop(n: int) -> int:
    return sum(range(1, n + 1))

# Both should give the same result
n = 100
print(f"Formula: {sum_formula(n)}")
print(f"Loop: {sum_loop(n)}")
```

This is just the beginning of what we can explore with math and code together!