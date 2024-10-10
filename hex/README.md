# Insights on the IEEE 754 standard

## 64 bits: Double-Precision Floating Point

![64 bits](img/double.png)

The standard requires a double-precision floating point number to be represented with a sign bit (S), 11 bits for the exponent of the number in base 2 with a bias of 1023 (E) and 52 bits for the fractional part of the number's mantissa in base 2 (F).



## 32 bits: Single-Precision Floating Point

![32 bits](img/single.png)

The standard requires a double-precision floating point number to be represented with a sign bit (S), 8 bits for the exponent of the number in base 2 with a bias of 127 (E) and 23 bits for the fractional part of the number's mantissa in base 2 (F).

Let's consider an example: let's represent the number: 0.2

## 1/3

By following the examples discussed so far it is immediate to write:

$$ \left \lfloor log_2(3) \right \rfloor \approx -2$$
$$ \frac{1}{3} = \frac{1}{4} * \frac{4}{3} = 2^{-2}\displaystyle \sum_{i=0}^{\infty}\  \frac{1}{2^{2i}}$$

And get the hexadecimal representation as discussed in the thesis
