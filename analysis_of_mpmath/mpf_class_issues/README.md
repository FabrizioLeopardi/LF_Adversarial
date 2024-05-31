# mpf class

The mpf class seems to present the same issues of the float class
The script `sum.py` shows that mpf presents the same issue as described in Zombori.

## computational times

The folder `computational_times` empirically shows that the multiplication of numbers with "many" digits uses an algorithm with the same complexity of Karatsuba algorithm. 
A more detailed analysis of mpmath source code supports the fact that such multiplication is indeed performed with the standard multiplication algorithm (i.e. Karatsuba).
