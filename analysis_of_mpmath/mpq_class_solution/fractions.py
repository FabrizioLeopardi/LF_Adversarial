from mpmath import *

#Useful class. mpmath source code writes that: "Exact rational type, currently only intended for internal use."
a = rational.mpq((1,2))
b = rational.mpq((2,3))
print(a*b)
