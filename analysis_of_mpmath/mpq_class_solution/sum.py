from mpmath import *

# rational.mpq class seems to solve also this issue...
# "Exact rational type, currently only intended for internal use."
# Unique problem "intended for internal use"

mp.dps = 1000
print(rational.mpq(10,1)**1000+rational.mpq(1.0,1)-rational.mpq(10,1)**1000)
print(rational.mpq(10,1)**100000+rational.mpq(1.0,1)-rational.mpq(10,1)**100000)
