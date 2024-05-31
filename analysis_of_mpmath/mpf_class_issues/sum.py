from mpmath import *

# This is the main issue with the class, as it may be exploited as in Zombori
mp.dps = 1000
print(mpf(10)**1000+mpf(1.0)-mpf(10)**1000)
print(mpf(10)**10000+mpf(1.0)-mpf(10)**10000)
