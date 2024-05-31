from mpmath import *

mp.dps = 100
f = lambda x: sin(x)
g = lambda x: quad(cos, [0,x])
h = lambda x: f(x)-g(x)
plot(h, [0,1])
