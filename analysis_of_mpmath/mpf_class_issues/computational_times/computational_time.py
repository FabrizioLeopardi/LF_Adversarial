import time
import random
from mpmath import *

def estimate_params(f,max_value):
    num_iter = 100
    alpha = 0.0
    epsilon = 0.0
    epsilon = float((log(f(max_value-1)) - log(f(max_value/2)))/(log(max_value-1)-log(max_value/2)))
    epsilon -= 1.0
    alpha = float(f(max_value-1)/(max_value-1)**(1+epsilon))
    return alpha, epsilon
        

#Approximate values obtain. Notice that alpha may depend on the computer architecture
#While epsilon depend on the complexity of the multiplication algorithm used.
#With this analysis it seems that the algorithm used for multiplication is Karatsuba

#1.0202285956234292e-10
#0.6103947706229054

alpha = 0.0
epsilon = 0.0
size = 200000
max_value = int(size/2-1)
mp.dps = size
x = []
elapsed_times = []

for i in range(max_value):
    x.append(mpf(10)**mpf(i-1)+1)
    
for i in range(max_value):
    start = time.time()
    y = x[i]*x[i]
    end = time.time()
    elapsed_times.append(end-start)

f = lambda x: elapsed_times[int(x)]

[alpha,epsilon] = estimate_params(f,max_value)
print(alpha)
print(epsilon)

#Put size to 10000 before uncommenting this
#g = lambda x: alpha*(x**(mpf(1)+epsilon))
#plot([f,g], [1,max_value])

plot(f, [1,max_value])
