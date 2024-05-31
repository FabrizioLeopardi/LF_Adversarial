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
    
# In python3 the int class is unbounded
def generate_number(i):
    p = 0
    q = 1
    for j in range(i-1):
        q *= 10
    p = q+1
    return p
    


#Approximate values obtain. Notice that alpha may depend on the computer architecture
#While epsilon depend on the complexity of the multiplication algorithm used.
#With this analysis it seems that the algorithm used for multiplication is Karatsuba
#Indeed mpf seems to rely upon inner gmpy or python multiplication (see mpmath/libmp/libmpf.py)... line 855 and 889: man = sman*tman
#In my case the backend is python
#mpf_mul() method is called inside mpmath/ctx_mp_python.py
#If type of sman and tman is integer python3 uses as default Karatsuba algorithm
#The graph plot in this script supports this statement
#(Using gmpy as backend? it should implement Schönhage–Strassen algorithm)

#1.0202285956234292e-10
#0.6103947706229054

alpha = 0.0
epsilon = 0.0
size = 50000
max_value = int(size/2-1)
mp.dps = size
x = []
x2 = []
elapsed_times = []
elapsed_times2 = []

for i in range(max_value):
    x.append(mpf(10)**mpf(i-1)+1)
    x2.append(generate_number(i))
    
for i in range(max_value):
    start = time.time()
    y = x[i]*x[i]
    end = time.time()
    elapsed_times.append(end-start)
    
for i in range(max_value):
    start = time.time()
    y = x2[i]*x2[i]
    end = time.time()
    elapsed_times2.append(end-start)

f = lambda x: elapsed_times[int(x)]
h = lambda x: elapsed_times2[int(x)]

#[alpha,epsilon] = estimate_params(f,max_value)
#print(alpha)
#print(epsilon)

#Put size to 10000 before uncommenting this
#g = lambda x: alpha*(x**(mpf(1)+epsilon))
#plot([f,g], [1,max_value])

plot([f, h], [1,max_value])
