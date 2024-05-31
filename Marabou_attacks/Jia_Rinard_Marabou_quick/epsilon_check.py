import numpy as np

file = open("raw_model/epsilon_value.txt","r")
  
epsilon = float(file.readline())

file.close()

img_0 = np.ones((28,28))
img_1 = np.ones((28,28))


def generate_first_number(a):
    if (a[1]==" "):
        return float(a[0])
        
    if (a[2]==" "):
        return float(float(a[0])*10.0+float(a[1]))
    
    if (a[3]==" "):
        return float(float(a[0])*100.0+float(a[1])*10.0+float(a[2]))

T = open("x_0.ppm","r")
        
T.readline()
T.readline()
T.readline()

for i in range(28):
    for j in range(28):
        img_0[i][j] = float(generate_first_number(T.readline()))
                
T.close()

S = open("x_adv.ppm","r")

S.readline()
S.readline()
S.readline()
    
for i in range(28):
    for j in range(28):
        img_1[i][j] = float(generate_first_number(S.readline()))
    
S.close()

condition = True

for i in range(28):
    for j in range(28):
        if (img_1[i][j] - img_0[i][j]>epsilon or img_1[i][j] - img_0[i][j]<-epsilon):
            print(img_1[i][j]-img_0[i][j])
            print(img_0[i][j])
            print(img_1[i][j])
            condition = False
            
            
print(condition)
if (condition):
    print("x_adv is in the valid neighborhood od x_0, it should have been safe...")
