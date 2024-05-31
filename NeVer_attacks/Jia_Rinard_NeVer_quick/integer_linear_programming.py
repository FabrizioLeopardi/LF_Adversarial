import tensorflow as tf
import numpy as np
import random as rn

epsilon = 5.0/255.0

def index_of_max(a):
  l = a[0]
  k = 0
  for i in range(10):
    if (a[i]>l):
      l = a[i]
      k = i
  return k

def main():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        ])
        
    weight_matrix_1 = np.zeros((10, 784))
    bias_1 = np.zeros(10)
    k = 0
    l = 0
    count = 0
    with open("data/NN_data.txt","r") as Neural:
        for line in Neural:
            count+=1
            if (k==784 and l<10):
                bias_1[l] = float(line)
                l += 1
            if (k<784):
                weight_matrix_1[l][k] = float(line)
                l += 1
                if (l==10):
                    l=0
                    k += 1
    print()
    print()
    for i in range(10):
        for j in range(784):
            indices = [[j,i]]
            model.weights[0].assign(tf.tensor_scatter_nd_update(model.weights[0],indices,[weight_matrix_1[i][j]]))
            
    for i in range(10):
        indices = [[i]]
        model.weights[1].assign(tf.tensor_scatter_nd_update(model.weights[1],indices,[bias_1[i]]))


    img_final = np.zeros((1,28,28))

    T = open("data/x_0_data.txt","r")
        
    for i in range(28):
        for j in range(28):
            img_final[0][i][j] = float(T.readline())
                
    T.close()
    
    
    for i in range(28):
        for j in range(28):
            if (weight_matrix_1[2][28*i+j]>0):
                img_final[0][i][j] -= float(5.0/255.0)
                if (img_final[0][i][j]<0):
                    img_final[0][i][j] = 0
            else:
                img_final[0][i][j] += float(5.0/255.0)
                if (img_final[0][i][j]>1):
                    img_final[0][i][j] = 1
            
            
    x_tf = tf.convert_to_tensor(img_final)
    a = model.predict(x_tf, verbose=0)
    print(a)
    print(index_of_max(a[0]))
            
    #my_x_tf = open("data/x_adv_data.txt","w")
    g = open("x_adv.ppm","w")
    
    g.write("P3\n")
    g.write("28 28\n")
    g.write("255\n")
    
    for aa in range(28):
        for bb in range(28):
            #my_x_tf.write(str(img_final[0][aa][bb])+"\n")
            g.write(str(round(img_final[0][aa][bb]*255.0))+" "+str(round(img_final[0][aa][bb]*255.0))+" "+str(round(img_final[0][aa][bb]*255.0))+"\n")
    #my_x_tf.close()
    g.close()

if __name__ == "__main__":
    main()
