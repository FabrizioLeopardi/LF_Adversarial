import tensorflow as tf
import numpy as np
import random as rn

def get_NN_data():
    NN_data2 = []
    
    x = input("Which NN would you like to try? 1) 85.5% accuracy or 2) 94.7% accuracy? ")
    
    my_file = open("epsilon_value.txt","w")
    my_file2 = open("x_value.txt","w")
    if (x == "1"):
        with open("NN_data.txt","r") as e:
            e.readline()
            for line in e:
                if (line!="]\n"):
                    NN_data2.append(float(line.replace(',',"")))
        my_file.write("5.0")
        my_file2.write(x)
    else:
        with open("betterNN/best.txt","r") as e:
            e.readline()
            for line in e:
                if (line!="]\n"):
                    NN_data2.append(float(line.replace(',',"")))
        my_file.write("1.0")
        my_file2.write(x)
        
    my_file.close()
    my_file2.close()
    return NN_data2


def main():
    NN_data2 = get_NN_data()
    #print(NN_data2)
    
    
    weight_matrix_1 = np.zeros((10, 784))
    bias_1 = np.zeros(10)

    k = 0
    l = 0
    count = 0
    for k in range(785):
      for l in range(10):
            count+=1
            if (k==784 and l<10):
              bias_1[l] = float(NN_data2[l+10*k])
              l += 1
            if (k<784):
              weight_matrix_1[l][k] = float(NN_data2[l+10*k])
              l += 1
              if (l==10):
                l=0
                k += 1

    model = tf.keras.models.Sequential([
            tf.keras.layers.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
            ])

    for i in range(10):
      for j in range(784):
        indices = [[j,i]]
        model.weights[0].assign(tf.tensor_scatter_nd_update(model.weights[0],indices,[weight_matrix_1[i][j]]))

    for i in range(10):
      indices = [[i]]
      model.weights[1].assign(tf.tensor_scatter_nd_update(model.weights[1],indices,[bias_1[i]]))
      
      
    
    model.export("../NN_model")
    #model.save("model.keras")
   
    
if __name__=="__main__":
    main()
