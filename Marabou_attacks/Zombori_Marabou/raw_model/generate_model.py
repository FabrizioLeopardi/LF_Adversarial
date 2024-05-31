import tensorflow as tf
import numpy as np

def generate_omegas(omega, n):
    mean = pow(omega,1/n)
    std_dev = pow(omega,1/n)/4
    x = np.random.normal(loc=mean,scale=std_dev)
    if x<0:
        x = generate_omegas(omega,n)
    
    return x


def main():
    sigma = -3
    n = 20
    omega = pow(2,60)
    omegas = np.zeros((2,n))
    
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(2, activation='relu'),

    ])
    
    
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    
    for i in range(n):
        model.add(tf.keras.layers.Dense(2, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    
    
   # BUILDING MODEL WEIGHTS:

    indices = [[0,0]]
    model.weights[0].assign(tf.tensor_scatter_nd_update(model.weights[0],indices,[1]))
    indices = [[0,1]]
    model.weights[0].assign(tf.tensor_scatter_nd_update(model.weights[0],indices,[1]))
    indices = [[0]]
    model.weights[1].assign(tf.tensor_scatter_nd_update(model.weights[1],indices,[-0.5]))
    indices = [[1]]
    model.weights[1].assign(tf.tensor_scatter_nd_update(model.weights[1],indices,[-2]))

    indices = [[0,0]]
    model.weights[2].assign(tf.tensor_scatter_nd_update(model.weights[2],indices,[sigma]))
    indices = [[0,1]]
    model.weights[2].assign(tf.tensor_scatter_nd_update(model.weights[2],indices,[0]))
    indices = [[1,0]]
    model.weights[2].assign(tf.tensor_scatter_nd_update(model.weights[2],indices,[0]))
    indices = [[1,1]]
    model.weights[2].assign(tf.tensor_scatter_nd_update(model.weights[2],indices,[1]))
    indices = [[0]]
    model.weights[3].assign(tf.tensor_scatter_nd_update(model.weights[3],indices,[1]))
    indices = [[1]]
    model.weights[3].assign(tf.tensor_scatter_nd_update(model.weights[3],indices,[1]))
    
    omegas_product = np.ones((2))
    for i in range(n):
        # Weights
        omegas[0][i]=generate_omegas(omega,n)
        omegas[1][i]=generate_omegas(omega,n)
        omegas_product[0] *= omegas[0][i]
        omegas_product[1] *= omegas[1][i]
        
        indices = [[0,0]]
        model.weights[2*i+4].assign(tf.tensor_scatter_nd_update(model.weights[2*i+4],indices,[omegas[0][i]]))
        indices = [[0,1]]
        model.weights[2*i+4].assign(tf.tensor_scatter_nd_update(model.weights[2*i+4],indices,[0]))
        indices = [[1,0]]
        model.weights[2*i+4].assign(tf.tensor_scatter_nd_update(model.weights[2*i+4],indices,[0]))
        indices = [[1,1]]
        model.weights[2*i+4].assign(tf.tensor_scatter_nd_update(model.weights[2*i+4],indices,[omegas[1][i]]))
        
        # Bias
        indices = [[0]]
        model.weights[2*i+5].assign(tf.tensor_scatter_nd_update(model.weights[2*i+5],indices,[0]))
        indices = [[1]]
        model.weights[2*i+5].assign(tf.tensor_scatter_nd_update(model.weights[2*i+5],indices,[0]))
    
    
    indices = [[0,0]]
    model.weights[2*n+4].assign(tf.tensor_scatter_nd_update(model.weights[2*n+4],indices,[omega/omegas_product[0]]))
    indices = [[1,0]]
    model.weights[2*n+4].assign(tf.tensor_scatter_nd_update(model.weights[2*n+4],indices,[-omega/omegas_product[1]]))
    indices = [[0]]
    model.weights[2*n+5].assign(tf.tensor_scatter_nd_update(model.weights[2*n+5],indices,[1]))
    
    indices = [[0,0]]
    model.weights[2*n+6].assign(tf.tensor_scatter_nd_update(model.weights[2*n+6],indices,[1]))
    indices = [[0,1]]
    model.weights[2*n+6].assign(tf.tensor_scatter_nd_update(model.weights[2*n+6],indices,[-2]))
    indices = [[0]]
    model.weights[2*n+7].assign(tf.tensor_scatter_nd_update(model.weights[2*n+7],indices,[0]))
    indices = [[1]]
    model.weights[2*n+7].assign(tf.tensor_scatter_nd_update(model.weights[2*n+7],indices,[1]))
    
    # ----------------------------------------------------------------------------------------- 
    
    
    """
    final = [0]
    x_tf = tf.convert_to_tensor(final)
    a = model.predict(x_tf, verbose=0)
    print(a)
    """
   
   
    model.export("../NN_model")
    


if __name__ == "__main__":
    main()
