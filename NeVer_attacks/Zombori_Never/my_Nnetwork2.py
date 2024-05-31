import numpy as np

import pynever.strategies.abstraction as abst
import pynever.nodes as nodes
import pynever.strategies.verification as ver
import pynever.networks as networks
import pynever.utilities as utils
import pynever.strategies.conversion as conv

# In order to prove that it is possible to evade the verifier i try a neural network as in Zombori_FoolingACompleteNeuralNetworkVerifier.pdf (chapter 4, figure 2)

def print_star_data(p_star: abst.Star):
    """
    Just to plot input constraints explicitly, function taken from examples/notebooks/bidimensional_example_FC_ReLU_Complete.ipynb
    """

    print("PREDICATE CONSTRAINTS:")
    for row in range(p_star.predicate_matrix.shape[0]):
        constraint = ""
        for col in range(p_star.predicate_matrix.shape[1]):
            if p_star.predicate_matrix[row, col] < 0:
                sign = "-"
            else:
                sign = "+"
            constraint = constraint + f"{sign} {abs(p_star.predicate_matrix[row, col])} * x_{col} "

        constraint = constraint + f"<= {p_star.predicate_bias[row, 0]}"
        print(constraint)

    print("VARIABLES EQUATIONS:")
    for row in range(p_star.basis_matrix.shape[0]):
        equation = f"z_{row} = "
        for col in range(p_star.basis_matrix.shape[1]):
            if p_star.basis_matrix[row, col] < 0:
                sign = "-"
            else:
                sign = "+"
            equation = equation + f"{sign} {abs(p_star.basis_matrix[row, col])} * x_{col} "

        if p_star.center[row, 0] < 0:
            c_sign = "-"
        else:
            c_sign = "+"
        equation = equation + f"{c_sign} {abs(p_star.center[row, 0])}"
        print(equation)
    
def main():
    my_Nnetwork = networks.SequentialNetwork('','')
    verifier = ver.NeverVerification("best_n_neurons", None, None)
    sigma = -3
    omega = pow(2,60)
    n = 20
    omegas = np.zeros((2,n))
    file1 = open("omegas1.txt","r")
    file2 = open("omegas2.txt","r")
    
    
    # Retrieve correct values from files
    for i in range(n):
        omegas[0,i] = float(file1.readline())
        omegas[1,i] = float(file2.readline())
    
    
    # INPUT STARSET DEFINITION
    C = np.zeros((2, 1))
    C[0, 0] = 1
    C[1, 0] = -1
    d = np.ones((2, 1))
    d[0, 0] = 1
    d[1, 0] = 0
    star = abst.Star(C, d)
    abs_input = abst.StarSet({star})
    print_star_data(star)
    
    
    # FIRST FULLY CONNECTED NODE DEFINITION
    weight_matrix_1 = np.ones((2, 1))
    bias_1 = np.zeros(2)
    bias_1[0] = -0.5
    bias_1[1] = -2
    current_node = nodes.FullyConnectedNode("FC_1", (1,), 2, weight_matrix_1, bias_1, True)
    
    
    # APPENDING THE LAYER TO THE NETWORK
    my_Nnetwork.add_node(current_node)
    
    
    # FIRST ReLU NODE DEFINITION
    current_node = nodes.ReLUNode("ReLU_1", (2,)) # <-- This DOES NOT overwrite the first element of the list: my_Nnetwork.nodes
    my_Nnetwork.add_node(current_node)
 
    
    # SECOND FC and ReLU layer
    weight_matrix_1 = np.zeros((2, 2))
    weight_matrix_1[0, 0] = sigma
    weight_matrix_1[1,1] = 1
    bias_1 = np.ones(2)
    current_node = nodes.FullyConnectedNode("FC_2", (2,), 2, weight_matrix_1, bias_1, True)
    my_Nnetwork.add_node(current_node)
    current_node = nodes.ReLUNode("ReLU_2", (2,))
    my_Nnetwork.add_node(current_node)
    prod1 = 1.0
    prod2 = 1.0
    
    # ADDING n FC and ReLU layers
    for i in range(n):
        weight_matrix_1 = np.zeros((2, 2))
        weight_matrix_1[0, 0] = omegas[0,i]
        weight_matrix_1[1,1] = omegas[1,i]
        bias_1 = np.zeros(2)
        current_node = nodes.FullyConnectedNode("FC_"+str(i+3), (2,), 2, weight_matrix_1, bias_1, True)
        my_Nnetwork.add_node(current_node)
        current_node = nodes.ReLUNode("ReLU_"+str(i+3), (2,))
        my_Nnetwork.add_node(current_node)
        prod1*=omegas[0,i] # <-- Computed here to save time
        prod2*=omegas[1,i] # <-- Computed here to save time
 
    
    # ADDING 2 Layers to trick numerically the verifier
    weight_matrix_1 = np.zeros((1, 2))
    weight_matrix_1[0, 0] = omega/prod1
    weight_matrix_1[0, 1] = -omega/prod2
    bias_1 = np.ones(1)
    current_node = nodes.FullyConnectedNode("FC_23", (2,), 1, weight_matrix_1, bias_1, True)
    my_Nnetwork.add_node(current_node)
    current_node = nodes.ReLUNode("ReLU_23", (1,))
    my_Nnetwork.add_node(current_node)
   
    # ADDING Final Layers
    weight_matrix_1 = np.ones((2, 1))
    weight_matrix_1[1, 0] = -2
    bias_1 = np.zeros(2)
    bias_1[1] = 1
    current_node = nodes.FullyConnectedNode("FC_24", (1,), 2, weight_matrix_1, bias_1, True)
    my_Nnetwork.add_node(current_node)
    current_node = nodes.ReLUNode("ReLU_24", (2,))
    my_Nnetwork.add_node(current_node)
    
    
    print()
    print("Neural Network has "+str(len(my_Nnetwork.nodes)) + " layers")
    print("Neural Network has " + str(my_Nnetwork.count_relu_layers()) + " ReLU Layers") # <-- function defined in class SequentialNetwork "pynever/networks.py"


    # VERIFIER
    print()
    C1 = np.zeros((1, 2))
    C1[0, 0] = -1
    d1 = np.zeros((1, 1))
    d1[0, 0] = -0.001
    property_to_verify = ver.NeVerProperty(C,d,[C1],[d1])
    condition = verifier.verify(my_Nnetwork, property_to_verify)
    print(condition) # <-- It will always return true: i.e. these weights successfully evade the verifier since the output of the NN is believed to be always equal to 0
    
    
if __name__ == "__main__":
    main()
