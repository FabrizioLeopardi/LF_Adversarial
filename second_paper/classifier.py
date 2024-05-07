import numpy as np

import pynever.strategies.abstraction as abst
import pynever.nodes as nodes
import pynever.strategies.verification as ver
import pynever.networks as networks
import pynever.utilities as utils
import pynever.strategies.conversion as conv

# CONSTANT DEFINITION (JIaRinard_ExploitingVerifiedNeuralNetworksViaFloatingPointNumericalError formalism, custom values):
epsilon = 0.000001

def print_star_data_simplified(p_star: abst.Star, mode: bool):
    """
    Just to plot input constraints explicitly, function taken from examples/notebooks/bidimensional_example_FC_ReLU_Complete.ipynb
    """

    print("PREDICATE CONSTRAINTS:")
    for row in range(p_star.predicate_matrix.shape[0]):
        constraint = ""
        for col in range(p_star.predicate_matrix.shape[1]):
            if p_star.predicate_matrix[row, col] != 0.0:
                if p_star.predicate_matrix[row, col] < 0:
                    sign = "-"
                else:
                    sign = "+"
                constraint = constraint + f"{sign} {abs(p_star.predicate_matrix[row, col])} * x_{col} "

        constraint = constraint + f"<= {p_star.predicate_bias[row, 0]}"
        print(constraint)
    
    if (mode):
        print("VARIABLES EQUATIONS:")
        for row in range(p_star.basis_matrix.shape[0]):
            equation = f"z_{row} = "
            for col in range(p_star.basis_matrix.shape[1]):
                if p_star.predicate_matrix[row, col] != 0.0:
                    if p_star.basis_matrix[row, col] < 0:
                        sign = "-"
                    else:
                        sign = "+"
                    equation = equation + f"{sign} {abs(p_star.basis_matrix[row, col])} * x_{col} "

            if p_star.center[row, 0] != 0:
                if p_star.center[row, 0] < 0:
                    c_sign = "-"
                else:
                    c_sign = "+"
                equation = equation + f"{c_sign} {abs(p_star.center[row, 0])}"
            print(equation)

def print_star_data_simplified_y(p_star: abst.Star, mode: bool):
    """
    Just to plot input constraints explicitly, function taken from examples/notebooks/bidimensional_example_FC_ReLU_Complete.ipynb
    """

    print("PREDICATE CONSTRAINTS:")
    for row in range(p_star.predicate_matrix.shape[0]):
        constraint = ""
        for col in range(p_star.predicate_matrix.shape[1]):
            if p_star.predicate_matrix[row, col] != 0.0:
                if p_star.predicate_matrix[row, col] < 0:
                    sign = "-"
                else:
                    sign = "+"
                constraint = constraint + f"{sign} {abs(p_star.predicate_matrix[row, col])} * y_{col} "

        constraint = constraint + f"<= {p_star.predicate_bias[row, 0]}"
        print(constraint)
    
    if (mode):
        print("VARIABLES EQUATIONS:")
        for row in range(p_star.basis_matrix.shape[0]):
            equation = f"z_{row} = "
            for col in range(p_star.basis_matrix.shape[1]):
                if p_star.predicate_matrix[row, col] != 0.0:
                    if p_star.basis_matrix[row, col] < 0:
                        sign = "-"
                    else:
                        sign = "+"
                    equation = equation + f"{sign} {abs(p_star.basis_matrix[row, col])} * y_{col} "

            if p_star.center[row, 0] != 0:
                if p_star.center[row, 0] < 0:
                    c_sign = "-"
                else:
                    c_sign = "+"
                equation = equation + f"{c_sign} {abs(p_star.center[row, 0])}"
            print(equation)

def main():
    my_Nnetwork = networks.SequentialNetwork('','')
    verifier = ver.NeverVerification("best_n_neurons", None, None)
    
    
    # Building the neural network
    Neural = open("data/NN_data.txt","r")
    weight_matrix_1 = np.zeros((10, 784))
    bias_1 = np.zeros(10)
    
    
    # FC
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
    current_node = nodes.FullyConnectedNode("FC_1", (784,), 10, weight_matrix_1, bias_1, True)
    my_Nnetwork.add_node(current_node)
    
    
    # ReLU
    current_node = nodes.ReLUNode("ReLU_1", (10,))
    my_Nnetwork.add_node(current_node)
    
    
    # X SEED data:
    count = 0
    x_s = []
    with open("data/x_seed_data.txt", "r") as f:
        for line in f:
            x_s.append(float(line))
        count += 1
    size = len(x_s)
    
    
    # INPUT STARSET DEFINITION: (Adv_epsilon(x))
    # Warning?? Why???
    C = np.zeros((4*size, size))
    for i in range(size):
        C[i, i] = 1
        C[size+i, i] = -1
        C[2*size+i, i] = -1
        C[3*size+i, i] = 1
    d = np.ones((4*size, 1))
    for i in range(size):
        d[i, 0] = x_s[i]+epsilon
        d[size+i] = -x_s[i]+epsilon
        d[2*size+i] = 0
        d[3*size+i] = 1
    star = abst.Star(C, d)
    abs_input = abst.StarSet({star})
    
    
    # Debug print
    print()
    print("Neural Network has "+str(len(my_Nnetwork.nodes)) + " layers")
    print("Neural Network has " + str(my_Nnetwork.count_relu_layers()) + " ReLU Layers")
    print()


    # Verify the property
    target_class = 2 # The image is classified as 2, because verifier (with "small" values of epsilon) gives False only when target_class = 2 otherwise gives True.
    C1 = np.zeros((9,10))
    for i in range(9):
        j = 0
        if (i>=target_class):
            j = 1
        C1[i,i+j] = 1
        C1[i,target_class] = -1
    d1 = np.zeros((9,1))
    star = abst.Star(C1, d1)
    abs_input = abst.StarSet({star})
    print_star_data_simplified_y(star, False)
    property_to_verify = ver.NeVerProperty(C,d,[C1],[d1])
    print(verifier.verify(my_Nnetwork, property_to_verify))
    # True --> there is no overlap with critical region (C1, d1)
    # False --> there is overlap with critical region (C1, d1)
    
if __name__ == "__main__":
    main()
