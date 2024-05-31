import numpy as np

import pynever.strategies.abstraction as abst
import pynever.nodes as nodes
import pynever.strategies.verification as ver
import pynever.networks as networks
import pynever.utilities as utils
import pynever.strategies.conversion as conv

# In order to prove that it is possible to evade the verifier i first try a neural network as in Zombori_FoolingACompleteNeuralNetworkVerifier.pdf (chapter 4, figure 2)

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

def generate_omegas(omega, n):
    mean = pow(omega,1/n)
    std_dev = pow(omega,1/n)/4
    x = np.random.normal(loc=mean,scale=std_dev)
    if x<0:
        x = generate_omegas(omega,n)
    
    return x
    
    
def main():
    
    number_true = 0
    number_false = 0
    
    # 100 cycles to have an idea of the frequency of successful evasions
    for cycle in range(100):
        my_Nnetwork = networks.SequentialNetwork('','')
        verifier = ver.NeverVerification("best_n_neurons", None, None)
        sigma = -3
        omega = pow(2,60)
        n = 20
        omegas = np.zeros((2,n))

        
        # INPUT STARSET DEFINITION
        C = np.zeros((2, 1))
        C[0, 0] = 1
        C[1, 0] = -1
        d = np.ones((2, 1))
        d[0, 0] = 1
        d[1, 0] = 0
        star = abst.Star(C, d)
        abs_input = abst.StarSet({star})


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
            omegas[0,i] = generate_omegas(omega,n)
            omegas[1,i] = generate_omegas(omega,n)
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
     
        
        # ADDING 2 Layers to trick the verifier
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
       
       
        # VERIFIER
        C1 = np.zeros((1, 2))
        C1[0, 0] = -1
        d1 = np.zeros((1, 1))
        d1[0, 0] = -0.001
        property_to_verify = ver.NeVerProperty(C,d,C1,d1)
        condition = verifier.verify(my_Nnetwork, property_to_verify)
        print(str(cycle)+": "+str(condition))
        if condition:
            number_true += 1
        else:
            number_false += 1
        
        
    # Final print
    print()
    print()
    print()
        
    print("total number of TRUE: "+str(number_true))
    print("total number of FALSE: "+str(number_false))
        
    
    
if __name__ == "__main__":
    main()
