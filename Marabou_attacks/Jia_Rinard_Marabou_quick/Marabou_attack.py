from maraboupy import Marabou
import numpy as np



#epsilon = 5.0/255.0 # Adversary capability
epsilon_r = 0.00000001

def import_x_seed_data():
    x_s = []
    with open("data/x_seed_data.txt", "r") as f:
        for line in f:
            x_s.append(float(line))
    return x_s


def main():

    file = open("raw_model/epsilon_value.txt","r")
    base = float(file.readline())
    epsilon = float(base/255.0)
    file.close()

    output_alphas = open("alphas.txt","w")
    filename = "onnx_model_NN/model.onnx"
    network = Marabou.read_onnx(filename)
    x_s = import_x_seed_data()
    alpha_prime = 0.0
    alpha = 1.0
    
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    while (alpha-alpha_prime>epsilon_r):
        network = Marabou.read_onnx(filename)
        scale = (alpha+alpha_prime)/2
        
        # Domain
        for i in range(len(x_s)):
            low = max(scale*x_s[i]-epsilon,0)
            high = min(scale*x_s[i]+epsilon,1)
            x = int(i/28)
            y = i%28
            network.setLowerBound(inputVars[x][y],low)
            network.setUpperBound(inputVars[x][y],high)
            
        # Codomain
        network.setUpperBound(outputVars[2], 0.0)
        
        # Verify
        """
        vals = network.solve("marabou.log",verbose=0)
        opt = Marabou.createOptions(1,5,0,2,500,1.5,2,False,'auto','auto',False,20,False)
        vals = network.solve("marabou.log",options=opt)
        """
       
        vals = network.solve("log/Marabou.log", verbose=False)
        if (len(vals)>0 and vals[0]=="sat"):
            alpha_prime = scale
        else:
            alpha = scale
        
        output_alphas.write(str(alpha_prime)+" "+str(alpha)+"\n")
    output_alphas.close()

    new_text_file = open("final_safe_value.txt","w")
    new_text_file.write(str(alpha_prime))
    new_text_file.close()
    
    
if __name__ == "__main__":
    main()
