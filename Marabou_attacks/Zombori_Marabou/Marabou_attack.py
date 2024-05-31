from maraboupy import Marabou
import numpy as np

def main():
    filename = "onnx_model_NN/model.onnx"
    network = Marabou.read_onnx(filename)
    
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]
    
    # Domain
    network.setLowerBound(inputVars[0],0)
    network.setUpperBound(inputVars[0],1)
            
    # Codomain
    network.setLowerBound(outputVars[0], 0.01)
        
     
    vals = network.solve("log/Marabou.log", verbose=True)
    
    txt_file = open("Result.txt","w")
    
    if (len(vals)>0 and vals[0]=="sat"):
        txt_file.write("Evasion didn't work")
    else:
        txt_file.write("Evasion worked!")
    
    # If sat: ok, evasion didn't work
    # If unsat: evasion worked!
    
    
    
if __name__ == "__main__":
    main()
