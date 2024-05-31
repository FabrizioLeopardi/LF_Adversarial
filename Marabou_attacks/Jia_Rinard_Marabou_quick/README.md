# Numerical attack
  
In this folder a numerical attack against a fully connected single layer Neural Network with ReLU activation is performed.

The Neural Network was previously trained on the MNIST dataset, the evasion is performed on the first image of the test set of the dataset. The image clearly represent an handwritten 2.

Inside the `../../colab_notebooks`folder the code for the training and testing of the network is provided.

The weights in human readable notation can be found in `raw_model/NN_data.txt"`.

In this version the user can try a new experiment with a stronger network inside the `raw_model/betterNN/best.txt` folder.
In this case the adversary capability had to be changed because the original image was recognized not to be safe with the original capability.

  

## The strategy

The strategy remains the same as the one in `../../NeVer_attacks/Jia_Rinard_NeVer_quick`

## How to run the scripts

For the Marabou version I automatized the process of running the various components of the evasion attack.
Just run `./execute_jia_rinard_attack.sh` and select the NN you want Marabou to analyse and be numerically "fooled".
  

## Integer linear programming problem solution

The concept behind this is the same as described in the Never_attacks/Jia_Rinard_Never_quick/README.md

## What's new

If you selected the original NN with 85.5% accuracy please try and go in the compare folder and run `./compare_results.sh`
This script will not give any output if the adversarial sample was the same found with the NeVer tool.

## The NN format

Marabou accepts both `.pb` and `.onnx` format.
I could have worked only with the `.pb` version of the NN BUT I decided to work with the `.onnx` format since in the future other tests may be carried out with different solvers. Therefore I believe it's going to be more clear if all the experiments can be easily repeated in the same format. 
Just make sure to have the package `tf2onnx` installed.

## Results with the new NN

The results with the new NN are more interesting since the logic relative to the fifth component of the output is greater than 0. (Notice that the original NN had a big issue in classifying the handwritten 5).
This is truly an adversarial example that clearly evade the NN and the verifier.
You can find more details about the new NN in the `../../my_best_NN_analysis` folder.
