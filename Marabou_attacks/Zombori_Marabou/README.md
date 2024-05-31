# Fooling an ad-hoc NN

  

This folder contains the implementation of the method described in Zombori's paper.

`my_Nnetwork2.py` verifies that an obfuscated neural network with 'normal' weights (`omegas1.txt` and `omegas2.txt`) can evade pynever.

This network is in general not the same as the one seen in `../../NeVer_attacks/Zombori_Never`. The weights are drawn exactly as described in the paper.

## The NN format

  

Marabou accepts both `.pb` and `.onnx` format.

I could have worked only with the `.pb` version of the NN BUT I decided to work with the `.onnx` format since in the future other tests may be carried out with different solvers. Therefore I believe it's going to be more clear if all the experiments can be easily repeated in the same format.

Just make sure to have the package `tf2onnx` installed.
