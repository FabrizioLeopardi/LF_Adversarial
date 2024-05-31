# Fooling an ad-hoc NN

This folder contains the implementation of the method described in Zombori's paper.
`my_Nnetwork2.py` verifies that an obfuscated neural network with 'normal' weights (`omegas1.txt` and `omegas2.txt`) can evade pynever.
`statistical_analysis.py`build a NN with the same structure 100 times varying each time randomly the weights as described in the paper. For simplicity $$\sigma = -3$$.
As shown in `results.txt` 89 out of 100 tests evaded NeVer verifier.

