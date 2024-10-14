# LF_Adversarial

This repository contains the source code for the experiments carried out during my master thesis in Robotics Engineering. 

##

The code is organized in folders that are independent one another.
In the end the following organization was chosen as it is believed to be semantically clear. A link to the papers that inspired these works is provided. The numbering of the reported references follows the one used in the thesis 

### &#9658; Marabou_attacks
This folder contains the attacks against the Marabou verifier.
The theoretical discussion of the attacks can be found at [[2](https://openreview.net/pdf?id=4IwieFS44l)] and [[3](https://arxiv.org/pdf/2003.03021)]
All the attacks were tested against the package version `maraboupy==1.0.0`.

### &#9658; Master_thesis
This folder contains the $\textcolor{red}{\text{FINAL VERSION}}$ of the master thesis (delivered on the 11th-12th October 2024) and the manim code for the animations inside the presentation.

###  &#9658; NeVer_attacks
This folder contains the attacks against the NeVer verifier.
The theoretical discussion of the attacks can be found at [[2](https://openreview.net/pdf?id=4IwieFS44l)] and [[3](https://arxiv.org/pdf/2003.03021)]
All the attacks were tested against the package version `pyNeVer==0.1.2`.
This code exploits star propagation for NN verification.

###  &#9658; analysis_of_mpmath

This folder contains an analysis of the strengths and strengths and weaknesses of the mpmath library. The main classes that have been analyzed are $\texttt{mpf}$ and $\texttt{mpq}$.

###  &#9658; colab_notebooks

This folder contains the code to train and test 784-10 feed-forward fully connected  neural networks with ReLU activation. Only the coarsest neural network found (in terms of accuracy) was used in the experiments.

###  &#9658; hex

This folder contains a script to retrieve the 64-bit and 32-bit hexadecimal representations of a real number following the IEEE754 standard

###  &#9658; my_best_NN_analysis

This folder contains the code to test the best 784-10 feed-forward fully connected  neural networks with ReLU activation. This neural network was found via the code developed in the folder `colab_notebooks`

### &#9658; new_NeVer_ver_attacks

This folder contains an attack against the NeVer verifier.
The theoretical discussion of the attack can be found at [[2](https://openreview.net/pdf?id=4IwieFS44l)].
All the attacks were tested against the package version `pyNeVer==0.1.2`.
This code does not exploit star propagation for NN verification.

### &#9658; pf

This folder contains the tests that were conducted on the PULP-Frontnet datasets.
Please refer to [[6](https://arxiv.org/pdf/2103.10873)] for the theoretical description of the case study.

 # 
 :warning: $\textcolor{red}{\text{NOTICE:}}$ This repository is currently being improved and new experiments will be added soon
 
