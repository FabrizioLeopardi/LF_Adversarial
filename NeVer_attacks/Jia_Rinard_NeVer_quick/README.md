# Numerical attack

In this folder a numerical attack against a fully connected single layer Neural Network with ReLU activation is performed.
The Neural Network was previously trained on the MNIST dataset, the evasion is performed on the first image of the test set of the dataset. The image clearly represent an handwritten 2.
Inside the `../../colab_notebooks`folder the code for the training and testing of the network is provided.
The weights in human readable notation can be found in `data/NN_data_raw.txt"`

## The strategy 

The basic idea underlying the attack is to adjust the brightness of the starting safe image (`x_seed.ppm`) in such a way to obtain a
darker unsafe image (`x_0.ppm`) for which the verifier wrongly believes no adversarial example exists, due to numerical error.
`x_adv.ppm` is shown to exist, it is the actual adversarial example.
In order to perform such an attack I partially relied on JIaRinard's paper. A custom variation to the method was performed because the `verify()` function of the NeverVerification class provides neither a complete nor an incomplete verifier. So the "proper" evasion was performed in a way similar to the one a real world attacker would perform.
Recently the SearchVerification class was introduced: this allows to follow any step of the algorithm described in the paper as its `verify()` function implement a complete classifier.

## How to run the scripts
0. (Optional) `python3 classifier.py`  It will classify the starting image as a 2.
1. (Optional) `chmod +x start_directory` 
(Optional) `./start_directory`
It will delete the already generated output files
2. (Optional) `python3 numerical_attack.py` it will find a proper value to adjust the brightness. (This value was already hardcoded inside x0_generator.py)
3. `python3 x0_generator.py`  It will create the x_0.ppm and data/x_0_data.txt files.
4. `python3 integer_linear_programming.py` It will create x_adv.ppm file
5.  (Optional check) `python3 epsilon_check.py` It will check the property $\textbf{x}^{adv} \in Adv_\epsilon(\textbf{x}^0)$


## Integer linear programming problem solution

In order to find `x_adv` i solved a very easy integer linear programming problem.
This is possible because the NN has only one layer, therefore the output of the network $\textbf{y}$ can be written as a function of the input  $\textbf{x}$ as:

$$ 
\textbf{y} = [y_0, y_1, ..., y_9]^T 
$$
$$
y_i = f_i(W\textbf{x}+b );\ \forall{i} \in { 0,...,9 } 
$$
being 
$$ 
f_i(\textbf{x}) = max(0,x_i) 
$$

From here I tried to evade minimizing $y_2$ by noticing that any component of vector function $\textbf{f}$ is monotonic:

$$\textbf{x}^{adv} =  argmin_{\textbf{x} \in Adv_\epsilon(\textbf{x}^0)} \{y_2\} = argmin\{f_2(W\textbf{x}+\textbf{b})\} 
= argmin\{ \braket{(W\textbf{x}+\textbf{b}) , \  \textbf{e}^2} \} 
= argmin\{ \textbf{w} \textbf{x}+b_2\} $$

subject to 
$$l_i \le x_i \le u_i$$


being $\textbf{e}^2 = [0,0,1,0,0,0,0,0,0,0]^T$, $\textbf{w} = \({\textbf{e}^2}\)^T W$ and $\braket{.,.}$
is the scalar product.
The solution of the problem can be found by letting $x_i=l_i$ whenever the coefficient $w_i$ that multiplies $x_i$ is greater than $0$ and $x_i = u_i$ otherwise.
The property the verifier assumed not to be possibile in the neighborhood of $\textbf{x}^0$ (`x_0.ppm`) was 
$$y_0 \ge y_i \ ,\forall i$$
In the end by minimising $y_2$ I obtained the logit vector: 
$\textbf{y} = \textbf{f}(\textbf{x}^{adv}) = [0,0,0,0,0,0,0,0,0,0]^T$ for which the property is satisfied even if $||\textbf{x}^{adv} - \textbf{x}^0  ||_\infty \le \epsilon = 5/255$ , $\textbf{x}^{adv} \le \textbf{1}$ and $\textbf{x}^{adv} \ge \textbf{0}$.
Notice that a classifier based on such a network that classifies the digit by finding the <mark>first value</mark> in the vector greater or equal to the others would classify the image as a 0.

Of course this method may not be successful for any input and adversary capability $\epsilon$. In particular the attack failed for the same input and $\epsilon = 0.02$




