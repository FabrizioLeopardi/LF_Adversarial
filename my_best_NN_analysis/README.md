# Best NN found

In this folder you can find some data regarding the best NN I found for the task of classifying the images of the MNIST dataset with the constraint of using one single layer with 10 neurons and ReLU activation. 
The learning and refinement phases were running in background while I developed the rest of the code for my thesis.

## Learning procedure

I trained the network for 3000 epochs to find a first candidate network the weights of which had to be refined.
Notice that during the learning phase the algorithm may be stuck in a local maxima (from the point of view of the accuracy).
After 100 epochs one is easily able to tell if the algorithm is stuck and can interrupt the learning.
Therefore it is easy to implement a procedure that reaches 3000 epochs only for the best candidate NN after 100 epochs.
Of course this is not a formal approach yet a good heuristics.
 
## Refinement procedure

The refinement procedure is nothing more that a custom gradient descent method implemented in such a way to optimise the only metric of accuracy on the whole dataset.


