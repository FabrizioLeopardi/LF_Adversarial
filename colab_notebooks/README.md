# Some notebooks

In this folder some details regarding the training of the NNs are explained.
The files Exercise.ipynb and Exercise2.ipynb present the weights of Best_NN_found.txt and Best_NN_found2.txt and prove that very high accuracies can be achieved even with only 10 neurons. These NNs were not used in the evasions against Never2.

<mark>WARNING:</mark> Testing.ipynb, Exercise.ipynb and Exercise2.ipynb have a vector of 7850 components hardcoded, consider open these file locally as colab may present some lag.
I hardcoded the data inside the notebooks since for compatibility reasons.

## SIMPLE_MNIST_Tensor_ipynb.ipynb

This file contains the basic training of the NNs.
It also contains the data of the image I chose for the evasion attacks.

## Testing.ipynb

This file contains Accuracies and Confusion matrices of the NN I used in "../second_paper" for the training and testing set of the MNIST database.
This NN have an Accuracy of approx. 85.5% on the training set.
This NN have an Accuracy of approx. 85.3% on the test set.
Seems not bad for having only 10 neurons, but it never classifies correctly class 5. (It classifies a total of 59804 images correctly)

## Exercise.ipynb

This file is a copy of Testing.ipynb with the weights of Best_NN_found.txt. 
This new NN have an Accuracy of approx. 94.2% on the training set.
This new NN have an Accuracy of approx. 92.4% on the test set.
Very accurate for having only 10 neurons but present a bit of generalisation error. (It classifies a total of 65754 images correctly)

## Exercise2.ipynb

This file is a copy of Testing.ipynb with the weights of Best_NN_found2.txt. 
This new NN have an Accuracy of approx. 93.8% on the training set.
This new NN have an Accuracy of approx. 93.0% on the test set.
Less accurate than Exercise.ipynb but more consistent in the generalisation. (It classifies a total of 65597 images correctly)

## raw_data folder

This folder contains the raw dumpy arrays of weights and biases as well as some code for the translation.
To run the code use the following:
`chmod +x Translate_as_vector.py`
`chmod +x translate.sh`
`./translate.sh`






