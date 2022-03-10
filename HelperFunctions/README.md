# HelperFunctions

This package consists of functions which are used in every neural network in the whole project. 

**The HelperFunctions.py consists of the following functions:** 

- creating a confusion matrix 
- calculating and printing an output of all accuracy for each class
- printing a progress bar and calculating the remaining time
- drawing a graph for tensor board
- printing out all wrong guesses of the model
- converting a float tensor to a long tensor
- loading the model

**The HelperPhases.py consists of the following functions:** 

- running the training phase
  - iterating through the data loader and updating the weights
  - updating progress bar
  - drawing accuracy and loss graph in tensor board
  - creating confusion matrix 
- running the training phase
  - iterating through the data loader
  - updating progress bar
  - creating confusion matrix 

Both phases are grouped into this file, because the training and testing phase is the same in every recognition application.