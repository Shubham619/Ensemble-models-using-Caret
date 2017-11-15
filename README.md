# Ensemble-models-using-Caret 
Ensemble technique is used for combining two or more algorithms of similar or dissimilar types.
Three main types of method explained in code are Averaging, majority votes and weighted average.
I have performed caret ensemble stacking which has multiple layers of machine learning models that are placed one over another where each of the models passes their predictions to the model in the layer above it and the top layer model takes decisions based on the below layers. I have used Recursive Feature elimination(rfe).which is a wrapper method to find the best subset of features to use for modeling.
I have used generalized linear model,random forest and gradient boosting machine using caret package.
