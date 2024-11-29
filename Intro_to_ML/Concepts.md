
Basic concepts
1. Overview
- fitting/training model: we need to use data to predict price, the step of capturing patterns from data is training the model.
- training data: data used to fit the model
- how to fit the model
- validate if model works on data other than test data 

2. Decision Tree
- more "splits" (if conditions) - called "deeper" tree
- leaf: the final prediction we make (bottom of the tree)

3. How to build and use a model
- Specify prediction target (y - SalePrice column in our example)
- Hold predictive features (x - a DataFrame to hold selected ata corresponding to the list of features we picked)
- Define the type of model will be (model=model imported from sklearn)
- Fit to capture patterns from data (model.fit(x, y))
- Predict expected results (model.predict(x))
- Evaluate how the model performs/accuracy (compare model.predict() with y values)

4. randonmness in model training
- We introduce some randomness in model training (random_state in sklearn), this is to avoid overfitting that the model is highly dependent on the training data we provided and causing low accuracy when evaluating using test data. 