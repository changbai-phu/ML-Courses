
Basic concepts
1. Overview
- fitting/training model: we need to use data to predict price, the step of capturing patterns from data is training the model.
- training data: data used to fit the model
- how to fit the model
- validate if model works on data other than test data 

2. Decision Tree
- more "splits" (if conditions) - called "deeper" tree
    - 10 splits: 2^10=1024 leaves (10th level)
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

5. Model Validation (meausre predictive accuracy)
- Metrics for summarizing model quality
    - Mean Absolute Error (MAE): take the average of the absolute errors, while each error = actual - predicted. 
    - sklearn has a function mean_absolute_error to calculate MAE
- The problem with "in-sample" scores: model may only accurate when evaluate using only training data, and very inaccurate when used in practice.
- Validation data: exclude some data from the model-building process, and then use those to test the model's accuracy.
- sklearn has a function train_test_split to break up the data into training data and validation data.

6. Underfitting and Overfitting
- Overfitting: a model matches the training data almost perfectly, but does poorly in validation and other new data.
- Underfitting: a model performs poorly in training data - failed to capture patterns.
- there is a sweet spot between underfitting and before go into overfitting.
- e.g., max_leaf_nodes argument allows us to control overfitting/underfitting since the more leaves we make, the more we are moving to the overfitting.

7. Random forests - class RandomForestRegressor()
- use multiple trees and then take the average of the predictions of each component tree.


# References mentioned in the Kaggle course
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html 