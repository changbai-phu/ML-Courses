Course outline
- tackle data types: missing values, categorical variables
- design pipelines to improve the quality of ML code
- advanced techniques (cross-validation) to validate model
- built models (XGBoost) that are widely used 
- avoid leakage

1. Deal with missing values - 3 approaches
- Drop columns with missing values (may lose a lot of useful info)
- Imputation: fill missing values with some other number, e.g., mean value of the column (may be far off from the real value, or missing itself is also meaningful info)
- An extension to imputation: fill missing values with some number and plus adding extra column to indicate which are the imputated entries.
- 2nd and 3rd perform bettern than 1st approach

2. Imputation
- not necessary to fill mean value only, sometimes depends on the dataset, may need to fill values other than mean, for example a value of 0. 
- use only SimpleImputator().fit_transform(X_train), but use SimpleImputator().transform(X_val)
    - Reason: to avoid data leakage. It doesn't mean that we keep missing values in test data still missing, but instead, we uses the previously calculated mean values (from the training data) to fill in the missing values in the test data. This way, the test data gets its missing values imputed, but no information from the test data is used to calculate the imputation values.
    - can also make predictions be more consistent with real-world data, and maintain fair evaluatioon by using pre-computed values only based on training data. 

3. Categorical Variables
- e.g., how oftern you eat breakfast, here are the options: Never, Rarely, Most days, or Everdays. 
- response can only fall into a fixed pre-set categories.

4. 3 approaches to deal with categorical variables
- Drop Categorical variables
- Ordinal Encoding (from sklearn.preprocessing import OrdinalEncoder): assign each unique value to a different integer. e.g., assign number from 0 to 3 to Never to Everydays.
- One-Hot Encoding (from sklearn.preprocessing import OneHotEncoder): create extra columns to indicate the presence or absence of each value in the original data. 
    - e.g., Mark 1 for the entries in the new col Red if it has color Red, Mark 0 in the new col Red if it has color other than Red.
    - Compare to ordinal encoding: One-Hot does not assume an ordering of the categories. 
    - Nominal variables: categorical variables that don't have an intrinsic ranking.
    - This approach doesn't work well if have large number of categorical variables.
    - We one-hot encode columns with relatively low cardinality. High cardinality columns can be dropped or use ordinal encoding. 

4. Cardinality: # of unique entries of a categorical variable

5. Pipelines: automate workflows that include multiple preprocessing steps, ensure consistency in preprocessing, make code clean and organized, use for cross-validation and hyperparameter tuning
- Define preprocessing steps (ColumnTransformer)
- Define the model 
- Create and Evaluate the pipeline (Pipeline)
    - can supply the processed features in X_valid to the predict() 

6. Cross-Validation (cross_val_score())
- So far, we decided what predictive variables to use, what types of models to use, what arguments to supply based on our measurement on model using the validation data set which we splitted from the orginal dataset. 
- In cross-validation: we run modeling process on different subsets of the data. 
    - e.g., divide the data into 5 pieces (folds), each 20% of the full dataset. Then, run experiment for each fold which is each time, we hold one fold as a validation set, the rest as training data and do the measurement. Repeat for all 5 folds. 
- when should we use cross-validation?
    - small datasets
    - for large datasets, one single validation set is usually sufficient. Or if multiple experiment yields the similar results. 

7. Hyperparameter optimization
- grid search: determine the best combination of parameters for a ML model (GridSearchCV())

8. Gradient Boosting
- ensemble methods:
    - (random forest method): combine the predictions of several models. 
    - gradient boosting is another ensemble method
- Gradient boosting: go through cycles to iteratively add models into an ensemble. It is sequential.
    - define a model, generate predictions
    - use predictions to calculate a loss function (e.g., mean squared error) -- use gradient descent on the loss function
    - use the loss function to train a new model
    - add the new model to ensemble
    - repeat 
- XGBoost: extreme gradient boosting (xgboost.XGBRegressor)

9. Parameter Tuning in XGBoost
- n_estimators = num of models in the ensemble (usually 100-1000)
- early_stopping_rounds: automatically find the ideal value for n_estimators.
    - stop iterating when the validation score stops improving. 
    - need to specify a number before stopping. (usually 5 is good)
- eval_set: set validation dataset
- learning_rate: multiply the predictions by a small number before add models to the ensemble
    - in general, small learning rate + large n_estimators -> more accurate XGBoost models.
    - default learning_rate = 0.1
- n_jobs: use parallelism to build models faster.
    - set that equal to num of cores of the machine
    