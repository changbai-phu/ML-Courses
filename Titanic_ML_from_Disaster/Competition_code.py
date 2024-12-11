'''
Experiment 1:
Based on Alexis's code, add new feature Age and Fare, 
use HistGradientBoostingClassifier due to missing values,
update max_depth=7 and add learning_rate=0.05.
Others left the same.
mae: 0.23318385650224216
Result: 0.77511 (same as the RandomForestClassifer)
'''
from sklearn.ensemble import HistGradientBoostingClassifier

y = train_data["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = HistGradientBoostingClassifier(max_iter=100, max_depth=7, random_state=1, learning_rate=0.05)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae) # 0.23318385650224216

model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")



'''
Experiment 2:
Split X into train and valid sets to calculate MAE for validation other than the Kaggle result.
Use for loop to find the n_estimators that leads to the min MAE (which is 300)
And plug 300 back to the original code (RandomForestClassifier) and then predict X_test.
mae: 0.2062780269058296
Result: 0.77511 (same as the RandomForestClassifer with n=100)
'''
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_score(n):
    model = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)

n_estimator_list = [100, 150, 200, 250, 300, 350, 400]
results = {num: get_score(num) for num in n_estimator_list}  # Dictionary comprehension
print(results)
'''
{100: 0.21524663677130046, 150: 0.21524663677130046, 200: 0.2242152466367713, 250: 0.21524663677130046, 
300: 0.2062780269058296, 350: 0.21524663677130046, 400: 0.2242152466367713}
'''

model_final = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1)
model_final.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

'''
Experiment 3: Since Exp1 and Exp2 show the same accuracy, simply adding extra features, or switch to 
different model or use different value for n_estimator don't really help. 
In this exp, will start from preprocessing the chosen features.
Then, use the same RandomForestClassifer but with n_estimators=300 which calculated from exp2. 
mae: 0.21524663677130046
Result: 0.76555 (decreased)
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin"]
X = train_data[features]
X_test = test_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define preprocessing for numerical features
numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Define preprocessing for categorical features
categorical_features = ['Cabin', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute Cabin with most_frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical variables
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
# Define the complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1))  # Replace with the model of your choice
])

pipeline.fit(train_X, train_y)
pred_pipeline = pipeline.predict(val_X)
mae = mean_absolute_error(val_y, pred_pipeline)
print(mae)

# train on whole dataset again and then make predictions
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_pipeline.csv', index=False)
print("Your submission was successfully saved!")


'''
Experiment 4: Use the same preprocessing (exp3) on the features (exp2)
to test if preprocess causing the reduction in accuracy.
mae: 0.21524663677130046
Result: 0.76555 (decreased)
'''
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = train_data[features]
X_test = test_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define preprocessing for categorical features
categorical_features = ['Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute Cabin with most_frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical variables
])
# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ]
)
# Define the complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1))  # Replace with the model of your choice
])

pipeline.fit(train_X, train_y)
pred_pipeline = pipeline.predict(val_X)
mae = mean_absolute_error(val_y, pred_pipeline)
print(mae)

# train on whole dataset again and then make predictions
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)

'''
Experiment 5: Test different pre-processing
Use LabelEncoder() on only 'Sex', remove other preprocess
MAE: 0.2242152466367713
Result: 0.77511 (same as exp1,2)
'''
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = train_data[features]
X_test = test_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Label encode the 'Sex' feature manually
label_encoder = LabelEncoder()
train_X['Sex'] = label_encoder.fit_transform(train_X['Sex'])
val_X['Sex'] = label_encoder.transform(val_X['Sex'])
X_test['Sex'] = label_encoder.transform(X_test['Sex'])

model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1)
model.fit(train_X, train_y)
preds = model.predict(val_X)
mae = mean_absolute_error(val_y, preds)
print(mae)

# Train the model on the whole dataset (train_X + val_X) and make predictions on X_test
train_full_X = pd.concat([train_X, val_X])
train_full_y = pd.concat([train_y, val_y])

model.fit(train_full_X, train_full_y)
predictions = model.predict(X_test)


'''
Experiment 6: Find optimal parameters 
Use GridSearch to find the optimal set of parameters
Test result: 0.7802690582959642
Result: 0.77511 (the same as exp 1 & 2)
'''
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = train_data[features]
X_test = test_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Label encode the 'Sex' feature
label_encoder = LabelEncoder()
train_X['Sex'] = label_encoder.fit_transform(train_X['Sex'])
val_X['Sex'] = label_encoder.transform(val_X['Sex'])
X_test['Sex'] = label_encoder.transform(X_test['Sex'])

# Define the model
model = RandomForestClassifier(random_state=1)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Evaluation metric
    verbose=2,
    n_jobs=-1  # Use all available processors
)

# Fit the grid search on training data
grid_search.fit(train_X, train_y)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
'''
Best Parameters: {'max_depth': 3, 'min_samples_split': 10, 'n_estimators': 200}
Best Score: 0.7994164515767029
'''
# Use the best estimator to predict
best_model = grid_search.best_estimator_
val_preds = best_model.predict(val_X)
accuracy = accuracy_score(val_y, val_preds)
print(f"Validation Accuracy: {accuracy}")

# Predict on the test set using the best model
test_preds = best_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_preds})
output.to_csv('submission_GridSearch.csv', index=False)
print("Your submission was successfully saved!")