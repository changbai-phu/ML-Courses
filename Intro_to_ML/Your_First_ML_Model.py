import pandas as pd

#### Tutorial ####
# 1. Select data for modeling
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# 2. Select the prediction target using dot notation 
# single col is stored in a Series (just like DataFrame but only one col of data)
y = melbourne_data.Price

# 3. Choose 'Features'
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()

# 4. Build your model
# library used: scikit-learn 
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))



#### Exercise ####
# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")

# Q1: Specify prediction target: Select the target variable, which corresponds to the sales price
# print the list of columns in the dataset to find the name of the prediction target
home_data.describe()
y = home_data.SalePrice

# Q2: Create X holding the predictive features
# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = home_data[feature_names]
X.head()

# Q2b: Review data
# print description or statistics from X
X.describe()
# print the top few lines
X.head()

# Q3: Specify and fit model
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit the model
iowa_model.fit(X, y)

# Q4: Make predictions
predictions = iowa_model.predict(X)
print(predictions)

# Think about your results
# Use the head method to compare the top few predictions to the actual home values (in y) for those same homes
y.head()