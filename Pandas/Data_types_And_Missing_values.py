# Data Types and Missing Values


#### Tutorials ####
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)

# dtype: data type for a column in DataFrame or a Series
reviews.price.dtype

# dtypes: returns the dtype of every column in the DataFrame
reviews.dtypes
# col consists entirely of strings will be given the 'object' type

# astype: possible to convert a col of one type into another
reviews.points.astype('float64')

reviews.index.dtype

# NaN: Not a Number, entries missing values, float64 dtype
# pd.isnull(): to select NaN entries; pd.notnull() to select non NaN
reviews[pd.isnull(reviews.country)]

# fillna(): replacing missing values
reviews.region_2.fillna("Unknown")
# backfill strategy: fill each missing value with the first non-null value that appears 
# sometime after the given record in the database

# replace(): replace non-null value
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")



#### Exercises ####
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.data_types_and_missing_data import *
print("Setup complete.")

#Q1 What is the data type of the points column in the dataset?
dtype = reviews.points.dtype

#Q2 Create a Series from entries in the points column, but convert the entries to strings
point_strings = reviews.points.astype('str')

#Q3 How many reviews in the dataset are missing a price?
missing_price_reviews = reviews[reviews.price.isnull()]
n_missing_prices = len(missing_price_reviews)
# or n_missing_prices = reviews.price.isnull().sum()
# or equivalently:
# n_missing_prices = pd.isnull(reviews.price).sum()

#Q4  Create a Series counting the number of times each value occurs in the region_1 field. 
# This field is often missing data, so replace missing values with Unknown. Sort in descending order. 
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)