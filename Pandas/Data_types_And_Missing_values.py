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