# Indexing, Selecting & Assigning 

#### Tutorial ####
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('display.max_rows', 5)

# select specific Series out of a DataFrame
reviews.country # or reviews['country']

# Indexing operator []
reviews['country'][0]

# Indexing in Pandas
## index-based selection:
reviews.iloc[0] # select first row of the data; row-first, col-second
reviews.iloc[:, 0] # to get a column; ':'operator means everything
reviews.iloc[:3, 0] # select the first 3 rows
reviews.iloc[1:3, 0] # select the second and third rows
reviews.iloc[[0, 1, 2], 0] # select first 3 rows as well 
reviews.iloc[-5:] # select the last five rows
## Label-based selection:
reviews.loc[0, 'country'] # get the first entry
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']] 

### iloc vs loc: 
### iloc: using Python stdlib indexing schema, include first but exclude the last
### iloc: 0:10--> 0,...,9
### loc: indexes inclusively 0:10 --> 0,...,10

# Manipulating the index
reviews.set_index("title")

# Conditional selection
reviews.country == 'Italy' # product a serires of true/false booleans based on the 'country' of each record
reviews.loc[reviews.country == 'Italy']
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)] # and
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)] # or
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]

# Assigning data
reviews['critic'] = 'everyone'
reviews['critic']
reviews['index_backwards'] = range(len(reviews), 0, -1) # decreaing value by 1 every iteration
reviews['index_backwards']


#### Exercise ####
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")

reviews.head()
desc = reviews.description
type(desc) # Series

# get first value from the description col of reviews
first_description = desc[0] # or reviews.description.iloc[0] or reviews.description[0]

first_row = reviews.iloc[0]

# get first 10 values from the description column in reviews
first_descriptions = reviews.description.iloc[0:10] # or reviews.description.iloc[:10] or desc.head(10) and reviews.loc[:9, "description"]

# Select the records with index labels 1, 2, 3, 5, and 8
sample_reviews = reviews.iloc[[1,2,3,5,8]]

df = reviews.loc[[0,1,10,100], ['country','province','region_1','region_2']]

df = reviews.loc[0:99, ['country','variety']] # or cols_idx = [0, 11] df = reviews.iloc[:100, cols_idx]

italian_wines = reviews.loc[reviews.country=='Italy']

## Note: don't forget the () inside [] 
top_oceania_wines = reviews.loc[(reviews.points>=95) & (reviews.country.isin(['Australia','New Zealand']))]
