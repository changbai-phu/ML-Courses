# Grouping and Sorting

#### Tutorial ####
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

reviews.groupby('points').points.count() # can also use value_counts()

reviews.groupby('points').price.min() # get the cheapest wine in point value

# select the name of the first wine reviewed from each winery
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

# select the best wine by country and province based on point value
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

# agg() allows running different functions simultaneously
reviews.groupby(['country']).price.agg([len, min, max])

## Multi-indexes
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
mi = countries_reviewed.index
type(mi)

# convert back to a regular index
countries_reviewed.reset_index()

## Sorting
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len') # default ascending order
countries_reviewed.sort_values(by='len', ascending=False)

countries_reviewed.sort_index()

countries_reviewed.sort_values(by=['country', 'len'])


#### Exercies ####
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.grouping_and_sorting import *
print("Setup complete.")

#Q1 Create a Series whose index is the taster_twitter_handle category from the dataset, 
# and whose values count how many reviews each person wrote.
reviews_written = reviews.groupby(['taster_twitter_handle']).taster_twitter_handle.count()
# or reviews_written = reviews.groupby('taster_twitter_handle').size()
# or reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

#Q2 Create a Series whose index is wine prices and whose values is the maximum number of points a wine costing 
# that much was given in a review. Sort the values by price, ascending
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

#Q3 Create a DataFrame whose index is the variety category from the dataset and whose values are the min and 
# max values thereof. 
price_extremes = reviews.groupby('variety').price.agg([min, max])

#Q4 Create a variable sorted_varieties containing a copy of the dataframe from the previous question where varieties 
# are sorted in descending order based on minimum price, then on maximum price (to break ties).
# Note: 'False' - uppercase matters
price_extremes = reviews.groupby('variety').price.agg([min, max])
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

#Q5 Create a Series whose index is reviewers and whose values is the average review score given out by that reviewer. 
reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()
# can run reviewer_mean_ratings.describe() to get more information about this new Series just created

#Q6 Create a Series whose index is a MultiIndexof {country, variety} pairs. For example, a pinot noir produced in the US 
# should map to {"US", "Pinot Noir"}. Sort the values in the Series in descending order based on wine count.
country_variety_counts = reviews.groupby(['country', 'variety']).variety.count().sort_values(ascending=False)