# Summary Functions and Maps

#### Tutorial ####
import pandas as pd
pd.set_option('display.max_rows', 5)
import numpy as np
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# describe() is type-aware; provide high level summary 
reviews.points.describe() 
reviews.taster_name.describe()

reviews.points.mean()
reviews.taster_name.unique()
reviews.taster_name.value_counts()

# Maps - map()
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean) # remean the scores to 0; returns a new Series
# Maps - apply() - transform a whole DataFrame 
def remean_points(row):
    row.points = row.points - review_points_mean
    return row
reviews.apply(remean_points, axis='columns') # if axis='index', then need to give a function to transform each colummn
# faster way
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean

# easy way of combining country and region information
reviews.country + " - " + reviews.region_1


#### Exercises ####
import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()

# Q1 What is the median of the points column in the reviews DataFrame?
median_points = reviews.points.median()

# Q2 What countries are represented in the dataset?
countries = reviews.country.unique()

# Q3 How often does each country appear in the dataset?
reviews_per_country = reviews.country.value_counts()

# Q4 Create variable centered_price containing a version of the price column with the mean price subtracted.
## 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.
review_points_mean = reviews.price.mean()
centered_price = reviews.price - review_points_mean

# Q5 Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

# Q6 Counting how many times each of these two words appears in the description column in the dataset
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()  # The map() function applies the lambda function to each description (desc) in this column.
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

# Q7  A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.
# Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')