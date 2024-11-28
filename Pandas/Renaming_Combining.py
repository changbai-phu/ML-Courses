# Renaming and Combining

#### Tutorial ####
import pandas as pd
pd.set_option('display.max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

reviews.rename(columns={'points': 'score'})

# rename col 1 rows
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

# Combining: concat(), join(), merge()
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])

## join(): combine DataFrame objects that have common index
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK')


#### Exercise ####
##load data
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.renaming_and_combining import *
print("Setup complete.")

reviews.head()

## Q1: Create a copy of reviews with these columns renamed to region and locale, respectively.
renamed = reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})

## Q2: Set the index name in the dataset to wines.
reindexed = reviews.rename_axis('wines')

## Q3: Create a DataFrame of products mentioned on either subreddit.
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"

combined_products = pd.concat([gaming_products, movie_products])
combined_products.head()

## Q4: Use unique key MeetID to combine the two tables into one
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

left = powerlifting_meets.set_index(['MeetID'])
right = powerlifting_competitors.set_index(['MeetID'])
powerlifting_combined = left.join(right, lsuffix='_meet', rsuffix='_competitors')
powerlifting_combined.head()
