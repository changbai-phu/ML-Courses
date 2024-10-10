### Creating, Reading and Writing

import pandas as pd

#### Tutorial ####
# DataFrame: table
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

# Series: list / a single column of DataFrame
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

# Read CSV: comma-separated values
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.shape # check the size
wine_reviews.head() # grab first 5 rows



#### Exercise ####
# Note: need to use [] 
fruits = pd.DataFrame({'Apples':[30], 
                       'Bananas':[21]})

fruit_sales = pd.DataFrame({'Apples':[35, 41],
                           'Bananas':[21, 34]},
                          index=['2017 Sales', '2018 Sales'])

ingredients = pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'], name='Dinner')

# use index_col to specify (so not showing the unnamed column)
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)

# write to csv
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv('cows_and_goats.csv')
