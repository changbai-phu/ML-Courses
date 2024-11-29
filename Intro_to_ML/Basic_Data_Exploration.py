# First: Get familiar with the data

import pandas as pd

#### Tutorial ####
# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()

# Data description:
# count: # of rows have non-missing values
# mean: average. std: standard deviation. 
# 25%: sort column from min to max, and go a quarter way through the list,
#   the number that is >25% of the values and smaller than 75% of the value.
#   pronounce "25th percentile"


#### Exercise ####
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")

# Step 1: Loading data
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Step 2: Review the data
# Print summary statistics in next line
home_data.describe()
# What is the average lot size (rounded to nearest integer)?
#   mean value of LotArea
avg_lot_size = 10517
# As of today, how old is the newest home (current year - the date in which it was built)
#   current year minus the max value of YearBuilt
newest_home_age = 14

# Step 3 Think About Your Data
'''The newest house in your data isn't that new. A few potential explanations for this:
1.They haven't built new houses where this data was collected.
2.The data was collected a long time ago. Houses built after the data publication wouldn't show up.
If the reason is explanation #1 above, does that affect your trust in the model you build with this data? What about if it is reason #2?
### before take a look at the data, I prefer it is most likely going to be #2 which does affect my trust in model, since the model we built
### based on that data won't be up-to-date, may not fit well and not doing good in prediction anymore. However, we can use new data to test
### if the model fit first to see how it performs and adjust or re-train the model later.
### if assume #1 is the case, it does affect as well since there must be some reasons preventing companies from building new houses, either the area
### is too packed with buildings, or policies, or nature disasters etc. If those are true, the model won't fit anymore.

How could you dig into the data to see which explanation is more plausible?
### if assume #1, should see freeze in the YrSold, but as we can tell, there are houses sold every year.
### so I will say #2 is more plausible.
'''
