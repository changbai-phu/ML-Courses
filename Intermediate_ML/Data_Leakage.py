#### Tutorial ####
import pandas as pd

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])
# Select target
y = data.card
# Select predictors
X = data.drop(['card'], axis=1)
print("Number of rows in the dataset:", X.shape[0])
X.head()

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')
print("Cross-validation accuracy: %f" % cv_scores.mean())
'''
Cross-validation accuracy: 0.981052
98% looks suspicious, should inspect closer on the data for target leakage
e.g., expenditure
'''
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
'''
Fraction of those who did not receive a card and had no expenditures: 1.00
Fraction of those who received a card and had no expenditures: 0.02
features 'share', 'active', 'majorcards' -- better to be safe than concerning, so should exclude those as well
'''
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')
print("Cross-val accuracy: %f" % cv_scores.mean())
'''Cross-val accuracy: 0.830919'''

#### Exercise ####
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex7 import *
print("Setup Complete")

# Q1: The data science of shoelaces
'''
My answer: 
the leather used feature constitutes a source of data leakage.
And it is target leakage. 
Correct Answer: 
This is tricky, and it depends on details of how data is collected (which is common when thinking about leakage). 
Would you at the beginning of the month decide how much leather will be used that month? 
If so, this is ok. But if that is determined during the month, you would not have access to it when you make the prediction. 
If you have a guess at the beginning of the month, and it is subsequently changed during the month, 
the actual amount used during the month cannot be used as a feature (because it causes leakage).
'''

# Q2: Return of the shoelaces
'''
My answer: 
It still depends. If the amount of leather ordered is made at the beginning of the month, it is not leakage.
Just like the amount of leather used case, if the amount of ordered changes during the month, it is target leakage.
Correct answer:
This could be fine, but it depends on whether they order shoelaces first or leather first. 
If they order shoelaces first, you won't know how much leather they've ordered when you predict their shoelace needs. 
If they order leather first, then you'll have that number available when you place your shoelace order, and you should be ok.
'''

# Q3: Getting rich with crptocurrencies
'''
My answer: no leakage in his model
Correct Answer:
There is no source of leakage here. These features should be available at the moment you want to make a predition, 
and they're unlikely to be changed in the training data after the prediction target is determined. 
But, the way he describes accuracy could be misleading if you aren't careful. (The value of the cryptocurrency in dollars has fluctuated up and down by over 100 in the last year, and yet his model's average error is less than 1. He says this is proof his model is accurate)
If the price moves gradually, today's price will be an accurate predictor of tomorrow's price, 
but it may not tell you whether it's a good time to invest.
 For instance, if it is 100 tomorrow may seem accurate, even if it can't tell you whether the price is going up or down from the current price. 
 A better prediction target would be the change in price over the next day. If you can consistently predict whether the price is about to go up or down 
 (and by how much), you may have a winning investment opportunity.
'''

# Q4: Preventing infections
'''
My answer:
Yes. Average infection rate may have problem with train-test contamination, and target leakage.
Correct answer:
This poses a risk of both target leakage and train-test contamination (though you may be able to avoid both if you are careful).
You have target leakage if a given patient's outcome contributes to the infection rate for his surgeon, which is then plugged back into the prediction model for whether that patient becomes infected. 
You can avoid target leakage if you calculate the surgeon's infection rate by using only the surgeries before the patient we are predicting for.
Calculating this for each surgery in your training data may be a little tricky.
You also have a train-test contamination problem if you calculate this using all surgeries a surgeon performed, including those from the test-set. 
The result would be that your model could look very accurate on the test set, even if it wouldn't generalize well to new patients after the model is deployed. This would happen because the surgeon-risk feature accounts for data in the test set. Test sets exist to estimate how the model will do when seeing new data. 
So this contamination defeats the purpose of the test set.
'''

# Q5: Housing prices
'''
Here are four features that could be used as predictors.
1.Size of the house (in square meters)
2.Average sales price of homes in the same neighborhood
3.Latitude and longitude of the house
4.Whether the house has a basement
'''
# Fill in the line below with one of 1, 2, 3 or 4.
potential_leakage_feature = 2
'''
Explanations:
2 is the source of target leakage. Here is an analysis for each feature:
1. The size of a house is unlikely to be changed after it is sold (though technically it's possible). 
But typically this will be available when we need to make a prediction, 
and the data won't be modified after the home is sold. So it is pretty safe.

2. We don't know the rules for when this is updated. 
If the field is updated in the raw data after a home was sold, and the home's sale is used to calculate the average, 
this constitutes a case of target leakage. At an extreme, if only one home is sold in the neighborhood, 
and it is the home we are trying to predict, then the average will be exactly equal to the value we are trying to predict. In general, for neighborhoods with few sales, the model will perform very well on the training data. But when you apply the model, the home you are predicting won't have been sold yet, so this feature won't work the same as it did in the training data.

3. These don't change, and will be available at the time we want to make a prediction. 
So there's no risk of target leakage here.

4. This also doesn't change, and it is available at the time we want to make a prediction. 
So there's no risk of target leakage here.
'''