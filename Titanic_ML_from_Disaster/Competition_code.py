'''
Experiment 1:
Based on Alexis's code, add new feature Age and Fare, 
use HistGradientBoostingClassifier due to missing values,
update max_depth=7 and add learning_rate=0.05.
Others left the same.
Result: 0.77511 (same as the RandomForestClassifer)
'''
from sklearn.ensemble import HistGradientBoostingClassifier

y = train_data["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = HistGradientBoostingClassifier(max_iter=100, max_depth=7, random_state=1, learning_rate=0.05)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

'''
Experiment 2:
Split X into train and valid sets to calculate MAE for validation other than the Kaggle result.
Use for loop to find the n_estimators that leads to the min MAE (which is 300)
And plug 300 back to the original code (RandomForestClassifier) and then predict X_test.
Result: 0.77511 (same as the RandomForestClassifer with n=100)
'''
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_score(n):
    model = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)

n_estimator_list = [100, 150, 200, 250, 300, 350, 400]
results = {num: get_score(num) for num in n_estimator_list}  # Dictionary comprehension
print(results)
'''
{100: 0.21524663677130046, 150: 0.21524663677130046, 200: 0.2242152466367713, 250: 0.21524663677130046, 
300: 0.2062780269058296, 350: 0.21524663677130046, 400: 0.2242152466367713}
'''

model_final = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1)
model_final.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")