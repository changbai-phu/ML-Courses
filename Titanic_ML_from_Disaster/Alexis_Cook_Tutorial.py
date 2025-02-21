'''
Codes below are copied from Alexis Cook's Titanic Tutorial
https://www.kaggle.com/code/alexisbcook/titanic-tutorial

Summary of keypoints: 
Dataset Overview: The tutorial uses the Titanic dataset, 
    which contains information about passengers 
    such as their age, sex, class, and whether they survived or not. 
    The goal is to predict survival using this data.

Data Exploration: It explains how to explore the data, 
    check for missing values, and understand the relationships 
    between different features and the target (survival).

Data Preprocessing: The tutorial walks through cleaning the data 
    by filling missing values (like age) and converting categorical data 
    (like gender) into numerical values so that the machine learning model can use them.

Modeling: The tutorial then builds a simple machine learning model 
    using Logistic Regression. It explains how to split the data into training 
    and testing sets, train the model, and evaluate its accuracy.

Submission: Lastly, it shows how to make predictions on the test set 
    and submit the results to Kaggle’s Titanic competition.

Overall, the tutorial provides a step-by-step guide on how to work with a dataset, 
clean and preprocess the data, build a model, and make predictions, 
all while introducing basic machine learning concepts.
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
'''
% of women who survived: 0.7420382165605095
'''

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)
'''
% of men who survived: 0.18890814558058924
'''

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")