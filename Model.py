import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset and Explore Data
df = pd.read_csv('medical-charges.csv')
print(df , "\n\n")
print(df.head() , "\n\n")
print(df.describe().T  , "\n\n")
print(df.columns)

#check missing values
print(df.isnull().sum())

# plt.bar() needs numeric values
# df.columns are strings
# The correct way is Count plot (recommended) or Target Variable distribution (example)
'''df.hist(figsize=(10,8))
plt.title("Medical Insurance Terms")
plt.show()'''

plt.hist(df['charges'] , bins=30)
plt.title("Medical Insurance features")
plt.show()

# EDA , Feature Engineering and Data Preprocessing

# seperate dependent and independent feature
X = df.drop(columns=['charges'])
y = df['charges']

print(X.head) 
print(y.head)

# Train and Test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , y ,test_size=0.3 , random_state=42)

print(X_train)
print(X_test) 
print(y_train)
print(y_test)


# standardizing the datset {In this dataset , few columns have string value so encode and column tranfer will change only numerical values}
# So you cannot use StandardScalar Directly
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)
#instead of 
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

#Import inbuild model from Sklearn

from sklearn.linear_model import LinearRegression

#for hyperparameter tuning, we can direct use lasso regression , ridge regression , elastic regression.
# But we will use cross validation

from sklearn.model_selection import cross_val_score

regression  = LinearRegression()
regression.fit(X_train_processed,y_train)

MSE = cross_val_score(regression, X_train_processed , y_train, scoring='neg_mean_squared_error' , cv=5)

print("Mean MSE : ", np.mean(MSE))

#Prediction
reg_pred=regression.predict(X_test_processed)
print(reg_pred)

# Compare this predicted values with truth values {y_test}

import seaborn as sns
sns.displot(reg_pred-y_test , kind='kde')
plt.title("Compare The difference")
plt.show()


from sklearn.metrics import r2_score
score = r2_score(reg_pred, y_test)
print(score)

import joblib

joblib.dump(regression, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

