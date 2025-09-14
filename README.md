🚢 Titanic Survival Prediction

This project is my implementation of the classic Titanic: Machine Learning from Disaster problem. It is a beginner-friendly machine learning project where the goal is to predict whether a passenger survived the Titanic shipwreck based on various features such as age, sex, class, and more.

📌 Problem Statement

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, resulting in the deaths of more than 1,500 passengers and crew.

The objective of this project is to predict the survival of passengers using passenger data (such as name, age, gender, socio-economic class, etc.).

📂 Dataset

I used the dataset from the Kaggle Titanic Competition
.

train.csv → Contains passenger details + survival label (Survived).

test.csv → Contains passenger details without labels (used for predictions).

gender_submission.csv → Sample submission file from Kaggle.

# Features:

PassengerId: Unique ID for each passenger

Survived: Survival (0 = No, 1 = Yes) → Target variable

Pclass: Passenger’s ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

Name: Passenger name

Sex: Gender of passenger

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Ticket: Ticket number

Fare: Passenger fare

Cabin: Cabin number (many missing values)

Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# 🛠️ Steps Involved
1. Exploratory Data Analysis (EDA)

Checked data types, null values, and distributions.

Plotted survival rates by Sex, Pclass, Age, Embarked, etc.

Key insights:

Females had much higher survival rates.

1st-class passengers survived more than 3rd-class.

Children had higher survival probability.

# 2. Data Preprocessing

Handled missing values (Age, Embarked, Cabin).

Dropped irrelevant columns (PassengerId, Ticket).

Created new features:

FamilySize = SibSp + Parch + 1

Title extracted from Name (Mr, Miss, Mrs, etc.).

Encoded categorical variables (Sex, Embarked, Title, etc.).

Scaled numerical features where necessary.

# 3. Model Building

Split dataset into training and validation sets.

Tried multiple classification models:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting (XGBoost, LightGBM)

Evaluated models using accuracy score and cross-validation.