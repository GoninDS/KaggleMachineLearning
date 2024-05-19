import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data_path = '../../Data/melb_data.csv'
data = pd.read_csv(data_path)
# Prints out all of the columns in the dataframe
print("Columns:")
print(data.columns)
print("\n")

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
data = data.dropna(axis=0)

# After cleaning the dataset we want a prediction target
y = data.Price

# We want columns to be inputted into our model for predictiones
# These are called "features", by convention this is the X
data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[data_features]

# Methods to check what data we are using
print("Describe X")
print(x.describe())
print("\n")

print("Head of X")
print(x.head())
print("\n")

# Steps for defining a model
# Define: What type of model? 
# Fit: Capture patterns from provided data.
# Predict
# Evaluate: Determine how accurate it was
# Setting a random state ensures you get the same results
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(x,y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("\n")
print("The predictions are")
print(melbourne_model.predict(x.head()))