import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data_path = '../../Data/melb_data.csv'
data = pd.read_csv(data_path)

# There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE). 
# For this we need a model
data = data.dropna(axis=0)
data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[data_features]
y = data.Price

# Define and fit the model
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(x, y)

# Let's calculate the mean absolute error
predicted_home_prices = melbourne_model.predict(x)
print("Mean absolute error:")
print(mean_absolute_error(y, predicted_home_prices))

# What's the problem?
# => In sampling, using the same data for training and evaluating.

