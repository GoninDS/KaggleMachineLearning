import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data_path = '../../Data/melb_data.csv'
data = pd.read_csv(data_path)

# We are going to split the data in training and testing
data = data.dropna(axis=0)
data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[data_features]
y = data.Price
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

# Define and fit the model with the training data
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_x, train_y)

# Get predicted prices with validation data
val_predictions = melbourne_model.predict(val_x)
print("Mean absolute error:")
print(mean_absolute_error(val_y, val_predictions))