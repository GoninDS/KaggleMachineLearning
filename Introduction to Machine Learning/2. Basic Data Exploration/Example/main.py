import pandas as pd

# Set up the data and print a summary
data_path = "melb_data.csv"
data = pd.read_csv(data_path)
print(data.describe())