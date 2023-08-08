import pandas as pd

# Load the csv file
df = pd.read_csv('unshuffled.csv', header=None)

# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataframe to a new csv file
df.to_csv('input.csv', index=False, header=False)
