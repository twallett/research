#%%
import os
import pandas as pd
import re

# Define the directory containing the reward txt files
folder_path = 'cum-rew'

# Temporary storage for the data and corresponding column keys
data = {}
columns = []

# Regex pattern to extract environment, model, and hyperparameters
pattern = re.compile(r'^(.*?)_(PolicyGradient|PPOClip)_(pi[\d.]+_v[\d.]+)_rewards\.txt$')

# Loop through all .txt files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        match = pattern.match(filename)
        if not match:
            print(f"Skipping unrecognized file format: {filename}")
            continue
        
        env, model, params = match.groups()
        col_key = (env, model, params)
        columns.append(col_key)

        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            rewards = [float(line.strip()) for line in file if line.strip()]
        data[col_key] = rewards

# Create a DataFrame from the data
df = pd.DataFrame.from_dict(data, orient='columns')

# Set the column MultiIndex
df.columns = pd.MultiIndex.from_tuples(columns, names=["Environment", "Model", "Alpha Settings"])

# Save to CSV
df.to_csv('acrobot_results.csv')
print("Successfully created acrobot_results.csv")
# %%