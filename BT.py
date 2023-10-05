import pandas as pd
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# load annotated data
df = pd.read_csv('IRT_annotation.csv')

# map model_id to (model_name, setting)
one = "gpt4"
two = "chatgpt"
three = "gpt3"
four = "bard"
five = "vicuna"
six = "alpaca"
de_entity = {
    1: two, 2: five, 3: six, 4: three, 5: one, 6: four,
    7: three, 8: five, 9: two, 10: four, 11: six, 12: one,
    13: three, 14: four, 15: one, 16: two, 17: five, 18: six,
    19: three, 20: four, 21: five, 22: one, 23: six, 24: two,
    25: two, 26: five, 27: one, 28: three, 29: four, 30: six,
    31: one, 32: two, 33: five, 34: three, 35: four, 36: six,
}
mapping = {one: 0, two: 1, three: 2, four: 3, five: 4, six: 5}
print([one, two, three, four, five, six])

topics = df['topic'].unique().tolist()
assert len(topics) == 31
stances = df['stance'].unique().tolist()
assert len(stances) == 2

# Replace values in the 'model_id' column using the mapping dictionary
df['model_id'] = df['model_id'].map(de_entity)


# Define the stances you want to analyze
stances = df['stance'].unique()

# Define the linear transformation function to map strengths to [1, 100]
def linear_transform(strength):
    return 1 + (99 * strength)

# Initialize a dictionary to store estimated strengths for each stance and setting
estimated_params = {}

# Iterate through stances
for stance in stances:

    # Filter data for the current stance
    stance_data = df[df['stance'] == stance]

    # Get settings within this stance
    settings = stance_data['setting'].unique()

    # Initialize a dictionary for this stance
    stance_estimated_params = {}

    # Iterate through settings within this stance
    for setting in settings:
        setting_data = stance_data[stance_data['setting'] == setting]

        # Get model names present in this setting
        present_model_names = setting_data['model_id'].unique()

        # Calculate all unique pairs for comparisons based on the present model names
        pairs = list(combinations(present_model_names, 2))

        # Create a DataFrame to store pairwise comparisons
        pairwise_df = pd.DataFrame({'Model1': [pair[0] for pair in pairs], 'Model2': [pair[1] for pair in pairs]})

        # Iterate through the data to fill the pairwise comparisons DataFrame
        for i, row in pairwise_df.iterrows():
            model1 = row['Model1']
            model2 = row['Model2']
            better_count = len(setting_data[(setting_data['model_id'] == model1) & (setting_data['ranking'] < setting_data['ranking'].max())]) - \
                          len(setting_data[(setting_data['model_id'] == model2) & (setting_data['ranking'] < setting_data['ranking'].max())])
            if better_count > 0:
                pairwise_df.at[i, 'Comparison'] = 1  # Model1 is better
            elif better_count < 0:
                pairwise_df.at[i, 'Comparison'] = -1  # Model2 is better
            else:
                pairwise_df.at[i, 'Comparison'] = 0  # Tie


        def bradley_terry(params):
            A = np.exp(params)
            n = len(A)
            likelihood = 0
            for i in range(len(pairwise_df)):
                model1 = pairwise_df.at[i, 'Model1']
                model2 = pairwise_df.at[i, 'Model2']
                comparison = pairwise_df.at[i, 'Comparison']
                i_index = present_model_names.tolist().index(model1)
                j_index = present_model_names.tolist().index(model2)
                if comparison == 1:
                    likelihood += np.log(A[i_index] / (A[i_index] + A[j_index]))
                elif comparison == -1:
                    likelihood += np.log(A[j_index] / (A[i_index] + A[j_index]))
                else:
                    likelihood += np.log(0.5)  # Tie
            return -likelihood

        # Initial values for model parameters
        params0 = np.zeros(len(present_model_names))

        # Run optimization to estimate strengths
        result = minimize(bradley_terry, params0, method='BFGS')

        # Apply linear transformation to map strengths to [1, 100], we finally decide only keep move 3 decimal places to the left
        transformed_strengths = linear_transform(result.x)

        # Store estimated strengths for this setting within this stance
        stance_estimated_params[setting] = {model: strength for model, strength in zip(present_model_names, transformed_strengths)}

    # Store estimated strengths for this stance
    estimated_params[stance] = stance_estimated_params

# Print the estimated model parameters (strengths of each model) for each stance and setting
for stance, stance_params in estimated_params.items():
    print(f"Stance: {stance}")
    for setting, params in stance_params.items():
        print(f"Estimated Model Parameters for Setting '{setting}' (Strengths):")
        for model, strength in params.items():
            print(f"Model {model}: {strength:.2f}")