import pandas as pd

# Load the uploaded team_raw.csv file
df = pd.read_csv("raw_data/team_raw.csv")

# Function to extract the enemy abbreviation from MATCHUP
def extract_enemy(matchup, team_abbr):
    parts = matchup.split()
    if parts[1] == '@' or parts[1] == 'vs.':
        if parts[0] == team_abbr:
            return parts[2]
        else:
            return parts[0]
    return None

def extract_home(matchup):
    parts = matchup.split()
    if parts[1] == '@':
        return 0
    else:
        return 1
    return None

# Apply the function to get enemy team
df['enemy'] = df.apply(lambda row: extract_enemy(row['MATCHUP'], row['TEAM_ABBREVIATION']), axis=1)
df['home'] = df.apply(lambda row: extract_home(row['MATCHUP']), axis=1)

# Create the new dataframe
result_df = df[['TEAM_ABBREVIATION', 'enemy', 'WL', 'home', 'PLUS_MINUS']].copy()
result_df.columns = ['team', 'enemy', 'win', 'home', 'diff']


# Convert 'WL' column to boolean: True for 'W', False for 'L'
result_df['win'] = result_df['win'].map({'W': 1, 'L': 0})

# Save the result to CSV
output_path = "team.csv"
result_df.to_csv(output_path, index=False)

output_path
