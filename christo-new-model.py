import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

team = pd.read_csv('data/clean_data/team.csv')

df = team.copy()
df['game_date'] = pd.to_datetime(df['game_date'], unit='s')
df = df.sort_values(by='game_date', ascending=True)

# Encode categorical features
le_home = LabelEncoder()
le_away = LabelEncoder()

# Encode categorical columns without dropping any columns
df['team_home'] = le_home.fit_transform(df['team_home'])
df['team_away'] = le_away.fit_transform(df['team_away'])


# Make new columns (time series)
# Average (for last 5 games of the same season) of all columns from fgm_home onwards. 
def add_rolling_avg(df, stat_col, team_col, season_col='season_year', date_col='game_date', window=5):
    """
    Adds a new column to the dataframe with the rolling average of a stat
    over the last `window` games for each team within the same season.
    
    Parameters:
    - df: DataFrame
    - stat_col: Name of the stat column to average (e.g. 'fgm_home')
    - team_col: Home status of the team column (e.g. 'team_id_home' or 'team_id_away')
    - season_col: Name of the season column (default: 'season_year')
    - date_col: Name of the date column (should be datetime)
    - window: Number of previous games to average (default: 5)
    
    Returns:
    - df with new column added: '{stat_col}_avg{window}'
    """
    
    # Define the new column name
    new_col = f"{stat_col}_avg{window}"
    
    # Calculate the rolling average (excluding current row via shift)
    df[new_col] = (
        df.groupby([season_col, team_col], group_keys=False)[stat_col]
        .apply(lambda x: x.shift().rolling(window=window, min_periods=window).mean())
    )
    
    return df

team_id = ['team_id_home','team_id_away']
cols = df.columns[8:]
i=2
for col in cols:
    df = add_rolling_avg(df, stat_col=col, team_col=team_id[i%2])
    i+=1

# Drop irrelevant columns and NaN entries
df = df.drop(columns=['season_year', 'game_id', 'game_date',
       'team_home', 'team_away', 'fgm_home', 'fgm_away', 'fga_home',
       'fga_away', 'fg3m_home', 'fg3m_away', 'fg3a_home', 'fg3a_away',
       'ftm_home', 'ftm_away', 'fta_home', 'fta_away', 'oreb_home',
       'oreb_away', 'dreb_home', 'dreb_away', 'ast_home', 'ast_away',
       'tov_home', 'tov_away', 'stl_home', 'stl_away', 'blk_home', 'blk_away',
       'blka_home', 'blka_away', 'pts_home', 'pts_away', 'plus_minus_home',
       'plus_minus_away'])
df = df.dropna()

# Features and target
X = df.drop(columns=['win_home'])  # Keep all columns except the target
y = df['win_home']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize logistic regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred, output_dict=True)

print(accuracy)