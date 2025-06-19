import pandas as pd

class TeamCleanup:
    def __init__(self):
        self.team_raw = pd.read_csv('raw_data/team_raw.csv') 
        self.team_raw.drop(columns=['TEAM_NAME','FG_PCT','FG3_PCT','FT_PCT','REB','PF','PFD'], inplace=True)
        self.team_raw.drop(self.team_raw.columns[-27:], axis=1, inplace=True)
        self.team_raw['WL'] = self.team_raw['WL'].replace({'W':1, 'L':0})
        self.team_raw['row_num'] = self.team_raw.groupby('GAME_ID').cumcount() + 1
        
        df_pivot = self.team_raw.pivot(index='GAME_ID', columns='row_num')
        df_pivot.columns = [f"{col}_{num}" for col, num in df_pivot.columns]
        self.team = df_pivot.reset_index() 
        
    
df = TeamCleanup().team

print(df.columns)