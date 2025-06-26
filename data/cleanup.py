import pandas as pd
import re

def to_snake_case(s):
    """Convert a string to snake_case."""
    return re.sub(r'[\W\s]+', '_', s).strip().lower()

class TeamCleanup:
    def __init__(self):
        self.team_raw = pd.read_csv('raw_data/team_raw.csv') 
        self.team_raw.drop(columns=['TEAM_NAME','FG_PCT','FG3_PCT','FT_PCT','REB','PF','PFD', 'MIN'], inplace=True)
        self.team_raw.drop(self.team_raw.columns[-27:], axis=1, inplace=True)
        self.team_raw['WL'] = self.team_raw['WL'].replace({'W':1, 'L':0})

        df_home = self.team_raw[self.team_raw['MATCHUP'].str.contains('vs.')].add_suffix('_home')
        df_away =  self.team_raw[self.team_raw['MATCHUP'].str.contains('@')].add_suffix('_away')

        df_merged = pd.merge(df_home, df_away, left_on="GAME_ID_home", right_on="GAME_ID_away") 
        # print(df_merged)

        #keep home column and delete home suffix
        df_merged.drop(columns=['SEASON_YEAR_away', 'GAME_ID_away','GAME_DATE_away', 'MATCHUP_home', 'MATCHUP_away', 'WL_away'], inplace=True) 
        df_merged.rename(columns={"SEASON_YEAR_home": "season_year", 
                        "GAME_ID_home": "game_id", 
                        "GAME_DATE_home": "game_date",
                        "WL_home":"win_home",
                        "TEAM_ABBREVIATION_home" : "team_home",
                        "TEAM_ABBREVIATION_away": "team_away"}, inplace=True) 

        #change game_date to datetime object
        df_merged['game_date'] = pd.to_datetime(
            df_merged['game_date']
            .str.replace('t', ' ', case=False)
            .str.replace('_', '-', n=2)
            .str.replace('_', ':', n=2),errors='coerce').apply(lambda x:int(x.timestamp()))
    
        # #Convert column names to snakecase
        df_merged.columns = [to_snake_case(col) for col in df_merged.columns]

        # #Convert all string values in the DataFrame to snake_case
        for col in df_merged.select_dtypes(include='object').columns:
            df_merged[col] = df_merged[col].apply(lambda x: to_snake_case(x) if isinstance(x, str) else x)
        
        df_merged['team_home'] = df_merged['team_home'].str.upper()
        df_merged['team_away'] = df_merged['team_away'].str.upper()

        df_merged = df_merged[['season_year', 'game_id', 'game_date', 'win_home',
                                'team_id_home', 'team_id_away',
                                'team_home', 'team_away',
                                'fgm_home', 'fgm_away',
                                'fga_home', 'fga_away',
                                'fg3m_home', 'fg3m_away',
                                'fg3a_home', 'fg3a_away',
                                'ftm_home', 'ftm_away',
                                'fta_home', 'fta_away',
                                'oreb_home', 'oreb_away',
                                'dreb_home', 'dreb_away',
                                'ast_home', 'ast_away',
                                'tov_home', 'tov_away',
                                'stl_home', 'stl_away',
                                'blk_home', 'blk_away',
                                'blka_home', 'blka_away',
                                'pts_home', 'pts_away',
                                'plus_minus_home', 'plus_minus_away']] 
        print(df_merged.dtypes)
        self.team = df_merged
        
df = TeamCleanup().team
#save to csv
df.to_csv('clean_data/team.csv', index= False) 
# print(df['game_date'].dtype)

# print(df.columns) 
