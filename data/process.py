import pandas as pd
#Set DEBUG to False to save data
DEBUG = True
DEBUG = False


OLDEST_SEASON = 2010
NEWEST_SEASON = 2023
TEAM_AMOUNT = 20
class Process:
    def __init__(self):

        #Sort data chronologically, from most recent to oldest
        self.raw_data = pd.read_csv('clean_data/team.csv')
        self.sorted_data = self.raw_data.sort_values(by = 'game_date', ascending = False)

        #Convert season_year into a more readable form, using the starting season year (e.g. 2011-2012 becomes 2011)
        self.floor_year = self.sorted_data.copy()
        years = self.floor_year.apply(self.start_year, axis = 1)
        self.floor_year["season_year"] = years

        #Filter games into discrete seasons, stores in a dictionary
        self.seasons = {}
        for i in range(OLDEST_SEASON, NEWEST_SEASON + 1):
            self.seasons[i] = self.floor_year[self.floor_year["season_year"] == i]
        
        #Scrapes data from the newest season to get the relationship between team ids and team names
        iter_frame = self.seasons[NEWEST_SEASON].itertuples()
        temp = {}
        while(len(temp) < TEAM_AMOUNT):
            row = next(iter_frame)
            if row.team_id_home not in temp.keys():
                temp[row.team_id_home] = row.team_home
            if row.team_id_away not in temp.keys():
                temp[row.team_id_away] = row.team_away
        self.teams = pd.DataFrame([{'team_id': k, 'team_name': v} for k, v in temp.items()])
        self.teams = self.teams.sort_values(by = 'team_name')


    def start_year(self,row):
        return int(row["season_year"][:4])

#Saves the dataframes into discrete CSVs when DEBUG is False
result = Process()
sorted = result.sorted_data
floor_year = result.floor_year
teams = result.teams

#Set DEBUG at top of program
if not DEBUG:
    sorted.to_csv('clean_data/subdata/sort.csv', index= False)
    floor_year.to_csv('clean_data/subdata/floor_year.csv', index= False)
    teams.to_csv('clean_data/subdata/teams.csv', index = False)
    for key, dataframe in result.seasons.items():
        dataframe.to_csv(f"clean_data/subdata/seasons/season_{key}.csv", index = False)
else:
    print(f"Sorted datatypes: \n {sorted.dtypes} \n")
    print(f"Floor_year datatypes: \n {floor_year.dtypes} \n")
