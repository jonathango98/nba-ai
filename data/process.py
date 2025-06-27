import pandas as pd
DEBUG = True
# DEBUG = False

class Process:
    def __init__(self): 
        self.raw_data = pd.read_csv('clean_data/team.csv')
        self.sorted_data = self.raw_data.sort_values(by = 'game_date', ascending = False)


        self.floor_year = self.sorted_data.copy()
        years = self.floor_year.apply(self.start_year, axis = 1)
        self.floor_year["season_year"] = years
        self.season_2023 = self.floor_year[self.floor_year["season_year"] == 2023]

    def start_year(self,row):
        return int(row["season_year"][:4])
    

result = Process()
sorted = result.sorted_data
floor_year = result.floor_year
season_2023 = result.season_2023
if not DEBUG:
    sorted.to_csv('clean_data/subdata/sort.csv', index= False)
    floor_year.to_csv('clean_data/subdata/floor_year.csv', index= False)
    season_2023.to_csv('clean_data/subdata/season_2023.csv', index= False)
else:
    print(f"Sorted datatypes: \n {sorted.dtypes} \n")
    print(f"Floor_year datatypes: \n {floor_year.dtypes} \n")
    print(f"Season_2023 datatypes: \n {season_2023.dtypes} \n")
