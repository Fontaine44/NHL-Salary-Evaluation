import pandas as pd

START_SEASON = 2013
END_SEASON = 2023

def main():
    salary_info_df = pd.read_csv("salary_info_merged.csv", encoding="utf-8")

    # Drop goalies
    salary_info_df = salary_info_df[salary_info_df["position"] != "G"]
    salary_info_df.drop(columns=["position"], inplace=True)

    # Calculate age at the start of the season
    player_birthdate = pd.to_datetime(salary_info_df["birthDate"])
    salary_info_df["age"] = salary_info_df["season"] - player_birthdate.dt.year
    salary_info_df.drop(columns=["birthDate"], inplace=True)

    # Calculate height in cm
    salary_info_df["height"] = salary_info_df["height"].apply(height_to_cm)
    
    # Create df with all stats
    stats_df_list = []
    for year in range(START_SEASON, END_SEASON + 1):
        yearly_stats_df = pd.read_csv(f"{year}-{year+1}.csv")

        # Replace ARI for UTA
        yearly_stats_df["team"] = yearly_stats_df["team"].str.replace("ARI", "UTA", regex=False)

        # Drop unpopulated stats
        yearly_stats_df.drop(columns=["xGoalsForAfterShifts", "xGoalsAgainstAfterShifts", "corsiForAfterShifts", "corsiAgainstAfterShifts", "fenwickForAfterShifts", "fenwickAgainstAfterShifts"], inplace=True)

        # Create season stats for all situations
        stats_df = yearly_stats_df[yearly_stats_df["situation"] == "all"]
        stats_df = stats_df.drop(columns=["situation"])
        
        # Add stats for other situations
        for situation in ["5on5", "5on4", "4on5", "other"]:

            # Take all stats for a situation
            situation_stats = yearly_stats_df[yearly_stats_df["situation"] == situation]

            # Drop situation column
            situation_stats = situation_stats.drop(columns=["situation", "position"])

            # Merge with other situations stats
            stats_df = pd.merge(stats_df, situation_stats, how="left", on=["name", "season", "team", "playerId"], suffixes=(None, f"_{situation}"))

        # Add season stats to list
        stats_df_list.append(stats_df)
    
    # Concatenate all seasons
    stats_df = pd.concat(stats_df_list, ignore_index=True)

    # Merge salary info with stats
    dataset_df = pd.merge(salary_info_df, stats_df, how="inner", on=["name", "season", "team", "playerId"])

    # Export dataset
    dataset_df.to_csv("dataset.csv", index=False)


# Convert height to centimeters
def height_to_cm(height):
    feet, inches = height.split("' ")
    feet = int(feet)
    inches = int(inches.replace('"', ''))
    return round(feet * 30.48 + inches * 2.54)

if __name__ == '__main__':
    main()