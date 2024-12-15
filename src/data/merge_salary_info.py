import pandas as pd

def main():

    # Open salaries
    salaries_df = pd.read_csv("salary.csv", encoding="utf-8")

    # Open players info
    players_df = load_player_lookup()

    # Merge the two dataframes
    merged_df = pd.merge(salaries_df, players_df, how="left", on="name")

    print(merged_df.head())
    print(merged_df.shape)

    # Save the merged dataframe to a csv
    merged_df.to_csv("salary_info_merged.csv", index=False)
    

def load_player_lookup():
    # Open players info
    players_df = pd.read_csv("allPlayersLookup.csv")

    # Drop unnecessary columns
    players_df.drop(columns=["team", "primaryPosition", "primaryNumber"], inplace=True)

    return players_df


if __name__ == '__main__':
    main()