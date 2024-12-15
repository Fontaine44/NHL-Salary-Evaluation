from bs4 import BeautifulSoup as bs4
import requests

SPOTRACT_URL = "https://www.spotrac.com/nhl"
TEAMS = {
    "anaheim-ducks": "ANA",
    "boston-bruins": "BOS",
    "buffalo-sabres": "BUF",
    "calgary-flames": "CGY",
    "carolina-hurricanes": "CAR",
    "chicago-blackhawks": "CHI",
    "colorado-avalanche": "COL",
    "columbus-blue-jackets": "CBJ",
    "dallas-stars": "DAL",
    "detroit-red-wings": "DET",
    "edmonton-oilers": "EDM",
    "florida-panthers": "FLA",
    "los-angeles-kings": "LAK",
    "minnesota-wild": "MIN",
    "montreal-canadiens": "MTL",
    "nashville-predators": "NSH",
    "new-jersey-devils": "NJD",
    "new-york-islanders": "NYI",
    "new-york-rangers": "NYR",
    "ottawa-senators": "OTT",
    "philadelphia-flyers": "PHI",
    "pittsburgh-penguins": "PIT",
    "san-jose-sharks": "SJS",
    "seattle-kraken": "SEA",
    "st-louis-blues": "STL",
    "tampa-bay-lightning": "TBL",
    "toronto-maple-leafs": "TOR",
    "utah-hockey-club": "UTA",
    "vancouver-canucks": "VAN",
    "vegas-golden-knights": "VGK",
    "washington-capitals": "WSH",
    "winnipeg-jets": "WPG"
}

NAMES_TO_CHANGE = {
    "Alexander Killorn": "Alex Killorn",
    "Nicholas Ritchie": "Nick Ritchie",
    "T.J. Brodie": "TJ Brodie",
    "T.J. Galiardi": "TJ Galiardi",
    "John Oduya": "Johnny Oduya",
    "Pierre-Alexandre Parenteau": "PA Parenteau",
    "Maxime Talbot": "Max Talbot",
    "Cameron Atkinson": "Cam Atkinson",
    "Dan Cleary": "Daniel Cleary",
    "Jacob Muzzin": "Jake Muzzin",
    "Mats Zuccarello-Aasen": "Mats Zuccarello",
    "Nicklas Grossmann": "Nick Grossmann",
    "Magnus Paajarvi-Svensson": "Magnus Paajarvi",
    "Brandon Crombeen": "B.J. Crombeen",
    "Philip Kessel": "Phil Kessel",
    "James Van Riemsdyk": "James van Riemsdyk",
    "Nikolai Kulemin": "Nikolay Kulemin",
    "Christopher Higgins": "Chris Higgins",
    "Tobias Enstrom": "Toby Enstrom",
    "Cameron Talbot": "Cam Talbot",
    "Alex Petrovic": "Alexander Petrovic",
    "Yevgeny Medvedev": "Evgeny Medvedev",
    "Thomas Wilson": "Tom Wilson",
    "Michael Condon": "Mike Condon",
    "Matthew Murray": "Matt Murray",
    "Phillip Grubauer": "Philipp Grubauer",
    "Matthew Grzelcyk": "Matt Grzelcyk",
    "Trevor Van Riemsdyk": "Trevor van Riemsdyk",
    "Matthew Benning": "Matt Benning",
    "Michael Matheson": "Mike Matheson",
    "Vincent Hinostroza": "Vinnie Hinostroza",
    "Joshua Morrissey": "Josh Morrissey",
    "Charles Mcavoy": "Charlie McAvoy",
    "J.T Compher": "J.T. Compher",
    "Zachary Werenski": "Zach Werenski",
    "Zach Sanford": "Zachary Sanford",
    "Mitchell Marner": "Mitch Marner",
    "Alex Kerfoot": "Alexander Kerfoot",
    "Alexander Debrincat": "Alex DeBrincat",
    "MacKenzie Blackwood": "Mackenzie Blackwood",
    "Alexander Georgiev": "Alexandar Georgiev",
    "Joshua Brown": "Josh Brown",
    "Samuel Blais": "Sammy Blais",
    "Dylan Demelo": "Dylan DeMelo",
    "Anthony DeAngelo": "Tony DeAngelo",
    "Jani Hakanpaa": "Jani Hakanp",
    "Michael Mcleod": "Michael McLeod",
    "Theodor Blueger": "Teddy Blueger",
    "Michael Anderson": "Mikey Anderson",
    "Jacob Middleton": "Jake Middleton",
    "Samuel Montembeault": "Sam Montembeault",
    "Daniel Vladar": "Dan Vladar",
    "Arvid Söderblom": "Arvid Soderblom",
    "Matthew Boldy": "Matt Boldy",
    "Alex Carrier": "Alexandre Carrier",
    "Kurtis Macdermid": "Kurtis MacDermid",
    "Alexis Lafrenière": "Alexis Lafreniere",
    "Tim Stützle": "Tim Stutzle",
    "Cameron York": "Cam York",
    "William Borgen": "Will Borgen",
    "Alexei Toropchenko": "Alexey Toropchenko",
    "Gabe Vilardi": "Gabriel Vilardi",
    'Thomas Novak': 'Tommy Novak',
    "Matthew Beniers": "Matty Beniers",
    "Janis Moser": "J.J. Moser",
    "Nicklas Grossman": "Nicklas Grossmann",
    "Zach Aston-Reese": "Zachary Aston-Reese",
    "Nicholas Paul": "Nick Paul",
}

NAMES_TO_DROP = [
    "Kodie Curran",
    "Yegor Zamula",
    "Daniil Tarasov",
    "",
]

SALARY_CAP = 83500000
START_SEASON = 2013
END_SEASON = 2023

def main():
    players = []

    # Get all players salaries from all teams from 2013 to 2023
    for year in range(START_SEASON, END_SEASON + 1):
        for team in TEAMS.keys():
            salaries = get_team_salary(team, year)
            players.extend(salaries)

    # Clean players (change names, drop names)
    players = [[NAMES_TO_CHANGE.get(player[0], player[0]), player[1], player[2], player[3], player[4], player[5]] for player in players if player[0] not in NAMES_TO_DROP]
    
    # Print to csv
    with open("salary.csv", "w", encoding="utf-8") as f:
        f.write("name,season,team,adjustedSalary,salary,capPercentage\n")
        for player in players:
            f.write(f"{player[0]},{player[1]},{player[2]},{player[3]},{player[4]},{player[5]}\n")

        f.seek(f.tell() - 1) # Remove the last newline

def get_team_salary(team, year):
    url = f"{SPOTRACT_URL}/{team}/overview/_/year/{year}"
    page = requests.get(url)
    soup = bs4(page.content, "html.parser")

    # Find the first table
    table = soup.find("table")

    # Check if table exists
    if not table:
        return []

    table_body = table.find("tbody")
    team = TEAMS[team]

    # Find all rows and add cap pct and name to a list
    players = []
    for row in table_body.find_all("tr"):
        name_td = row.find("td", class_="text-left sticky left-0")
        name_a = name_td.find("a")
        salary_td = row.find("td", class_="text-center contract contract-cap_total2 highlight")
        cap_pct_td = row.find("td", class_="text-center contract contract-cap_total_league_pct text-center-important")

        try:
            salary = int(salary_td.text.strip().replace('$', '').replace(',', ''))
        except ValueError:
            continue
        
        # Discard players with salary < 975k to not include rookies
        if salary > 975000:
            name = name_a.text.replace('\n', ' ').strip()
            cap_pct = round(float(cap_pct_td.text.strip().replace('%', '')) / 100, 4)
            adjusted_salary = round(cap_pct * SALARY_CAP)
            players.append([name, year, team, adjusted_salary, salary, cap_pct])

    return players

if __name__ == "__main__":
    main()
