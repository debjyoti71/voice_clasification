import requests
from bs4 import BeautifulSoup

# Predefined Player ID Map
player_id_map = {
    "sachin tendulkar": 25,
    "virat kohli": 1413,
    "ms dhoni": 265,
    "hardik pandya": 9647
}

def get_player_stats_by_id(player_name):
    player_name = player_name.lower().strip()
    if player_name not in player_id_map:
        return {"error": f"No ID found for player '{player_name}'"}

    url = f"https://www.cricbuzz.com/profiles/{player_id_map[player_name]}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        page = requests.get(url, headers=headers).text
        soup = BeautifulSoup(page, "lxml")

        # Player basic section
        profile = soup.find("div", id="playerProfile")
        info = profile.find("div", class_="cb-col cb-col-100 cb-bg-white")

        name = info.find("h1", class_="cb-font-40").text.strip()
        country = info.find("h3", class_="cb-font-18 text-gray").text.strip()
        image_url = info.find("img")["src"]

        personal = soup.find_all("div", class_="cb-col cb-col-60 cb-lst-itm-sm")
        role = personal[2].text.strip() if len(personal) > 2 else "N/A"

        # ICC Rankings
        icc = soup.find_all("div", class_="cb-col cb-col-25 cb-plyr-rank text-right")
        batting_ranks = [icc[i].text.strip() if i < len(icc) else "-" for i in range(3)]
        bowling_ranks = [icc[i].text.strip() if i < len(icc) else "-" for i in range(3, 6)]

        # Stats
        summary = soup.find_all("div", class_="cb-plyr-tbl")
        batting_stats, bowling_stats = {}, {}

        if len(summary) > 0:
            for row in summary[0].find("tbody").find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 13:
                    fmt = cols[0].text.strip().lower()
                    batting_stats[fmt] = {
                        "matches": cols[1].text.strip(),
                        "innings": cols[2].text.strip(),
                        "runs": cols[3].text.strip(),
                        "highest": cols[5].text.strip(),
                        "average": cols[6].text.strip(),
                        "strike_rate": cols[7].text.strip(),
                        "50s": cols[11].text.strip(),
                        "100s": cols[12].text.strip(),
                    }

        if len(summary) > 1:
            for row in summary[1].find("tbody").find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 12:
                    fmt = cols[0].text.strip().lower()
                    bowling_stats[fmt] = {
                        "matches": cols[1].text.strip(),
                        "innings": cols[2].text.strip(),
                        "balls": cols[3].text.strip(),
                        "runs": cols[4].text.strip(),
                        "wickets": cols[5].text.strip(),
                        "best_innings": cols[9].text.strip(),
                        "economy": cols[7].text.strip(),
                        "5w": cols[11].text.strip()
                    }

        return {
            "name": name,
            "country": country,
            "image": image_url,
            "role": role,
            "profile_url": url,
            "rankings": {
                "batting": {"test": batting_ranks[0], "odi": batting_ranks[1], "t20": batting_ranks[2]},
                "bowling": {"test": bowling_ranks[0], "odi": bowling_ranks[1], "t20": bowling_ranks[2]}
            },
            "batting_stats": batting_stats,
            "bowling_stats": bowling_stats
        }

    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    player = input("Enter player name: ")
    stats = get_player_stats_by_id(player)

    print("\n📊 Player Stats:")
    for key, val in stats.items():
        print(f"{key}: {val}")
