import lxml
import requests
from bs4 import BeautifulSoup
import time

link = "https://www.cricbuzz.com/cricket-match/live-scores"

while True:
    try:
        source = requests.get(link).text
        page = BeautifulSoup(source, "lxml")

        page = page.find("div", class_="cb-col cb-col-100 cb-bg-white")
        matches = page.find_all("div", class_="cb-scr-wll-chvrn cb-lv-scrs-col")

        live_matches = []

        for match in matches:
            live_matches.append(match.text.strip())

        if live_matches:
            print("⏱️ Live Score Update:", live_matches[0])
        else:
            print("No live matches found.")

    except Exception as e:
        print("⚠️ Error:", e)

    time.sleep(1)
