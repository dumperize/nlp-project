import requests, bs4


def download_list():
    headers = {"User-agent": "your bot 0.1"}

    urls = []
    page_number = 0

    while 1:
        page_number += 1

        res = requests.get(
            f"https://www.ted.com/talks?sort=newest&page={str(page_number)}",
            headers,
        )

        soup = bs4.BeautifulSoup(res.text)
        divs = soup.select("div.container.results div.col")

        if len(divs) == 0:
            break

        for div in divs:
            urls.append(
                "https://www.ted.com"
                + div.select("div.media__image a.ga-link")[0].get("href")
            )

    return urls
