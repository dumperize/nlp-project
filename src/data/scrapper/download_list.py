from email.policy import default
import bs4
import click
import os

from src.data.scrapper.utils import repeated_request


FILE_PATH_SCRAPPER = 'data/interim/scrapper/list_urls.txt'

@click.command()
@click.argument("output_file_path", default=FILE_PATH_SCRAPPER)
def download_list(output_file_path=FILE_PATH_SCRAPPER):
    urls = []
    page_number = 0

    while 1:
        page_number += 1

        res = repeated_request(f"https://www.ted.com/talks?sort=newest&page={str(page_number)}", 5)

        soup = bs4.BeautifulSoup(res.text)
        divs = soup.select("div.container.results div.col")

        if len(divs) == 0:
            break

        for div in divs:
            urls.append(
                "https://www.ted.com"
                + div.select("div.media__image a.ga-link")[0].get("href")
            )

    os.makedirs(os.path.dirname(FILE_PATH_SCRAPPER), exist_ok=True)
    f = open(output_file_path, 'w')
    f.write('\n'.join(urls))
    f.close()

if __name__ == "__main__":
    download_list()