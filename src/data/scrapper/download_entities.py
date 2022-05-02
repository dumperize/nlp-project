import click
import pandas as pd

from multiprocessing import Process, Manager
from src.data.scrapper.download_entity import download_entity


INPUT_FILE_PATH = "data/interim/scrapper/list_urls.txt"
OUTPUT_FILE_PATH = "data/interim/scrapper/TED_Talks.xlsx"


@click.command()
@click.argument("list_urls_file_path", default=INPUT_FILE_PATH, type=click.Path(exists=True))
@click.argument("output_file_path", default=OUTPUT_FILE_PATH)
def download_entities(list_urls_file_path=INPUT_FILE_PATH, output_file_path=OUTPUT_FILE_PATH):
    file = open(list_urls_file_path, 'r')
    urls = file.readlines()

    csv_list_ = []
    with Manager() as manager:
        csv_list = manager.list()
        Processess = []

        concurrency_count = 3
        urls_ = [
            urls[
                (i * (len(urls) // concurrency_count)): ((i + 1) * (len(urls) // concurrency_count))
            ]
            for i in range(concurrency_count)
        ]

        leftovers = urls[
            (concurrency_count * (len(urls) // concurrency_count)): len(urls)
        ]
        for i in range(len(leftovers)):
            urls_[i] += [leftovers[i]]

        for (id_, urls__) in enumerate(urls_):
            p = Process(target=download_entity, args=(urls__, id_, csv_list))
            Processess.append(p)
            p.start()

        # block until all the threads finish (i.e. block until all **download** function calls finish)
        for t in Processess:
            t.join()

        csv_list_ = list(csv_list)

    df = pd.DataFrame(csv_list_)
    df.to_excel(output_file_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    download_entities()
