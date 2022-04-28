from multiprocessing import Process, Manager
from download_entity import download_entity


def download_entities(urls):
    csv_list_ = []
    with Manager() as manager:
        csv_list = manager.list()
        Processess = []

        concurreny_count = 100
        urls_ = [
            urls[
                (i * (len(urls) // concurreny_count)) : (
                    (i + 1) * (len(urls) // concurreny_count)
                )
            ]
            for i in range(concurreny_count)
        ]

        leftovers = urls[
            (concurreny_count * (len(urls) // concurreny_count)) : len(urls)
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

    return csv_list_
