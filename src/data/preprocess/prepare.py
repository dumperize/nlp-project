import click
import pandas as pd


INPUT_FILE = 'data/interim/scrapper/TED_Talk.xlsx'
OUTPUT_FILE = 'data/processed/full_dataset.xlsx'

@click.command()
@click.argument("input_file_path", default=INPUT_FILE, type=click.Path(exists=True))
@click.argument("output_file_path", default=OUTPUT_FILE)
def clean_data(input_file_path = INPUT_FILE, output_file_path=OUTPUT_FILE):
    stop_list = [43148,  9985,   196,  1156, 42819,  1323,  2028, 42461, 42548,
        23943, 37985, 26265,  2273, 42546,  1677,  2147, 39095, 15814,
            2611,   117,  1464,   115, 82299,   729,   109,   179, 70428,
            364,   995,    99, 42464,    81,   988,  2684,  2366]

    df = pd.read_excel(input_file_path)

    # удалим все пустые
    df = df[df['transcript'].notna()]
    df = df[df['talk__description'].notna()]

    # удалим все короткие
    df = df[df['transcript'].apply(lambda x: len(x.strip()) > 10)]
    df = df[df['talk__description'].apply(lambda x: len(x.strip()) > 10)]

    # переименуем столбцы
    df = df.rename(columns={'transcript':'text',"talk__description": "summary", 'talk__name': "name", "talk__id": "id"})

    # возьмем только нужные столбцы
    df = df[['id', "name", "summary", 'text']]

    # уберем строки из стоплиста
    df = df.drop(df[df['id'].isin(stop_list)].index, axis=0)

    # приведем все в нижний регистр
    df['summary'] = df['summary'].apply(lambda x:x.lower())
    df['text'] = df['text'].apply(lambda x:x.lower())

    print("Size full dataset: ", df.shape[0])
    df.to_excel(output_file_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    clean_data()