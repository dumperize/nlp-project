import click
import pandas as pd


@click.command()
@click.argument("input_file_path", type=click.Path(exists=True))
@click.argument("train_files_path")
@click.argument("test_files_path")
@click.argument("val_files_path")
def split_dataset(input_file_path, train_files_path, test_files_path, val_files_path):
    df = pd.read_excel(input_file_path)

    train_dataset = df.sample(frac=0.8)
    rest_part = df.drop(train_dataset.index)

    test_dataset = rest_part.sample(frac=0.5)
    val_dataset = rest_part.drop(test_dataset.index)

    print('Train dataset: ', train_dataset.shape[0])
    print('Test dataset: ', test_dataset.shape[0])
    print('Val dataset: ', val_dataset.shape[0])

    train_dataset.to_excel(train_files_path, encoding='utf-8', index=False)
    test_dataset.to_excel(test_files_path, encoding='utf-8', index=False)
    val_dataset.to_excel(val_files_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    split_dataset()
