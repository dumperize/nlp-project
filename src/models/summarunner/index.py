import youtokentome as yttm
import pandas as pd
import torch
import click
import json

from src.models.summarunner.iterator import BatchIterator
from src.models.summarunner.model import SentenceTaggerRNN
from src.models.summarunner.train_model import train_model


@click.command()
@click.argument("bpe_file_path", default="data/interim/helper/bpe.bin", type=click.Path(exists=True))
@click.argument("train_file_path", default="data/interim/helper/train_dataset_with_info.xlsx", type=click.Path(exists=True))
@click.argument("val_file_path", default="data/interim/helper/val_dataset_with_info.xlsx", type=click.Path(exists=True))
def calc_summarunner_scores(
        bpe_file_path,
        train_file_path,
        val_file_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_records = pd.read_excel(train_file_path)
    val_records = pd.read_excel(val_file_path)

    train_records['sentences'].apply(lambda x: json.loads(x))
    val_records['sentences'].apply(lambda x: json.loads(x))
    train_records['greedy_summary_sentences'].apply(lambda x: json.loads(x))
    val_records['greedy_summary_sentences'].apply(lambda x: json.loads(x))

    bpe_processor = yttm.BPE(bpe_file_path)
    vocabulary = bpe_processor.vocab()
    train_iterator = BatchIterator(train_records, vocabulary, 32, bpe_processor, device=device)
    val_iterator = BatchIterator(val_records, vocabulary, 32, bpe_processor, device=device)

    print(train_iterator)

    model = SentenceTaggerRNN(len(vocabulary))
    train_model(model, train_iterator, val_iterator, vocabulary=vocabulary,
                bpe_processor=bpe_processor, device_name=device)


if __name__ == "__main__":
    calc_summarunner_scores()