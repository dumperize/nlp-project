import youtokentome as yttm
import pandas as pd
import torch
import click
import json

from src.models.summarunner.iterator import BatchIterator
from src.models.summarunner.model import SentenceTaggerRNN
from src.models.summarunner.train_model import train_model
from src.models.summarunner.utils import json_convert


@click.command()
@click.argument("bpe_file_path", type=click.Path(exists=True))
@click.argument("train_file_path", type=click.Path(exists=True))
@click.argument("val_file_path", type=click.Path(exists=True))
@click.argument("model_path")
def trin_summarunner(bpe_file_path,train_file_path,val_file_path, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_records = pd.read_excel(train_file_path)
    val_records = pd.read_excel(val_file_path)

    train_records['sentences'] = train_records['sentences'].apply(json_convert)
    val_records['sentences'] = val_records['sentences'].apply(json_convert)
    train_records['greedy_summary_sentences'] = train_records['greedy_summary_sentences'].apply(json_convert)
    val_records['greedy_summary_sentences'] = val_records['greedy_summary_sentences'].apply(json_convert)

    bpe_processor = yttm.BPE(bpe_file_path)
    vocabulary = bpe_processor.vocab()
    train_iterator = BatchIterator(train_records, vocabulary, 32, bpe_processor, device=device)
    val_iterator = BatchIterator(val_records, vocabulary, 32, bpe_processor, device=device)

    model = SentenceTaggerRNN(len(vocabulary))
    train_model(model, train_iterator, val_iterator, vocabulary=vocabulary,
                bpe_processor=bpe_processor, device_name=device)

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    trin_summarunner()