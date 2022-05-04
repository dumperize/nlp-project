import click
import pandas as pd
import youtokentome as yttm


DEFAULT_DATA_PATH = 'dataset/data.xlsx'
DEFAULT_MODEL_PATH = "dataset/model_BPE.bin"
DEFAULT_VOCAB_SIZE = 10000


@click.command()
@click.argument('data_path', default=DEFAULT_DATA_PATH, type=click.Path(exists=True))
@click.argument('model_path', default=DEFAULT_MODEL_PATH)
@click.argument('vocab_size', default=DEFAULT_VOCAB_SIZE)
def train_bpe(data_path=DEFAULT_DATA_PATH, model_path=DEFAULT_MODEL_PATH, vocab_size=DEFAULT_VOCAB_SIZE):
    records = pd.read_excel(data_path)

    temp_file_name = 'temp.txt'
    with open(temp_file_name, "w") as temp:
        for i, record in records.iterrows():
            text, summary = record['text'].lower(), record ['summary'].lower()
            temp.write(text+'\n')
            temp.write(summary+'\n')
    yttm.BPE.train(data=temp_file_name, vocab_size=vocab_size, model=model_path)
    test()

def test():
    print('-------------')
    print('Test:')
    bpe_proc = yttm.BPE(DEFAULT_MODEL_PATH)
    result = bpe_proc.encode(['Corner stone seq2seq with attention (using bidirectional ltsm )'], output_type=yttm.OutputType.SUBWORD)
    print(result)

if __name__ == '__main__':
    train_bpe()
    