import nltk
import math
import torch
import numpy as np
from rouge import Rouge

from src.models.summarunner.extractive import build_summary_greedy


class BatchIterator():
    def __init__(
            self,
            records,
            vocabylary,
            batch_size,
            bpe_processor,
            shuffle=True,
            max_sentences=30,
            max_sentence_length=50,
            device=torch.device('cuda')):
        self.records = records
        self.n_samples = records.shape[0]
        self.batch_size = batch_size
        self.bpe_processor = bpe_processor
        self.shuffle = shuffle
        self.batches_count = int(math.ceil(self.n_samples / batch_size))
        self.rouge = Rouge()
        self.vocabylary = vocabylary
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.device = device

    def __len__(self):
        return self.batches_count

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start+self.batch_size, self.n_samples)
            batch_indices = indices[start:end]

            batch_inputs = []
            batch_outputs = []
            max_sentence_length = 0
            max_sentences = 0
            batch_records = []

            for data_ind in batch_indices:
                record = self.records.iloc[data_ind]
                batch_records.append(record)
                text, summary = record['text'], record['summary']

                # if 'sentences' not in record:
                #     sentences = nltk.sent_tokenize(text)
                # else:
                sentences = record['sentences']
                max_sentences = max(len(sentences), max_sentences)

                # if 'summary_greedy' not in record:
                #     sentences_indeces = build_summary_greedy(
                #     text, summary, calc_score=lambda x,y: self.rouge.get_scores([x], [y], avg=True)['rouge-2']['f'])
                # else:
                sentences_indeces = record['greedy_summary_sentences']
                
                inputs = [self.bpe_processor.encode(
                    sentence)[:self.max_sentence_length]for sentence in sentences]
                max_sentence_length = max(max_sentence_length, max(
                    [len(token) for token in inputs]))

                outputs = [int(i in sentences_indeces)
                           for i in range(len(sentences))]
                batch_inputs.append(inputs)
                batch_outputs.append(outputs)

            tensor_inputs = torch.zeros(
                (self.batch_size, max_sentences, max_sentence_length), dtype=torch.long, device=self.device)
            tensor_outputs = torch.zeros(
                (self.batch_size, max_sentences), dtype=torch.float32, device=self.device)

            for i, inputs in enumerate(batch_inputs):
                for j, sentence_tokens in enumerate(inputs):
                    tensor_inputs[i][j][:len(sentence_tokens)] = torch.LongTensor(
                        sentence_tokens)
            for i, outputs in enumerate(batch_outputs):
                tensor_outputs[i][:len(outputs)] = torch.LongTensor(outputs)

            yield {
                'inputs': tensor_inputs,
                'outputs': tensor_outputs,
                'records': batch_records,
            }
