import torch.nn as nn
import torch


class SentenceEncoderRNN(nn.Module):
    def __init__(self, input_size, embeding_dim, hidden_size, n_layers=3, dropout=0.3, bidirectional=True):
        super().__init__()

        num_directional = 2 if bidirectional else 1
        assert hidden_size % num_directional == 0
        hidden_size = hidden_size//num_directional

        self.embeding_dim = embeding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embeding_layer = nn.Embedding(input_size, embeding_dim)
        self.rnn_layers = nn.LSTM(embeding_dim, hidden_size, n_layers,
                                  dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs, hidden=None):
        embedded = self.embeding_layer(inputs)
        outputs, _ = self.rnn_layers(embedded, hidden)
        sentences_embeddings = torch.mean(outputs, 1)
        return sentences_embeddings


class SentenceTaggerRNN(nn.Module):
    def __init__(self, vocabulary_size, token_embedding_dim=256, semtence_encoder_hidden_size=256,
                 hidden_size=256, bidirectional=True, sentence_encoder_n_layer=2,
                 sentence_encoder_dropout=0.3, sentence_encoder_biderectional=True,
                 n_layers=1, dropout=0.3) -> None:
        super().__init__()

        num_directional = 2 if bidirectional else 1
        assert hidden_size % num_directional == 0
        hidden_size = hidden_size//num_directional

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.sentences_encoder = SentenceEncoderRNN(vocabulary_size, token_embedding_dim,
                                                    semtence_encoder_hidden_size, sentence_encoder_n_layer,
                                                    sentence_encoder_dropout, sentence_encoder_biderectional)

        self.rnn_layer = nn.LSTM(semtence_encoder_hidden_size, hidden_size,
                                 n_layers, dropout=dropout,
                                 bidirectional=bidirectional, batch_first=True)

        self.dropout_layer = nn.Dropout(dropout)
        self.content_linear_layer = nn.Linear(hidden_size * 2, 1)
        self.document_linear_layer = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.salience_linear_layer = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.tanh_layer = nn.Tanh()

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size(0)
        sentencrs_count = inputs.size(1)
        tokens_count = inputs.size(2)
        inputs = inputs.reshape(-1, tokens_count)

        emedded_sentences = self.sentences_encoder(inputs)
        emedded_sentences = emedded_sentences.reshape(batch_size, sentencrs_count, -1)

        outputs, _ = self.rnn_layer(emedded_sentences, hidden)
        outputs = self.dropout_layer(outputs)

        document_emedding = self.tanh_layer(self.document_linear_layer(torch.mean(outputs, 1)))

        content = self.content_linear_layer(outputs).squeeze(2)

        salience = torch.bmm(outputs,self.salience_linear_layer(document_emedding).unsqueeze(2)).squeeze(2)

        return content + salience
