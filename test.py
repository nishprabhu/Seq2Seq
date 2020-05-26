""" Example Usage """

import torch
import torch.nn as nn
from seq2seq import Decoder


class RNN(Decoder):
    """ RNN class """

    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=5, padding_idx=0)
        self.rnn = nn.LSTM(5, 5, 2)
        self.output_layer = nn.Linear(5, 10)

    def forward_step(self, encoder_output, decoder_input):
        """ A single time step """
        output = self.embedding(decoder_input)
        output, _ = self.rnn(output)
        output = self.output_layer(output)
        return output


def main():
    """ Main function """
    model = RNN()
    encoder_output = torch.randn(3, 4, 5)
    decoder_input = torch.ones(3, 6, dtype=torch.long)
    output = model(encoder_output, decoder_input, predict=False)
    print(output.shape)
    output = model(encoder_output, decoder_input, predict=True)
    print(output.shape)


if __name__ == "__main__":
    main()
