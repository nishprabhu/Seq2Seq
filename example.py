""" Example Usage """

import torch
import torch.nn as nn
from decoder import Decoder


class RNN(Decoder):
    """ RNN class """

    def __init__(self, embedding_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=5, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(embedding_dim, output_dim)

    def forward(self, encoder_output, decoder_input):
        """ A single time step. Define the function as you would for training with teacher forcing. """
        output = self.embedding(decoder_input)
        output, _ = self.rnn(output)
        output = self.output_layer(output)
        return output


def main():
    """ Main function """

    # Hyperparamters
    embedding_dim = 5
    num_layers = 2
    output_dim = 10
    batch_size = 3
    encoder_seq_length = 4
    decoder_seq_length = 6

    # Model Definition
    model = RNN(embedding_dim, num_layers, output_dim)

    # Inputs
    encoder_output = torch.randn(batch_size, encoder_seq_length, embedding_dim)
    decoder_input = torch.ones(batch_size, decoder_seq_length, dtype=torch.long)

    # Forward and Generate
    forward_output = model(encoder_output, decoder_input)
    generate_output = model.generate(encoder_output, decoder_input)
    print("Forward: {}".format(forward_output.shape))
    print("Generate: {}".format(generate_output.shape))


if __name__ == "__main__":
    main()
