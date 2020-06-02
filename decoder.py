""" Decoder wrapper written in PyTorch 1.5.0 """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ Decoder base class """

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, encoder_output, decoder_input, **kwargs):
        """
        A single forward step

        Arguments:
        ---------
        encoder_output: Tensor of shape (batch_size, e, embedding_size)
        decoder_input: Tensor of shape (batch_size, d)
        [e and d are encoder and decoder sequence lengths respectively.]

        Output:
        ------
        outputs: Tensor of shape (batch_size, d, vocabulary_size)
        """
        raise NotImplementedError

    def generate(self, encoder_output, decoder_input, max_target_length=30, **kwargs):
        """
        Forward pass

        Arguments:
        ---------
        encoder_output: Tensor of shape (batch_size, e, embedding_size)
        decoder_input: Tensor of shape (batch_size, d)
                       Provide a vector of shape (batch_size, 1) consisting of <START> tokens for prediction.
        max_target_length: Integer indicating the number of time-steps to be run during inference.
        [e and d are encoder and decoder sequence lengths respectively.]
        
        Output:
        ------
        outputs: Tensor of shape (batch_size, d+max_target_length-1, vocabulary_size)
                 max_target_length new tokens are generated.
                 outputs sequence length has -1 because the <START> tokens are dropped from the output.
        """
        current_input = decoder_input
        for _ in range(max_target_length):
            current_output = self.forward(encoder_output, current_input, **kwargs)
            indices = torch.argmax(F.softmax(current_output, dim=-1), dim=-1)[:, -1:]
            current_input = torch.cat([current_input, indices], dim=-1)
        return current_output
