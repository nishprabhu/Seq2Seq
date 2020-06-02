# Decoder
## A PyTorch wrapper for seq2seq decoders.

This wrapper implements a decoder module that runs teacher forcing during training (forward method) and iterative generation during prediction (generate method). Subclass this decoder module to avoid re-writing the iterative generation code every time. 
