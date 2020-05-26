# Seq2Seq
## A PyTorch wrapper for seq2seq models.

This wrapper implements a decoder module that runs teacher forcing during training and iterative generation during prediction. Subclass this decoder module to avoid re-writing the iterative generation code every time. 
