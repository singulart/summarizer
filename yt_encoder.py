"""Byte pair encoding utilities"""
import os
import youtokentome as yttm
import hashlib
from transformers.tokenization_utils import PreTrainedTokenizer
import shutil
import regex as re
from os.path import samefile

NEW_LINE = '<|n|>'


class YTEncoder(PreTrainedTokenizer):
    def_name = 'encoder.model'

    def __init__(self, filename, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # no default special tokens - you can update this value if you add special tokens
        self.max_len_single_sentence = 1024
        # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = 1024

        if os.path.isdir(filename): filename = os.path.join(filename, self.def_name)

        self.bpe = yttm.BPE(filename)
        self.hash = hashlib.sha512(open(filename, 'rb').read()).hexdigest()[:10]
        self.filename = filename

    def encode(self, text, **kwargs):
        if text and text[0] != ' ':
            text = ' ' + text
        text = re.sub(r'(?=[^ ])([\W])([\w])', r'\g<1> \g<2>', text)
        text = text.replace('\n', f' {NEW_LINE} ')

        return self.bpe.encode([text], output_type=yttm.OutputType.ID)[0]

    def decode(self, tokens, **kwargs):  # I hate regexps
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        result = self.bpe.decode(tokens)[0]
        result = re.sub(r'( )?(<\|n\|>)( )?', r'\n', result)
        result = re.sub(r'([\n(]) (\w)', r'\g<1>\g<2>', result)
        result = re.sub(r'(\W)([«"''\n(]|^) (\w)', r'\g<1>\g<2>\g<3>', result)
        result = re.sub(r'(\w)- (\w)', r'\g<1>-\g<2>', result)
        return result

    def tokenize(self, text, **kwargs):
        return self.encode(text)

    def _tokenize(self, text, **kwargs):
        return self.encode(text, kwargs)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls(*inputs, **kwargs)

    @staticmethod
    def add_special_tokens_single_sentence(token_ids):
        return token_ids

    def save_pretrained(self, save_directory):
        src = self.filename
        dst = os.path.join(save_directory, self.def_name)
        if src != dst:
            shutil.copyfile(src, dst)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return token

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return index

    @property
    def vocab_size(self):
        return self.bpe.bpe_cython.vocab_size()


