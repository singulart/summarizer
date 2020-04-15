import torch
from torch.utils.data import DataLoader, Dataset
import youtoken2me as yttm
from pathlib import Path

model_path = 'rubin_yttm.model'


class RubinDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = yttm.BPE(model_path)
        # tokenizer._tokenizer.post_processor = BertProcessing(
        #     ("</s>", tokenizer.token_to_id("</s>")),
        #     ("<s>", tokenizer.token_to_id("<s>")),
        # )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("./eo_data/").glob("*-eval.txt") if evaluate else Path("./eo_data/").glob("1.txt")
        for src_file in src_files:
            print("🔥", src_file)
        lines = src_file.read_text(encoding="utf-8").splitlines()
        self.examples += [x.ids for x in tokenizer.encode_batch(lines)]


def __len__(self):
    return len(self.examples)


def __getitem__(self, i):
    # We’ll pad at the batch level.
    return torch.tensor(self.examples[i])
