from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save(".", "rubinberto")

tokenizer = ByteLevelBPETokenizer(
    "rubinberto-vocab.json",
    "rubinberto-merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(
    tokenizer.encode(
        "А можно вспоминать не о событиях, а, например, о чувствах, испытываемых нами за «отчетный период»."
    ).tokens
)
