import youtokentome as yttm

train_data_path = "eo_data/1.txt"
model_path = 'models/rubin_yttm/rubin_yttm.model'

# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)

# Loading model
bpe = yttm.BPE(model=model_path)
print(
    bpe.encode(["Сейчас об этой дате непростительно и стыдливо подзабыли. "],
               output_type=yttm.OutputType.SUBWORD)
)

