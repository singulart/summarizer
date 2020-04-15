### Using YTEncoder to train the language model from scratch

```yt_encoder.py``` was copied from 
https://github.com/singulart/ru_transformers with minimal changes

```run_language_modeling.py``` was brought from official repo https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py amd modified to enable training from scratch.


Create a YTTM vocabulary (same as what youtoken2me.py does):
```
yttm bpe --data eo_data\rubin.txt --model models\rubin_yttm\rubin_yttm.model --vocab_size 4000 
```

Command line to start training from scratch which worked for me (uses vocabulary from previous step)

``` 
python run_language_modeling.py --model_type gpt2 --tokenizer_class YTEncoder --tokenizer_name models\rubin_yttm\rubin_yttm.model --do_train --learning_rate 1e-4 --num_train_epochs 5 --save_total_limit 2 --save_steps 2000 --per_gpu_train_batch_size 1 --evaluate_during_training --seed 42 --train_data_file eo_data\rubin.txt --output_dir models\Output
```
