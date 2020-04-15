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
--model_type gpt2 --tokenizer_class YTEncoder --tokenizer_name models\rubin_yttm\rubin_yttm.model --do_train --learning_rate 1e-4 --num_train_epochs 5 --save_total_limit 2 --save_steps 2000 --per_gpu_train_batch_size 1 --evaluate_during_training --seed 42 --logging_steps 10 --train_data_file eo_data\rubin.txt --eval_data_file eo_data\01-eval.txt --output_dir models\Output
```

Results in generation stage are not satisfactory (after 250 epochs andMovingLoss=0.21, Perplexity=1.24):
```
python run_generation.py --model_type gpt2 --model_name_or_path models\Output\checkpoint-2000 --length 30 
```

Examples:

- и новогочи пологической словом понима этот приобретаетется захватить бы понымстра Тексты распро З?
- хорошо наверного приялотка неде уста чем-то совершенноства.
- хтом, подростго сказать уж значдемительный глупо было,е,вид конечно, была имрени случаях,браят рмеством Разве Медведе ставить своб

Questions: why perplexity on eval stage is high?
```
04/15/2020 22:29:24 - INFO - __main__ -   ***** Eval results  *****
04/15/2020 22:29:24 - INFO - __main__ -     perplexity = tensor(2800.4243)
```