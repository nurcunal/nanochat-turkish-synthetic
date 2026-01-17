## Tokenizer training
timestamp: 2026-01-16 20:47:10

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- force: True
- train_time: 133.2790
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 64
- token_bytes_mean: 8.0283
- token_bytes_std: 3.6869


## Base model training
timestamp: 2026-01-17 01:16:13

- run: DSAI585-Project
- optim: adamw
- device_type: 
- depth: 10
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 20
- device_batch_size: 32
- total_batch_size: 524,288
- embedding_lr: 0.3000
- unembedding_lr: 0.0040
- weight_decay: 0.2000
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- adam_beta1: 0.8000
- adam_beta2: 0.9500
- warmup_ratio: 0.0000
- warmdown_ratio: 0.4000
- final_lr_frac: 0.0000
- resume_from_step: -1
- eval_every: 250
- eval_tokens: 10,485,760
- core_metric_every: -1
- core_metric_max_per_task: 500
- sample_every: 2000
- save_every: 1000
- model_tag: None
- Number of parameters: 133,038,100
- Number of FLOPs per token: 6.488064e+08
- Calculated number of iterations: 5075
- Number of training tokens: 2,660,761,600
- Tokens : Params ratio: 20.0000
- DDP world size: 1
- warmup_ratio: 0.0000
- warmdown_ratio: 0.4000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8396
- Final validation bpb: 0.8396
- CORE metric estimate: None
- MFU %: 36.11%
- Total training flops: 1.726319e+18
- Total training time: 256.16m
- Peak memory usage: 26394.42MiB


## Base model loss
timestamp: 2026-01-17 01:20:23

- model: base_model (step 5075)
- train bpb: 0.8359
- val bpb: 0.8384
- sample 0: <|bos|>Fransa'nın başkenti Paris anlattıBu anlattıBu anlattıBu anlattıBu anlattıBu anlattıBu anlattıBu anlattı
- sample 1: <|bos|>Altının kimyasal sembolü,
- sample 2: <|bos|>Dün Cuma idiyse yarın hangi gün olur?
.
- sample 3: <|bos|>Sıcağın zıttı,
- sample 4: <|bos|>Güneş sistemindeki gezegenler şunlardır: Güneş anlattıBu anlattıBu anlattıBu anlattıBu anlattıBu anlattıBu anlattıBu anlattı
- sample 5: <|bos|>En sevdiğim renk,
- sample 6: <|bos|>5*x + 3 = 13 ise x kaçtır?
.


## Midtraining
timestamp: 2026-01-17 01:28:43

- run: DSAI585-Project
- device_type: 
- dtype: bfloat16
- model_tag: None
- model_step: None
- num_iterations: -1
- max_seq_len: 2048
- device_batch_size: 32
- total_batch_size: 524,288
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 1.0000
- eval_every: 150
- eval_tokens: 10,485,760
- train_suite: None
- dry_run: False
- Number of iterations: 100
- DDP world size: 1
- Minimum validation bpb: 0.5787


## Chat evaluation mid
timestamp: 2026-01-17 01:29:37

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- eval_suite: None
- device_type: 
- XNLI-TR: 0.3335
- XCOPA-TR: 0.5000
- Belebele-TR: 0.2711
- ChatCORE metric: 0.0095


## Chat SFT
timestamp: 2026-01-17 01:33:04

- run: DSAI585-Project
- device_type: 
- dtype: bfloat16
- source: mid
- model_tag: None
- model_step: None
- num_epochs: 1
- num_iterations: -1
- device_batch_size: 4
- target_examples_per_step: 32
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- eval_suite: None
- train_suite: None
- Training rows: 6337
- Number of iterations: 198
- Training loss: 1.7082
- Validation loss: 2.1124


## Chat evaluation sft
timestamp: 2026-01-17 01:33:52

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- eval_suite: None
- device_type: 
- XNLI-TR: 0.3477
- XCOPA-TR: 0.5000
- Belebele-TR: 0.2689
- ChatCORE metric: 0.0156


## Summary

[bloat data missing]

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| ChatCORE        | -        | 0.0095   | 0.0156   | -        |

Total wall clock time: unknown
