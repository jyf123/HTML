## Preparation

Please refer to https://github.com/snap-stanford/pretrain-gnns#dataset-download to download chem dataset.


## Training & Evaluation
### Pre-training: ###
```
./pretrain.sh $GPU_ID $a $b
```

### Finetuning: ###
```
./finetune.sh $GPU_ID $DATASET_NAME $input_model_file
```


