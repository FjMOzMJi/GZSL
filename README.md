# Generalized Zero-Shot Intent Classification

## Data

### Datasets

All data can be downloaded from here:  https://www.hostize.com/v/O2cakOCXHR.

Place the downloaded data in the `./data` directory.

1. Atis (atis)

    Preprocessed dataset stored in downloaded  `data/atis` directory.

2. MultiWoZ (multiwoz)

    Preprocessed dataset stored in downloaded  `data/multiwoz `directory.

3. CLINC (clinc)

    Preprocessed dataset stored in downloaded  `data/clinc` directory.

4. Banking77 (bank)

    Preprocessed dataset stored in downloaded  `data/bank` directory.

### Data directory structure

The data for each dataset is stored in `all.csv` w/o splitting.

All predefined splits stored in dataset root directory.

All information about intents stored in `intent_info` folder. 

`intent_info/descriptions` contains intent descriptions and different types of patterns.

Intent and utterance similarity matrices for negative sampling stored in `intent_info/intent_similarity` and `uttr_similarity` directories respectively.


## Train and Evaluate

### For training and evaluation step1

```
python classification/train.py dataset={dataname} experiment.name=/path/to/experiment/name experiment.step=step1

python classification/evaluate.py dataset={dataname} experiment.name=/path/to/experiment/name experiment.step=step1
```


### For training and evaluation step2
```
python classification/train.py dataset={dataname} experiment.name=/path/to/experiment/name experiment.step=step2 checkpoint.save_model=step1/model/saved/path

python classification/evaluate.py dataset={dataname} experiment.name=/path/to/experiment/name experiment.step=step2 checkpoint.save_model=step1/model/saved/path
```

## Configs

### Reproducibility

**Specific setups**

The default hyper-parameters settings to reproduce experiments for a specific dataset are detailed in the corresponding documentation:

`./classfication/conf/dataset/{dataname}.yaml`

### Config directory structure
```
conf
|-- config.yaml
|   dataset
|   |-- atis.yaml
|   |-- bank.yaml
|   |-- clinc.yaml
|   |-- multiwoz.yaml
```
### Parameters
| Parameter                     |                Default               | Description                                                                                |
|------------------------------|:------------------------------------:|--------------------------------------------------------------------------------------------|
| **dataset**                  |  |                                                                                            |
| dataset.name                 |                                      | Dataset and it's config name                                                               |
| dataset.path                 |                                      | Relative path to split data or whole dataset                                               |
| dataset.intent_info_path     |                                      | Relative path to intent information data                                                   |
| dataset.description_type     |                                      | Type of intent description to use. Ex: `names`, `d1_pattern`                 |
| dataset.uttr_len             |                                      | Max length of utterance in tokens. Longer utterance would be truncated.                    |
| dataset.desc_len             |                                      | Max length of intent description in tokens. Longer utterance would be truncated.           |
| **model**                    |                                      |                                                                                            |
| model.base_model             |             roberta-base             | Contextualized encoder model name or path                                                  |
| model.dropout                |                  0.5                 | Linear classifier head dropout                                                             |
| model.embedding_dim          |                  768                 | Contextualized encoder embedding size                                                      |
| **experiment**           |                                      |                                                                                            |
| experiment.step          |                  step1                  | training step for the model                                                               |
| experiment.root_dir          |                  ./                  | Root path for experiments                                                                  |
| experiment.name              |                  ???                 | Experiment name - needs to specify                                                         |
| experiment.seed              |                   0                  | Random seed                                                                                |
| experiment.epochs            |        <specified for dataset>       | Epochs to train                                                                            |
| experiment.batch_size        |        <specified for dataset>       | Batch size                                                                                 |
| experiment.accum_steps       |        <specified for dataset>       | Number of gradient accumulation steps                                                      |
| experiment.k_negative        |                   7                  | Number of examples for negative sampling                                                   |
| experiment.train_only_seen   |                 True                 | Train only with seen intent descriptions or not.                                           |
| experiment.intent_desc_first |                 false                 | Is intent description above utterance in sentence  pair encoding                          |
| experiment.test_epoch        |                 None                 | Specify epoch for evaluation. Default: best loss epoch                                     |
| experiment.temperature | 0.5 | Temperature for feature space contrastive learning           |
| experiment.mlm_percent | 0.2 | The proportion of tokens masked in the sentence |
| experiment.mlm_param | 1 | the trade-off hyperparameter 𝜇 |
| experiment.prompt_len | 4 | prompt length for second step |
| **scheduler**                |                                      |                                                                                            |
| scheduler.lr                 |                 2e-5                 | Learning rate                                                                              |
| scheduler.warmup_steps       |                 0.15                 | Scheduler warmup iterations                                                                |
| **checkpoint**         |                                      |                                                                                            |
| checkpoint.save_from_epoch   |                 None                 | Specified epoch to save checkpoint from. Default: save only best loss checkpoint           |
| checkpoint.saved_model       |                 None                 | Epoch to load model checkpoint from. Default: load from best loss checkpoint.              |
| log.print_every              |                 1000                 | Number of iterations to log loss.                                                          |





