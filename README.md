# Which Side Are You On? A Multi-task Dataset for End-to-End Argument Summarisation and Evaluation

This work followed the paper publised on ACL 2024

## Ethic approved

This work has received ethical review from the institution under the license number 2024-16577-36845

Due to the suggestion, we will consider release task 1 and task 2's data in a more safety ways. It will be release very soon!

## Requirements
- python
- pytorch
- sentence-transformers

## Run Baselines
### Task1

### Task2
- Classification models
```
XXX
```
- Contrastive Learning models
```
# roberta-large
python train.py --model_name roberta-large \
  --train_dataset_file PATH/TO/train_convincingness.csv \
  --dev_dataset_file PATH/TO/dev_convincingness.csv \
  --test_dataset_file PATH/TO/test_convincingness.csv \
  --output_path PATH/TO/OUTPUT \
  --num_epochs  10 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --max_input_length 256 \
  --add_special_tokens "<SEP>" \
  --learning_rate 3e-5 \
  --task_name 'task2'

# sentence-t5-large or sentence-t5-xl
 python train.py --model_name sentence-transformers/sentence-t5-large \
  --train_dataset_file PATH/TO/train_convincingness.csv \
  --dev_dataset_file PATH/TO/dev_convincingness.csv \
  --test_dataset_file PATH/TO/test_convincingness.csv \
  --output_path PATH/TO/OUTPUT \
  --num_epochs  10 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --max_input_length 256 \
  --add_special_tokens "</s>" \
  --learning_rate 3e-5 \
  --sentence_transformer \
  --task_name 'task2'
```

### Task3
```
# roberta-large
python train.py --model_name roberta-large \
  --train_dataset_file PATH/TO/task3_trainset.csv \
  --dev_dataset_file PATH/TO/task3_devset.csv \
  --test_dataset_file PATH/TO/task3_testset.csv \
  --output_path PATH/TO/OUTPUT \
  --num_epochs  10 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --max_input_length 512 \
  --learning_rate 3e-5 \
  --task_name 'task3'

# sentence-t5-large or sentence-t5-xl
 python train.py --model_name sentence-transformers/sentence-t5-large \
  --train_dataset_file PATH/TO/task3_trainset.csv \
  --dev_dataset_file PATH/TO/task3_devset.csv \
  --test_dataset_file PATH/TO/task3_testset.csv \
  --output_path PATH/TO/OUTPUT \
  --num_epochs  10 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --max_input_length 512 \
  --learning_rate 3e-5 \
  --sentence_transformer \
  --task_name 'task3'
```

### Task4
