"""
This script trains sentence transformers with a triplet loss function.

Codes are adapted from https://github.com/marcelbra/argmining-21-keypoint-analysis-sharedtask-code-2
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime

import csv
import logging
import os
import sys
import argparse
from torch import nn

from custom_evaluator import CustomTripletEvaluator, CustomRankingEvaluator
from custom_loss import CustomRankingLoss

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, default='distilbert-base-uncased', help="Name of the pre-trained model.")
    parser.add_argument("--train_dataset_file", type=str, required=True, default=None, help="Path to the train dataset file.")
    parser.add_argument("--dev_dataset_file", type=str, default=None, help="Path to the dev dataset file.")
    parser.add_argument("--test_dataset_file", type=str, default=None, help="Path to the test dataset file.")
    parser.add_argument("--output_path", type=str, default='output', help="Path to the output directory.")

    # training-related parameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--evaluation_steps", type=int, default=500, help="Number of steps to evaluate the model.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--max_input_length", type=int, default=256, help="Maximum input sequence length.")
    parser.add_argument("--add_special_token", type=str, default=None, help="The special token to add, e.g., <SEP>, between input triplets.")
    # parser.add_argument("--loss", type=str, default='Triplet', help="Loss function to use for training. Choose from Triplet, Contrastive, or OnlineContrastive.")
    parser.add_argument("--sentence_transformer", action="store_true", help="Whether to use model supported by sentence-transformers instead of transformers.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--distance_metric", type=str, default='euclidean', help="Distance metric to use for training. Choose from cosine, euclidean, or manhattan.")
    parser.add_argument("--triplet_margin", type=float, default=5, help="Margin for triplet loss.")
    parser.add_argument("--task_name", type=str, default='task2', help="Task name. Choose from task2 or task3.")

    args = parser.parse_args()
    assert args.train_dataset_file, "Please specify a path to the train dataset file."
    logger.info('Arguments: {}'.format(args))

    output_path = args.output_path + "-" + args.model_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.sentence_transformer:
        model = SentenceTransformer(args.model_name)
        model.max_seq_length = args.max_input_length
        # commented out because by specifying the token to add as argument, we can align it with what's in the tokenizer
        # if args.add_special_token:
        #     model.tokenizer.add_tokens([args.add_special_token], special_tokens=True)
        #     model.resize_token_embeddings(len(model.tokenizer))
    else:
        word_embedding_model = models.Transformer(args.model_name)
        word_embedding_model.max_seq_length = args.max_input_length
        if args.add_special_token:
            word_embedding_model.tokenizer.add_tokens([args.add_special_token], special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)
        
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    logger.info("Done with model preparation.")

    train_examples = []
    if args.task_name == 'task2':
        logger.info("Read Triplet Evidences train dataset.")
        with open(args.train_dataset_file, encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                if row['Label']:
                    l = float(row['Label'])
                    if l == 0.5:
                        # skip if label is 0.5
                        continue
                    elif int(l) == 1:
                        if args.add_special_token:
                            train_examples.append(InputExample(texts=[row['Topic'] + ' ' + args.add_special_token + ' ' + row['Argument'], row['Evidence_1'], row['Evidence_2']], label=0))
                        else:
                            train_examples.append(InputExample(texts=[row['Topic'] + ' ' + row['Argument'], row['Evidence_1'], row['Evidence_2']], label=0))
                    elif int(l) == 0:
                        if args.add_special_token:
                            train_examples.append(InputExample(texts=[row['Topic'] + ' ' + args.add_special_token + ' ' + row['Argument'], row['Evidence_2'], row['Evidence_1']], label=0))
                        else:
                            train_examples.append(InputExample(texts=[row['Topic'] + ' ' + row['Argument'], row['Evidence_2'], row['Evidence_1']], label=0))
                    else:
                        raise ValueError('Label must be one of [None, 0.5, 0, 1].')
                else:
                    # skip if label is None
                    continue
    elif args.task_name == 'task3':
        logger.info("Read Ranking Summaries train dataset.")
        with open(args.train_dataset_file, encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                train_examples.append(InputExample(texts=[row['Input'], row['Summary_1'], row['Summary_2'], row['Summary_3']], label=1))
    else:
        raise ValueError('Task name must be one of [task2, task3].')
    
    # print('Train examples evidence 1: ', [e.texts[1] for e in train_examples])
    logger.info('Number of train examples: {}'.format(len(train_examples)))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.train_batch_size)
    if args.distance_metric == 'cosine':
        distance_metric = losses.TripletDistanceMetric.COSINE
    elif args.distance_metric == 'euclidean':
        distance_metric = losses.TripletDistanceMetric.EUCLIDEAN
    elif args.distance_metric == 'manhattan':
        distance_metric = losses.TripletDistanceMetric.MANHATTAN
    else:
        raise ValueError('Distance metric must be one of [cosine, euclidean, manhattan].')
    
    if args.task_name == 'task2':
        train_loss = losses.TripletLoss(model, distance_metric=distance_metric, triplet_margin=args.triplet_margin)
    elif args.task_name == 'task3':
        train_loss = CustomRankingLoss(model, distance_metric=distance_metric, margin=args.triplet_margin)

    if args.dev_dataset_file:
        if args.task_name == 'task2':
            logger.info("Read Triplet Evidences dev dataset.")
            dev_anchors, dev_poss, dev_negs, dev_labels = [], [], [], []
            with open(args.dev_dataset_file, encoding='utf8') as devF:
                dev_reader = csv.DictReader(devF, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in dev_reader:
                    if row['Label']:
                        l = float(row['Label'])
                        if l == 0.5:
                            # skip if label is 0.5
                            continue
                        if args.add_special_token:
                            dev_anchors.append(row['Topic'] + ' ' + args.add_special_token + ' ' + row['Argument'])
                        else:
                            dev_anchors.append(row['Topic'] + ' ' + row['Argument'])
                        dev_poss.append(row['Evidence_1'])
                        dev_negs.append(row['Evidence_2'])
                        dev_labels.append(int(l))
                    else:
                        # skip if label is None
                        continue
            evaluator = CustomTripletEvaluator(
                dev_anchors, dev_poss, dev_negs, dev_labels,
                name='dev', 
                batch_size=args.eval_batch_size, 
                show_progress_bar=True,
                write_csv=True,
            )
        elif args.task_name == 'task3':
            logger.info("Read Ranking Summaries dev dataset.")
            dev_anchors, dev_exp1, dev_exp2, dev_exp3, dev_labels = [], [], [], [], []
            with open(args.dev_dataset_file, encoding='utf8') as devF:
                dev_reader = csv.DictReader(devF, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in dev_reader:
                    dev_anchors.append(row['Input'])
                    dev_exp1.append(row['Summary_1'])
                    dev_exp2.append(row['Summary_2'])
                    dev_exp3.append(row['Summary_3'])
                    dev_labels.append(1)
            evaluator = CustomRankingEvaluator(
                dev_anchors, dev_exp1, dev_exp2, dev_exp3, dev_labels,
                name='dev',
                batch_size=args.eval_batch_size,
                show_progress_bar=True,
                write_csv=True,
            )
        else:
            raise ValueError('Task name must be one of [task2, task3].')
        
        logger.info('Number of dev examples: {}'.format(len(dev_labels)))
    else:
        evaluator = None

    warmup_steps = int(len(train_dataloader) * args.num_epochs * 0.1) # 10% of train data for warm-up

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=args.evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_path,
        optimizer_params={'lr': args.learning_rate},
    )

    # Load the stored model and evaluate its performance on the test set
    if args.test_dataset_file:
        if args.task_name == 'task2':
            logger.info("Read Triplet Evidences test dataset.")
            test_anchors, test_poss, test_negs, test_labels = [], [], [], []
            with open(args.test_dataset_file, encoding='utf8') as testF:
                test_reader = csv.DictReader(testF, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in test_reader:
                    if row['Label']:
                        l = float(row['Label'])
                        if l == 0.5:
                            # skip if label is 0.5
                            continue
                        if args.add_special_token:
                            test_anchors.append(row['Topic'] + ' ' + args.add_special_token + ' ' + row['Argument'])
                        else:
                            test_anchors.append(row['Topic'] + ' ' + row['Argument'])
                        test_poss.append(row['Evidence_1'])
                        test_negs.append(row['Evidence_2'])
                        test_labels.append(int(l))
                    else:
                        # skip if label is None
                        continue
            test_evaluator = CustomTripletEvaluator(
                test_anchors, test_poss, test_negs, test_labels,
                name='test',
                batch_size=args.eval_batch_size,
                show_progress_bar=True,
                write_csv=True,
                task_name=args.task_name,
            )
        elif args.task_name == 'task3':
            logger.info("Read Ranking Summaries test dataset.")
            test_anchors, test_exp1, test_exp2, test_exp3, test_labels = [], [], [], [], []
            with open(args.test_dataset_file, encoding='utf8') as testF:
                test_reader = csv.DictReader(testF, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in test_reader:
                    test_anchors.append(row['Input'])
                    test_exp1.append(row['Summary_1'])
                    test_exp2.append(row['Summary_2'])
                    test_exp3.append(row['Summary_3'])
                    test_labels.append(1)
            test_evaluator = CustomRankingEvaluator(
                test_anchors, test_exp1, test_exp2, test_exp3, test_labels,
                name='test',
                batch_size=args.eval_batch_size,
                show_progress_bar=True,
                write_csv=True,
            )
        else:
            raise ValueError('Task name must be one of [task2, task3].')
        logger.info('Number of test examples: {}'.format(len(test_labels)))

        model = SentenceTransformer(output_path)
        test_evaluator(model, output_path)
