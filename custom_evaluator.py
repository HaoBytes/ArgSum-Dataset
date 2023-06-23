"""
Script for customized evaluator.
Codes are adapted from TripletEvaluator.
"""

from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers import util
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List
from sentence_transformers.readers import InputExample

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class CustomTripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        labels: List[int],
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.labels = np.array(labels)
        self.name = name

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhattan", "accuracy_euclidean"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []
        labels = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        if example.label is not None:
            labels = example.label
        else:
            labels = None
        return cls(anchors, positives, negatives, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        num_triplets = 0
        # num_correct_cos_triplets, num_correct_manhattan_triplets, num_correct_euclidean_triplets = 0, 0, 0
        cos_predictions, manhattan_predictions, euclidean_predictions = [], [], []

        embeddings_anchors = model.encode(
            self.anchors, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_positives = model.encode(
            self.positives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_negatives = model.encode(
            self.negatives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )

        # Cosine distance
        pos_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_positives)
        neg_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_negatives)
        # print('pos_cos_distance: ', pos_cos_distance)
        # print('neg_cos_distances: ', neg_cos_distances)

        # Manhattan
        pos_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_positives)
        neg_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_negatives)
        # print('pos_manhattan_distance: ', pos_manhattan_distance)
        # print('neg_manhattan_distances: ', neg_manhattan_distances)

        # Euclidean
        pos_euclidean_distances = paired_euclidean_distances(embeddings_anchors, embeddings_positives)
        neg_euclidean_distances = paired_euclidean_distances(embeddings_anchors, embeddings_negatives)
        # print('pos_euclidean_distance: ', pos_euclidean_distance)
        # print('neg_euclidean_distances: ', neg_euclidean_distances)


        for idx in range(len(pos_cos_distances)):
            num_triplets += 1

            if pos_cos_distances[idx] < neg_cos_distances[idx]:
                # num_correct_cos_triplets += 1
                cos_predictions.append(1)
            else:
                cos_predictions.append(0)

            if pos_manhattan_distances[idx] < neg_manhattan_distances[idx]:
                # num_correct_manhattan_triplets += 1
                manhattan_predictions.append(1)
            else:
                manhattan_predictions.append(0)

            if pos_euclidean_distances[idx] < neg_euclidean_distances[idx]:
                # num_correct_euclidean_triplets += 1
                euclidean_predictions.append(1)
            else:
                euclidean_predictions.append(0)

        cos_predictions = np.array(cos_predictions)
        manhattan_predictions = np.array(manhattan_predictions)
        euclidean_predictions = np.array(euclidean_predictions)
        # accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_cos = accuracy_score(self.labels, cos_predictions)
        # accuracy_manhattan = num_correct_manhattan_triplets / num_triplets
        accuracy_manhattan = accuracy_score(self.labels, manhattan_predictions)
        # accuracy_euclidean = num_correct_euclidean_triplets / num_triplets
        accuracy_euclidean = accuracy_score(self.labels, euclidean_predictions)

        # logger.info("Accuracy Cosine Distance:   \t{:.2f}".format(accuracy_cos * 100))
        # logger.info("Accuracy Manhattan Distance:\t{:.2f}".format(accuracy_manhattan * 100))
        # logger.info("Accuracy Euclidean Distance:\t{:.2f}\n".format(accuracy_euclidean * 100))

        logger.info("[Cosine Distance Report] accuracy: \t{:.2f}, f1: \t{:.2f}, precision: \t{:.2f}, recall: \t{:.2f}".format(accuracy_cos* 100, f1_score(self.labels, cos_predictions)* 100, precision_score(self.labels, cos_predictions)* 100, recall_score(self.labels, cos_predictions)* 100))
        logger.info("[Manhattan Distance Report] accuracy: \t{:.2f}, f1: \t{:.2f}, precision: \t{:.2f}, recall: \t{:.2f}".format(accuracy_manhattan*100, f1_score(self.labels, manhattan_predictions)*100, precision_score(self.labels, manhattan_predictions)*100, recall_score(self.labels, manhattan_predictions)*100))
        logger.info("[Euclidean Distance Report] accuracy: \t{:.2f}, f1: \t{:.2f}, precision: \t{:.2f}, recall: \t{:.2f}\n".format(accuracy_euclidean*100, f1_score(self.labels, euclidean_predictions)*100, precision_score(self.labels, euclidean_predictions)*100, recall_score(self.labels, euclidean_predictions)*100))

        logger.info("Classification Report Cosine Distance:   \n{}".format(classification_report(self.labels, cos_predictions, digits=4)))
        logger.info("Classification Report Manhattan Distance:\n{}".format(classification_report(self.labels, manhattan_predictions, digits=4)))
        logger.info("Classification Report Euclidean Distance:\n{}\n".format(classification_report(self.labels, euclidean_predictions, digits=4)))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])
            
            if self.name == 'test':
                # write predictions file
                preds_df = pd.DataFrame({'Argument': self.anchors, 'Evidence_1': self.positives, 'Evidence_2': self.negatives, 'Label': self.labels, 
                                         'Evidence_1_Cosine': pos_cos_distances, 'Evidence_2_Cosine': neg_cos_distances, 'Cosine': cos_predictions, 
                                         'Evidence_1_Manhattan': pos_manhattan_distances, 'Evidence_2_Manhattan': neg_manhattan_distances, 'Manhattan': manhattan_predictions, 
                                         'Evidence_1_Euclidean': pos_euclidean_distances, 'Evidence_2_Euclidean': neg_euclidean_distances, 'Euclidean': euclidean_predictions})
                preds_df.to_csv(os.path.join(output_path, self.name+'_predictions.csv'), index=False)

        if self.main_distance_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhattan
        if self.main_distance_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhattan, accuracy_euclidean)




class CustomRankingEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a multiple inputs: (sentence, example1, example2, example3).
        Checks if distance(sentence, example_i) < distance(sentence, example_j) if i < j.
    """

    def __init__(
        self,
        anchors: List[str],
        examples_1: List[str],
        examples_2: List[str],
        examples_3: List[str],
        labels: List[int],
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.anchors = anchors
        self.examples_1 = examples_1
        self.examples_2 = examples_2
        self.examples_3 = examples_3
        self.labels = np.array(labels)
        self.name = name

        assert len(self.anchors) == len(self.examples_1)
        assert len(self.anchors) == len(self.examples_2)
        assert len(self.anchors) == len(self.examples_3)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "ranking_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhattan", "accuracy_euclidean"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        examples_1 = []
        examples_2 = []
        examples_3 = []
        labels = []

        for example in examples:
            anchors.append(example.texts[0])
            examples_1.append(example.texts[1])
            examples_2.append(example.texts[2])
            examples_3.append(example.texts[3])
        if example.label is not None:
            labels = example.label
        else:
            labels = None
        return cls(anchors, examples_1, examples_2, examples_3, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("RankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        num_examples = 0
        num_correct_cos_examples, num_correct_manhattan_examples, num_correct_euclidean_examples = 0, 0, 0

        embeddings_anchors = model.encode(
            self.anchors, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_examples_1 = model.encode(
            self.examples_1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_examples_2 = model.encode(
            self.examples_2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_examples_3 = model.encode(
            self.examples_3, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )

        # Cosine distance
        exp1_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_examples_1)
        exp2_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_examples_2)
        exp3_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_examples_3)

        # Manhattan
        exp1_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_examples_1)
        exp2_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_examples_2)
        exp3_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_examples_3)

        # Euclidean
        exp1_euclidean_distances = paired_cosine_distances(embeddings_anchors, embeddings_examples_1)
        exp2_euclidean_distances = paired_cosine_distances(embeddings_anchors, embeddings_examples_2)
        exp3_euclidean_distances = paired_cosine_distances(embeddings_anchors, embeddings_examples_3)


        for idx in range(len(embeddings_anchors)):
            num_examples += 1

            if exp1_cos_distances[idx] < exp2_cos_distances[idx] and exp2_cos_distances[idx] < exp3_cos_distances[idx]:
                num_correct_cos_examples += 1

            if exp1_manhattan_distances[idx] < exp2_manhattan_distances[idx] and exp2_manhattan_distances[idx] < exp3_manhattan_distances[idx]:
                num_correct_manhattan_examples += 1

            if exp1_euclidean_distances[idx] < exp2_euclidean_distances[idx] and exp2_euclidean_distances[idx] < exp3_euclidean_distances[idx]:
                num_correct_euclidean_examples += 1

        accuracy_cos = num_correct_cos_examples / num_examples
        accuracy_manhattan = num_correct_manhattan_examples / num_examples
        accuracy_euclidean = num_correct_euclidean_examples / num_examples

        logger.info("Accuracy Cosine Distance:   \t{:.2f}".format(accuracy_cos * 100))
        logger.info("Accuracy Manhattan Distance:\t{:.2f}".format(accuracy_manhattan * 100))
        logger.info("Accuracy Euclidean Distance:\t{:.2f}\n".format(accuracy_euclidean * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])
            
            if self.name == 'test':
                # write predictions file
                preds_df = pd.DataFrame({'Input': self.anchors, 'Summary_1': self.examples_1, 'Summary_2': self.examples_2, 'Summary_3': self.examples_3,
                                         'Summary_1_Cosine': exp1_cos_distances, 'Summary_2_Cosine': exp2_cos_distances, 'Summary_3_Cosine': exp3_cos_distances, 
                                         'Summary_1_Manhattan': exp1_manhattan_distances, 'Summary_2_Manhattan': exp2_manhattan_distances, 'Summary_3_Manhattan': exp3_manhattan_distances, 
                                         'Summary_1_Euclidean': exp1_euclidean_distances, 'Summary_2_Euclidean': exp2_euclidean_distances, 'Summary_3_Euclidean': exp3_euclidean_distances})
                preds_df.to_csv(os.path.join(output_path, self.name+'_predictions.csv'), index=False)

        if self.main_distance_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhattan
        if self.main_distance_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhattan, accuracy_euclidean)