import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict
from sentence_transformers import SentenceTransformer, losses

class CustomRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, distance_metric=losses.TripletDistanceMetric.EUCLIDEAN, margin: float = 0):
        super(CustomRankingLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.margin = margin
    
    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(losses.TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'margin': self.margin}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_example_1, rep_example2, rep_example_3 = reps
        distance_1 = self.distance_metric(rep_anchor, rep_example_1)
        distance_2 = self.distance_metric(rep_anchor, rep_example2)
        distance_3 = self.distance_metric(rep_anchor, rep_example_3)

        losses = F.relu(distance_1 - distance_2 + self.margin).mean() + F.relu(distance_1 - distance_3 + self.margin).mean() + F.relu(distance_2 - distance_3 + self.margin).mean()
        return losses
