from torch import nn
from torch.nn import functional as F


class ActionConceptLoss(nn.Module):
    def forward(self, similarity, intent_action, intent_concept):
        loss_action = F.binary_cross_entropy(similarity, intent_action, reduction='sum')
        loss_concept = F.binary_cross_entropy(similarity, intent_concept, reduction='sum')
        return loss_concept + loss_action

def get_ce_loss(temperature):
    class SoftenedCrossEntropyLoss(nn.Module):
        def __init__(self, temperature):
            super(SoftenedCrossEntropyLoss, self).__init__()
            self.temperature = temperature

        def forward(self, logits, labels):
            softmax_logits = F.softmax(logits / self.temperature, dim=1)
            loss = F.cross_entropy(softmax_logits, labels)
            return loss

    return SoftenedCrossEntropyLoss(temperature)

def get_loss(cfg):
    return nn.BCELoss(reduction='sum')

