import torch.nn as nn
import torch

class MSEMarginLoss(nn.Module):
    def __init__(self):
        super(MSEMarginLoss, self).__init__()

    def forward(self, positive_scores, negative_scores, positive_labels, negative_labels):
        """
        A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
        All inputs should be tensors of equal size
        """     
        loss = torch.mean(torch.pow((positive_scores - negative_scores) - (positive_labels - negative_labels), 2))
        return loss