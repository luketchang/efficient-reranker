import torch
import torch.nn as nn
import torch.nn.functional as F

def margin_mse_loss(positive_scores, negative_scores, positive_labels, negative_labels):
    """
    A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
    positive_scores (torch.Tensor): Tensor of shape (batch_size,) containing relevance scores for the positive document.
    negative_scores (torch.Tensor): Tensor of shape (batch_size * num_negatives,) containing relevance scores for negative documents.
    positive_labels (torch.Tensor): Tensor of shape (batch_size,) containing relevance labels for the positive document.
    negative_labels (torch.Tensor): Tensor of shape (batch_size * num_negatives,) containing relevance labels for negative documents.
    """
    # repeat positives so that we can compute the difference with all negatives
    num_negs_per_pos = len(negative_scores) // len(positive_scores)
    positive_scores_repeated = positive_scores.repeat_interleave(num_negs_per_pos)
    positive_labels_repeated = positive_labels.repeat_interleave(num_negs_per_pos)

    loss = torch.mean(torch.pow((positive_scores_repeated - negative_scores) - (positive_labels_repeated - negative_labels), 2))
    return loss
    
def info_nce_loss(positive_scores, negative_scores, temperature=0.07):
    """
    Computes InfoNCE loss as per the formula.
    
    Args:
        positive_scores (torch.Tensor): Tensor of shape (batch_size,) containing relevance scores for the positive document.
        negative_scores (torch.Tensor): Tensor of shape (batch_size, num_negatives) containing relevance scores for negative documents.
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Scalar InfoNCE loss.
    """
    # Divide the scores by temperature
    positive_scores = positive_scores / temperature
    negative_scores = negative_scores / temperature

    # Concatenate positive and negative scores along the second dimension (so we get [positive | negatives])
    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)  # Shape: (batch_size, 1 + num_negatives)

    # The target is that the first logit (index 0) should be the correct one, which is the positive score
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

    # Compute cross-entropy loss (this does the log-softmax for us internally)
    loss = F.cross_entropy(logits, labels)
    
    return loss
    