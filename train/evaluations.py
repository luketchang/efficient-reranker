import torch
import torchmetrics
import torchmetrics.retrieval

def evaluate_model_by_loss(model, eval_data_loader, loss_fn, accelerator):
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in eval_data_loader:
            # Extract inputs and labels from batch
            positives = batch["positives"]
            negatives = batch["negatives"]
            positive_labels = batch["positive_labels"]
            negative_labels = batch["negative_labels"]

            # Forward pass
            positive_outs = model(**positives)
            negative_outs = model(**negatives)

            positive_logits = positive_outs.logits
            negative_logits = negative_outs.logits
            
            # Calculate loss
            loss = loss_fn(positive_logits, negative_logits, positive_labels, negative_labels)

            # Gather the loss across all processes
            accelerator.print("Gathering losses")
            gathered_loss = accelerator.gather(loss)  # Gather loss from all processes

            # Compute the mean loss on the gathered losses
            eval_loss += gathered_loss.sum().item()

    # Now compute the average eval loss
    # Total number of samples in the eval set (gathered across all processes)
    num_samples = len(eval_data_loader.dataset)
    avg_eval_loss = eval_loss / num_samples

    model.train()
    return avg_eval_loss

def evaluate_model_by_ndcg(model, eval_data_loader, accelerator):
    model.eval()

    calc_ndcg = torchmetrics.retrieval.RetrievalNormalizedDCG(top_k=10)

    all_preds = []
    all_labels = []
    all_indexes = []

    with torch.no_grad():
        for i, batch in enumerate(eval_data_loader):
            accelerator.print(f"Processing batch {i}/{len(eval_data_loader)}")

            qids = batch["qids"]
            labels = batch["labels"]
            pairs = batch["pairs"]
            
            outputs = model(**pairs)
            output_logits = outputs.logits

            if output_logits.dim() > 1:
                preds_for_batch = output_logits.squeeze(1)
            else:
                preds_for_batch = output_logits

            labels_for_batch = labels.long()
            indexes_for_batch = qids.long()

            all_preds.append(preds_for_batch)
            all_labels.append(labels_for_batch)
            all_indexes.append(indexes_for_batch)

    # combine subarrays into single tensor
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_indexes = torch.cat(all_indexes).long()

    accelerator.print("Gathering losses")
    all_preds = accelerator.gather(all_preds)
    all_labels = accelerator.gather(all_labels)
    all_indexes = accelerator.gather(all_indexes)

    ndcg = calc_ndcg(all_preds, all_labels, all_indexes)

    model.train()
    return ndcg.item()