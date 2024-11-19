import torch
from beir.retrieval.evaluation import EvaluateRetrieval

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

def evaluate_model_by_ndcgs(model, eval_data_loaders, accelerator):
    model.eval()

    ndcgs = []
    for i, eval_data_loader in enumerate(eval_data_loaders):
        all_qids = []
        all_pids = []
        all_preds = torch.tensor([]).to(accelerator.device)

        with torch.no_grad():
            for j, batch in enumerate(eval_data_loader):
                accelerator.print(f"Processing batch {j}/{len(eval_data_loader)}")

                qids = batch["qids"]
                pids = batch["pids"]
                pairs = batch["pairs"]

                outputs = model(**pairs)
                output_logits = outputs.logits

                if output_logits.dim() > 1:
                    preds_for_batch = output_logits.squeeze(1)
                else:
                    preds_for_batch = output_logits


                all_qids = all_qids + qids
                all_pids = all_pids + pids
                all_preds = torch.cat([all_preds, preds_for_batch])

        accelerator.print("Gathering losses")
        all_qids = accelerator.gather_for_metrics(all_qids)
        all_pids = accelerator.gather_for_metrics(all_pids)
        all_preds = accelerator.gather_for_metrics(all_preds)

        print(len(eval_data_loader.dataset.qrels))
        qrels = eval_data_loader.dataset.get_qrels(0)
        ndcg, _map, recall, precision = calc_metrics(all_qids, all_pids, all_preds, qrels)
        accelerator.print(f"NDCGs: {ndcg}")
        ndcgs.append(ndcg["NDCG@10"])

    model.train()
    return ndcgs

def calc_metrics(qids, pids, preds, qrels, k_values=[1, 5, 10, 50, 100]):
    # Make sure all tensors are on the CPU
    preds = preds.cpu()

    rank_results = {}

    for i in range(len(qids)):
        qid = str(qids[i])
        pid = str(pids[i])
        pred = float(preds[i].item())
        

        if qid not in rank_results:
            rank_results[qid] = {}

        rank_results[qid][pid] = pred

    eval_results = EvaluateRetrieval.evaluate(qrels, rank_results, k_values)
    return eval_results