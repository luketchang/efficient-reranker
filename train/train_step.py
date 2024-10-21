def train_step(model, batch, loss_fn, optimizer, accelerator, gradient_accumulation_steps, grad_clip_max_norm, global_step):
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
    loss = loss_fn(positive_logits, negative_logits, positive_labels, negative_labels) / gradient_accumulation_steps
    avg_loss = loss.item() / len(positive_labels)
    
    # Perform backpropagation with automatic mixed precision (handled by accelerator)
    accelerator.backward(loss)
    
    # Apply gradient clipping for stability
    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
    
    if global_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return avg_loss