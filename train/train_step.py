from loss import margin_mse_loss, info_nce_loss, combined_loss

def train_step_margin_mse(model, batch, optimizer, accelerator, gradient_accumulation_steps, grad_clip_max_norm, writer, global_step):
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
    loss = margin_mse_loss(positive_logits, negative_logits, positive_labels, negative_labels) / gradient_accumulation_steps
    avg_loss = loss.item() * gradient_accumulation_steps / len(positive_labels)
    
    # Perform backpropagation with automatic mixed precision (handled by accelerator)
    accelerator.backward(loss)
    
    # Apply gradient clipping for stability
    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
    
    if global_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return avg_loss

def train_step_info_nce(model, batch, optimizer, accelerator, gradient_accumulation_steps, grad_clip_max_norm, writer, global_step):
    # Extract inputs and labels from batch
    positives = batch["positives"]
    negatives = batch["negatives"]
    
    # Forward pass
    positive_outs = model(**positives)
    negative_outs = model(**negatives)

    positive_logits = positive_outs.logits
    negative_logits = negative_outs.logits
    
    # Calculate loss
    loss = info_nce_loss(positive_logits, negative_logits) / gradient_accumulation_steps
    avg_loss = loss.item() * gradient_accumulation_steps / len(positive_logits)
    
    # Perform backpropagation with automatic mixed precision (handled by accelerator)
    accelerator.backward(loss)
    
    # Apply gradient clipping for stability
    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
    
    if global_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return avg_loss

def train_step_combined_margin_mse_info_nce(model, batch, optimizer, accelerator, gradient_accumulation_steps, grad_clip_max_norm, writer, global_step):
    # Extract inputs and labels from the batch
    positives = batch["positives"]
    negatives = batch["negatives"]
    positive_labels = batch["positive_labels"]
    negative_labels = batch["negative_labels"]

    # Forward pass
    positive_outs = model(**positives)
    negative_outs = model(**negatives)

    positive_logits = positive_outs.logits
    negative_logits = negative_outs.logits

    # Compute individual losses
    mse_loss = margin_mse_loss(positive_logits, negative_logits, positive_labels, negative_labels)
    nce_loss = info_nce_loss(positive_logits, negative_logits)

    # print and write individual losses to tensorboard
    accelerator.print(f"Step {global_step}: MSE Loss: {mse_loss.item()}, NCE Loss: {nce_loss.item()}")
    writer.add_scalar("Loss/MSE", mse_loss.item(), global_step)
    writer.add_scalar("Loss/NCE", nce_loss.item(), global_step)

    # Combine the losses using the combined loss function
    combined = combined_loss(mse_loss, nce_loss) / gradient_accumulation_steps

    # Perform backpropagation
    accelerator.backward(combined)

    # Apply gradient clipping for stability
    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

    if global_step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Average loss for logging
    avg_combined_loss = combined.item() * gradient_accumulation_steps
    return avg_combined_loss
