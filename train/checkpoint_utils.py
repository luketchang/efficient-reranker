import json
import os
import shutil

def save_global_step(step):
    with open(os.path.join("global_step.json"), 'w') as f:
        json.dump({ "global_step": step }, f)

def load_global_step():
    path = "global_step.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return data['global_step']

    return 0 # no global step

def save_eval_metric(save_path, eval_metric):
    with open(os.path.join(save_path, "best_eval_metric.json"), 'w') as f:
        json.dump({ "eval_metric": eval_metric }, f)

def load_best_eval_metric(save_path, is_loss=False):
    if not save_path:
        if is_loss:
            return float('inf')
        else:
            return 0
    
    best_loss_file = os.path.join(save_path, "best_eval_metric.json")
    if os.path.exists(best_loss_file):
        with open(best_loss_file, 'r') as f:
            data = json.load(f)
            return data['eval_metric']
            
    return float('inf') # Return inf loss if not found

def checkpoint_path_to_prefix(checkpoint_path):
    # Check if the path ends with '-train' or '-inference' and remove them
    if checkpoint_path.endswith('-train'):
        return checkpoint_path[:-6]  # Strip the last 6 characters ('-train')
    elif checkpoint_path.endswith('-inference'):
        return checkpoint_path[:-10]  # Strip the last 10 characters ('-inference')
    return checkpoint_path

def save_new_checkpoint_and_delete_old(accelerator, model, eval_metric, new_checkpoint_prefix, old_checkpoint_prefix):
    accelerator.print("Saving accelerator state and model")
    save_checkpoint(accelerator, model, eval_metric, new_checkpoint_prefix)
    accelerator.print("Saved accelerator state and model")

    # Only main process can remove checkpoint
    if old_checkpoint_prefix and accelerator.is_main_process:
        accelerator.print("Deleting old checkpoint")
        delete_old_checkpoint(old_checkpoint_prefix)
        accelerator.print("Deleted old checkpoint")

def save_checkpoint(accelerator, model, eval_metric, checkpoint_prefix):
    state_path = f'{checkpoint_prefix}-train'
    bin_path = f'{checkpoint_prefix}-inference.pth'

    # Save accelerator state
    accelerator.print("Saving accelerator state")
    accelerator.save_state(state_path)
    save_eval_metric(state_path, eval_metric)
    accelerator.print("Saved accelerator state")

    # Unwrap the model
    unwrapped_model = accelerator.unwrap_model(model)

    # Save the entire model (both Hugging Face part and custom layers)
    accelerator.print("Saving the entire model")
    accelerator.save(unwrapped_model.state_dict(), bin_path)
    accelerator.print(f"Saved model to {bin_path}")

def delete_old_checkpoint(checkpoint_prefix):
    state_path = f'{checkpoint_prefix}-train'
    bin_path = f'{checkpoint_prefix}-inference'
    if os.path.exists(state_path):
        shutil.rmtree(state_path)
    else:
        print(f"Attempted to delete state. Directory {state_path} does not exist.")

    if os.path.exists(bin_path):
        shutil.rmtree(bin_path)
    else:
        print(f"Attempted to model bin. Directory {bin_path} does not exist.")