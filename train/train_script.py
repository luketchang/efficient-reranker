import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, set_seed
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import argparse
from datasets.teacher_triples import TeacherTriplesDataset
from checkpoint_utils import save_global_step, load_global_step, load_best_eval_metric, save_new_checkpoint_and_delete_old, checkpoint_path_to_prefix
from train_step import train_step
from evaluations import evaluate_model_by_loss
from torch.optim import AdamW
from loss import MSEMarginLoss

def training_loop(model_name, checkpoint_path, lr, weight_decay, dropout_prob, num_epochs, batch_size, seed, queries_path, corpus_path, train_positive_rank_results_path, train_negative_rank_results_path, eval_positive_rank_results_path, eval_negative_rank_results_path, eval_every_n_batches, model_bf16, mixed_precision, use_ds, ds_config_path):
    save_path = f'new-{model_name}'

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,           # Use ZeRO stage 2 (stage 3 offloads even more, but is slower)
        offload_optimizer_device="none",  # Whether to offload optimizer state to CPU (reduce GPU VRAM)
        offload_param_device="none",       # Whether to offload parameters to CPU (reduce GPU VRAM)
        hf_ds_config=ds_config_path
    ) if use_ds else None
    
    # Initialize accelerator with mixed precision
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision=mixed_precision, device_placement=True)
    accelerator.print(f"State: {accelerator.state}")
    
    # Lower learning rate for mixed precision stability
    gradient_accumulation_steps = 1 # TODO: maybe add to CLI later
    set_seed(seed)

    # Load train data
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=True)
    train_dataset = TeacherTriplesDataset(queries_path, corpus_path, positive_rank_results_path=train_positive_rank_results_path, negative_rank_results_path=train_negative_rank_results_path, tokenizer=tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
    accelerator.print(f"train data loader len: {len(train_data_loader)}")

    # Load eval data
    eval_dataset = TeacherTriplesDataset(queries_path, corpus_path, positive_rank_results_path=eval_positive_rank_results_path, negative_rank_results_path=eval_negative_rank_results_path, tokenizer=tokenizer)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate_fn)
    
    # Instantiate the model
    if model_bf16:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, num_labels=1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    model.config.attention_dropout = dropout_prob
    model.config.resid_pdrop = dropout_prob
    model.config.embd_pdrop = dropout_prob
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-3)

    # Prepare everything for distributed mixed precision
    model, optimizer, train_data_loader, eval_data_loader = accelerator.prepare(model, optimizer, train_data_loader, eval_data_loader)
    accelerator.print("accelerator prepared")
    accelerator.print(f"train data loader len (accelerator): {len(train_data_loader)}")

    # Load previous state if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.print(f"Loading state from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
    else:
        accelerator.print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/{save_path}')

    # Loss function
    loss_function = MSEMarginLoss()

    # Load best eval loss and global step
    best_avg_eval_loss = load_best_eval_metric(checkpoint_path, is_loss=True)
    global_step = load_global_step()
    accelerator.print(f"Best evaluation loss so far: {best_avg_eval_loss}")
    accelerator.print(f"Starting at global step: {global_step}")

    checkpoint_prefix = checkpoint_path_to_prefix(checkpoint_path) if checkpoint_path else None

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):
            accelerator.print(f"Processing batch {step}/{len(train_data_loader)}")
            avg_train_loss = train_step(model, batch, loss_function, optimizer, accelerator, gradient_accumulation_steps, global_step)
            if accelerator.is_main_process:
                accelerator.print(f'Avg train loss: {avg_train_loss:.4g}')
                writer.add_scalar('Loss/train', avg_train_loss, global_step)

            if step % eval_every_n_batches == 0 and step > 0:
                accelerator.print("Evaluating model")
                avg_eval_loss = evaluate_model_by_loss(model, eval_data_loader, loss_function, accelerator)
                accelerator.print("Evaluated model")

                if accelerator.is_main_process:
                    accelerator.print(f'Avg evaluation loss: {avg_eval_loss:.4g}')
                    writer.add_scalar('Loss/eval', avg_eval_loss, global_step)

                if avg_eval_loss < best_avg_eval_loss:
                    new_checkpoint_prefix = f'{save_path}-step-{global_step}'
                    save_new_checkpoint_and_delete_old(accelerator, model, avg_eval_loss, new_checkpoint_prefix, checkpoint_prefix)

                    checkpoint_prefix = new_checkpoint_prefix
                    best_avg_eval_loss = avg_eval_loss
                

            global_step += 1
        
        accelerator.print(f"End of epoch {epoch}")
        accelerator.wait_for_everyone()

        accelerator.print("Evaluating model at the end of the epoch")
        avg_eval_loss = evaluate_model_by_loss(model, eval_data_loader, loss_function, accelerator)
        if avg_eval_loss < best_avg_eval_loss:
            new_checkpoint_prefix = f'{save_path}-step-{global_step}'
            save_new_checkpoint_and_delete_old(accelerator, model, avg_eval_loss, new_checkpoint_prefix, checkpoint_prefix)

            checkpoint_prefix = new_checkpoint_prefix
            best_avg_eval_loss = avg_eval_loss
        
        accelerator.wait_for_everyone()
        
        save_global_step(global_step)
        
    accelerator.end_training()

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script")
    
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to the checkpoint to resume training from")
    parser.add_argument("--lr", type=float, default=0.0002, required=False, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.001, required=False, help="Weight decay for optimizer")
    parser.add_argument("--dropout_prob", type=float, default=0.1, required=False, help="Dropout probability")
    parser.add_argument("--num_epochs", type=int, default=3, required=False, help="Number of epochs for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, required=False, help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (optional, default=42)")
    
    parser.add_argument("--eval_every_n_batches", type=int, default=5, help="Evaluate model every n batches (optional, default=32)")
    parser.add_argument("--model_bf16", type=str, default=None, help="Load model in bf16")
    parser.add_argument("--mixed_precision", type=str, default=None, help="Mixed precision (fp16, bf16)")
    parser.add_argument("--use_ds", type=str, default=None, help="Use DeepSpeed")
    parser.add_argument("--ds_config_path", type=str, required=False, help="Deepspeed config file")

    args = parser.parse_args()

    # Call the training loop with parsed arguments
    training_loop(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_prob=args.dropout_prob,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size_per_gpu,
        seed=args.seed,
        train_path=args.train_file_path,
        eval_path=args.eval_file_path,
        eval_every_n_batches=args.eval_every_n_batches,
        model_bf16=args.model_bf16,
        mixed_precision=args.mixed_precision,
        use_ds=args.use_ds,
        ds_config_path=args.ds_config_path
    )

if __name__ == "__main__":
    main()