import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, set_seed
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from models.deberta_v3_reranker import DeBERTaReranker
import os
import argparse
from datasets.pos_neg import PositiveNegativeDataset
from datasets.query_passage_pair import QueryPassagePairDataset
from checkpoint_utils import save_global_step, load_global_step, load_best_eval_metric, save_checkpoint, delete_old_checkpoint, checkpoint_path_to_prefix
from train.train_step import train_step_margin_mse, train_step_info_nce
from evaluations import evaluate_model_by_ndcgs
from torch.optim import AdamW

def training_loop(model_name, pooling, checkpoint_path, num_neg_per_pos, lr, weight_decay, dropout_prob, num_epochs, batch_size, seed, delete_old_checkpoint_steps, queries_paths, corpus_paths, train_positive_rank_results_paths, train_negative_rank_results_paths, train_qid_bases, eval_rank_results_paths, eval_qrels_paths, eval_queries_paths, eval_qid_bases, eval_every_n_batches, model_bf16, mixed_precision, grad_accumulation_steps, grad_clip_max_norm, use_ds, ds_config_path):
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
    
    # Set seed
    set_seed(seed)

    # Instantiate the model
    model = DeBERTaReranker(model_name, pooling)
    if model_bf16:
        model = model.to(torch.bfloat16)
    
    model.config.attention_dropout = dropout_prob
    model.config.resid_pdrop = dropout_prob
    model.config.embd_pdrop = dropout_prob
    model.config.use_cache = False
    model.deberta.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) # TODO: fix hack

    # Load train data
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=True)
    train_dataset = PositiveNegativeDataset(queries_paths, corpus_paths, positive_rank_results_paths=train_positive_rank_results_paths, negative_rank_results_paths=train_negative_rank_results_paths, tokenizer=tokenizer, qid_bases=train_qid_bases, max_seq_len=model.config.max_position_embeddings, seed=seed, num_neg_per_pos=num_neg_per_pos)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
    accelerator.print(f"train data loader len: {len(train_data_loader)}")

    # Load eval data
    eval_datasets = [QueryPassagePairDataset([query_path], [corpus_path], rank_results_paths=[rank_results_path], qrels_paths=[qrels_path], tokenizer=tokenizer, qid_bases=eval_qid_bases, max_seq_len=model.config.max_position_embeddings) for query_path, corpus_path, rank_results_path, qrels_path in zip(eval_queries_paths, corpus_paths, eval_rank_results_paths, eval_qrels_paths)]
    eval_data_loaders = [DataLoader(eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate_fn) for eval_dataset in eval_datasets]
    
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare everything for distributed mixed precision
    model, optimizer, train_data_loader, eval_data_loaders = accelerator.prepare(model, optimizer, train_data_loader, eval_data_loaders)
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

    # Load best eval loss and global step
    best_eval_metric = load_best_eval_metric(checkpoint_path, is_loss=False)
    global_step = load_global_step()
    last_saved_global_step = global_step
    accelerator.print(f"Best evaluation metric so far: {best_eval_metric}")
    accelerator.print(f"Starting at global step: {global_step}")

    checkpoint_prefix = checkpoint_path_to_prefix(checkpoint_path) if checkpoint_path else None

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):
            accelerator.print(f"Processing batch {step}/{len(train_data_loader)}")
            avg_train_loss = train_step_margin_mse(model, batch, optimizer, accelerator, grad_accumulation_steps, grad_clip_max_norm, global_step)
            if accelerator.is_main_process:
                accelerator.print(f'Avg train loss: {avg_train_loss:.4g}')
                writer.add_scalar('Loss/train', avg_train_loss, global_step)

            if step % eval_every_n_batches == 0 and step > 0:
                accelerator.print("Evaluating model")
                eval_ndcgs = evaluate_model_by_ndcgs(model, eval_data_loaders, accelerator)
                avg_eval_ndcg = sum(eval_ndcgs) / len(eval_ndcgs) # take average
                accelerator.print("Evaluated model")

                if accelerator.is_main_process:
                    accelerator.print(f'Avg ndcg: {avg_eval_ndcg:.4g}')
                    writer.add_scalar('ndcg/eval', avg_eval_ndcg, global_step)

                    for i, ndcg in enumerate(eval_ndcgs):
                        accelerator.print(f'Ndcg eval {i}: {ndcg:.4g}')
                        writer.add_scalar(f'ndcg/eval_{i}', ndcg, global_step)

                if avg_eval_ndcg > best_eval_metric:
                    new_checkpoint_prefix = f'{save_path}-step-{global_step}'
                    save_checkpoint(accelerator, model, avg_eval_ndcg, new_checkpoint_prefix)

                    # NOTE: only delete old checkpoint if new one is within delete_old_checkpoint_steps
                    if global_step - last_saved_global_step < delete_old_checkpoint_steps:
                        delete_old_checkpoint(accelerator, checkpoint_prefix)

                    checkpoint_prefix = new_checkpoint_prefix
                    best_eval_metric = avg_eval_ndcg
                    last_saved_global_step = global_step

            global_step += 1
        
        accelerator.print(f"End of epoch {epoch}")
        accelerator.wait_for_everyone()

        accelerator.print("Evaluating model at the end of the epoch")
        eval_ndcgs = evaluate_model_by_ndcgs(model, eval_data_loaders, accelerator)
        avg_eval_ndcg = sum(eval_ndcgs) / len(eval_ndcgs) # take average

        if accelerator.is_main_process:
            accelerator.print(f'Avg ndcg: {avg_eval_ndcg:.4g}')
            writer.add_scalar('ndcg/eval', avg_eval_ndcg, global_step)

            for i, ndcg in enumerate(eval_ndcgs):
                accelerator.print(f'Ndcg eval {i}: {ndcg:.4g}')
                writer.add_scalar(f'ndcg/eval_{i}', ndcg, global_step)

        if avg_eval_ndcg > best_eval_metric:
            new_checkpoint_prefix = f'{save_path}-step-{global_step}'
            save_checkpoint(accelerator, model, avg_eval_ndcg, new_checkpoint_prefix)

            # NOTE: only delete old checkpoint if new one is within delete_old_checkpoint_steps
            if global_step - last_saved_global_step < delete_old_checkpoint_steps:
                delete_old_checkpoint(accelerator, checkpoint_prefix)

            checkpoint_prefix = new_checkpoint_prefix
            best_eval_metric = avg_eval_ndcg
            last_saved_global_step = global_step
        
        accelerator.wait_for_everyone()
        
        save_global_step(global_step)
        
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description="Training script")
    
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load")
    parser.add_argument("--pooling", type=str, default="mean", required=False, help="Pooling strategy for the model")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to the checkpoint to resume training from")
    parser.add_argument("--num_neg_per_pos", type=int, default=1, required=False, help="Number of negatives to sample per positive")
    parser.add_argument("--lr", type=float, default=0.00002, required=False, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False, help="Weight decay for optimizer")
    parser.add_argument("--dropout_prob", type=float, default=0.1, required=False, help="Dropout probability")
    parser.add_argument("--num_epochs", type=int, default=3, required=False, help="Number of epochs for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, required=False, help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (optional, default=42)")
    parser.add_argument("--delete_old_checkpoint_steps", type=int, default=8192, help="Delete old checkpoint if new one is within this number of steps (optional, default=8192)")

    # NOTE: can take multiple items per arg
    parser.add_argument("--queries_paths", type=str, nargs='+', required=True, help="Paths to the queries JSONL files")
    parser.add_argument("--corpus_paths", type=str, nargs='+', required=True, help="Paths to the corpus JSONL files")
    parser.add_argument("--train_positive_rank_results_paths", type=str, nargs='+', required=True, help="Paths to the train positive rank results files")
    parser.add_argument("--train_negative_rank_results_paths", type=str, nargs='+', required=True, help="Paths to the train negative rank results files")
    parser.add_argument("--train_qid_bases", type=int, nargs='+', required=True, help="Base of the qid interpreted as int")

    # NOTE: can take multiple items per arg
    parser.add_argument("--eval_qrels_paths", type=str, nargs='+', required=True, help="Paths to the eval qrels files")
    parser.add_argument("--eval_queries_paths", type=str, nargs='+', required=True, help="Paths to the eval queries files")
    parser.add_argument("--eval_rank_results_paths", type=str, nargs='+', required=True, help="Paths to the eval rank results files")
    parser.add_argument("--eval_qid_bases", type=int, nargs='+', required=True, help="Base of the qid interpreted as int")

    parser.add_argument("--eval_every_n_batches", type=int, default=5, help="Evaluate model every n batches (optional, default=32)")
    parser.add_argument("--model_bf16", type=str, default=None, help="Load model in bf16")
    parser.add_argument("--mixed_precision", type=str, default=None, help="Mixed precision (fp16, bf16)")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--use_ds", type=str, default=None, help="Use DeepSpeed")
    parser.add_argument("--ds_config_path", type=str, required=False, help="Deepspeed config file")

    args = parser.parse_args()

    training_loop(
        model_name=args.model_name,
        pooling=args.pooling,
        checkpoint_path=args.checkpoint_path,
        num_neg_per_pos=args.num_neg_per_pos,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_prob=args.dropout_prob,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size_per_gpu,
        seed=args.seed,
        delete_old_checkpoint_steps=args.delete_old_checkpoint_steps,
        queries_paths=args.queries_paths,
        corpus_paths=args.corpus_paths,
        train_positive_rank_results_paths=args.train_positive_rank_results_paths,
        train_negative_rank_results_paths=args.train_negative_rank_results_paths,
        train_qid_bases=args.train_qid_bases,
        eval_rank_results_paths=args.eval_rank_results_paths,
        eval_qrels_paths=args.eval_qrels_paths,
        eval_queries_paths=args.eval_queries_paths,
        eval_qid_bases=args.eval_qid_bases,
        eval_every_n_batches=args.eval_every_n_batches,
        model_bf16=args.model_bf16,
        mixed_precision=args.mixed_precision,
        grad_accumulation_steps=args.grad_accumulation_steps,
        grad_clip_max_norm=args.grad_clip_max_norm,
        use_ds=args.use_ds,
        ds_config_path=args.ds_config_path
    )

if __name__ == "__main__":
    main()