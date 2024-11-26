import argparse
import torch
import time
from collections import defaultdict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer
from custom_datasets.query_passage_pair import QueryPassagePairDataset
from models.deberta_v3_reranker import DeBERTaReranker
from data_utils import strip_prefixes

def main(model_name, precision, checkpoint_path, qrels_path, rank_results_path, queries_path, corpus_path, batch_size, output_path, hits_per_query, qid_base, flush_interval):
    accelerator = Accelerator(device_placement=True)

    model = DeBERTaReranker(model_name=model_name, precision=precision)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(state_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=True)
    dataset = QueryPassagePairDataset([queries_path], [corpus_path], [rank_results_path], [qrels_path], qid_bases=[qid_base], tokenizer=tokenizer, max_seq_len=model.config.max_position_embeddings, hits_per_query=hits_per_query)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    new_rank_results = defaultdict(list)
    
    # Initialize latency tracking
    total_latency = 0
    total_batches = 0
    
    with open(output_path, 'a') as f:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(f"Processing batch {i}/{len(dataloader)}")

                inputs = batch["pairs"]
                
                # Start timing the ranking step
                start_time = time.perf_counter()
                outputs = model(**inputs)
                output_logits = outputs.logits
                # End timing and convert to milliseconds
                batch_latency = (time.perf_counter() - start_time) * 1000
                total_latency += batch_latency
                total_batches += 1

                for j in range(len(output_logits)):
                    qid = batch["qids"][j]
                    pid = batch["pids"][j]
                    score = output_logits[j].item()
                    new_rank_results[qid].append({"pid": pid, "score": score})

                # Write results to file every `batch_write_interval` batches
                if (i + 1) % flush_interval == 0:
                    print(f"Writing results to file after batch {i + 1}")
                    print(f"Average ranking latency: {total_latency/total_batches:.2f} ms over {total_batches} batches")
                    for qid in sorted(new_rank_results.keys(), key=lambda k: int(strip_prefixes(k), qid_base)):
                        sorted_pid_and_scores = sorted(new_rank_results[qid], key=lambda x: x['score'], reverse=True)
                        for item in sorted_pid_and_scores:
                            f.write(f"{qid}\t{item['pid']}\t{item['score']}\n")
                    f.flush()  # Ensure data is written to disk
                    new_rank_results.clear()  # Clear results to start fresh for the next interval

            # Write any remaining results that didn't reach the batch_write_interval
            if new_rank_results:
                print("Writing remaining results to file")
                print(f"Final average ranking latency: {total_latency/total_batches:.2f} ms over {total_batches} batches")
                for qid in sorted(new_rank_results.keys(), key=lambda k: int(strip_prefixes(k), qid_base)):
                    sorted_pid_and_scores = sorted(new_rank_results[qid], key=lambda x: x['score'], reverse=True)
                    for item in sorted_pid_and_scores:
                        f.write(f"{qid}\t{item['pid']}\t{item['score']}\n")
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank passages using a pre-trained model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"], help="Precision for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--rank_results_path", type=str, required=True, help="Path to the rank results file")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to the qrels file path")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--hits_per_query", type=int, default=100, help="Number of hits per query")
    parser.add_argument("--qid_base", type=int, default=10, help="Base of qid (e.g. 10, 16)")
    parser.add_argument("--flush_interval", type=int, default=256, help="Interval of batches to write results to file")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    main(
        model_name=args.model_name,
        precision=args.precision,
        checkpoint_path=args.checkpoint_path,
        rank_results_path=args.rank_results_path,
        qrels_path=args.qrels_path,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
        hits_per_query=args.hits_per_query,
        qid_base=args.qid_base,
        flush_interval=args.flush_interval
    )
