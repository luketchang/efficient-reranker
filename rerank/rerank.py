import argparse
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer
from datasets.query_passage_pair import QueryPassagePairDataset
from data_utils import strip_prefixes
from models.deberta_v3_reranker import DeBERTaReranker

def main(model_name, checkpoint_path, qrels_path, rank_results_path, queries_path, corpus_path, batch_size, output_path, qid_prefix, pid_prefix, hits_per_query):
    accelerator = Accelerator(device_placement=True)

    model = DeBERTaReranker(model_name=model_name)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = QueryPassagePairDataset(queries_path, corpus_path, rank_results_path, qrels_path, tokenizer=tokenizer, max_seq_len=model.config.max_position_embeddings, hits_per_query=hits_per_query)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    new_rank_results = defaultdict(list)
    for i, batch in enumerate(dataloader):
        print(f"Processing batch {i}/{len(dataloader)}")

        inputs = batch["pairs"]
        
        outputs = model(**inputs)
        output_logits = outputs.logits

        for i in range(len(output_logits)):
            qid = int(batch["qids"][i])
            pid = int(batch["pids"][i])
            score = output_logits[i].item()
            new_rank_results[qid].append({"pid": pid, "score": score})

    # Sort the outer dictionary by pid and then the internal list by score
    with open(output_path, 'w') as f:
        for qid in sorted(new_rank_results.keys()):  # Sort by qid
            # Sort internal list by pid first, and then by score
            sorted_pid_and_scores = sorted(new_rank_results[qid], key=lambda x: (x['score']), reverse=True)
            for item in sorted_pid_and_scores:
                f.write(f"{qid_prefix}{qid}\t{pid_prefix}{item['pid']}\t{item['score']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank passages using a pre-trained model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--rank_results_path", type=str, required=True, help="Path to the rank results file")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to the qrels file path")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--qid_prefix", type=str, default="", help="qid prefix (e.g. train)")
    parser.add_argument("--pid_prefix", type=str, default="", help="pid prefix (e.g. doc)")
    parser.add_argument("--hits_per_query", type=int, default=100, help="Number of hits per query")


    args = parser.parse_args()

    main(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        rank_results_path=args.rank_results_path,
        qrels_path=args.qrels_path,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
        qid_prefix=args.qid_prefix,
        pid_prefix=args.pid_prefix,
        hits_per_query=args.hits_per_query
    )
