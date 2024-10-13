import random

def sample_jsonl(input_file, output_file, sample_size=10000):
    # Read the entire file and count the number of lines
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Take a random sample of 5k lines
    sampled_lines = random.sample(lines, sample_size)

    # Write the sampled lines to a new file
    with open(output_file, 'w') as f:
        for line in sampled_lines:
            f.write(line)

# Usage
input_file = 'data/nq-train/queries.jsonl'
output_file = 'data/nq-train/queries-10k.jsonl'

sample_jsonl(input_file, output_file)