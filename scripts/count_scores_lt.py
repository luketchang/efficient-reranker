import csv
import argparse

def main(rank_results_path, lt_value):
    # List to store rows with scores < lt_value
    rows_with_scores_below_lt_value = []

    # Open the TSV file and read line by line
    with open(rank_results_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        
        for row in reader:
            try:
                # The score is in the third column (index 2), convert it to a float
                score = float(row[2])
                
                # Check if the score is less than lt_value
                if score < lt_value:
                    rows_with_scores_below_lt_value.append(row)
            except ValueError:
                # In case there's an issue converting the score to a float
                print(f"Could not convert score for row: {row}")

    # Print the lines with scores below lt_value
    print(f"Rows with scores below {lt_value} (Total: {len(rows_with_scores_below_lt_value)}):")
    for row in rows_with_scores_below_lt_value:
        print("\t".join(row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter rows with scores below a specified value.')
    parser.add_argument('--rank_results_path', type=str, required=True, help='Path to the rank results TSV file')
    parser.add_argument('--lt_value', type=float, required=True, help='Threshold value to filter scores')

    args = parser.parse_args()
    main(args.rank_results_path, args.lt_value)