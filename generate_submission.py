import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--language', help="select language")
args = parser.parse_args()

label_file_path = f'./CLEF_infer_output/{args.language}'
query_file_path = f'./CLEF_data/1A_checkworthy/{args.language}/test_query.tsv'
output_file_path = f"subtask1A_checkworthy_{args.language}.tsv"

query_list = list()
label_list = list()

# query list processing
query_data = pd.read_csv(query_file_path, sep='\t', dtype={"tweet_id": str})
for query_info in zip(query_data.topic, query_data.tweet_id, query_data.tweet_text):
    query_list += [query_info]

# label list processing
with open(label_file_path) as label_file:
    for line in label_file.readlines():
        label_list += [line.strip()]

print(f'{len(query_list)} should be equal with {len(label_list)}')

with open(output_file_path, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['topic', 'tweet_id', 'class_label', 'run_id'])
    for tweet_info, label in zip(query_list, label_list):
        topic = tweet_info[0]
        tweet_id = tweet_info[1]
        class_label = "1" if label == "yes" else "0"
        run_id = "mt5"
        tsv_writer.writerow([topic, tweet_id, class_label, run_id])
    
print("Done!")

