from tabulate import tabulate
import json 

# Loop through datasets
tasks = ["cola","rte","mrpc","stsb"]
seeds = [41,42,43,44,45]
models = ["distilbert-base-uncased","albert-base-v1"] #"albert/albert-base-v1","squeezebert/squeezebert-uncased","openai-community/gpt2","xlnet/xlnet-base-cased","google-t5/t5-base"

task_to_metrics = {
    "boolq" : "accuracy",
    "cb" : "accuracy",
    "wic" : "accuracy",
    "wsc" : "accuracy",
    "copa" : "accuracy",
    "rte" : "accuracy",
    "mrpc" : "accuracy",
    "stsb" : "pearson",
    "sst2" : "accuracy",
    "cola" : "matthews_correlation",
    "qnli" : "accuracy"
}

global_results = {}

job_ids = ['baselines_SPLIT','lookahead_5'] # ensemble3x_v12

for task in tasks:
    global_results[task] = {}
    for model in models:

        global_results[task][model] = {}
        
        for job_id in job_ids:
            
            folder_path = f"./saves/{job_id}"
            eval_save_file_name = f'{folder_path}/results_rg_{task}_{model}.json'
            try:
                results = json.load( open(eval_save_file_name,'r') )
            except:
                continue

            if '40' in results:
                del results['40']
            
            results_per_seed = [ ( v, v[ task_to_metrics[task] ] ) for k,v in results.items() ]
            results_per_seed.sort(key = lambda x : x[1])
            try:
                assert len(results_per_seed) == 5
                global_results[task][model][job_id] = results_per_seed[2]
                
            except:
                continue


print(global_results)
data = global_results

# Prepare the data for tabulation
headers = ["Model"] + list(data.keys()) + ["Average Score"]
rows = []

# Loop through each task and model configuration
for task, models in data.items():
    for model, configs in models.items():
        for config, values in configs.items():
            # Creating model+config as the row label
            row_label = f"{model} ({config})"
            # Initialize the row if it doesn't exist
            if not any(row[0] == row_label for row in rows):
                rows.append([row_label] + [None] * len(data) + [None])
            # Find the index for the current task
            task_index = headers.index(task)
            # Format metrics into a single string
            metrics = ', '.join(f"{v:.4f}" for k, v in values[0].items())
            # Find the row and set the value
            for row in rows:
                if row[0] == row_label:
                    row[task_index] = metrics

# Calculate the average score for each row
for row in rows:
    scores = []
    for i in range(1, len(headers) - 1):
        if row[i] is not None:
            # Extract the score from the formatted string
            score = float(row[i].split(',')[0])
            scores.append(score)
    if scores:
        average_score = sum(scores) / len(scores)
        row[-1] = f"{average_score:.4f}"

# Print the table using tabulate
print(tabulate(rows, headers=headers, tablefmt='github'))
