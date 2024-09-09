import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Function to read JSON files and extract time_per_batch
def read_json_files(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.startswith("coarse_time_m") and filename.endswith(".json"):
            parts = filename[:-5].split("_")
            model = parts[2][1:]
            task = parts[3][1:]
            optimizer = parts[4][1:]
            epochs = int(parts[5][1:])
            
            with open(os.path.join(directory, filename), 'r') as f:
                file_data = json.load(f)
                time_per_batch = file_data["time_per_batch"]
                
            if model not in data:
                data[model] = {}
            if task not in data[model]:
                data[model][task] = {}
            if optimizer not in data[model][task]:
                data[model][task][optimizer] = {}
            
            data[model][task][optimizer][epochs] = time_per_batch
    
    return data

# Read data from JSON files
data = read_json_files("saves")

# Prepare data for plotting
models = list(data.keys())
tasks = list(data[models[0]].keys())
optimizers = list(data[models[0]][tasks[0]].keys())
epochs = list(data[models[0]][tasks[0]][optimizers[0]].keys())

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Set the width of each bar and the positions of the bars
bar_width = 0.8 / len(optimizers)
r = np.arange(len(models))

# Plot bars for each optimizer
optimizer_name_formating = {
    'adadelta': 'Adadelta',
    'adagrad' : 'AdaGrad',
    'adam'    : 'Adam',
    'adamw'   : 'AdamW',
    'asgd'    : 'ASGD',
    'sgd'     : 'SGD',
    'rmsprop' : 'RMSprop',
    'agni'    : 'AdamW + AGNI'
}
print(data)
for i, optimizer in enumerate(optimizers):
    times = [data[model][tasks[0]][optimizer][epochs[0]] for model in models]
    pos = r + i * bar_width
    ax.bar(pos, times, width=bar_width, label= optimizer_name_formating[optimizer] if optimizer != "agni" else "AdamW + AGNI")

# Customize the plot
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Time per Batch (seconds)', fontsize=12)
ax.set_title(f'Comparison of Time per Batch for Different Optimizers\n(Task: {tasks[0]}, Epochs: {epochs[0]})', fontsize=14)
ax.set_xticks(r + bar_width * (len(optimizers) - 1) / 2)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(title='Optimizers', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
