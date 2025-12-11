#!/usr/bin/env python3
import os
import csv
from prettytable import PrettyTable

# Base directory where the output logs are stored
base_dir = 'linear_probing_logs/percent_data_0.3/'

# List to hold all the summary data
summary_data = []

# Traverse the base directory
for model_dataset in os.listdir(base_dir):
    model_dataset_dir = os.path.join(base_dir, model_dataset)
    if os.path.isdir(model_dataset_dir):
        # Split the directory name to get model and dataset
        # Use underscore as separator and take the last part as dataset
        parts = model_dataset.split('_')
        if len(parts) < 2:
            print(f"Invalid model-dataset directory name: {model_dataset}")
            continue

        # Dataset is the last part, model is everything before that
        dataset = parts[-1]
        model = '_'.join(parts[:-1])

        # Look for CSV files directly in the model_dataset_dir
        csv_filename = f"{model}_{dataset}_results.csv"
        csv_filepath = os.path.join(model_dataset_dir, csv_filename)

        if os.path.exists(csv_filepath):
            with open(csv_filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data = {
                        'Model': model,
                        'Dataset': dataset,
                        'Checkpoint': 'Latest',  # Since no checkpoint subdirectories, use 'Latest'
                        'W_F1': row.get('W_F1', ''),
                        'AUROC': row.get('AUROC', ''),
                        'BACC': row.get('BACC', ''),
                        'ACC': row.get('ACC', ''),
                        'AUPR': row.get('AUPR', '')
                    }
                    summary_data.append(data)
        else:
            print(f"CSV file not found: {csv_filepath}")

# Sort the summary data by Dataset first, then Model, and Checkpoint
summary_data.sort(key=lambda x: (x['Dataset'], x['Model'], x['Checkpoint']))

# Create a PrettyTable instance
fieldnames = ['Model', 'Dataset', 'Checkpoint', 'W_F1', 'AUROC', 'BACC', 'ACC', 'AUPR']
table = PrettyTable()
table.field_names = fieldnames
table.align = 'l'

# Add rows to the table with formatted numeric values and separators between datasets
current_dataset = None
for i, data in enumerate(summary_data):
    # Format numeric values to four decimal places
    for metric in ['W_F1', 'AUROC', 'BACC', 'ACC', 'AUPR']:
        if data[metric]:
            try:
                data[metric] = f"{float(data[metric]):.4f}"
            except ValueError:
                pass  # Keep the original value if it can't be converted to float

    # Add separator row when dataset changes (except for the first row)
    if current_dataset is not None and current_dataset != data['Dataset']:
        # Add a separator row with dashes
        separator_row = ['-' * 20 for _ in fieldnames]
        table.add_row(separator_row)

    # Update current dataset
    current_dataset = data['Dataset']

    # Add the data row
    row = [data.get(field, '') for field in fieldnames]
    table.add_row(row)

# Print the table
print(table)

# Optionally, write the table to a text file
output_txt = 'linear_probing_logs/summary_results_0.3.txt'
with open(output_txt, 'w') as txtfile:
    txtfile.write(str(table))

print(f"Summary results written to {output_txt}")

# Also create a more readable version with clear dataset headers
print("\n" + "=" * 80)
print("DATASET-GROUPED RESULTS")
print("=" * 80)

# Group by dataset for cleaner display
from collections import defaultdict

grouped_data = defaultdict(list)
for data in summary_data:
    grouped_data[data['Dataset']].append(data)

for dataset_name, dataset_rows in grouped_data.items():
    print(f"\n📊 DATASET: {dataset_name}")
    print("-" * 60)

    # Create a table for this dataset
    dataset_table = PrettyTable()
    dataset_fieldnames = ['Model', 'W_F1', 'AUROC', 'BACC', 'ACC',
                          'AUPR']  # Remove Dataset column since it's in the header
    dataset_table.field_names = dataset_fieldnames
    dataset_table.align = 'l'

    for data in dataset_rows:
        row = [data.get('Model', ''), data.get('W_F1', ''), data.get('AUROC', ''),
               data.get('BACC', ''), data.get('ACC', ''), data.get('AUPR', '')]
        dataset_table.add_row(row)

    print(dataset_table)

print("\n" + "=" * 80)