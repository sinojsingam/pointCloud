import pandas as pd
import sys

file_path = '../working/files/geometric_features_2.0.csv'
output_file = '../working/chunks/features_2_0.csv'
chunk_size = 1000
df = chunks = pd.read_csv(file_path, chunksize=chunk_size)

# Write the first few rows to a new CSV file
with open(output_file, 'w') as f:
    for i, chunk in enumerate(chunks):
        if i == 0:  # Write only the first chunk
            chunk.to_csv(f, index=False)
        else:
            break

