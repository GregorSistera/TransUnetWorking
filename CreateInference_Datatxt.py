import os

# Define source folder and output file path
npz_folder = './inference_data'
output_list_path = './lists/lists_MyEpithelialDataset/infer.txt'

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_list_path), exist_ok=True)

# List all .npz files
npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
npz_files.sort()  # Optional: sort alphabetically

# Write to infer.txt
with open(output_list_path, 'w') as f:
    for file in npz_files:
        f.write(file + '\n')

print(f"✅ Created list with {len(npz_files)} files at {output_list_path}")
