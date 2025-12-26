import csv
import random
import sys

def generate_dummy_probs(dataset_path, output_path):
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {dataset_path} not found.")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            x, y = parts[0], parts[1]
            
            # Random probabilities
            p_route = random.random()
            p_assign = random.random()
            p_loss = random.random()
            
            # Normalize roughly
            total = p_route + p_assign + p_loss
            p_route /= total
            p_assign /= total
            p_loss /= total
            
            writer.writerow([x, y, p_assign, p_route, p_loss])

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
    else:
        dataset = "formatted_dataset/eil51.txt"
        
    generate_dummy_probs(dataset, "attention_probs.csv")
