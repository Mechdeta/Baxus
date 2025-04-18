import pandas as pd
import json

csv_file = "501 Bottle Dataset - Sheet1.csv"
df = pd.read_csv(csv_file)

dataset = []
for _, row in df.iterrows():
    if pd.isna(row['image_url']):  # Skip rows with NaN URLs
        continue
    entry = {
        "name": row["name"],
        "image_path": row["image_url"],
        "label_text": row["name"],
        "brand": str(row["brand_id"])
    }
    dataset.append(entry)

output_json = "baxus_dataset.json"
with open(output_json, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Created {output_json} with {len(dataset)} entries")
