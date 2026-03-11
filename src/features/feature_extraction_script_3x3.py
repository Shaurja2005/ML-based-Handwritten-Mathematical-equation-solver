import os
import json
import csv
import math

def calculate_features(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    strokes = data.get("strokes", [])
    label = data.get("label", "unknown")
    
    if not strokes:
        return None
        
    num_strokes = len(strokes)
    all_x, all_y = [], []
    total_length = 0.0
    
    grid_counts = [0] * 9 
    total_points = 0
    
    for stroke in strokes:
        for i in range(len(stroke)):
            x = stroke[i]['x']
            y = stroke[i]['y']
            
            all_x.append(x)
            all_y.append(y)
            total_points += 1
            
            if i > 0:
                dx = x - stroke[i-1]['x']
                dy = y - stroke[i-1]['y']
                total_length += math.sqrt(dx**2 + dy**2)
                
            col = min(int(x * 3), 2) 
            row = min(int(y * 3), 2)
            grid_index = row * 3 + col
            grid_counts[grid_index] += 1
                
    if not all_x or total_points == 0:
        return None
        
    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    aspect_ratio = width / (height + 0.0001)
    area = width * height
    
    grid_density = [count / total_points for count in grid_counts]
    
    features = {
        "label": label,
        "num_strokes": num_strokes,
        "width": round(width, 4),
        "height": round(height, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "total_length": round(total_length, 4),
        "area": round(area, 4)
    }
    

    for i in range(9):
        features[f"grid_zone_{i+1}"] = round(grid_density[i], 4)
        
    return features

def build_csv_dataset(input_folder, output_csv):
    
    headers = [
        "label", "num_strokes", "width", "height", "aspect_ratio", "total_length", "area",
        "grid_zone_1", "grid_zone_2", "grid_zone_3", 
        "grid_zone_4", "grid_zone_5", "grid_zone_6", 
        "grid_zone_7", "grid_zone_8", "grid_zone_9"
    ]
    processed_count = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for filename in os.listdir(input_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(input_folder, filename)
                features = calculate_features(filepath)
                
                if features:
                    writer.writerow(features)
                    processed_count += 1
                    
    print("="*40)
    print(f" Feature Extraction Complete!")
    print(f"Successfully saved {processed_count} rows with GRID FEATURES to '{output_csv}'")
    print("="*40)

if __name__ == "__main__":
    CLEAN_DATA_FOLDER = r"data\dataset_preprocessed" 
    FINAL_CSV_FILE = r"data\dataset_3x3.csv" 
    build_csv_dataset(CLEAN_DATA_FOLDER, FINAL_CSV_FILE)