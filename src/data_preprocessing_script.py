import os
import json
import numpy as np

def resample_stroke(stroke, num_points=30):
    
    x = np.array([point['x'] for point in stroke])
    y = np.array([point['y'] for point in stroke])
    
    if len(x) == 1:
        return np.column_stack((np.repeat(x, num_points), np.repeat(y, num_points)))
        
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]
    
    if total_length == 0:
        return np.column_stack((np.repeat(x[0], num_points), np.repeat(y[0], num_points)))
        
    target_dist = np.linspace(0, total_length, num_points)
    
    new_x = np.interp(target_dist, cumulative_dist, x)
    new_y = np.interp(target_dist, cumulative_dist, y)
    
    return np.column_stack((new_x, new_y))

def preprocess_json_drawing(filepath, target_points_per_stroke=30):
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    strokes = data.get("strokes", [])
    if not strokes:
        return None
        
    resampled_strokes = []
    for stroke in strokes:
        if len(stroke) > 0:
            new_stroke = resample_stroke(stroke, target_points_per_stroke)
            resampled_strokes.append(new_stroke)
            
    all_points = np.vstack(resampled_strokes)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    max_dim = max(width, height)
    if max_dim == 0: 
        max_dim = 1 
        
    normalized_strokes = []
    for stroke in resampled_strokes:
        norm_x = (stroke[:, 0] - min_x) / max_dim
        norm_y = (stroke[:, 1] - min_y) / max_dim
        normalized_strokes.append(np.column_stack((norm_x, norm_y)))
        
    return normalized_strokes

def process_entire_dataset(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created new directory: {output_folder}")
        
    count = 0
    
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".json"):
                input_path = os.path.join(root, filename)
                

                output_path = os.path.join(output_folder, filename)
                
                clean_strokes = preprocess_json_drawing(input_path)
                
                if clean_strokes is not None:
                    formatted_strokes = []
                    for stroke in clean_strokes:
                        formatted_stroke = [{"x": float(pt[0]), "y": float(pt[1])} for pt in stroke]
                        formatted_strokes.append(formatted_stroke)
                    
                    with open(input_path, 'r', encoding='utf-8') as f:
                        new_json_data = json.load(f)
                        
                    new_json_data["strokes"] = formatted_strokes
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(new_json_data, f, indent=2)
                        
                    count += 1
                    #Prints progress 
                    if count % 100 == 0:
                        print(f"Processed {count} files...")

    print("="*30)
    print(f"✅ Preprocessing Complete!")
    print(f"Successfully cleaned and saved {count} files to '{output_folder}'.")
    print("="*30)

if __name__ == "__main__":
    RAW_DATA_FOLDER = r"E:\Handwritten-Mathematical-equation-solver\data\cleaned-dataset" 
    
    # All processed files will dump into this single folder
    CLEAN_DATA_FOLDER = r"E:\Handwritten-Mathematical-equation-solver\data\dataset_preprocessed" 
    
    process_entire_dataset(RAW_DATA_FOLDER, CLEAN_DATA_FOLDER)