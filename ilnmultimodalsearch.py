from flask import Flask, render_template, request, jsonify, url_for
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os
import pandas as pd
import traceback
import webbrowser
from threading import Timer

app = Flask(__name__, static_folder='static', static_url_path='/static')

class ImageSearchApp:
    def __init__(self, model_name, embeddings_path, image_paths_file):
        self.device = "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_embeddings = torch.load(embeddings_path, map_location=self.device)
        
        with open(image_paths_file, 'r') as f:
            self.image_paths = json.load(f)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        return text_embedding

    def compute_similarity(self, text_embedding):
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        image_embeddings = self.image_embeddings / self.image_embeddings.norm(dim=-1, keepdim=True)
        similarities = torch.matmul(image_embeddings, text_embedding.T).squeeze()
        return similarities

    def search_images(self, query, top_k=5):
        text_embedding = self.get_text_embedding(query)
        similarities = self.compute_similarity(text_embedding)
        top_k_indices = torch.topk(similarities, min(top_k, len(similarities))).indices
        top_k_image_paths = [self.image_paths[i] for i in top_k_indices]
        top_k_similarities = [similarities[i].item() for i in top_k_indices]
        return list(zip(top_k_image_paths, top_k_similarities))

# Initialize the ImageSearchApp
model_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
embeddings_path = 'embeddings/OpenClipILNfull.pt'
image_paths_file = 'data/image_paths.json'
search_app = ImageSearchApp(model_name, embeddings_path, image_paths_file)

# Load the metadata CSV into a pandas DataFrame
metadata_path = 'data/iln_text_date_volume_issue_page.csv'
metadata_df = pd.read_csv(metadata_path)

# Normalize filenames in the DataFrame (remove file extension for comparison)
metadata_df['normalized_filename'] = metadata_df['cropped_filename'].apply(
    lambda x: os.path.splitext(x)[0]
)

def generate_archive_url(metadata):
    base_url = "https://archive.org/details/sim_illustrated-london-news"
    
    # Extract relevant metadata
    date = metadata.get('date', '')
    
    # Convert volume and issue to strings and remove .0 if present
    volume = str(metadata.get('volume', ''))
    volume = volume.rstrip('0').rstrip('.') if '.' in volume else volume
    
    issue = str(metadata.get('issue', ''))
    issue = issue.rstrip('0').rstrip('.') if '.' in issue else issue
    
    page = str(metadata.get('page_number', ''))
    
    print(date)
    print(volume)
    
    # Construct the URL
    url = f"{base_url}_{date}_{volume}_{issue}/page/n{page}"
    return url

def search_images_with_metadata(query, top_k=5):
    search_results = search_app.search_images(query, top_k)
    
    enriched_results = []
    
    for image_path, similarity in search_results:
        normalized_image_path = os.path.splitext(os.path.basename(image_path))[0]
        
        matching_row = metadata_df[metadata_df['normalized_filename'] == normalized_image_path]

        if not matching_row.empty:
            metadata = matching_row.iloc[0].to_dict()
            metadata = {k: 'Not available' if pd.isna(v) else v for k, v in metadata.items()}
            
            # Generate the Internet Archive URL
            archive_url = generate_archive_url(metadata)
            
            enriched_results.append({
                'image_path': image_path,
                'similarity': similarity,
                'metadata': metadata,
                'archive_url': archive_url
            })
        else:
            enriched_results.append({
                'image_path': image_path,
                'similarity': similarity,
                'metadata': None,
                'archive_url': None
            })
    
    return enriched_results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form['query']
        top_k = int(request.form['top_k'])
        
        if not query:
            return jsonify({"error": "Query is empty"}), 400
        
        if top_k < 1 or top_k > 20:
            return jsonify({"error": "Number of results should be between 1 and 20"}), 400
        
        results = search_images_with_metadata(query, top_k)
        print(results)
        
        if not results:
            return jsonify({"error": "No results found"}), 404
        
        return jsonify(results)
    
    except ValueError as ve:
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal server error occurred"}), 500
    
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)