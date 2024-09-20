from flask import Flask, render_template, request, jsonify, url_for
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os
import pandas as pd
import traceback
import webbrowser
from threading import Timer
from PIL import Image
import logging

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageSearchApp:
    def __init__(self, model_name, embeddings_path, image_paths_file, metadata_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_embeddings = torch.load(embeddings_path, map_location=self.device)
        
        with open(image_paths_file, 'r') as f:
            self.image_paths = json.load(f)

        # Load metadata
        self.metadata_df = pd.read_csv(metadata_path)
        self.metadata_df['normalized_filename'] = self.metadata_df['cropped_filename'].apply(
            lambda x: os.path.splitext(x)[0]
        )
        self.metadata_df['year'] = pd.to_datetime(self.metadata_df['date']).dt.year

        # Ensure alignment between image_paths and metadata
        self.align_data()

    def align_data(self):
        # Create a mapping from image paths to their indices
        path_to_index = {os.path.splitext(os.path.basename(path))[0]: i for i, path in enumerate(self.image_paths)}
        
        # Filter metadata to only include images we have embeddings for
        self.metadata_df = self.metadata_df[self.metadata_df['normalized_filename'].isin(path_to_index.keys())]
        
        # Add the embedding index to the metadata
        self.metadata_df['embedding_index'] = self.metadata_df['normalized_filename'].map(path_to_index)
        
        # Sort the metadata by the embedding index to ensure alignment
        self.metadata_df = self.metadata_df.sort_values('embedding_index').reset_index(drop=True)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        return text_embedding

    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        return image_embedding.cpu()

    def compute_similarity(self, embedding):
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        image_embeddings = self.image_embeddings / self.image_embeddings.norm(dim=-1, keepdim=True)
        similarities = torch.matmul(image_embeddings, embedding.T).squeeze()
        return similarities

    def filter_by_date_range(self, start_year, end_year):
        if start_year is None and end_year is None:
            return list(range(len(self.image_paths)))
        
        filtered_indices = self.metadata_df[
            (self.metadata_df['year'] >= start_year if start_year else True) &
            (self.metadata_df['year'] <= end_year if end_year else True)
        ]['embedding_index'].tolist()
        
        return filtered_indices

    def search_images(self, query, top_k=5, mode='text', start_year=None, end_year=None):
        logger.info(f"Searching images with mode: {mode}, start_year: {start_year}, end_year: {end_year}")
        
        if mode == 'text':
            embedding = self.get_text_embedding(query)
        elif mode == 'image':
            embedding = self.get_image_embedding(query)
        else:
            raise ValueError("Invalid search mode. Use 'text' or 'image'.")

        similarities = self.compute_similarity(embedding)
        
        # Filter by date range
        filtered_indices = self.filter_by_date_range(start_year, end_year)
        filtered_similarities = similarities[filtered_indices]
        
        top_k_indices = torch.topk(filtered_similarities, min(top_k, len(filtered_similarities))).indices
        top_k_original_indices = [filtered_indices[i] for i in top_k_indices]
        top_k_image_paths = [self.image_paths[i] for i in top_k_original_indices]
        top_k_similarities = [filtered_similarities[i].item() for i in top_k_indices]
        
        results = []
        for path, similarity in zip(top_k_image_paths, top_k_similarities):
            metadata = self.metadata_df[self.metadata_df['normalized_filename'] == os.path.splitext(os.path.basename(path))[0]].iloc[0].to_dict()
            metadata = {k: 'Not available' if pd.isna(v) else v for k, v in metadata.items()}
            archive_url = generate_archive_url(metadata)
            results.append({
                'image_path': path,
                'similarity': similarity,
                'metadata': metadata,
                'archive_url': archive_url
            })
        
        return results

def generate_archive_url(metadata):
    base_url = "https://archive.org/details/sim_illustrated-london-news"
    
    date = metadata.get('date', '')
    volume = str(metadata.get('volume', ''))
    volume = volume.rstrip('0').rstrip('.') if '.' in volume else volume
    issue = str(metadata.get('issue', ''))
    issue = issue.rstrip('0').rstrip('.') if '.' in issue else issue
    page = str(metadata.get('page_number', ''))
    
    url = f"{base_url}_{date}_{volume}_{issue}/page/n{page}"
    return url

# Initialize the ImageSearchApp
model_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
embeddings_path = 'embeddings/OpenClipILNfull.pt'
image_paths_file = 'data/image_paths.json'
metadata_path = 'data/iln_text_date_volume_issue_page.csv'
search_app = ImageSearchApp(model_name, embeddings_path, image_paths_file, metadata_path)

@app.route('/')
def home():
    return render_template('indextextimage.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form['query']
        top_k = int(request.form['top_k'])
        start_year = int(request.form['start_date']) if request.form.get('start_date') else None
        end_year = int(request.form['end_date']) if request.form.get('end_date') else None
        
        logger.info(f"Received search request. Query: {query}, Top K: {top_k}, Start Year: {start_year}, End Year: {end_year}")
        
        if not query:
            return jsonify({"error": "Query is empty"}), 400
        
        if top_k < 1 or top_k > 20:
            return jsonify({"error": "Number of results should be between 1 and 20"}), 400
        
        results = search_app.search_images(query, top_k, mode='text', start_year=start_year, end_year=end_year)
        
        if not results:
            logger.warning("No results found for the query")
            return jsonify({"error": "No results found"}), 404
        
        return jsonify(results)
    
    except ValueError as ve:
        logger.error(f"ValueError in search: {str(ve)}")
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400
    except Exception as e:
        logger.error(f"An error occurred in search: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal server error occurred"}), 500

@app.route('/image_search', methods=['POST'])
def image_search():
    try:
        logger.info("Received image search request")
        if 'image' not in request.files:
            logger.error("No image file provided")
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("No image file selected")
            return jsonify({"error": "No image file selected"}), 400

        top_k = int(request.form['top_k'])
        start_year = int(request.form['start_date']) if request.form.get('start_date') else None
        end_year = int(request.form['end_date']) if request.form.get('end_date') else None

        if top_k < 1 or top_k > 20:
            logger.error(f"Invalid top_k value: {top_k}")
            return jsonify({"error": "Number of results should be between 1 and 20"}), 400

        # Open and process the image
        image = Image.open(image_file).convert("RGB")
        logger.info(f"Image processed: {image.size}")

        # Perform image search
        results = search_app.search_images(image, top_k, mode='image', start_year=start_year, end_year=end_year)

        if not results:
            logger.warning("No results found for the image search")
            return jsonify({"error": "No results found"}), 404

        return jsonify(results)

    except ValueError as ve:
        logger.error(f"ValueError in image_search: {str(ve)}")
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400
    except Exception as e:
        logger.error(f"An error occurred in image_search: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal server error occurred"}), 500

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)