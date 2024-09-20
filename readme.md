# Illustrated London News Multimodal Search

This Flask application provides a multimodal search engine for exploring images from the Illustrated London News using textual queries.

## Getting the Project

You can get this project in two ways: cloning the repository using Git or downloading it directly from the GitHub website.

### Option 1: Cloning the Repository (for Git users)

1. Open your command line interface (Command Prompt for Windows or Terminal for Mac).
2. Navigate to the directory where you want to store the project.
3. Run the following command:
   ```
   git clone https://github.com/tpsmi/ilnmultimodalsearch.git
   ```
4. Navigate into the project directory:
   ```
   cd ilnmultimodalsearch
   ```

### Option 2: Downloading from GitHub (for non-Git users)

1. Open your web browser and go to https://github.com/tpsmi/ilnmultimodalsearch
2. Click on the green "Code" button near the top-right of the page.
3. In the dropdown menu, click "Download ZIP".
4. Once the download is complete, locate the ZIP file on your computer (usually in your Downloads folder).
5. Extract the ZIP file to your desired location.
6. Open your command line interface:
   - For Windows: Press Win+R, type `cmd`, and press Enter.
   - For Mac: Open the Terminal app (you can find it in Applications > Utilities > Terminal).
7. Navigate to the extracted project folder:
   - For Windows: 
     ```
     cd path\to\ilnmultimodalsearch
     ```
   - For Mac:
     ```
     cd /path/to/ilnmultimodalsearch
     ```
   Replace "path\to" or "/path/to" with the actual path where you extracted the ZIP file.

After following either option, continue with the "Installation" steps in the next section.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
   or
   ```
   conda create --name ilnmultimodal
   conda activate ilnmultimodal
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: This will install CPU-only versions of PyTorch and related packages. No additional installations are needed for the automatic browser opening feature.

## Data Setup

1. Download the large embedding file:
   - `OpenClipILNfull.pt` (File size: 211.2 MB)
   - Download link: https://filesender.surf.nl/?s=download&token=df6f6951-552d-4a06-a94f-3296960a7a0f
   - Please contact me if the download link is no longer active. 

   After downloading, place this file in the `embeddings/` directory.

   Note: This file is too large for GitHub, so it needs to be downloaded separately.

## Navigating to the Project Directory

### For Windows Users:

1. Open the Command Prompt:
   - Press `Win + R`, type `cmd`, and press Enter, or
   - Search for "Command Prompt" in the Start menu

2. Navigate to the cloned repository:
   ```
   cd path\to\iln-multimodal-search
   ```
   Replace `path\to` with the actual path where you cloned the repository.

   Tip: You can copy the full path from File Explorer by navigating to the folder, clicking in the address bar, and copying the text.

### For Mac Users:

1. Open the Terminal:
   - Click on the Launchpad icon in the Dock, type "Terminal" in the search field, then click Terminal, or
   - In the Finder, go to Applications > Utilities > Terminal

2. Navigate to the cloned repository:
   ```
   cd /path/to/iln-multimodal-search
   ```
   Replace `/path/to` with the actual path where you cloned the repository.

   Tip: You can drag the folder from Finder into the Terminal window to automatically insert the path.

## Running the Application

1. From the project root directory, run:
   ```
   python ilntextimage.py
   ```

2. The application will automatically open in your default web browser. If it doesn't, manually navigate to `http://127.0.0.1:5000/` in your web browser.

Note: The automatic browser opening feature uses Python's built-in `webbrowser` module and doesn't require any additional installations.

## Usage

1. Enter a search query in the text box or upload an image. 
2. Select the number of results you want to see.
3. Give a specific date range (in years).
4. Click the "Search" button.
5. The results will display with links to the corresponding pages on the Internet Archive.

## Troubleshooting

- If the browser doesn't open automatically, ensure that you have a default web browser set on your system.
- If you encounter any issues with automatic browser opening, you can always manually navigate to `http://127.0.0.1:5000/` in your web browser after running the application.
- If you're running the script in an environment without a graphical interface (e.g., a remote server), the automatic browser opening feature won't work. In this case, you'll need to access the application through a web browser on a machine that can reach the server.

## Notes

- This application uses a CLIP model for multimodal search, optimized for CPU usage.
- The app doesn't display images directly but provides links to the Internet Archive.
- Due to CPU-only optimization, search operations may take longer compared to GPU-accelerated versions.

## Cite

If you need a reference, please cite our paper: 

## License

[Specify your license here]
