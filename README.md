# Fashion Recommender System

## Project Overview
The Fashion Recommender System is a content‑based recommendation engine that helps shoppers discover apparel similar to a reference item.  
It converts each product image into a 2,048‑dimensional embedding using a pre‑trained ResNet50 convolutional neural network.  
These embeddings are indexed with a k‑nearest neighbors model so that, given a query image, the system returns the most visually similar products.  
E‑commerce platforms can integrate this approach to show alternatives when an item is out of stock or to surface complementary styles that match a customer's preferences.

The repository contains two main components:
1. **`app.py`** – processes the dataset to generate and store feature embeddings (`embeddings.pkl`) along with image filenames (`filenames.pkl`).
2. **`main.py`** – a Streamlit web interface that loads the stored embeddings, accepts an image upload, and displays the top matches.

## Dataset
The project uses the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle.  
It contains more than 44,000 catalog photos spanning 143 categories and includes metadata such as gender, sub‑category, and base color.  
This project only requires the images, but the metadata can be incorporated for filtering or advanced experiments.

1. Download the dataset from Kaggle (requires a Kaggle account).
2. Unzip the images and place them in an `images/` directory at the root of this repository.
3. The directory should contain files like `images/0001.jpg`, `images/0002.jpg`, etc.

## Features
- Content‑based product recommendations driven by deep CNN features.
- Script to preprocess the dataset and persist embeddings for fast lookups.
- K‑nearest neighbors search to retrieve visually similar items.
- Streamlit web app for uploading an image and viewing recommendations in the browser.
- Modular design—swap in a different backbone or similarity metric with minimal changes.

## Installation
```bash
git clone https://github.com/Pavankumarmanagoli/Fashion-recommender-system.git
cd Fashion-recommender-system
python3 -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. **Generate embeddings** (run once after downloading the dataset).  
   This script scans the `images/` directory, extracts features with ResNet50, and saves `embeddings.pkl` and `filenames.pkl` to disk.
   ```bash
   python app.py
   ```
2. **Launch the Streamlit app**.  
   The app loads the saved embeddings and lets you upload a query image to retrieve the nearest matches.
   ```bash
   streamlit run main.py
   ```
   After the app opens in your browser, upload a fashion image and the system will display similar products from the dataset.

## Example
Below is a minimal example showing how to load the embeddings and retrieve similar items programmatically:
```python
import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

# Load stored data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Build feature extractor
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

query_features = extract_features('path/to/query.jpg')
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
indices = neighbors.kneighbors([query_features])[1]

for idx in indices[0]:
    print(filenames[idx])
```

## Contributing
Contributions are welcome!
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request describing your changes.

## License
This project is licensed under the [MIT License](LICENSE).
