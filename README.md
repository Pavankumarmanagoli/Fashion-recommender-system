# Fashion Recommender System

## Project Overview
The Fashion Recommender System is a machine learning application that suggests visually similar fashion products. It uses a pre-trained ResNet50 model to extract feature embeddings from product images and recommends related items using nearest-neighbor search. This approach can be used to enhance shopping experiences by surfacing alternatives or complementary products.

## Dataset
The project uses the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle.

1. Download the dataset from Kaggle (requires a Kaggle account).
2. Unzip the images and place them in the `images/` directory at the root of this repository.
3. Make sure the directory structure resembles `images/0001.png`, `images/0002.png`, etc.

## Features
- Extracts deep feature embeddings from fashion images using ResNet50.
- Finds similar items with a k-nearest neighbors search.
- Interactive Streamlit web app for uploading an image and viewing recommendations.
- Utility script to generate image embeddings for the dataset.

## Installation
```bash
git clone https://github.com/Pavankumarmanagoli/Fashion-recommender-system.git
cd Fashion-recommender-system
python3 -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. **Generate embeddings** (run once after downloading the dataset):
   ```bash
   python app.py
   ```
2. **Launch the Streamlit app**:
   ```bash
   streamlit run main.py
   ```
   Upload a fashion image and the app will display similar items from the dataset.

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
