# Plant Doctor: AI for Healthy Crops 🌿
This project is a high-tech "plant doctor" that helps farmers protect their crops from disease. By using deep learning, we can automatically analyze a photo of a plant leaf and tell you if it's healthy or sick, and even what disease it has! This early detection system helps farmers take action fast, saving their harvest and hard-earned money.

# 🚀 Why We Built This
Empowering Farmers: We want to give farmers a powerful, easy-to-use tool right in their hands. Traditional methods of disease diagnosis are slow and can be costly, but our system provides an instant, accurate answer.

Breaking Language Barriers: Our project supports multiple languages! This is crucial because not all farmers speak the same language. We want this technology to be accessible to everyone, no matter where they are or what language they speak.

Cutting-Edge Tech: We used a Convolutional Neural Network (CNN), the gold standard for image classification, to create a smart, accurate model that can classify various plant leaf diseases.

# 🛠️ Get It Running (For the Tech-Savvy!)
Ready to try it out? Here’s how to get this project up and running on your machine.

## Prerequisites
Python 3.8+

pip

A strong internet connection (for downloading the model and dataset)

## Installation
### 1. Clone the repository:

Bash

git clone https://github.com/Silverfang180/Crop_Disease_Detection-/tree/main.git
- cd your-repo-name

### 2. Set up a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

### 3. Install dependencies:

Bash

pip install -r requirements.txt

# 💻 How to Use It
## Classify a Leaf Image
Simply run the main script with the path to your image:

Bash

python predict.py --image_path "path/to/your/leaf_image.jpg"
The system will output the predicted class (e.g., "Healthy," "Early Blight," "Late Blight") and a confidence score.

## Supported Languages
Our app provides output in several languages to serve a global audience. You can specify the desired language with a simple command-line flag.

Bash

python predict.py --image_path "path/to/your/leaf_image.jpg" --lang "es"
(Currently supports: en for English, es for Spanish, hi for Hindi, etc.)

# 🧠 The Magic Behind the Scenes
The core of this project is a CNN model that has been trained on a massive dataset of healthy and diseased plant leaf images.

Here's how it works:

1. Image Input: You provide a clear image of a plant leaf.

2. Convolutional Layers: The CNN uses a series of filters to break down the image and identify key features like texture, color, and patterns of disease.

3. Pooling Layers: These layers compress the data, making the model more efficient and robust to slight variations in the image (like different angles or lighting).

4. Classification: The final layers take all the learned features and classify the leaf into one of the trained categories (e.g., a specific disease or "healthy").

5. This process allows the model to learn the subtle visual clues that distinguish a healthy leaf from a diseased one, just like a trained botanist would.

# 🤝 Join Our Mission
We believe this technology can make a real difference. If you're a developer, a data scientist, or just someone passionate about sustainable agriculture, we'd love for you to contribute! You can:

Add more languages.

Improve the model's accuracy.

Suggest new features.

Feel free to open an issue or submit a pull request!

# 📄 License
This project is open-source and licensed under the MIT License. Use it, learn from it, and help us build a healthier future for our planet's crops.







