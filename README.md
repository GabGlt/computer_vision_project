# Final Project for Computer Vision Course

This repository contains the code for a computer vision project on **detecting and classifying poultry feces images to support early disease detection in poultry farming** using:

- **Support Vector Machine (SVM)** as the classification model  
- **Image processing and texture-based feature extraction**  
- **Multiclass classification** with the following labels:  
  - `0` = Coccidiosis  
  - `1` = Healthy  
  - `2` = Salmonella  

## Features

The application provides:

1. **Feces Disease detection and classification** – choose and image from your devicees and get immediate detection and prediction  
2. **Prediction confidence** – see probability scores for each class  
3. **Clean and simple UI** – easy to interact with using Streamlit 

## Installing Dependencies
    pip install -r requirements.txt

## Usage
### 1. Fine-tune the Model
Open the training notebook:
```
train.ipnyb
```
Run all cells to perform feature extraction, model training, and evaluation.

### 2. Run the Streamlit App Locally
To run the app on your local machine, follow these steps:

1. **Activate your Python environment (adjust the path to your own environment)**:

```bash
& "PATH/TO/YOUR/ENV/Scripts/Activate.ps1"
```

2. **Run the Streamlit application**:

```bash
streamlit run app.py
```

3. **Open the application in your browser**:

After running the above command, Streamlit will display a local URL in the terminal, usually:
```bash
http://localhost:8501
```
Open this URL in your browser to start interacting with the app.


### 3. Access the Deployed App
The app is also deployed online and can be accessed here:
```
🔗 https://poultry-feces-classifier.streamlit.app/
```
