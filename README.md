# Steel-Defect-Detection

**Overview**
This project is an end-to-end steel defect detection system leveraging a pre-trained ResNet152V2 deep learning model. The system identifies different types of defects on steel surfaces, providing bounding boxes and defect classifications.

 It performs two main tasks:
1. **Bounding Box Prediction:** Detects the location of defects on steel surfaces.
2. **Defect Classification:** Classifies the type of defect from multiple categories.

The model is trained using a custom dataset with annotated bounding boxes and defect classes. The dataset includes images and corresponding XML annotation files.

## Project Structure

- `images/`: Directory containing the steel defect images.
- `label/`: Directory containing XML annotation files with bounding box and class information.
- `resnet152v2.h5`: The trained model (not included in this repository).
- `steel_defect_detection.py`: Jupyter notebook for training and saving the trained model.
- `link.txt`: Link to Python script to load the model and perform inference.

## How to Use

1. Clone the repository:
  
   git clone https://github.com/MohiniDeshpande/steel-defect-detection.git
   
   cd steel-defect-detection
 
   pip install -r requirements.txt

3. Train the model or run inference:

To train the model, use the provided Jupyter notebook.

To run inference, use the [colab link](https://colab.research.google.com/drive/1-Q5jQzuoSq1Bd_pj_wWrUNzi5Yvxh71Y?usp=sharing).


## Model Architecture
The model is based on the ResNet152V2 architecture and is pre-trained on ImageNet. The model has been adapted for object detection (bounding box prediction) and classification of defects.

**Layers:**
Base Model: Pre-trained ResNet152V2 (Frozen for initial training).

Bounding Box Prediction: Dense layers predicting xmin, ymin, xmax, ymax.

Defect Classification: Dense layers predicting the defect type.

## Model Performance
The model achieved high accuracy on defect classification and precise bounding box predictions for defect locations.
<img width="666" alt="image" src="https://github.com/user-attachments/assets/029a8b30-8c7e-427f-a9fe-f3f57912a3d2">

