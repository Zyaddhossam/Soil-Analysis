# Soil Analysis 🌱🖼️

This project features two AI models: one for classifying the type of soil, and another for determining whether the soil is suitable for planting.

## Description

This project aims to help farmers and support the agricultural industry through the use of two AI models. The first model classifies the type of soil based on a small image of the sample, while the second evaluates whether the identified soil is suitable for planting by analyzing the proportions of certain components, such as manganese.
This system can be especially useful for farmers, agronomists, and agricultural researchers looking to make informed, data-driven planting decisions.

## Features
- Predicts soil type from uploaded images
- Returns probabilities for each soil class
- Evaluates whether the identified soil is suitable for planting
- Suggests the most suitable types of plants for the given soil (future Work)

### Model Details

#### 1. Soil Classification Model
- Input: Image of soil
- Output: Soil type
- Classes: Clay Soil, Alluvial Soil, Red Soil, Black Soil
- Description: This model analyzes a small image of the soil and classifies it into one of the predefined soil types. It is useful for quickly identifying soil type without manual testing.

#### 2. Soil Suitability Model
- Input: Numeric values for soil components (e.g., manganese, nitrogen)
- Output: Soil suitability For Planting
- Classes: Very Suitable, Suitable, Not Suitable
- Description: This model evaluates whether the soil is suitable for planting based on the proportions of key components. It helps determine the best use of the soil based on its nutrient composition.

### Technologies Used
- **FastAPI** – For building the web application  
- **Uvicorn** – For serving the FastAPI app  
- **Scikit-learn** – For implementing machine learning models  
- **Pandas** – For data manipulation and analysis  
- **NumPy** – For numerical operations  
- **Joblib** – For model serialization  
- **Pydantic** – For data validation and parsing  
- **Matplotlib** – For visualizing data  
- **Seaborn** – For statistical data visualization  
- **TensorFlow** – For building and training the deep learning model  
- **Microsoft Azure** – For cloud services and deployment  

### Project structure
```
Soil-Analysis
├── Deployment
│   └── API.py #code for API 
│   └── Model.h5 #The Soil Type trained model
│   └── random_forest_model.joblib #Soil Suitability trained Model
│   └── Requirments.txt #Requirements for the model
├── Soil-Suitability-Model
│   └── model.py #the code for the model itself (random_forest_model.joblib)
│   └── dataset.csv #dataset for the model   
│   └── Class_names #for class names in the model
├── Soil-Types 
│   └── Soil.ipynb #code for Model itself (Model.h5)
│   └── Class_names #for class names in the model
│   └── Cloud
│          └── Resource_Group #image for resource group on microsoft azure
│          └── Virtual Machine Inbound #image for VM inbound traffic on microsoft azure
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/Zyaddhossam/Soil_Model.git
cd Soil-Analysis/Deployment/
pip install -r requirements.txt
```
