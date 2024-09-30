# Machine Learning Prediction of Material Properties for Different Crystal Structures Using Random Forest Regression and XGBoost

Creating a well-structured `README.md` file for your project is crucial for guiding users on how to use, understand, and contribute to the repository. Below is a template that you can customize for your project **"Machine Learning Prediction of Material Properties for Different Crystal Structures Using Random Forest Regression and XGBoost"**.

---

# Machine Learning Prediction of Material Properties for Different Crystal Structures

### Repository for Predicting Material Properties Using Random Forest Regression and XGBoost

![License](https://img.shields.io/badge/license-MIT-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8-blue) ![ML](https://img.shields.io/badge/MachineLearning-RandomForest-orange)

## Table of Contents
- [Project Overview](#project-overview)
- [Crystal Structures](#crystal-structures)
- [Data](#data)
- [Modeling Techniques](#modeling-techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

---

## Project Overview
This project leverages machine learning techniques, specifically **Random Forest Regression** and **XGBoost**, to predict material properties based on various crystal structures. The aim is to provide accurate predictions for key material characteristics, allowing for enhanced material design and discovery.

### Objective:
- **Predict material properties** such as band gap, hardness, thermal properties, and more for different crystal structures using ML models.
- Use **Random Forest** and **XGBoost** algorithms to build models based on input features extracted from crystal structures.

---

## Crystal Structures
The following crystal structures are considered in this project:
1. **Cubic**  
2. **Hexagonal**
3. **Monoclinic**
4. **Orthorhombic**
5. **Tetragonal**
6. **Triclinic**
7. **Trigonal**

Each structure has different physical and chemical characteristics that are captured in the dataset for model training.

---

## Data
The datasets used in this project consist of **27 input features** representing material properties, and the target variable is the **property of interest** (e.g., band gap, hardness). The data was collected from reliable sources or created using computational simulations.

- **Input Features:**  
  Numerical features extracted from each crystal structure, such as atomic composition, lattice parameters, bond lengths, and more.
  
- **Target Variable:**  
  Specific material property (e.g., hardness, band gap) that we aim to predict.

Each crystal structure's dataset is stored in individual Excel files (e.g., `cubic.xlsx`, `hexagonal.xlsx`, etc.).

---

## Modeling Techniques
Two main machine learning models are used for predicting material properties:

### 1. Random Forest Regression
   - **Description**: A powerful ensemble learning method based on decision trees. Random Forests build multiple decision trees and aggregate their predictions for improved accuracy and stability.
   - **Parameters**:  
     - `n_estimators`: 391  
     - `max_depth`: 100  
     - `min_samples_split`: 2  
     - `max_features`: 'sqrt'

### 2. XGBoost
   - **Description**: A highly efficient and scalable gradient-boosted decision tree algorithm. Known for its strong performance on structured/tabular data.
   - **Parameters**:
     - Default hyperparameters with optimized settings for this dataset.

The models are trained and evaluated using **Mean Absolute Percentage Error (MAPE)** for model performance.

---

## Installation

### Prerequisites:
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`

### Installation Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/nithin1024/Prediction-of-Material-Properties-for-Different-Crystal-Structures-.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Prediction-of-Material-Properties-for-Different-Crystal-Structures-
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model:
To train the Random Forest or XGBoost model on the dataset, follow these steps:

1. **Preprocess the data:**
   The data is read from Excel files for each crystal structure, and preprocessed for model training.

2. **Train the models:**
   The script `train.py` provides options to train models on different crystal structures.

   ```bash
   python train.py --model random_forest --data cubic.xlsx
   ```

3. **Make Predictions:**
   Once the model is trained, you can use the `predict.py` script to make predictions on new data.

   ```bash
   python predict.py --model random_forest --input_data new_data.xlsx
   ```

4. **Visualize Results:**
   The project provides scripts to visualize the predicted vs. actual values using `matplotlib`.

   ```bash
   python plot_results.py
   ```

---

## Results
- The **Mean Absolute Percentage Error (MAPE)** achieved by the models is recorded for each crystal structure.
- The predicted vs. actual values are plotted for visual comparison.
- The model performs well across all crystal structures, with **Random Forest** showing slight improvements over **XGBoost** in most cases.

Sample MAPE for cubic structure:
```plaintext
MAPE: 5.23%


## Contributors
- **Nithin Kumar**  
  - Email: nithinkumarbandaru2020@gmail.com
  - GitHub: [nithin1024](https://github.com/nithin1024)


