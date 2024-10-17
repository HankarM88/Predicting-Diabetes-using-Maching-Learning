# Diabetes Prediction using Machine Learning

This project is focused on predicting the likelihood of a patient having diabetes using several machine learning models. The dataset used in this project contains various health measurements such as glucose level, blood pressure, BMI, etc., collected from patients. Machine learning models are built to classify whether a patient has diabetes or not.

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Dataset
The dataset used in this project is the **Pima Indians Diabetes Database**, which is available on Kaggle:  
[Diabetes Dataset on Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

### Features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic history)
- **Age**: Age (years)
- **Outcome**: Class variable (0 if non-diabetic, 1 if diabetic)

## Project Workflow
1. **Data Preprocessing**:
   - Handle missing values by imputing or removing rows
   - Scale the data for models that require normalization
   - Split the data into training and test sets

2. **Model Selection**:
   - Train multiple machine learning models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K Nearest Neighbors (KNN)
     - Decision Tree
     - Random Forest
   - Evaluate the models using metrics like accuracy, precision, recall, and F1 score.

3. **Model Evaluation**:
   - Compare the performance of different models to select the best one for predicting diabetes.

4. **Final Model Deployment**:
   - Save the trained model and make it ready for predictions on new data.

## Installation

To run this project locally, you'll need the following Python libraries:
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib** (for visualization)
- **Seaborn** (for more advanced plots)

Install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
## Usage 
1. Clone the Repository
```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd Predicting-Diabetes-with-Maching-and-Deep-learning
```
2. Customize the jupyter notebook
You can edit the script to experiment with different machine learning models or change the hyperparameters for better performance.
