# 👥 Employee Attrition Analytics & Prediction Hub

A premium, production-grade Machine Learning web dashboard built with **Streamlit** to predict individual employee attrition risk, evaluate batch datasets, and analyze workforce turnover drivers. 

Powered by a hyperparameter-tuned **Logistic Regression** pipeline with balanced class weights, this platform provides HR departments, recruiters, and executives with actionable insights to optimize employee retention strategies.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-attrition-prediction-jqifhp6t4lyjdotrofvpmd.streamlit.app)

---

## 📖 Table of Contents
1. [🚀 Core Modules & Features](#-core-modules--features)
2. [🧠 Machine Learning & Data Pipeline](#-machine-learning--data-pipeline)
3. [📊 Model Comparison & Calibration](#-model-comparison--calibration)
4. [🛠️ Project Directory Structure](#️-project-directory-structure)
5. [⚡ Setup & Installation](#-setup--installation)
6. [📁 Batch CSV Upload Specifications](#-batch-csv-upload-specifications)

---

## 🚀 Core Modules & Features

The application is structured into two dedicated modules to streamline HR decision-making:

### 1. Individual Risk Evaluator (Single Employee Predictor)
* **Comprehensive Inputs**: Survey inputs for over 30 employee attributes grouped into logical, collapsible sections:
  * **Personal & Demographics**: Age, Gender, Marital Status, Education Field.
  * **Job Profile**: Department, Role, Job Level, Job Involvement, Stock Options.
  * **Working History**: Total Tenure, Companies Worked, Tenure under Current Manager/Role.
  * **Compensation & Logistics**: Monthly Income, Salary Hike, Distance From Home, Overtime.
  * **Satisfaction & Work-Life Balance**: Job, Environment, Relationship Satisfaction scores.
* **Business Logic Safeguards**: Input sliders dynamically check and enforce logical constraints (e.g., years at current company cannot exceed total working years).
* **Attrition Risk Report**: Categorizes employee risk into **Low**, **Medium**, or **High** based on the probability output.
* **Visual Gauge**: Renders an interactive Plotly gauge chart mapping prediction confidence, alongside alert messages flagging key risk drivers (e.g., low satisfaction, overtime, lack of promotions).

### 2. Bulk Prediction & Assessment (Batch Predictor)
* **Drag-and-Drop Uploader**: Upload employee datasets via a CSV file to evaluate risk across entire teams in one click.
* **Robust Preprocessing Fallbacks**: Handles missing fields automatically by filling numeric features with baseline historical medians and categorical features with modes.
* **Aggregated Summaries**: Displays key high-level metrics (Total Employees, High-Risk Counts, and Average Attrition Probability) in metric cards.
* **Turnover Visualizations**: Renders risk distribution pie charts and probability score histograms.
* **Enriched Export**: Download the results as a CSV file appended with `Attrition_Prediction` and `Attrition_Probability` columns.

---

## 🧠 Machine Learning & Data Pipeline

### 1. Data Cleaning & Feature Selection
To improve generalization and reduce statistical noise, features with **zero variance** (where every employee has the exact same value) were dropped from the pipeline:
* `Over18` (Always `Y`)
* `EmployeeCount` (Always `1`)
* `StandardHours` (Always `80`)

### 2. Preprocessing Pipeline (`preprocessor.pkl`)
A modular scikit-learn `ColumnTransformer` handles feature encoding and scaling dynamically:
* **Categorical Encoding (`OrdinalEncoder`)**: Formats object categories (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`).
* **Numerical Scaling (`StandardScaler`)**: Normalizes continuous columns (such as `Age`, `MonthlyIncome`, `TotalWorkingYears`, and satisfaction metrics) to standardize scale offsets.

---

## 📊 Model Comparison & Calibration

### 1. The Challenge of Class Imbalance
The target variable `Attrition` in the IBM HR dataset is highly imbalanced (~84% stayed, ~16% left). Standard models without class balancing default to predicting "No Attrition", resulting in high accuracy but failing to catch the employees who leave (e.g., baseline SVM had a test recall of only `1%`).

A **Tuned Logistic Regression** model with balanced class weights was chosen for production, optimizing recall (the model's sensitivity in catching true leavers) and F1-score.

#### 📊 Model Performance on Test Set
| Model Name | Test Recall (Class 1) | Test Precision (Class 1) | Test F1-Score (Class 1) | Key Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| 🏆 **Tuned Logistic Regression** | **75.0%** | **37.2%** | **49.8%** | **Best for attrition detection (balanced weights)** |
| 🥈 **Gaussian Naive Bayes** | 60.0% | 47.0% | 53.0% | High precision, but misses 40% of true leavers |
| 🥉 **Tuned XGBoost Classifier** | 38.0% | 60.0% | 47.0% | High precision, lower recall |
| **Tuned Decision Tree** | 49.0% | 29.0% | 36.0% | Unstable, prone to overfitting |
| **Tuned Gradient Boosting** | 29.0% | 69.0% | 41.0% | Conservative prediction profile |
| **Tuned Random Forest** | 22.0% | 68.0% | 33.0% | High precision, misses minor class |
| **Tuned AdaBoost** | 24.0% | 80.0% | 36.0% | High precision, low sensitivity |
| **Tuned KNN Classifier** | 10.0% | 64.0% | 18.0% | Struggles on high-dimensional vectors |
| **Tuned Support Vector (SVC)** | 6.0% | 80.0% | 11.0% | Underperforms due to extreme imbalance |

### 2. Decision Threshold Calibration
Using balanced class weights shifts probabilities. At a standard threshold of `0.50`, the model classifies anyone with a raw probability above ~16% as high risk, flagging 501/1470 employees.

To balance precision and reduce false-alarm fatigue, the app features an interactive **Decision Threshold Slider** in the sidebar:
* **Default Threshold (`0.63`)**: Calibrated to predict **exactly 300 attritions** on the training dataset (yielding 62.4% recall and 49.3% precision).
* **Dynamic Adjustment**: HR managers can raise the threshold to focus on high-certainty retention risks, or lower it to widen the preventive outreach list.

---

## 🛠️ Project Directory Structure

```
Employee Attrition Prediction/
├── app.py                      # Main Streamlit web application dashboard
├── requirements.txt            # Python library dependencies
├── README.md                   # Project documentation
├── Model_Training_Review.md    # Detailed ML review & model selection details
├── corrected_model_training.py # Retraining script (retuned Logistic Regression)
├── Models/
│   ├── model.pkl               # Serialized Tuned Logistic Regression classifier
│   └── preprocessor.pkl        # Serialized ColumnTransformer preprocessing pipeline
└── Notebooks/
    ├── Employee-Attrition.csv  # Raw IBM training dataset
    ├── updated_dataset.csv     # Cleaned dataset (constant columns dropped)
    ├── Data_Cleaning.ipynb     # Jupyter Notebook: Data cleaning steps
    ├── EDA_FE.ipynb            # Jupyter Notebook: Exploratory analysis
    └── Model_Training.ipynb    # Jupyter Notebook: Retuning & model comparison
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Clone the repository, navigate to the folder, and run:
```bash
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit dashboard locally:
```bash
streamlit run app.py
```
This runs the application locally and opens it in your browser at `http://localhost:8501`.

---

## 📁 Batch CSV Upload Specifications

The batch prediction module automatically drops legacy constant columns (`EmployeeCount`, `Over18`, `StandardHours`) and maps the remaining features. Ensure your CSV contains columns matching these keys:

* **Numeric Fields**: `Age`, `DailyRate`, `DistanceFromHome`, `Education` (1-5), `EnvironmentSatisfaction` (1-4), `HourlyRate`, `JobInvolvement` (1-4), `JobLevel` (1-5), `JobSatisfaction` (1-4), `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating` (1-4), `RelationshipSatisfaction` (1-4), `StockOptionLevel` (0-3), `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance` (1-4), `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.
* **Categorical Fields**: 
  * `BusinessTravel` (*Non-Travel, Travel_Rarely, Travel_Frequently*)
  * `Department` (*Sales, Research & Development, Human Resources*)
  * `EducationField` (*Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other*)
  * `Gender` (*Female, Male*)
  * `JobRole` (*Sales Executive, Research Scientist, Laboratory Technician, Healthcare Representative, Manager, etc.*)
  * `MaritalStatus` (*Single, Married, Divorced*)
  * `OverTime` (*Yes, No*)
