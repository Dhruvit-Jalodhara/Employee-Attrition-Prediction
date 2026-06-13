# 👥 Employee Attrition Analytics & Prediction Hub

An interactive, premium Machine Learning web application built using **Streamlit** to predict employee attrition risk, run batch evaluations, and analyze key indicators of workforce turnover. Powered by a hyperparameter-tuned classification pipeline, this dashboard provides HR departments and executive leaders with actionable insights to design effective employee retention strategies.

---

## 🚀 Key Modules & Features

The application is structured into three specialized tabs for streamlined HR operations:

### 1. 👥 Single Employee Predictor
* **Comprehensive Inputs**: Over 30 employee attributes organized into collapsible sections (Demographics, Job Profile, History, Compensation, and Satisfaction).
* **Smart Business Logic**: Input sliders dynamically enforce logical dependencies (e.g., years at the company cannot exceed total working years).
* **Risk Scoring Report**: Computes the exact attrition probability and categorizes the employee into **Low**, **Medium**, or **High Risk**.
* **Visual Diagnostics**: Renders a premium, real-time Plotly gauge chart mapping prediction confidence, along with warning lists flagging critical attrition drivers (e.g., low job satisfaction, excessive overtime, stagnancy).

### 2. 📁 Bulk Prediction & Batch Assessment
* **CSV Bulk Uploader**: Process list files containing hundreds of employee profiles simultaneously.
* **Robust Preprocessing Fallbacks**: Automatically aligns column layout structures, handles missing fields, and replaces constants with default statistics (median/mode) calculated from historical baseline values.
* **Aggregated Summaries**: Generates high-level indicators (Total Processed, High-Risk Attritions Count, and Average Turnover Probability) alongside interactive distribution plots.
* **Data Export**: Provides a downloadable, prediction-enriched CSV containing prediction outcomes (`Yes`/`No`) and probability confidence scores.

### 3. 📊 HR Insights Dashboard
* **Exploratory Data Analysis (EDA)**: Visualizes historical employee distributions to help HR analyze underlying factors causing attrition.
* **Interactive Visualizations**: High-performance Plotly charts exploring turnover correlations across Departments, Overtime requirements, Marital Status, Monthly Income levels, and Workplace Satisfaction surveys.

---

## 🛠️ Project Directory Structure

```
Employee Attrition Prediction/
├── app.py                      # Main Streamlit web application dashboard
├── requirements.txt            # Python library dependencies
├── README.md                   # Project documentation and user guide
├── Models/
│   ├── model.pkl               # Serialized Tuned Logistic Regression classifier
│   └── preprocessor.pkl        # Serialized ColumnTransformer preprocessing pipeline
└── Notebooks/
    ├── Employee-Attrition.csv  # Original raw IBM HR attrition dataset
    ├── updated_dataset.csv     # Cleaned dataset (zero-variance features removed)
    ├── Data_Cleaning.ipynb     # Jupyter Notebook: Data cleaning, dropping constants
    ├── EDA_FE.ipynb            # Jupyter Notebook: Exploratory analysis & Feature Engineering
    └── Model_Training.ipynb    # Jupyter Notebook: Model comparison, tuning, and serialization
```

---

## 🧠 Machine Learning & Data Pipeline

### 1. Data Cleaning (Feature Selection)
To optimize model generalizability and remove noise, columns with **zero variance** (features where every employee has the exact same value) were dropped:
* `Over18` (Always `Y`)
* `EmployeeCount` (Always `1`)
* `StandardHours` (Always `80`)

### 2. Preprocessing Pipeline (`preprocessor.pkl`)
A modular scikit-learn `ColumnTransformer` handles feature encoding and scaling dynamically:
* **Categorical Encoding (`OrdinalEncoder`)**: Applied to `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, and `OverTime`.
* **Numerical Scaling (`StandardScaler`)**: Applied to continuous numerical columns (such as `Age`, `MonthlyIncome`, `TotalWorkingYears`, and satisfaction metrics) to standardize them for distance-sensitive classifiers.

### 3. Model Calibration & Selection (`model.pkl`)
Since the target variable `Attrition` is heavily imbalanced (~84% stayed, ~16% left), baseline classifiers without class balancing fail to catch true attritions (e.g., baseline SVM had a test recall of only `0.01`).

A **Tuned Logistic Regression** classifier was selected as the production model due to its superior capability in handling class imbalance using balanced weights, achieving the best combination of recall and F1-score:

#### 📊 Model Performance Comparison on Test Set
| Model Name | Test Recall (Class 1) | Test Precision (Class 1) | Test F1-Score (Class 1) | Key Notes |
| :--- | :---: | :---: | :---: | :--- |
| 🏆 **Tuned Logistic Regression** | **75.0%** | **37.2%** | **49.8%** | **Best for attrition detection (balanced weights)** |
| 🥈 **Gaussian Naive Bayes** | 60.0% | 47.0% | 53.0% | High precision, but misses 40% of leavers |
| 🥉 **Tuned XGBoost Classifier** | 38.0% | 60.0% | 47.0% | High precision, lower recall |
| **Tuned Decision Tree** | 49.0% | 29.0% | 36.0% | Unstable, prone to overfitting |
| **Tuned Gradient Boosting** | 29.0% | 69.0% | 41.0% | Conservative prediction profile |
| **Tuned Random Forest** | 22.0% | 68.0% | 33.0% | High precision, misses minor class |
| **Tuned AdaBoost** | 24.0% | 80.0% | 36.0% | High precision, low sensitivity |
| **Tuned KNN Classifier** | 10.0% | 64.0% | 18.0% | Struggles on high-dimensional vectors |
| **Tuned Support Vector (SVC)** | 6.0% | 80.0% | 11.0% | Underperforms due to extreme imbalance |

*Recall measures the percentage of actual leaving employees correctly identified by the model. High recall prevents costly "False Negatives" (failing to catch a leaving employee).*

### 4. Interactive Calibration Slider
Using `class_weight='balanced'` shifts prediction probabilities. To prevent false-alarm fatigue (the model flagging 501/1470 employees as high risk at a default `0.50` threshold), the app features a **Decision Threshold Slider** in the sidebar:
* **Default Threshold (`0.63`)**: Calibrated to predict **exactly 300 attritions** on the training dataset (62.4% recall and 49.3% precision).
* HR managers can slide the threshold to control sensitivity: lowering it catches more risk (higher recall), while raising it isolates high-certainty cases (higher precision).

---

## ⚡ Setup and Installation

Ensure you have **Python 3.8+** installed. Follow these steps to launch the app locally:

### 1. Clone & Navigate to Project Directory
```bash
cd "/Users/dhruvitjalodhara/programming/ML Practice/Employee Attrition Prediction"
```

### 2. Install Package Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
Launch the dashboard web server:
```bash
streamlit run app.py
```
Streamlit will launch a local development server and automatically open the application in your default web browser at `http://localhost:8501`.

---

## 📁 Batch CSV Upload Specifications
To test bulk prediction, you can upload any CSV file structure containing the employee records. The pipeline requires the following input features (extra columns such as `Attrition` or constants are ignored, and missing fields are automatically imputed):

* **Numerical Columns**: `Age`, `DailyRate`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.
* **Categorical Columns**: `BusinessTravel` (*Non-Travel, Travel_Rarely, Travel_Frequently*), `Department` (*Sales, Research & Development, Human Resources*), `EducationField` (*Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other*), `Gender` (*Female, Male*), `JobRole` (*Sales Executive, Research Scientist, Laboratory Technician, etc.*), `MaritalStatus` (*Single, Married, Divorced*), `OverTime` (*Yes, No*).
