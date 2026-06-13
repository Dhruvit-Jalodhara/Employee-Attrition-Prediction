import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Employee Attrition Hub",
    layout="wide",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# --- CUSTOM BRAND STYLING ---
st.markdown("""
<style>
    /* Custom style for clean container-cards */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1d3557, #457b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    /* Hover effect for buttons */
    .stButton>button {
        background-color: #1d3557 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #457b9d !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(69, 123, 157, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & PREPROCESSORS ---
@st.cache_resource
def load_model_artifacts(model_path, preprocessor_path, model_mtime, prep_mtime):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

@st.cache_data
def load_historical_data():
    csv_path = "Notebooks/Employee-Attrition.csv"
    if not os.path.exists(csv_path):
        csv_path = "/Users/dhruvitjalodhara/programming/ML Practice/Employee Attrition Prediction/Notebooks/Employee-Attrition.csv"
    return pd.read_csv(csv_path)

# Load artifacts with cache invalidation on file change
model_path = "Models/model.pkl"
preprocessor_path = "Models/preprocessor.pkl"
if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    model_path = "/Users/dhruvitjalodhara/programming/ML Practice/Employee Attrition Prediction/Models/model.pkl"
    preprocessor_path = "/Users/dhruvitjalodhara/programming/ML Practice/Employee Attrition Prediction/Models/preprocessor.pkl"

try:
    model_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
    prep_mtime = os.path.getmtime(preprocessor_path) if os.path.exists(preprocessor_path) else 0
    model, preprocessor = load_model_artifacts(model_path, preprocessor_path, model_mtime, prep_mtime)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading prediction models: {e}")
    model_loaded = False

try:
    df_history = load_historical_data()
    data_loaded = True
except Exception as e:
    st.warning(f"Could not load historical dataset for insights: {e}")
    data_loaded = False

# --- SIDEBAR SETTINGS ---
st.sidebar.markdown("## ⚙️ Model Controls")
st.sidebar.write("Adjust classification sensitivity to control the balance between Recall (catching all risks) and Precision (minimizing false alarms).")

# Risk threshold slider
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.63,
    step=0.01,
    help="Default is 0.63. Adjust this value to calibrate the number of predicted attritions. Higher threshold increases precision; lower threshold increases recall."
)

if data_loaded and model_loaded:
    try:
        @st.cache_data
        def get_historical_predictions():
            X_hist = df_history.drop('Attrition', axis=1, errors='ignore')
            for col in ['EmployeeCount', 'StandardHours', 'Over18']:
                if col in X_hist.columns:
                    X_hist = X_hist.drop(col, axis=1)
            expected_cols = [
                'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
                'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
            ]
            X_hist = X_hist[expected_cols]
            X_proc = preprocessor.transform(X_hist)
            return model.predict_proba(X_proc)[:, 1]
            
        probs_hist = get_historical_predictions()
        y_hist = df_history['Attrition'].map({'Yes': 1, 'No': 0})
        preds_hist = (probs_hist >= threshold).astype(int)
        
        tp = ((preds_hist == 1) & (y_hist == 1)).sum()
        fp = ((preds_hist == 1) & (y_hist == 0)).sum()
        fn = ((preds_hist == 0) & (y_hist == 1)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        st.sidebar.markdown("### 📊 Metrics on Historical Data")
        st.sidebar.markdown(f"**Recall (Sensitivity)**: `{recall:.1%}`  \n*(percentage of actual leavers caught)*")
        st.sidebar.markdown(f"**Precision (Accuracy)**: `{precision:.1%}`  \n*(percentage of predicted leavers who actually left)*")
        st.sidebar.markdown(f"**F1-Score**: `{f1:.3f}`")
        st.sidebar.markdown(f"**Predicted turnover count**: `{preds_hist.sum()}` / `{len(probs_hist)}` ({preds_hist.sum() / len(probs_hist):.1%})")
    except Exception as e:
        pass

# --- APP LAYOUT ---
st.markdown("<h1 class='main-title'>Employee Attrition Analytics Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict risk, analyze key indicators, and optimize retention strategies powered by Machine Learning.</p>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3 = st.tabs(["👥 Single Employee Predictor", "📁 Batch Prediction", "📊 HR Insights Dashboard"])

# --- TAB 1: SINGLE EMPLOYEE PREDICTOR ---
with tab1:
    if not model_loaded:
        st.error("Model artifacts not available. Please verify model file paths.")
    else:
        st.markdown("### Individual Risk Evaluation Form")
        st.write("Fill in the employee details below. The algorithm will evaluate the inputs and predict the probability of the employee leaving the organization.")
        
        # Define department/roles dictionary to maintain professional consistency
        dept_roles = {
            "Sales": ["Sales Executive", "Sales Representative", "Manager"],
            "Research & Development": [
                "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                "Healthcare Representative", "Research Director", "Manager"
            ],
            "Human Resources": ["Human Resources", "Manager"]
        }
        
        # Form structure
        with st.form("single_prediction_form"):
            # We group the fields into logical columns using expanders to avoid overwhelming the user
            
            exp_personal = st.expander("👤 Personal & Demographics", expanded=True)
            with exp_personal:
                col1, col2, col3 = st.columns(3)
                with col1:
                    age = st.slider("Age", min_value=18, max_value=60, value=30)
                    gender = st.radio("Gender", options=["Female", "Male"], horizontal=True)
                with col2:
                    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
                    education = st.selectbox("Education Level", options=[1, 2, 3, 4, 5], 
                                             format_func=lambda x: {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}[x],
                                             index=2)
                with col3:
                    education_field = st.selectbox("Education Field", options=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            
            exp_job = st.expander("💼 Job Profile & Role Details", expanded=True)
            with exp_job:
                col1, col2, col3 = st.columns(3)
                with col1:
                    department = st.selectbox("Department", options=list(dept_roles.keys()))
                    # Dynamic roles based on selected department
                    job_role = st.selectbox("Job Role", options=dept_roles[department])
                with col2:
                    job_level = st.slider("Job Level", min_value=1, max_value=5, value=1)
                    job_involvement = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3,
                                                       format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
                with col3:
                    performance_rating = st.selectbox("Performance Rating", options=[3, 4], 
                                                       format_func=lambda x: {3: "Excellent", 4: "Outstanding"}[x])
                    stock_option_level = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], index=0)
            
            exp_tenure = st.expander("⏳ Working History & Tenure", expanded=True)
            with exp_tenure:
                col1, col2 = st.columns(2)
                with col1:
                    total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=8)
                    # Enforce that YearsAtCompany <= TotalWorkingYears
                    years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=min(4, total_working_years))
                    num_companies_worked = st.slider("Number of Companies Worked", min_value=0, max_value=9, value=1)
                with col2:
                    # Bound sub-tenures by YearsAtCompany
                    years_in_current_role = st.slider("Years in Current Role", min_value=0, max_value=18, value=min(2, years_at_company))
                    years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=min(0, years_at_company))
                    years_with_curr_manager = st.slider("Years with Current Manager", min_value=0, max_value=17, value=min(2, years_at_company))
            
            exp_compensation = st.expander("💵 Compensation & Logistics", expanded=True)
            with exp_compensation:
                col1, col2, col3 = st.columns(3)
                with col1:
                    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
                    monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=14000, step=250)
                with col2:
                    daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=50)
                    hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65, step=5)
                with col3:
                    percent_salary_hike = st.slider("Percent Salary Hike (%)", min_value=11, max_value=25, value=14)
                    overtime = st.radio("Overtime", options=["No", "Yes"], horizontal=True)
                    business_travel = st.selectbox("Business Travel", options=["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
                    distance_from_home = st.slider("Distance From Home (miles)", min_value=1, max_value=29, value=8)

            exp_satisfaction = st.expander("❤️ Satisfaction & Work-Life Balance", expanded=True)
            with exp_satisfaction:
                col1, col2 = st.columns(2)
                with col1:
                    environment_satisfaction = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3,
                                                                 format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
                    job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3,
                                                        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
                with col2:
                    relationship_satisfaction = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3,
                                                                 format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
                    work_life_balance = st.select_slider("Work Life Balance", options=[1, 2, 3, 4], value=3,
                                                         format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x])
            
            # Additional constants & hidden columns required by preprocessor
            employee_number = 9999
            training_times_last_year = 2 # Setting sensible default
            
            # Ensure sanity rule checks (in case sliders were adjusted manually)
            years_at_company = min(years_at_company, total_working_years)
            years_in_current_role = min(years_in_current_role, years_at_company)
            years_since_last_promotion = min(years_since_last_promotion, years_at_company)
            years_with_curr_manager = min(years_with_curr_manager, years_at_company)

            submit_button = st.form_submit_button(label="Evaluate Attrition Risk")
            
        if submit_button:
            # Construct dictionary mapping exact features
            feature_dict = {
                "Age": age,
                "BusinessTravel": business_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance_from_home,
                "Education": education,
                "EducationField": education_field,
                "EmployeeNumber": employee_number,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies_worked,
                "OverTime": overtime,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StockOptionLevel": stock_option_level,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times_last_year,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_last_promotion,
                "YearsWithCurrManager": years_with_curr_manager
            }
            
            # Convert to DataFrame matching model columns exactly
            input_df = pd.DataFrame([feature_dict])
            
            # Predict
            try:
                # Transform inputs
                processed_inputs = preprocessor.transform(input_df)
                
                # Model Predict
                prob = model.predict_proba(processed_inputs)[0]
                pred = 1 if prob[1] >= threshold else 0
                
                # Show results in nice metrics card
                st.markdown("---")
                st.markdown("### Risk Analysis Report")
                
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    attrition_prob_pct = prob[1] * 100
                    
                    if attrition_prob_pct >= threshold * 100:
                        st.error(f"🔴 **HIGH RISK** of Attrition detected.")
                        risk_level = "High"
                    elif attrition_prob_pct >= max(10.0, (threshold - 0.20) * 100):
                        st.warning(f"🟡 **MEDIUM RISK** of Attrition detected.")
                        risk_level = "Medium"
                    else:
                        st.success(f"🟢 **LOW RISK** of Attrition detected.")
                        risk_level = "Low"
                        
                    st.write(f"The model estimates a **{attrition_prob_pct:.1f}%** probability that the employee will leave the company.")
                    
                    # Highlight critical risk indicators based on input values
                    st.markdown("##### Key Indicators for this Employee:")
                    indicators = []
                    if overtime == "Yes":
                        indicators.append("⚠️ **Overtime**: Working overtime is highly correlated with attrition.")
                    if job_satisfaction <= 2:
                        indicators.append("⚠️ **Low Job Satisfaction**: Employee has reported low job satisfaction.")
                    if environment_satisfaction <= 2:
                        indicators.append("⚠️ **Low Environment Satisfaction**: Workplace environment satisfaction is low.")
                    if monthly_income < 4000:
                        indicators.append("⚠️ **Below Average Compensation**: Monthly income is lower than typical thresholds.")
                    if work_life_balance <= 1:
                        indicators.append("⚠️ **Poor Work-Life Balance**: Reported work-life balance rating is poor.")
                    if years_since_last_promotion >= 5:
                        indicators.append("⚠️ **Promotion Stagnancy**: No promotion received in 5 or more years.")
                        
                    if len(indicators) > 0:
                        for ind in indicators:
                            st.write(ind)
                    else:
                        st.write("✓ No standard high-risk flags identified in survey scores.")
                        
                with col_res2:
                    # Create Plotly gauge chart for premium look
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = attrition_prob_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Attrition Probability (%)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#1d3557"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#d8f3dc'},
                                {'range': [30, 50], 'color': '#ffe3e3'},
                                {'range': [50, 100], 'color': '#ffccd5'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': attrition_prob_pct
                            }
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.markdown("### Bulk Prediction & Batch Assessment")
    st.write("Upload a CSV file containing employee records to predict attrition probability in bulk. The CSV should ideally match the structure of standard HR records.")
    
    # Template download help
    st.info("💡 **Formatting tip**: The file must contain columns matching standard feature names. Missing constant columns (like StandardHours) will be automatically filled.")
    
    uploaded_file = st.file_uploader("Upload employee CSV", type=["csv"])
    
    if uploaded_file is not None and model_loaded:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Validate columns
            req_cols = [
                'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
                'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
                'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
            ]
            
            missing_cols = [col for col in req_cols if col not in batch_df.columns]
            
            if len(missing_cols) > 0:
                st.warning(f"⚠️ Missing columns: {missing_cols}. They will be populated with default/mean values.")
                for col in missing_cols:
                    if col in ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
                              'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 
                              'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                              'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                              'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
                        # Fill numeric fields with mean or standard values
                        batch_df[col] = df_history[col].median() if data_loaded else 3
                    elif col == 'Gender':
                        batch_df[col] = 'Male'
                    elif col in ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']:
                        batch_df[col] = 'Other' if col == 'EducationField' else (df_history[col].mode()[0] if data_loaded else 'Sales')
                    elif col == 'OverTime':
                        batch_df[col] = 'No'
            
            # Fill mandatory constants
            if 'EmployeeNumber' not in batch_df.columns:
                batch_df['EmployeeNumber'] = np.arange(1, len(batch_df) + 1)
                
            # Align column ordering exactly as expected
            final_cols = [
                'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
                'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
            ]
            
            eval_df = batch_df[final_cols].copy()
            
            # Run Preprocessor and Predict
            processed_batch = preprocessor.transform(eval_df)
            probs = model.predict_proba(processed_batch)
            preds = np.where(probs[:, 1] >= threshold, 1, 0)
            
            # Add results to batch DataFrame
            batch_df['Attrition_Prediction'] = np.where(preds == 1, 'Yes', 'No')
            batch_df['Attrition_Probability'] = probs[:, 1]
            
            # Display stats
            num_high_risk = (batch_df['Attrition_Probability'] >= threshold).sum()
            num_med_risk = ((batch_df['Attrition_Probability'] >= max(0.10, threshold - 0.20)) & (batch_df['Attrition_Probability'] < threshold)).sum()
            num_low_risk = (batch_df['Attrition_Probability'] < max(0.10, threshold - 0.20)).sum()
            
            st.markdown("### Batch Prediction Results")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Total Employees Processed", len(batch_df))
            with col_b2:
                st.metric(f"High Risk Attrition (Prob >= {int(threshold*100)}%)", num_high_risk, delta=int(num_high_risk), delta_color="inverse")
            with col_b3:
                st.metric("Average Attrition Probability", f"{batch_df['Attrition_Probability'].mean()*100:.1f}%")
                
            # Render a summary layout
            col_chart1, col_chart2 = st.columns([1, 1])
            with col_chart1:
                # Pie chart of risk profiles
                risk_df = pd.DataFrame({
                    "Risk Level": [
                        f"Low Risk (<{int(max(0.10, threshold - 0.20)*100)}%)", 
                        f"Medium Risk ({int(max(0.10, threshold - 0.20)*100)}-{int(threshold*100)}%)", 
                        f"High Risk (>={int(threshold*100)}%)"
                    ],
                    "Count": [num_low_risk, num_med_risk, num_high_risk]
                })
                fig_pie = px.pie(risk_df, names="Risk Level", values="Count", 
                                 title="Distribution of Attrition Risk Profiles",
                                 color_discrete_sequence=['#d8f3dc', '#ffe3e3', '#ffccd5'])
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                # Attrition probability distribution chart
                fig_hist = px.histogram(batch_df, x="Attrition_Probability", nbins=20,
                                        title="Distribution of Attrition Probability Scores",
                                        labels={"Attrition_Probability": "Attrition Probability Score"},
                                        color_discrete_sequence=['#1d3557'])
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Display Table of Results
            st.markdown("#### Detailed Risk Roster")
            show_cols = ['EmployeeNumber', 'Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'Attrition_Probability', 'Attrition_Prediction']
            # Find which fields are actually in the source file
            valid_show_cols = [c for c in show_cols if c in batch_df.columns]
            st.dataframe(batch_df[valid_show_cols].sort_values(by='Attrition_Probability', ascending=False).style.format({
                'Attrition_Probability': '{:.1%}'
            }))
            
            # Download results button
            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV File",
                data=csv_export,
                file_name="employee_attrition_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Failed to process batch CSV: {e}")

# --- TAB 3: HR INSIGHTS DASHBOARD ---
with tab3:
    if not data_loaded:
        st.warning("Historical dataset not loaded. Insights dashboard is unavailable.")
    else:
        st.markdown("### Historical HR Attrition Exploratory Dashboard")
        st.write("Analyze patterns and drivers of attrition based on standard employee database records.")
        
        # Summary KPI metrics
        total_records = len(df_history)
        attrition_count = (df_history['Attrition'] == 'Yes').sum()
        overall_attrition_rate = attrition_count / total_records * 100
        avg_income = df_history['MonthlyIncome'].mean()
        avg_age = df_history['Age'].mean()
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("Total Records Analyzed", total_records)
        with col_kpi2:
            st.metric("Overall Attrition Rate", f"{overall_attrition_rate:.1f}%")
        with col_kpi3:
            st.metric("Average Monthly Salary", f"${avg_income:,.2f}")
        with col_kpi4:
            st.metric("Average Employee Age", f"{avg_age:.1f} Years")
            
        st.markdown("---")
        
        # Row 1: Charts
        col_row1_1, col_row1_2 = st.columns(2)
        
        with col_row1_1:
            # Attrition by Department
            dept_attrition = df_history.groupby('Department')['Attrition'].value_counts(normalize=True).unstack() * 100
            dept_attrition = dept_attrition.reset_index()
            fig_dept = px.bar(dept_attrition, x="Department", y="Yes", 
                              title="Attrition Rate by Department (%)",
                              labels={"Yes": "Attrition Rate (%)"},
                              color_discrete_sequence=['#e63946'])
            st.plotly_chart(fig_dept, use_container_width=True)
            
        with col_row1_2:
            # Attrition by Marital Status & Gender
            marital_attrition = df_history.groupby(['MaritalStatus', 'Attrition']).size().reset_index(name='Count')
            fig_marital = px.bar(marital_attrition, x="MaritalStatus", y="Count", color="Attrition",
                                 title="Attrition counts by Marital Status",
                                 color_discrete_map={'Yes': '#e63946', 'No': '#1d3557'},
                                 barmode="group")
            st.plotly_chart(fig_marital, use_container_width=True)
            
        # Row 2: Monthly Income vs Overtime & Attrition
        col_row2_1, col_row2_2 = st.columns(2)
        
        with col_row2_1:
            # Overtime vs Attrition
            ot_attrition = df_history.groupby('OverTime')['Attrition'].value_counts(normalize=True).unstack() * 100
            ot_attrition = ot_attrition.reset_index()
            fig_ot = px.bar(ot_attrition, x="OverTime", y="Yes",
                            title="Attrition Rate: Overtime vs No Overtime (%)",
                            labels={"Yes": "Attrition Rate (%)"},
                            color_discrete_sequence=['#e63946'])
            st.plotly_chart(fig_ot, use_container_width=True)
            
        with col_row2_2:
            # Box plot for Monthly Income distribution by Attrition
            fig_income = px.box(df_history, x="Attrition", y="MonthlyIncome", color="Attrition",
                                title="Monthly Income Distribution: Staid vs Left",
                                color_discrete_map={'Yes': '#e63946', 'No': '#1d3557'})
            st.plotly_chart(fig_income, use_container_width=True)

        # Row 3: Satisfaction score correlations
        st.markdown("#### Attrition Rates by Satisfaction Scores & Environment")
        col_row3_1, col_row3_2, col_row3_3 = st.columns(3)
        
        with col_row3_1:
            job_sat_attr = df_history.groupby('JobSatisfaction')['Attrition'].value_counts(normalize=True).unstack() * 100
            job_sat_attr = job_sat_attr.reset_index()
            fig_sat1 = px.bar(job_sat_attr, x="JobSatisfaction", y="Yes",
                              title="Attrition vs Job Satisfaction Score",
                              labels={"Yes": "Attrition Rate (%)", "JobSatisfaction": "Satisfaction (1-Low, 4-High)"},
                              color_discrete_sequence=['#457b9d'])
            st.plotly_chart(fig_sat1, use_container_width=True)
            
        with col_row3_2:
            env_sat_attr = df_history.groupby('EnvironmentSatisfaction')['Attrition'].value_counts(normalize=True).unstack() * 100
            env_sat_attr = env_sat_attr.reset_index()
            fig_sat2 = px.bar(env_sat_attr, x="EnvironmentSatisfaction", y="Yes",
                              title="Attrition vs Environment Satisfaction",
                              labels={"Yes": "Attrition Rate (%)", "EnvironmentSatisfaction": "Satisfaction (1-Low, 4-High)"},
                              color_discrete_sequence=['#457b9d'])
            st.plotly_chart(fig_sat2, use_container_width=True)
            
        with col_row3_3:
            wlb_attr = df_history.groupby('WorkLifeBalance')['Attrition'].value_counts(normalize=True).unstack() * 100
            wlb_attr = wlb_attr.reset_index()
            fig_sat3 = px.bar(wlb_attr, x="WorkLifeBalance", y="Yes",
                              title="Attrition vs Work-Life Balance",
                              labels={"Yes": "Attrition Rate (%)", "WorkLifeBalance": "WLB Score (1-Poor, 4-Best)"},
                              color_discrete_sequence=['#457b9d'])
            st.plotly_chart(fig_sat3, use_container_width=True)
