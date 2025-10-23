from utils.libraries import st
from utils.explore_page import show_explore_page
from utils.model_page import compare_model_page
from utils.predict_page import show_predict_page
from utils.functions import get_approved_choices, run_notebook_from_github, display_notebook_results

import pandas as pd

@st.cache_data(show_spinner=False)
def load_job_dataset_from_github(url: str):
    try:
        df = pd.read_csv(url)
        st.success("Dataset loaded successfully from GitHub.")
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

# If working, place your name here, and add other TO DOs for others
# IVY - Add Randoms Bits Forest

# Let's make the incoming parameters optional. Write them to the UI for now.
# We could select dropdowns based on the incoming args.
# index= is used to select dropdowns below.
# Let's match the short arg "bees" to a name "Honey Bees" rather than hardcoing an index value.
# The name matching could reside in a .csv file that defines a long list of data sources.

# import os
# import sys
# import random
# import argparse

# parser = argparse.ArgumentParser(description='Pull in 3 parameters: feature, target, and models')

# parser.add_argument('features', type=str, help="github path to the feature category")
# sort_order_choices = ('up', 'down', 'random')
# parser.add_argument('targets', type=str, help="target dataset")
# parser.add_argument('models', type=str, help="matching learning model type")

# try:
#     args = parser.parse_args()
# except SystemExit as e:
#     # This exception will be raised if --help or invalid command line arguments
#     # are used. Currently streamlit prevents the program from exiting normally
#     # so we have to do a hard exit.
#     os._exit(e.code)

#st.write("[Model.Earth](https://model.earth)")

# Should work, but doesn't
st.write(f'<link type="text/css" rel="stylesheet" href="https://model.earth/localsite/css/base.css" id="/localsite/css/base.css" /><script type="text/javascript" src="https://model.earth/localsite/js/localsite.js?showheader=true"></script>', unsafe_allow_html=True)

# st.experimental_get_query_params("test")

features = st.sidebar.selectbox("Features", ("Local Industries", "Local Places", "Local Products", "Job Descriptions", "Brain Voxels"), index=3)
targets = st.sidebar.selectbox("Target", ("Honey Bees", "Job Growth", "Wage Growth", "High Wages", "Real Job Listings", "Tree Canopy", "Eye Blinks"), index=4)

GITHUB_RAW_URL = "https://raw.githubusercontent.com/ayushmunjial/reality.streamlit.large/main/input/jobs/fake_job_postings.csv"

if targets == "Real Job Listings":
    df_jobs = load_job_dataset_from_github(GITHUB_RAW_URL)
    if not df_jobs.empty:
        st.subheader("Preview of 'Fake vs Real Job Listings' dataset")
        # st.dataframe(df_jobs.head())
        st.dataframe(df_jobs)
        st.write("Total rows:", len(df_jobs))

MODEL_CODE = {
    "Logistic Regression (lr)": "lr",
    "Random Forest Classifier (rfc)": "rfc",
    "Random Bits Forest (rbf)": "rbf",
    "Support Vector Machines (svm) - Runs Slow": "svm",
    "Neural Network Multi-Layer Perceptron (mlp)": "mlp",
    "XGBoost (xgboost)": "xgboost",
}

models = st.sidebar.selectbox("Model", list(MODEL_CODE.keys()), index=0)
selected_code = MODEL_CODE[models]



with st.spinner(f"Executing {models}..."):
    parameters = {
        "model_type": selected_code,  # 👈 this is key
        "test_size": 0.3,
        "max_features": 500,
        "random_state": 42,
        "oversample": True
    }
    success, output_path, error = run_notebook_from_github("Run-Models-bkup.ipynb", parameters)


# Add execution logic that works with existing "Run" buttons in pages
if 'execute_selected_model' not in st.session_state:
    st.session_state.execute_selected_model = False

# This will be triggered by existing "Run" buttons in the model pages
if st.session_state.get('execute_selected_model', False):
    st.session_state.execute_selected_model = False  # Reset flag
    
    # Execute the selected model
    parameters = {
        "test_size": 0.3,
        "max_features": 500,
        "random_state": 42,
        "oversample": True
    }
    
    with st.spinner(f"Executing {models}..."):
        parameters["model_type"] = selected_code  # 👈 pass correct short code
        success, output_path, error = run_notebook_from_github("Run-Models-bkup.ipynb", parameters)
        
        if success:
            st.success(f"✅ {models} executed successfully!")
            display_notebook_results(output_path)
        else:
            st.error(f"❌ Execution failed: {error}")

