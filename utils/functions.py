from utils.libraries import *
import streamlit as st
import requests
import base64
import time
import json
import random
import papermill as pm
import tempfile
import os

# Creating a Dataframe with word-vectors in TF-IDF form and Target values

github_token = st.secrets["GITHUB_TOKEN"]

def final_df(df, is_train, vectorizer, column):

    # TF-IDF form
    if is_train:
        x = vectorizer.fit_transform(df.loc[:,column])
    else:
        x = vectorizer.transform(df.loc[:,column])

    # TF-IDF form to Dataframe
    temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

    # Droping the text column
    df.drop(df.loc[:,column].name, axis = 1, inplace=True)

    # Returning TF-IDF form with target
    return pd.concat([temp, df], axis=1)


# Training the model with various combination and returns y_test and y_pred

def train_model(df, input, target, test_size, over_sample, vectorizer, model):

    X = df.drop(target, axis=1)
    y = df[target]
    print("Splitted Data into X and Y.")

    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Splitted Data into Train and Test.")
    
    # Training Preprocessing
    X_train = final_df(X_train, True, vectorizer, input)
    X_train.dropna(inplace=True)
    print("Vectorized Training Data.")

    if over_sample:
        sm = SMOTE(random_state = 2)
        X_train, Y_train = sm.fit_resample(X_train, Y_train.ravel())
        print("Oversampling Done for Training Data.")

    # Testing Preprocessing
    x_test = final_df(x_test, False, vectorizer, input)
    x_test.dropna(inplace=True)
    print("Vectorized Testing Data.")

    # fitting the model
    model = model.fit(X_train, Y_train)
    print("Model Fitted Successfully.")

    # calculating y_pred
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)

    return model, x_test, y_test, y_pred_prob

def evaluate(y_test, y_pred, y_pred_prob):
    roc_auc = round(roc_auc_score(y_test, y_pred_prob[:, 1]), 2)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)
    
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    
    # locate the index of the largest g-mean
    ix = argmax(gmeans)

    y_pred = (y_pred > thresholds[ix])

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"*ROC-AUC Score* \t\t: {roc_auc*100} %")
        st.write('*Best Threshold* \t\t: %.3f' % (thresholds[ix]))
    with col2:
        st.write('*G-Mean* \t\t\t: %.3f' % (gmeans[ix]))
        st.write(f"*Model Accuracy* : {round(accuracy,2,)*100} %")

    st.write("*Classification Report:*")
    st.text(classification_report(y_test, y_pred))


def get_cf_report_dict(y_test, y_pred):
    cf_report_dict = classification_report(y_test, y_pred, output_dict = True)
    return cf_report_dict


def trainer(df, test_size, over_sample, vectorizer, model):
    model, x_test, y_test, y_pred_prob = train_model(
        df=df, 
        input='text', 
        target='fraudulent', 
        test_size=test_size,
        over_sample=over_sample, 
        vectorizer=vectorizer, 
        model=model)

    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)

    evaluate(y_test, y_pred, y_pred_prob)
    
    #get classification report variable
    cf_report_dict = get_cf_report_dict(y_test,y_pred)

     #generate json file
    report_json_str = json.dumps(cf_report_dict)

    report_bytes = report_json_str.encode('utf-8')
    report_base64 = base64.b64encode(report_bytes).decode('utf-8')

    #Generate the name of the file
    model_name = str(model._class.name_)
    random_int = random.randint(1, 1000000)

    #defne the repo route
    url = f"https://api.github.com/repos/Ultracatx/RealityStream/contents/output/user_generated_json/{model_name}_{random_int}"
    headers = {
            "Authorization": f"token {github_token}",
            #"Accept": "application/vnd.github.v3+json",
        }
    data = {
        "message": "testing json file upload",
        "content": report_base64,
        "branch": "interaction_test",
           }
    
    #determine upload status
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 201:
        st.success("Report successfully uploaded to GitHub.")
    else:
        st.error(f"Failed to upload: {response.content}")

    #create a download button for user to download the json file
    st.download_button(
    label ="Download Classification Report as JSON",
    data = report_json_str,
    file_name ="classification_report.json",
    mime = "application/json")
    


nlp = spacy.load('en_core_web_sm')

# Text Preprocessing with varoius combination

def spacy_process(text):
  # Converts to lowercase
  text = text.strip().lower()

  # passing text to spacy's nlp object
  doc = nlp(text)
    
  # Lemmatization
  lemma_list = []
  for token in doc:
    lemma_list.append(token.lemma_)
  
  # Filter the stopword
  filtered_sentence =[] 
  for word in lemma_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
      filtered_sentence.append(word)
    
  # Remove punctuation
  punctuations="?:!.,;$\'-_"
  for word in filtered_sentence:
    if word in punctuations:
      filtered_sentence.remove(word)

  return " ".join(filtered_sentence)

# For Loading the Pickle File
def load_model():
    with open('output/jobs/saved/notebook_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

'''
def save_report_to_github(image_content, filename, repo_name, path_in_repo, commit_message, github_token):
    url = f"https://api.github.com/repos/{repo_name}/contents/{path_in_repo}/{filename}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "message": commit_message,
        "content": base64.b64encode(image_content).decode('utf-8'),
        "branch": "main",
    }
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 201:
        st.success("Image successfully uploaded to GitHub.")
    else:
        st.error(f"Failed to upload image: {response.content}") 
'''

# GitHub Notebook Integration Functions

# Approved notebooks for RealityStream
APPROVED_NOTEBOOK_CHOICES = [
    "Run-Models-bkup.ipynb"
]

def get_approved_choices():
    """
    Return the approved RealityStream notebook choices.
    """
    return APPROVED_NOTEBOOK_CHOICES


def detect_cpu_only():
    """
    Detect if the system has GPU/CUDA support.
    Returns True if CPU-only mode should be used.
    """
    try:
        import subprocess
        # Try to detect NVIDIA GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode != 0  # True if nvidia-smi fails (no GPU)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True  # No nvidia-smi found, assume CPU-only


def prepare_notebook_for_cpu(notebook_content):
    """
    Modify notebook to skip GPU-intensive cells when running on CPU.
    Adds a parameter cell and conditional logic.
    """
    import json

    nb = json.loads(notebook_content)

    # Add a parameter cell at the beginning (after cell 0)
    parameter_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "tags": ["parameters"]
        },
        "outputs": [],
        "source": [
            "# Papermill parameters - DO NOT EDIT MANUALLY\n",
            "use_cpu = False  # Will be set to True for CPU-only execution\n",
            "model_type = 'lr'  # Model to run\n",
            "test_size = 0.3\n",
            "max_features = 500\n",
            "random_state = 42\n",
            "oversample = True\n"
        ]
    }

    # Insert parameter cell after the first cell (index 1)
    if len(nb['cells']) > 1:
        nb['cells'].insert(1, parameter_cell)

    # Add CPU check cell after parameter cell
    cpu_check_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Skip GPU setup if use_cpu is True\n",
            "import sys\n",
            "if use_cpu:\n",
            "    print('⚠️ Running in CPU-only mode. GPU cells will be skipped.')\n",
            "    print('⚠️ This notebook requires GPU for full functionality.')\n",
            "    print('⚠️ Some features may not work correctly in CPU mode.')\n",
            "    sys.exit(0)  # Exit early to prevent RAPIDS installation\n"
        ]
    }

    nb['cells'].insert(2, cpu_check_cell)

    return json.dumps(nb)


def run_notebook_from_github(
    notebook_name,
    parameters=None,
    repo_owner="ModelEarth",
    repo_name="realitystream",
    path="models"
):
    """
    Download and execute a Jupyter notebook from GitHub using papermill.
    Automatically detects CPU-only systems and provides appropriate error messages.
    """

    # Security check
    if notebook_name not in APPROVED_NOTEBOOK_CHOICES:
        error_msg = f"Notebook '{notebook_name}' not in approved list"
        st.error(error_msg)
        return False, None, error_msg

    # Detect if CPU-only mode is needed
    use_cpu = detect_cpu_only()
    if use_cpu:
        st.warning(
            "⚠️ *CPU-only system detected*\n\n"
            "This notebook requires GPU/CUDA support which is not available. "
            "The execution will fail at the RAPIDS installation step.\n\n"
            "*Recommended actions:*\n"
            "- Run the notebook in Google Colab with GPU runtime (T4 GPU)\n"
            "- Use a cloud instance with NVIDIA GPU support\n"
            "- Wait for a CPU-compatible version of the models"
        )

    temp_input_path, temp_output_path = None, None

    try:
        # Download notebook content from GitHub
        st.info(f"Downloading notebook '{notebook_name}' from GitHub...")

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}/{notebook_name}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if "GITHUB_TOKEN" in st.secrets:
            headers["Authorization"] = f"token {st.secrets['GITHUB_TOKEN']}"

        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            error_msg = f"Failed to download notebook: HTTP {response.status_code}"
            st.error(error_msg)
            return False, None, error_msg

        file_data = response.json()

        # Handle different encoding types from GitHub API
        if file_data.get("encoding") == "base64":
            notebook_content = base64.b64decode(file_data["content"]).decode("utf-8")
        elif file_data.get("encoding") == "none":
            # For large files (>1MB), GitHub returns encoding="none" and provides download_url
            download_url = file_data.get("download_url")
            if not download_url:
                error_msg = "No download URL provided for large notebook file"
                st.error(error_msg)
                return False, None, error_msg

            st.info(f"Downloading large notebook from raw URL...")
            download_response = requests.get(download_url, timeout=60)
            if download_response.status_code != 200:
                error_msg = f"Failed to download notebook: HTTP {download_response.status_code}"
                st.error(error_msg)
                return False, None, error_msg

            notebook_content = download_response.text
        else:
            error_msg = f"Notebook content encoding '{file_data.get('encoding')}' not supported"
            st.error(error_msg)
            return False, None, error_msg

        # Prepare notebook for CPU execution if needed
        if use_cpu:
            st.info("Preparing notebook for CPU-only execution...")
            notebook_content = prepare_notebook_for_cpu(notebook_content)

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as temp_input:
            temp_input.write(notebook_content)
            temp_input_path = temp_input.name

        temp_output_path = temp_input_path.replace(".ipynb", "_output.ipynb")

        # Merge use_cpu parameter with user parameters
        exec_parameters = parameters or {}
        exec_parameters['use_cpu'] = use_cpu

        st.info("Executing notebook with papermill...")

        # Execute notebook with papermill
        pm.execute_notebook(
            input_path=temp_input_path,
            output_path=temp_output_path,
            parameters=exec_parameters,
            progress_bar=False,
            log_output=True
        )

        st.success(f"Notebook '{notebook_name}' executed successfully!")
        return True, temp_output_path, None

    except pm.PapermillExecutionError as e:
        # Papermill execution errors with detailed info
        error_msg = str(e)

        # Check if it's a RAPIDS/CUDA/GPU related error
        if any(keyword in error_msg.lower() for keyword in ['rapids', 'cuda', 'gpu', 'cudf', 'cuml', 'cupy']):
            error_msg = (
                "⚠️ GPU/RAPIDS execution error detected.\n\n"
                "The notebook requires GPU support (RAPIDS/CUDA) which is not available on your system.\n\n"
                "*Solutions:*\n"
                "1. Run the notebook in Google Colab with GPU runtime\n"
                "2. Use a cloud instance with NVIDIA GPU support\n"
                "3. Wait for CPU-only version of the notebook (coming soon)\n\n"
                f"Technical details: {str(e)[:500]}"
            )

        st.error(error_msg)
        return False, temp_output_path, error_msg

    except Exception as e:
        error_msg = f"Unexpected error running notebook: {str(e)}"
        st.error(error_msg)
        return False, temp_output_path, error_msg

    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except Exception as e:
                st.warning(f"Could not clean up temporary file: {str(e)}")


def display_notebook_results(output_path):
    """
    Display notebook execution results in Streamlit.
    """
    if not output_path or not os.path.exists(output_path):
        st.warning("No output notebook to display")
        return

    try:
        with open(output_path, "r") as f:
            notebook_data = json.load(f)

        st.subheader("Notebook Execution Results")

        for i, cell in enumerate(notebook_data.get("cells", [])):
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                st.write(f"*Cell {i + 1}:*")

                if cell.get("source"):
                    code = "".join(cell["source"])
                    st.code(code, language="python")

                for output in cell["outputs"]:
                    if output.get("output_type") == "stream":
                        st.text("".join(output.get("text", [])))
                    elif output.get("output_type") == "display_data":
                        if "text/plain" in output.get("data", {}):
                            st.text("".join(output["data"]["text/plain"]))
                    elif output.get("output_type") == "execute_result":
                        if "text/plain" in output.get("data", {}):
                            st.text("".join(output["data"]["text/plain"]))

                st.divider()

        try:
            os.unlink(output_path)
        except Exception as e:
            st.warning(f"Could not clean up output file: {str(e)}")

    except Exception as e:
        st.error(f"Error displaying notebook results: {str(e)}")