import streamlit as st
import pandas as pd
import requests
import io
import json
import base64 # For SFTP - will only be used conceptually
from pathlib import Path # Added for accessing the script path
# import paramiko # You would need to install paramiko for actual SFTP: pip install paramiko

# --- Configuration (Optional: for real app, consider .streamlit/secrets.toml) ---
# For demonstration purposes, API keys and credentials are not hardcoded.
# In a real app, use st.secrets for sensitive information.
# Example:
# [api_credentials]
# my_api_key = "your_api_key_here"
# sftp_username = "sftp_user"

# --- Utility Functions ---

def load_data_from_uploaded_file(uploaded_file):
    """
    Loads data from various file formats (CSV, Excel, JSON) into a pandas DataFrame.
    Handles common file extensions and provides basic error reporting.
    """
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        st.info(f"Attempting to load file: {uploaded_file.name} with extension: .{file_extension}")
        try:
            if file_extension == 'csv':
                # Use io.StringIO for text-based files
                return pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            elif file_extension in ['xls', 'xlsx']:
                # Use io.BytesIO for binary files
                return pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                # Use io.StringIO for text-based JSON
                return pd.read_json(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            else:
                st.error(f"Unsupported file type for {uploaded_file.name}: .{file_extension}. Please upload CSV, Excel, or JSON.")
                return None
        except Exception as e:
            st.error(f"Error reading file '{uploaded_file.name}': {e}. Please check file format and content.")
            return None
    return None

def call_api(url, method='GET', headers=None, data=None, json_data=None):
    """
    Calls an external API using the requests library.
    Supports GET, POST, PUT, and DELETE methods.
    Returns the JSON response or None on error.
    """
    st.info(f"Calling API: {method} {url}")
    try:
        response = None
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=data)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, data=data, json=json_data)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, data=data, json=json_data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers, data=data, json=json_data)
        else:
            st.error("Unsupported HTTP method specified.")
            return None

        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred during API call: {http_err} - Response: {response.text}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error during API call: {conn_err}. Check URL or network connectivity.")
        return None
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"API call timed out: {timeout_err}. The server took too long to respond.")
        return None
    except requests.exceptions.RequestException as req_err:
        st.error(f"An unexpected API request error occurred: {req_err}")
        return None
    except json.JSONDecodeError:
        st.warning(f"API response is not valid JSON. Raw response: {response.text}")
        return response.text # Return raw text if not JSON
    except Exception as e:
        st.error(f"An unexpected error occurred while calling the API: {e}")
        return None

def sftp_upload_simulation(host, port, username, password, remote_path, file_content_bytes):
    """
    Simulates an SFTP file upload.
    In a real application, you would use paramiko here.
    """
    st.info(f"Simulating SFTP upload to {username}@{host}:{port}{remote_path}")
    st.warning("This is a simulation. Actual SFTP transfer requires a backend script with SFTP client (e.g., Paramiko).")
    # Example of what real SFTP code would look like (requires paramiko)
    # try:
    #     transport = paramiko.Transport((host, port))
    #     transport.connect(username=username, password=password)
    #     sftp = paramiko.SFTPClient.from_transport(transport)
    #     
    #     with sftp.open(remote_path, 'wb') as f:
    #         f.write(file_content_bytes)
    #     
    #     sftp.close()
    #     transport.close()
    #     st.success(f"Successfully uploaded file to SFTP: {remote_path}")
    # except Exception as e:
    #     st.error(f"SFTP upload failed: {e}")
    st.success(f"Simulated SFTP upload of file (size: {len(file_content_bytes)} bytes) to {remote_path} completed.")
    return True

def api_push_simulation(url, method, headers, payload):
    """
    Simulates pushing data to an API endpoint.
    In a real application, you would use requests here.
    """
    st.info(f"Simulating API push to {url} using {method} method.")
    st.warning("This is a simulation. Actual API push requires a backend script with HTTP client (e.g., requests).")
    try:
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=payload)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, json=payload)
        else:
            st.error("Unsupported API push method.")
            return False
        
        response.raise_for_status()
        st.success(f"Simulated API push successful! Response status: {response.status_code}")
        st.json(response.json())
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Simulated API push failed: {e}")
        return False

# --- Streamlit App Setup ---

st.set_page_config(
    page_title="Universal Data Integrator & Reporter",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Universal Data Integrator & Reporter")
st.markdown("This application allows you to ingest data, transform it, generate reports, and configure automated workflows.")

# Initialize session state variables if they don't exist
# Session state is crucial for persisting data across Streamlit reruns
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {} # Stores ingested DataFrames {name: df}
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None # Stores the result of a merge operation
if 'report_df' not in st.session_state:
    st.session_state.report_df = None # Stores the final DataFrame prepared for reporting
if 'workflow_config' not in st.session_state:
    st.session_state.workflow_config = {} # Stores configuration for automated publishing

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["1. Data Ingestion", "2. Data Transformation", "3. Report Generation", "4. Automated Workflows"]
)

# --- Page: 1. Data Ingestion ---
if page == "1. Data Ingestion":
    st.header("1. Data Ingestion: Upload Files or Call APIs")
    st.markdown("---")

    st.subheader("Upload Data Files")
    uploaded_files = st.file_uploader(
        "Choose CSV, Excel (.xls, .xlsx), or JSON files",
        type=["csv", "xls", "xlsx", "json"],
        accept_multiple_files=True,
        help="Upload multiple files at once. Each will be stored as a separate DataFrame."
    )

    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            # Generate a unique name for the DataFrame based on original file name
            file_name_clean = uploaded_file.name.split('.')[0].replace(' ', '_').replace('-', '_')
            df_name = f"Uploaded_File_{file_name_clean}_{i+1}"
            
            # Check if a DataFrame with this exact name already exists in session_state
            # This prevents overwriting if a user uploads the exact same file multiple times
            if df_name in st.session_state.dataframes:
                # Append a number if name already exists
                j = 1
                while f"{df_name}_{j}" in st.session_state.dataframes:
                    j += 1
                df_name = f"{df_name}_{j}"
            
            df = load_data_from_uploaded_file(uploaded_file)
            if df is not None:
                st.session_state.dataframes[df_name] = df
                st.success(f"✅ Successfully loaded '{uploaded_file.name}' as DataFrame: **'{df_name}'**")
                with st.expander(f"Preview of {df_name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
                    st.dataframe(df.head(10)) # Show first 10 rows
            else:
                st.error(f"❌ Failed to load '{uploaded_file.name}'. See error message above.")
    else:
        st.info("No files uploaded yet. Please use the uploader above.")

    st.markdown("---")
    st.subheader("Call External APIs")
    api_url = st.text_input(
        "API Endpoint URL",
        "https://jsonplaceholder.typicode.com/posts/1",
        help="Example: https://jsonplaceholder.typicode.com/posts/1 (GET) or https://jsonplaceholder.typicode.com/posts (POST)"
    )
    api_method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])

    col_headers, col_body = st.columns(2)
    with col_headers:
        api_headers_input = st.text_area(
            "Headers (JSON format, optional)",
            "{}",
            help="Enter headers as a JSON object, e.g., {\"Content-Type\": \"application/json\"}"
        )
    with col_body:
        api_body_input = st.text_area(
            "Request Body (JSON format, optional)",
            "{}",
            help="Enter request body as a JSON object for POST/PUT requests."
        )

    if st.button("🚀 Call API"):
        try:
            headers = json.loads(api_headers_input)
            json_data_payload = None
            if api_method.upper() in ['POST', 'PUT']:
                json_data_payload = json.loads(api_body_input)

            api_response = call_api(api_url, method=api_method, headers=headers, json_data=json_data_payload)

            if api_response:
                st.success("API call successful!")
                st.json(api_response) # Display raw JSON response

                if st.checkbox("Convert API response to DataFrame?", key="convert_api_to_df"):
                    try:
                        # Attempt to convert various API response structures to DataFrame
                        api_df = None
                        if isinstance(api_response, list) and all(isinstance(item, dict) for item in api_response):
                            api_df = pd.DataFrame(api_response)
                        elif isinstance(api_response, dict):
                            api_df = pd.DataFrame([api_response]) # Single dict to DataFrame
                        else:
                            st.warning("API response format not directly convertible to DataFrame (expected list of dicts or single dict).")

                        if api_df is not None:
                            api_df_name_input = st.text_input(
                                "Name for this API DataFrame",
                                f"API_Data_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                            )
                            if st.button("Save API Data as DataFrame", key="save_api_df_btn"):
                                if api_df_name_input:
                                    st.session_state.dataframes[api_df_name_input] = api_df
                                    st.success(f"API response saved as DataFrame: **'{api_df_name_input}'**")
                                    with st.expander(f"Preview of {api_df_name_input}"):
                                        st.dataframe(api_df.head())
                                else:
                                    st.warning("Please provide a name for the API DataFrame.")
                    except Exception as e:
                        st.error(f"Error converting API response to DataFrame: {e}")
            else:
                st.error("API call failed or returned no data.")
        except json.JSONDecodeError:
            st.error("Invalid JSON format detected in Headers or Request Body. Please correct it.")
        except Exception as e:
            st.error(f"An unexpected error occurred during API call setup: {e}")

    st.markdown("---")
    st.subheader("Currently Loaded DataFrames")
    if st.session_state.dataframes:
        for name, df in st.session_state.dataframes.items():
            st.write(f"- **{name}**: {df.shape[0]} rows, {df.shape[1]} columns")
            with st.expander(f"Show details for {name}"):
                st.dataframe(df.head())
                st.write(f"Columns: {', '.join(df.columns)}")
                st.write("Data Types:")
                st.write(df.dtypes)
    else:
        st.info("No DataFrames loaded yet. Upload files or call APIs above.")

# --- Page: 2. Data Transformation ---
elif page == "2. Data Transformation":
    st.header("2. Data Transformation: Merge, Calculate & Rename")
    st.markdown("---")

    if not st.session_state.dataframes:
        st.warning("Please ingest data in the '1. Data Ingestion' section first to perform transformations.")
    else:
        st.subheader("Merge DataFrames")
        df_keys = list(st.session_state.dataframes.keys())
        if len(df_keys) >= 2:
            st.info("Select two DataFrames to merge. The app will automatically suggest common columns for merging.")
            
            col_merge1, col_merge2 = st.columns(2)
            with col_merge1:
                df1_name = st.selectbox("Select **First** DataFrame for Merge", df_keys, key="merge_df1_select")
            with col_merge2:
                df2_name = st.selectbox("Select **Second** DataFrame for Merge", df_keys, key="merge_df2_select")

            if df1_name and df2_name and df1_name != df2_name:
                df1 = st.session_state.dataframes[df1_name]
                df2 = st.session_state.dataframes[df2_name]

                common_cols = list(set(df1.columns) & set(df2.columns))
                
                if common_cols:
                    st.success(f"Common columns identified: `{', '.join(common_cols)}`")
                    merge_on_col = st.selectbox("Select column to merge on", common_cols, key="merge_on_col_select")
                    merge_type = st.selectbox("Select merge type (how)", ["inner", "left", "right", "outer"], key="merge_type_select")

                    if st.button("Merge DataFrames", key="execute_merge_btn"):
                        try:
                            # Add suffixes to differentiate columns with same names after merge
                            st.session_state.merged_df = pd.merge(
                                df1, df2, on=merge_on_col, how=merge_type, 
                                suffixes=(f'_{df1_name.replace(" ", "_")}', f'_{df2_name.replace(" ", "_")}')
                            )
                            st.success("✅ DataFrames merged successfully! The result is stored as 'Merged DataFrame'.")
                            st.write("Preview of **Merged DataFrame**:")
                            st.dataframe(st.session_state.merged_df.head(10))
                            st.write(f"Shape of merged DataFrame: {st.session_state.merged_df.shape}")
                        except Exception as e:
                            st.error(f"❌ Error merging DataFrames: {e}. Ensure the merge column data types are compatible.")
                else:
                    st.warning("⚠️ No common columns found between selected DataFrames. Cannot perform a direct merge.")
                    st.info("You might need to manually rename columns or create a common key in the 'Create New Data Points' section before merging.")
            else:
                st.info("Please select two *different* DataFrames to enable merging options.")
        else:
            st.info("Please upload at least two DataFrames in '1. Data Ingestion' to enable merging.")

    st.markdown("---")
    st.subheader("Create New Data Points by Calculations")
    
    # Create a list of available DataFrames, including the merged_df if it exists
    available_dfs_for_calc = list(st.session_state.dataframes.keys())
    if st.session_state.merged_df is not None:
        available_dfs_for_calc.insert(0, "Merged DataFrame") # Add merged_df as the first option

    if not available_dfs_for_calc:
        st.info("No DataFrames available for calculations. Please ingest data first.")
    else:
        target_df_calc_name = st.selectbox(
            "Select DataFrame to perform calculations on",
            available_dfs_for_calc,
            key="calc_target_df_select"
        )
        
        selected_df_for_calc = None
        if target_df_calc_name == "Merged DataFrame" and st.session_state.merged_df is not None:
            selected_df_for_calc = st.session_state.merged_df
        elif target_df_calc_name in st.session_state.dataframes:
            selected_df_for_calc = st.session_state.dataframes[target_df_calc_name]

        if selected_df_for_calc is not None:
            st.write(f"Current columns in **'{target_df_calc_name}'**: `{', '.join(selected_df_for_calc.columns)}`")
            st.dataframe(selected_df_for_calc.head())

            new_column_name = st.text_input("Enter New Column Name (e.g., 'Total_Amount')", key="new_col_name_input")
            calculation_expression = st.text_input(
                "Enter Calculation Expression (e.g., `df[\"Price\"] * df[\"Quantity\"]` or `df[\"ColA\"] + df[\"ColB\"]`)",
                value="df[\"Column1\"] + df[\"Column2\"]",
                help="Use `df` to refer to the selected DataFrame. Column names must be exact and enclosed in double quotes. Basic arithmetic and pandas operations are supported."
            )

            if st.button("✨ Apply Calculation", key="apply_calculation_btn"):
                if new_column_name and calculation_expression:
                    try:
                        # Use a copy to avoid modifying the original DataFrame until successful
                        temp_df_for_eval = selected_df_for_calc.copy()
                        
                        # Evaluate the expression within a safe namespace
                        # 'df' is made available for evaluation
                        temp_df_for_eval[new_column_name] = eval(calculation_expression, {"df": temp_df_for_eval, "pd": pd})
                        
                        # Update the session state only if successful
                        if target_df_calc_name == "Merged DataFrame":
                            st.session_state.merged_df = temp_df_for_eval
                        else:
                            st.session_state.dataframes[target_df_calc_name] = temp_df_for_eval

                        st.success(f"✅ New column **'{new_column_name}'** created successfully in **'{target_df_calc_name}'**!")
                        st.dataframe(temp_df_for_eval.head(10))
                    except NameError as ne:
                        st.error(f"❌ Calculation error: Column not found. {ne}. Please check your column names in the expression.")
                    except SyntaxError as se:
                        st.error(f"❌ Calculation error: Invalid Python syntax. {se}. Check your expression format.")
                    except Exception as e:
                        st.error(f"❌ An error occurred applying calculation: {e}. Review your expression and data types.")
                else:
                    st.warning("Please provide both a new column name and a calculation expression.")
        else:
            st.info("Please select a DataFrame from the dropdown to perform calculations.")

    st.markdown("---")
    st.subheader("Change Data Headers (Rename Columns)")
    
    available_dfs_for_rename = list(st.session_state.dataframes.keys())
    if st.session_state.merged_df is not None:
        available_dfs_for_rename.insert(0, "Merged DataFrame")

    if not available_dfs_for_rename:
        st.info("No DataFrames available for renaming columns. Please ingest data first.")
    else:
        rename_target_df_name = st.selectbox(
            "Select DataFrame to rename columns in",
            available_dfs_for_rename,
            key="rename_target_df_select"
        )

        selected_df_for_rename = None
        if rename_target_df_name == "Merged DataFrame" and st.session_state.merged_df is not None:
            selected_df_for_rename = st.session_state.merged_df
        elif rename_target_df_name in st.session_state.dataframes:
            selected_df_for_rename = st.session_state.dataframes[rename_target_df_name]
        
        if selected_df_for_rename is not None:
            st.write(f"Current columns in **'{rename_target_df_name}'**: `{', '.join(selected_df_for_rename.columns)}`")
            st.dataframe(selected_df_for_rename.head())

            col_old_name, col_new_name = st.columns(2)
            with col_old_name:
                old_col = st.selectbox("Select Old Column Name", selected_df_for_rename.columns.tolist(), key="old_col_select")
            with col_new_name:
                new_col = st.text_input("Enter New Column Name", key="new_col_input")

            if st.button("✏️ Rename Column", key="rename_column_btn"):
                if old_col and new_col:
                    if old_col in selected_df_for_rename.columns:
                        try:
                            df_renamed = selected_df_for_rename.rename(columns={old_col: new_col})
                            if rename_target_df_name == "Merged DataFrame":
                                st.session_state.merged_df = df_renamed
                            else:
                                st.session_state.dataframes[rename_target_df_name] = df_renamed
                            st.success(f"✅ Column **'{old_col}'** renamed to **'{new_col}'** successfully in **'{rename_target_df_name}'**!")
                            st.dataframe(df_renamed.head(10))
                        except Exception as e:
                            st.error(f"❌ Error renaming column: {e}.")
                    else:
                        st.error(f"❌ Column '{old_col}' not found in the selected DataFrame. Please select an existing column.")
                else:
                    st.warning("Please select an old column and provide a new column name.")
        else:
            st.info("Please select a DataFrame from the dropdown to rename columns.")

# --- Page: 3. Report Generation ---
elif page == "3. Report Generation":
    st.header("3. Report Generation: Prepare and Review")
    st.markdown("---")

    st.subheader("Select Data Source for Report")
    
    available_dfs_for_report = list(st.session_state.dataframes.keys())
    if st.session_state.merged_df is not None:
        available_dfs_for_report.insert(0, "Merged DataFrame")

    if not available_dfs_for_report:
        st.warning("No DataFrames available to create a report. Please ingest and transform data first.")
        report_df_candidate = None
    else:
        report_source_df_name = st.selectbox(
            "Choose a DataFrame as the base for your report",
            available_dfs_for_report,
            key="report_source_df_select"
        )

        report_df_candidate = None
        if report_source_df_name == "Merged DataFrame" and st.session_state.merged_df is not None:
            report_df_candidate = st.session_state.merged_df.copy()
        elif report_source_df_name in st.session_state.dataframes:
            report_df_candidate = st.session_state.dataframes[report_source_df_name].copy()
        
        if report_df_candidate is not None:
            st.write(f"Preview of selected data from **'{report_source_df_name}'** for report:")
            st.dataframe(report_df_candidate.head(10))
            st.write(f"Shape: {report_df_candidate.shape}")

            st.markdown("---")
            st.subheader("Filter Rows and Select Columns for Report")
            
            all_columns = report_df_candidate.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to include in the report (leave empty for all columns)",
                all_columns,
                default=all_columns,
                key="report_cols_multiselect"
            )

            # Optional: Add row filtering based on a column value
            st.markdown("##### Optional: Filter Rows")
            filter_column = st.selectbox("Select column to filter by (optional)", ["None"] + all_columns, key="filter_col_select")
            
            filtered_df_temp = report_df_candidate.copy() # Start with a copy for filtering

            if filter_column != "None":
                unique_values = report_df_candidate[filter_column].unique().tolist()
                filter_value_type = st.radio(
                    "Filter by:",
                    ["Select from list", "Enter custom value"],
                    key="filter_value_type_radio"
                )

                if filter_value_type == "Select from list":
                    selected_filter_values = st.multiselect(
                        f"Select values in '{filter_column}' to include",
                        unique_values,
                        key="filter_values_multiselect"
                    )
                    if selected_filter_values:
                        filtered_df_temp = filtered_df_temp[filtered_df_temp[filter_column].isin(selected_filter_values)]
                else: # Enter custom value
                    custom_filter_value = st.text_input(
                        f"Enter value for '{filter_column}' to include (exact match)",
                        key="custom_filter_value_input"
                    )
                    if custom_filter_value:
                        # Attempt to cast filter value to column dtype if possible, for exact match
                        try:
                            dtype = report_df_candidate[filter_column].dtype
                            if pd.api.types.is_numeric_dtype(dtype):
                                custom_filter_value = float(custom_filter_value) if '.' in custom_filter_value else int(custom_filter_value)
                            elif pd.api.types.is_bool_dtype(dtype):
                                custom_filter_value = custom_filter_value.lower() == 'true'
                        except ValueError:
                            pass # Keep as string if cannot convert
                        filtered_df_temp = filtered_df_temp[filtered_df_temp[filter_column] == custom_filter_value]
                    else:
                        st.info("Enter a value to apply the filter.")
            
            if selected_columns:
                st.session_state.report_df = filtered_df_temp[selected_columns].copy()
                st.success("✅ Report data prepared for review!")
                st.write("Final Report Preview:")
                st.dataframe(st.session_state.report_df.head(10))
                st.write(f"Final Report Shape: {st.session_state.report_df.shape}")

                st.markdown("---")
                st.subheader("Generate Downloadable Report")
                report_format = st.selectbox(
                    "Choose report format for download",
                    ["CSV", "Excel", "JSON"],
                    key="download_report_format_select"
                )

                if st.button("⬇️ Generate & Download Report", key="download_report_btn"):
                    if report_format == "CSV":
                        csv_data = st.session_state.report_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV Report",
                            data=csv_data,
                            file_name="generated_report.csv",
                            mime="text/csv",
                            key="download_csv_btn"
                        )
                    elif report_format == "Excel":
                        excel_buffer = io.BytesIO()
                        st.session_state.report_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                        excel_buffer.seek(0) # Rewind the buffer to the beginning
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_buffer,
                            file_name="generated_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel_btn"
                        )
                    elif report_format == "JSON":
                        json_data = st.session_state.report_df.to_json(orient='records').encode('utf-8')
                        st.download_button(
                            label="Download JSON Report",
                            data=json_data,
                            file_name="generated_report.json",
                            mime="application/json",
                            key="download_json_btn"
                        )
                    st.success("Report generated and ready for download!")
            else:
                st.warning("Please select at least one column for the report.")
        else:
            st.info("No data selected for report generation. Please choose a DataFrame above.")

# --- Page: 4. Automated Workflows ---
elif page == "4. Automated Workflows":
    st.header("4. Automated Workflows: Publish Reports")
    st.markdown("---")

    st.warning("Automated workflows (SFTP, API Push) require external scheduling and execution outside of this Streamlit application itself. This section helps you configure the parameters for such a workflow.")

    st.subheader("Review Report for Publishing")
    if st.session_state.report_df is None:
        st.info("Please prepare and review a report in the '3. Report Generation' section first.")
    else:
        st.write("This is the report currently prepared for publishing:")
        st.dataframe(st.session_state.report_df.head(10))
        st.write(f"Report size: {st.session_state.report_df.shape[0]} rows, {st.session_state.report_df.shape[1]} columns.")
        st.success("Your report is ready to be configured for a publishing workflow.")

        st.markdown("---")
        st.subheader("Configure Publishing Options")

        publish_method = st.selectbox(
            "Select publishing method",
            ["Local Drive (Download)", "SFTP (Configure Only)", "API Push (Configure Only)"],
            key="publish_method_select"
        )

        st.session_state.workflow_config['publish_method'] = publish_method

        if publish_method == "Local Drive (Download)":
            st.info("This option simulates generating the report and saving it locally. In an automated setup, this means saving to a designated file path.")
            report_format_local = st.selectbox(
                "Choose local report format",
                ["CSV", "Excel", "JSON"],
                key="local_report_format_select"
            )
            st.session_state.workflow_config['local_format'] = report_format_local

            if st.button("Simulate Local Report Generation", key="simulate_local_btn"):
                st.info(f"Simulating local report generation in {report_format_local} format...")
                try:
                    output_bytes = None
                    file_name = f"published_workflow_report.{report_format_local.lower()}"
                    mime_type = ""

                    if report_format_local == "CSV":
                        output_bytes = st.session_state.report_df.to_csv(index=False).encode('utf-8')
                        mime_type = "text/csv"
                    elif report_format_local == "Excel":
                        excel_buffer = io.BytesIO()
                        st.session_state.report_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                        excel_buffer.seek(0)
                        output_bytes = excel_buffer.getvalue()
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    elif report_format_local == "JSON":
                        output_bytes = st.session_state.report_df.to_json(orient='records').encode('utf-8')
                        mime_type = "application/json"
                    
                    if output_bytes:
                        st.download_button(
                            label=f"Download Simulated {report_format_local} Report",
                            data=output_bytes,
                            file_name=file_name,
                            mime=mime_type,
                            key="simulate_local_download_btn"
                        )
                        st.success("✅ Simulated local report generation and download link provided.")
                    else:
                        st.error("Failed to generate report for local simulation.")
                except Exception as e:
                    st.error(f"Error during local simulation: {e}")


        elif publish_method == "SFTP (Configure Only)":
            st.subheader("SFTP Connection Details")
            st.info("These details will be used by an external script for actual SFTP transfer.")
            st.session_state.workflow_config['sftp_host'] = st.text_input("SFTP Host (e.g., sftp.example.com)", key="sftp_host_input")
            st.session_state.workflow_config['sftp_port'] = st.number_input("SFTP Port", value=22, key="sftp_port_input")
            st.session_state.workflow_config['sftp_username'] = st.text_input("SFTP Username", key="sftp_username_input")
            st.session_state.workflow_config['sftp_password'] = st.text_input("SFTP Password", type="password", key="sftp_password_input")
            st.session_state.workflow_config['sftp_remote_path'] = st.text_input("Remote Path (e.g., /reports/daily/report.csv)", key="sftp_remote_path_input")
            st.session_state.workflow_config['sftp_file_format'] = st.selectbox("SFTP File Format", ["CSV", "Excel", "JSON"], key="sftp_file_format_select")

            if st.button("Simulate SFTP Upload", key="simulate_sftp_btn"):
                sftp_config = st.session_state.workflow_config
                if all(k in sftp_config for k in ['sftp_host', 'sftp_username', 'sftp_password', 'sftp_remote_path', 'sftp_file_format']):
                    try:
                        file_content_bytes = None
                        if sftp_config['sftp_file_format'] == "CSV":
                            file_content_bytes = st.session_state.report_df.to_csv(index=False).encode('utf-8')
                        elif sftp_config['sftp_file_format'] == "Excel":
                            excel_buffer = io.BytesIO()
                            st.session_state.report_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                            excel_buffer.seek(0)
                            file_content_bytes = excel_buffer.getvalue()
                        elif sftp_config['sftp_file_format'] == "JSON":
                            file_content_bytes = st.session_state.report_df.to_json(orient='records').encode('utf-8')

                        if file_content_bytes:
                            sftp_upload_simulation(
                                sftp_config['sftp_host'],
                                sftp_config['sftp_port'],
                                sftp_config['sftp_username'],
                                sftp_config['sftp_password'],
                                sftp_config['sftp_remote_path'],
                                file_content_bytes
                            )
                        else:
                            st.error("Failed to prepare file content for SFTP simulation.")
                    except Exception as e:
                        st.error(f"Error during SFTP simulation: {e}")
                else:
                    st.warning("Please fill in all SFTP configuration details to simulate.")

        elif publish_method == "API Push (Configure Only)":
            st.subheader("API Push Details")
            st.info("These details will be used by an external script to push data to your target API.")
            st.session_state.workflow_config['api_push_url'] = st.text_input("Target API URL for Push", key="api_push_url_input")
            st.session_state.workflow_config['api_push_method'] = st.selectbox("API Push Method", ["POST", "PUT"], key="api_push_method_select")
            st.session_state.workflow_config['api_push_headers'] = st.text_area("API Push Headers (JSON format)", "{\"Content-Type\": \"application/json\"}", key="api_push_headers_input")
            st.session_state.workflow_config['api_push_payload_type'] = st.selectbox("Payload Type", ["JSON (records)", "CSV (as string)"], key="api_push_payload_type_select")
            
            if st.button("Simulate API Push", key="simulate_api_push_btn"):
                api_push_config = st.session_state.workflow_config
                if all(k in api_push_config for k in ['api_push_url', 'api_push_method', 'api_push_headers', 'api_push_payload_type']):
                    try:
                        headers = json.loads(api_push_config['api_push_headers'])
                        payload = None

                        if api_push_config['api_push_payload_type'] == "JSON (records)":
                            payload = st.session_state.report_df.to_dict(orient='records')
                        elif api_push_config['api_push_payload_type'] == "CSV (as string)":
                            payload = {"data": st.session_state.report_df.to_csv(index=False)}
                        
                        if payload:
                            api_push_simulation(
                                api_push_config['api_push_url'],
                                api_push_config['api_push_method'],
                                headers,
                                payload
                            )
                        else:
                            st.error("Failed to prepare payload for API push simulation.")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format detected in API Push Headers. Please correct it.")
                    except Exception as e:
                        st.error(f"Error during API push simulation: {e}")
                else:
                    st.warning("Please fill in all API Push configuration details to simulate.")

        st.markdown("---")
        st.subheader("Review and Confirm Workflow Configuration")
        if st.session_state.workflow_config:
            st.write("Current Workflow Configuration:")
            st.json(st.session_state.workflow_config)
            st.success("Configuration saved in session state. You can copy this for your automation scripts.")
        else:
            st.info("No workflow configured yet. Select a publishing method above.")

        st.markdown("---")
        st.markdown("### ⚙️ How to Achieve True Automation (Beyond This App)")
        st.markdown("""
        To run the configured workflow automatically and consistently (e.g., daily, hourly), you would typically:
        
        1.  **Create a dedicated Python script** (`run_workflow.py`): This script would contain the core logic for:
            * Loading initial data (from defined sources, not interactive uploads).
            * Applying data merges, calculations, and column renaming based on pre-defined rules or configuration.
            * Generating the final report DataFrame.
            * Executing the selected publishing action (SFTP, API push, or local file save) using the saved configurations.
            
        2.  **External Orchestration**:
            * **Cron Jobs (Linux/macOS) / Task Scheduler (Windows)**: For basic, scheduled execution on a server.
            * **Cloud Functions (AWS Lambda, Google Cloud Functions, Azure Functions)**: For serverless, event-driven execution, highly scalable and cost-effective for intermittent tasks.
            * **Workflow Orchestrators (Apache Airflow, Prefect, Dagster)**: For complex, data-pipeline-centric automation with monitoring, retries, and dependencies.
            * **CI/CD Pipelines (GitHub Actions, GitLab CI/CD)**: For triggering workflows as part of code deployments or on a schedule.

        3.  **Secure Credentials**: Never hardcode API keys or SFTP passwords in your scripts. Use environment variables, a secure vault, or cloud-specific secret management services.

        This Streamlit app serves as a powerful interactive configurator and preview tool for defining these complex data workflows.
        """)

# --- Feature: Show App Code ---
st.markdown("---")
st.header('App Source Code', divider='gray')

current_script_path = Path(__file__)

try:
    with open(current_script_path, 'r') as f:
        app_code = f.read()
    with st.expander("Click to view the Python code for this app"):
        st.code(app_code, language='python')
except Exception as e:
    st.error(f"Could not load app source code: {e}")
