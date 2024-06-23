import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def lookup_employee_code(df, employee_name, reporting_manager_col="Reporting Manager Name", employee_code_col="Employee Code", not_found="no employee code"):
    filtered_df = df[df[reporting_manager_col] == employee_name]
    if filtered_df.empty:
        return not_found
    
    employee_code = filtered_df.iloc[0][employee_code_col]
    return str(employee_code) if not pd.isnull(employee_code) else not_found

def validate_employee_data(df):
    expected_columns = {
        'Employee Code': 'int64',
        'Employee Name': 'object',
        'Reporting Manager Name': 'object',
        'DOB': 'datetime64[ns]',
        'Designation': 'object',
        'Main Function': 'object',
        'Final Function': 'object',
        'Location': 'object',
        'CTC in lakhs': 'float64',
        'RO Code': 'int64',
        'RO Name': 'object',
        'RO Designation': 'object',
        'Reporting Manager Code': 'object'
    }

    results = {"column_check": "", "data_type_check": "", "null_check": "", "missing_check": "", "length_check": "", "changes_summary": ""}

    actual_columns = df.columns.tolist()
    missing_columns = [col for col in expected_columns.keys() if col not in actual_columns]
    extra_columns = [col for col in actual_columns if col not in expected_columns.keys()]
    
    if not missing_columns and not extra_columns:
        results["column_check"] = "Column names are correct."
    else:
        if missing_columns:
            results["column_check"] += f"Missing columns: {missing_columns}\n"
        if extra_columns:
            results["column_check"] += f"Extra columns: {extra_columns}\n"

            for col in extra_columns:
                dtype = st.selectbox(f"Select data type for '{col}'", ['int64', 'float64', 'object', 'datetime64[ns]'])
                expected_columns[col] = dtype

    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')

    actual_dtypes = df.dtypes
    data_type_mismatches = 0

    for col, expected_dtype in expected_columns.items():
        actual_dtype = actual_dtypes[col]
        if actual_dtype != expected_dtype:
            data_type_mismatches += 1
            results["data_type_check"] += f"Data type mismatch for column '{col}': Expected {expected_dtype}, Got {actual_dtype}\n"

    if data_type_mismatches == 0:
        results["data_type_check"] = "All data types are correct."
    else:
        results["data_type_check"] += f"There are {data_type_mismatches} data type mismatches."

    null_count_data = {'Column': [], 'Null Count': [], 'Employee Names 2 Rows Ahead': []}

    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            names_2_ahead = []
            for idx in df[df[col].isnull()].index:
                if idx + 2 < len(df):
                    names_2_ahead.append(df.loc[idx, 'Employee Name'])
                else:
                    names_2_ahead.append(None)
            null_count_data['Column'].append(col)
            null_count_data['Null Count'].append(null_count)
            null_count_data['Employee Names 2 Rows Ahead'].append(names_2_ahead)

    null_count_df = pd.DataFrame(null_count_data)
    results["null_check"] = null_count_df

    missing_both = df[df['Employee Code'].isnull() & df['Employee Name'].isnull()]

    if not missing_both.empty:
        results["missing_check"] = f"Rows where both 'Employee Code' and 'Employee Name' are missing found:\n{missing_both.to_string(index=False)}\nRows have been deleted."
        df = df.drop(missing_both.index)
    else:
        results["missing_check"] = "No rows found where both 'Employee Code' and 'Employee Name' are missing."

    invalid_employee_name = []
    invalid_employee_code = []
    invalid_main_function = []

    for index, row in df.iterrows():
        if pd.notna(row['Employee Name']) and len(row['Employee Name']) >= 25:
            invalid_employee_name.append(index)
        if pd.notna(row['Employee Code']) and row['Employee Code'] >= 20000:
            invalid_employee_code.append(index)
        if pd.notna(row['Main Function']) and len(row['Main Function']) >= 100:
            invalid_main_function.append(index)

    employee_code_lengths = df['Employee Code'].dropna().astype(str).apply(len)
    if len(employee_code_lengths.unique()) > 1:
        results["length_check"] = ""
    else:
        results["length_check"] = "All Employee Codes have the same number of digits."

    if invalid_employee_name:
        results["length_check"] += "\n\nRows with Employee Name length >= 25 characters:\n"
        results["length_check"] += df.loc[invalid_employee_name].to_string(index=False)
    else:
        results["length_check"] += "\n\nAll Employee Names are less than 25 characters."

    if invalid_employee_code:
        results["length_check"] += "\n\nRows with Employee Code >= 10000:\n"
        results["length_check"] += df.loc[invalid_employee_code].to_string(index=False)
    else:
        results["length_check"] += "\n\nAll Employee Codes are less than 10000."

    if invalid_main_function:
        results["length_check"] += "\n\nRows with Main Function length >= 100 characters:\n"
        results["length_check"] += df.loc[invalid_main_function].to_string(index=False)
    else:
        results["length_check"] += "\n\nAll Main Functions are less than 100 characters."

    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    df['Age'] = ((today - df['DOB']).dt.days / 365).astype('float64')
    df['AgeBracket'] = pd.cut(df['Age'], bins=range(0, int(df['Age'].max()) + 5, 5), right=False)
    df['AgeBracket'] = df['AgeBracket'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
    

    results["changes_summary"] += "- Added 'Age' column based on DOB.\n"
    results["changes_summary"] += "- Added 'AgeBracket' column to categorize ages.\n"
    results["changes_summary"] += "- Added 'Reporting Manager Code' column.\n"

    return df, results

def assign_employee_layers(df, head_codes):
    df['Employee Layer'] = None
    
    # Find the head employees' names
    head_names = df[df['Employee Code'].isin(head_codes)]['Employee Name'].tolist()

    # Assign N-0 to the head employees
    df.loc[df['Employee Code'].isin(head_codes), 'Employee Layer'] = "N-0"

    # Dictionary to store employee names and their corresponding layers
    layer_dict = {name: "N-0" for name in head_names}

    current_layer = 0
    while True:
        # Get the employees in the current layer
        current_layer_employees = df[df['Employee Layer'] == f"N-{current_layer}"]['Employee Name'].tolist()
        
        # If no employees found in the current layer, break the loop
        if not current_layer_employees:
            break
        
        # Increment to the next layer
        next_layer = current_layer + 1
        
        # Iterate over current layer employees to find their subordinates
        for employee in current_layer_employees:
            # Find subordinates of the current employee
            subordinates = df[df['Reporting Manager Name'] == employee]['Employee Name'].tolist()
            
            # Assign layers to subordinates and update layer_dict
            for subordinate in subordinates:
                layer_dict[subordinate] = f"N-{next_layer}"
                df.loc[df['Employee Name'] == subordinate, 'Employee Layer'] = f"N-{next_layer}"
        
        current_layer = next_layer

    return df

def assign_ro_layer(df):
    df['RO Layer'] = None
    max_layer = df['Employee Layer'].str.extract(r'N-(\d+)').astype(float).max()
    max_layer = max_layer.fillna(0)
    for layer in range(int(max_layer) + 1):
        employees_in_layer = df[df['Employee Layer'] == f"N-{layer}"]['Employee Name'].tolist()
        for employee in employees_in_layer:
            if layer > 0:  # For N-0, keep RO Layer as null
                df.loc[df['Employee Name'] == employee, 'RO Layer'] = f"N-{layer - 1}"
    return df

def assign_pm_ic_span(df):
    df['Span'] = df['Employee Name'].apply(lambda x: len(df[df['Reporting Manager Name'] == x]))

    df['PM/IC'] = df['Span'].apply(lambda x: 'PM' if x > 0 else 'IC')

    span_columns = ['Span =1', 'Span =2', 'Span =3', 'Span =4', 'Span 5-8', 'Span >10']
    for col in span_columns:
        df[col] = 0
    
    df['Span =1'] = df['Span'].apply(lambda x: 1 if x == 1 else 0)
    df['Span =2'] = df['Span'].apply(lambda x: 1 if x == 2 else 0)
    df['Span =3'] = df['Span'].apply(lambda x: 1 if x == 3 else 0)
    df['Span =4'] = df['Span'].apply(lambda x: 1 if x == 4 else 0)
    df['Span =9'] = df['Span'].apply(lambda x: 1 if x == 9 else 0)
    df['Span 5-8'] = df['Span'].apply(lambda x: 1 if 5 <= x <= 8 else 0)
    df['Span >10'] = df['Span'].apply(lambda x: 1 if x > 10 else 0)

    return df

def save_and_download_figure(fig, filename):
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64
    b64 = base64.b64encode(buf.read()).decode()
    
    # Provide a download link
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)
    buf.close()

def create_visualizations(cleaned_df):
    # Get unique locations
    locations = cleaned_df['Location'].unique()

    # Add 'Overall' option to locations
    locations = ['Overall'] + list(locations)

    # Multiselect for selecting multiple locations with a unique key
    selected_locations = st.multiselect("Select Location", locations, default=["Overall"], key="location_multiselect")

    # Filter data based on selected locations
    if "Overall" in selected_locations:
        location_filtered_df = cleaned_df
    else:
        location_filtered_df = cleaned_df[cleaned_df['Location'].isin(selected_locations)]
        
    # Employee Layer vs Span DataFrame
    span_columns = ['Span =1', 'Span =2', 'Span =3', 'Span =4', 'Span 5-8', 'Span >10']
    span_df = location_filtered_df.groupby('Employee Layer')[span_columns].sum().reset_index()
    st.write("Employee Layer vs Span")
    st.dataframe(span_df)

    # Age Bracket Group vs Designation with Location Dropdown
    age_bracket_designation = location_filtered_df.groupby(['AgeBracket', 'Designation']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    age_bracket_designation.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"Age Bracket Group vs Designation for Location: {', '.join(selected_locations)}")

    ax.set_ylabel("Count")
    ax.set_xlabel("Age Bracket")
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')
    st.pyplot(fig)
    save_and_download_figure(fig, 'age_bracket_vs_designation.png')
    plt.clf()

    # Age Group vs Span DataFrame
    age_span_df = location_filtered_df.groupby('AgeBracket')[span_columns].sum().reset_index()
    st.write("Age Group vs Span")
    st.dataframe(age_span_df)

    # CTC Range Designation Wise Vertical Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='CTC in lakhs', y='Designation', data=location_filtered_df, orient='h', ax=ax)
    ax.set_title("CTC Range Designation Wise")
    st.pyplot(fig)
    save_and_download_figure(fig, 'ctc_range_designation_wise.png')
    plt.clf()

    # Number of People in Each Age Group for Each Layer Wise
    age_layer_df = location_filtered_df.groupby(['AgeBracket', 'Employee Layer']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    age_layer_df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Number of People in Each Age Group for Each Layer")
    ax.set_ylabel("Count")
    ax.set_xlabel("Age Bracket")
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')
    st.pyplot(fig)
    save_and_download_figure(fig, 'age_group_vs_layer.png')
    plt.clf()

def plot_employee_layers_by_location(cleaned_df):
    st.write("### Box Plot of Number of Employees by Employee Layer")
    
    # Get unique locations
    locations = cleaned_df['Location'].unique()

    # Add 'Overall' option to locations
    locations = ['Overall'] + list(locations)

    # Multiselect for selecting multiple locations
    selected_locations = st.multiselect("Select Location", locations, default=["Overall"])

    # Filter data based on selected locations
    if "Overall" in selected_locations:
        filtered_df = cleaned_df
        location_text = "Overall"
    else:
        filtered_df = cleaned_df[cleaned_df['Location'].isin(selected_locations)]
        location_text = ', '.join(selected_locations)

    # Prepare data for plotting
    employee_layer_counts = filtered_df['Employee Layer'].value_counts().sort_index()
    layer_names = employee_layer_counts.index
    counts = employee_layer_counts.values

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(layer_names, counts, align='center', color='b')
    ax.barh(layer_names, -counts, align='center', color='b')
    ax.invert_yaxis()
    ax.set_title(f'Symmetrical Number of Employees by Employee Layer for {location_text}')
    ax.set_xlabel('Number of Employees')
    ax.set_ylabel('Employee Layer')
    ax.xaxis.set_visible(False)
    for i, v in enumerate(counts):
        ax.text(v + 0.2, i, str(v), color='blue', va='center')
        ax.text(-v - 0.2, i, str(v), color='blue', va='center')
    st.pyplot(fig)
    save_and_download_figure(fig, 'symmetrical_employee_layer.png')

def create_dataframes(cleaned_df):
    span_columns = ['Span =1', 'Span =2', 'Span =3', 'Span =4', 'Span 5-8', 'Span >10']
    span_df = cleaned_df.groupby('Employee Layer')[span_columns].sum().reset_index()
    
    location = st.selectbox("Select Location", cleaned_df['Location'].unique(), key="location_selectbox")
    location_filtered_df = cleaned_df[cleaned_df['Location'] == location]
    age_bracket_designation = location_filtered_df.groupby(['AgeBracket', 'Designation']).size().unstack().fillna(0)
    
    age_span_df = cleaned_df.groupby('AgeBracket')[span_columns].sum().reset_index()
    
    age_layer_df = cleaned_df.groupby(['AgeBracket', 'Employee Layer']).size().unstack().fillna(0)
    
    return span_df, age_bracket_designation, age_span_df, age_layer_df

import uuid
def create_age_dataframes(cleaned_df):
    # Get unique locations
    locations = cleaned_df['Location'].unique()

    # Add 'Overall' option to locations
    locations = ['Overall'] + list(locations)

    # Generate a unique key for the multiselect widget
    multiselect_key = "location_multiselect_" + str(uuid.uuid4())

    # Multiselect for selecting multiple locations with a unique key
    selected_locations = st.multiselect("Select Location", locations, default=["Overall"], key=multiselect_key)

    # Filter data based on selected locations
    if "Overall" in selected_locations:
        location_filtered_df = cleaned_df
    else:
        location_filtered_df = cleaned_df[cleaned_df['Location'].isin(selected_locations)]

    # Layer-wise dataframe with age bracket
    layer_age_df = location_filtered_df.groupby(['Employee Layer', 'AgeBracket']).size().unstack().fillna(0)
    st.write("### Layer-wise Dataframe with Age Bracket")
    st.dataframe(layer_age_df)

    # Designation-wise dataframe with age bracket
    designation_age_df = location_filtered_df.groupby(['Designation', 'AgeBracket']).size().unstack().fillna(0)
    st.write("### Designation-wise Dataframe with Age Bracket")
    st.dataframe(designation_age_df) 

import pandas as pd
import pygraphviz as pgv
import matplotlib.pyplot as plt
import streamlit as st
import io

def build_org_chart(df, head_codes):
    G = pgv.AGraph(strict=False, directed=True)
    
    # Set default node attributes with increased font size
    G.node_attr.update(shape='rectangle', style='filled', fillcolor='#D3D3D3', fontname='Helvetica', fontsize=20)
    G.edge_attr.update(fontname='Helvetica', fontsize=20)
    
    for index, row in df.iterrows():
        G.add_node(row['Employee Name'], label=row['Employee Name'])
        if not pd.isnull(row['Reporting Manager Name']):
            G.add_edge(row['Reporting Manager Name'], row['Employee Name'])

    # Adjust node and rank separation
    G.layout(prog='dot', args='-Gnodesep=2.0 -Granksep=3.0')  
    G.draw('employee_hierarchy_chart.png')  # Generate the graph image

    # Read the image and increase resolution using Matplotlib
    image = plt.imread('employee_hierarchy_chart.png')
    fig, ax = plt.subplots(figsize=(200, 200))  # Increase figure size further
    ax.imshow(image)
    ax.axis('off')

    st.pyplot(fig)

def preprocess_data(df):
    df = df.dropna(subset=['Employee Name', 'Employee Code'])
    df['Reporting Manager Name'] = df['Reporting Manager Name'].fillna('')
    return df
def generate_department_chart(df, selected_department):
    # Initialize graph
    G = pgv.AGraph(strict=False, directed=True)
    
    # Set default node attributes
    G.node_attr.update(shape='rectangle', style='filled', fillcolor='#D3D3D3', fontname='Helvetica', fontsize=20)
    G.edge_attr.update(fontname='Helvetica', fontsize=20)

    # Create a list of head code employees
    head_codes = ['A123', 'B456', 'C789']  # Replace with actual head codes
    
    # Add head code employees to the graph
    for head_code in head_codes:
        if head_code in df['Employee Code'].values:  # Ensure head code exists in the dataframe
            head_code_name = df.loc[df['Employee Code'] == head_code, 'Employee Name'].values[0]
            G.add_node(head_code_name, label=head_code_name)

    # Create a sub-dataframe for the selected department
    df_selected_dept = df[df['Department'] == selected_department]

    # Add nodes and edges for employees in the selected department
    for index, row in df_selected_dept.iterrows():
        if row['Employee Code'] in head_codes:
            continue  # Skip head code employees already added
        G.add_node(row['Employee Name'], label=row['Employee Name'])
        if not pd.isnull(row['Reporting Manager Name']):
            if row['Reporting Manager Name'] in df_selected_dept['Employee Name'].values:
                G.add_edge(row['Reporting Manager Name'], row['Employee Name'])

    # Layout adjustments
    G.layout(prog='dot')  # Use default layout (dot)

    # Create a figure
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')

    # Draw the graph
    G.draw('temp_department_chart.png')

    # Read and display the image
    department_chart_image = plt.imread('temp_department_chart.png')
    ax.imshow(department_chart_image)

    # Save the final figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    return fig, buf


# Define lookup_employee_code function and other necessary functions here

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Function definitions and imports are unchanged

# Set page configuration and title
st.set_page_config(page_title="EY Project 1", page_icon="🌐")

# EY logo and title
st.image("EY_logo_2019.svg.png", width=200)
st.title('EY\'s Automated Data Cleaning and Validation')

page_bg_img = '''
    <style>
    body {
    background-color: #daedf4;
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar description
st.sidebar.write("""
This automated data cleaning and validation tool processes employee data to ensure accuracy and consistency. It includes functionalities for:
- Data Validation: Checks for correct column names, data types, and identifies missing values.
- Hierarchy Assignment: Automatically assigns hierarchical layers and spans based on reporting relationships.
- Visualization: Generates interactive visualizations such as employee layer distribution, age demographics, and CTC ranges.
- Organizational Charts: Automatically builds and visualizes organizational hierarchy charts based on input data.

Upload your employee dataset, specify head employee codes, and explore various analytics and insights tailored to your organization's structure and demographics.
""")

# File upload and initial data display
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Uploaded File Details:")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("Column names:")
    st.write(df.columns.tolist())

    st.write("### Uploaded DataFrame:")
    st.write(df)

    # Employee codes input and processing
    head_codes = st.text_input("Enter Employee Codes of the heads (comma-separated):", "")

    if head_codes:
        head_codes = [int(code.strip()) for code in head_codes.split(",")]
        head_df = df[df['Employee Code'].isin(head_codes)]
        st.write("### Head Employees:")
        st.write(head_df)

        continue_with_heads = st.radio("Do you wish to continue with these employees as heads?", ("Yes", "No"), key="head_confirm")

        if continue_with_heads == "No":
            head_names = st.text_input("Enter names of employee heads (comma-separated):", "", key="head_names_input")

            if head_names:
                head_names_list = [name.strip() for name in head_names.split(",")]
                head_codes = [lookup_employee_code(df, name) for name in head_names_list]
                head_codes = [code for code in head_codes if code != "no employee code"]

        cleaned_df, results = validate_employee_data(df)

        # Data cleaning and validation results
        cleaned_df = assign_employee_layers(cleaned_df, head_codes)
        cleaned_df = assign_ro_layer(cleaned_df)
        cleaned_df = assign_pm_ic_span(cleaned_df)

        st.write("### Validation Results:")
        st.write("**Column Check Results:**")
        st.write(results["column_check"])
            
        st.write("**Data Type Check Results:**")
        for line in results["data_type_check"].strip().split('\n'):
            st.write(line)

        st.write("**Null Value Check and Count:**")
        st.write(results["null_check"])

        st.write("**Check for Rows with Missing 'Employee Code' and 'Employee Name':**")
        st.write(results["missing_check"])

        for line in results["length_check"].split('\n'):
            st.write(line)

        st.write("### Summary of Changes Made:")
        st.write(results["changes_summary"])
        st.write("- Added 'Employee Layer' column to indicate hierarchy level.")
        st.write("- Added 'RO Layer' column to indicate hierarchy level of Reporting Officers.")
        st.write("- Added 'PM/IC' column to indicate if the employee is a PM or IC.")
        st.write("- Added 'Span' column to indicate number of direct subordinates.")
        st.write("- Added 'Span =1', 'Span =2', 'Span =3', 'Span =4', 'Span 5-8', 'Span >10' columns to indicate span ranges.")

        st.write("### Cleaned DataFrame:")
        st.write(cleaned_df)

        plot_employee_layers_by_location(cleaned_df)
        create_visualizations(cleaned_df)
        create_dataframes(cleaned_df)
        create_age_dataframes(cleaned_df)

        if st.button('Generate Employee Hierarchy Chart'):
            build_org_chart(cleaned_df, head_codes)

        # Department selection and chart generation
        departments = cleaned_df['Department'].unique()
        selected_department = st.selectbox("Select Department:", departments)

        # Filtered department chart
        filtered_df = cleaned_df[cleaned_df['Department'] == selected_department]
        fig, buf = generate_department_chart(filtered_df,selected_department)
        st.pyplot(fig)

        # Download button for organizational chart
        st.download_button(
            label="Download Image",
            data=buf,
            file_name="employee_hierarchy_chart.png",
            mime="image/png"
        )
