<<<<<<< HEAD
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def process_data(df):
    # Replace negative values with NaN
    df[['GHI', 'DNI', 'DHI']] = df[['GHI', 'DNI', 'DHI']].applymap(lambda x: np.nan if x < 0 else x)
    # Drop "Comments" (No use)
    df.drop("Comments", axis=1, inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)

def data_info(df):
    results = {}
    results['summary_stats'] = Describe_data(df)
    results['data_types'] = Data_type(df)
    results['null_counts'] = Sum_of_null_datas(df)
    results['negative_counts'] = negative_counts(df)
    return results

def visualize_correlation(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    return plt

def Calculate_Z_Score(df):
    data = df
    numeric_columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'TModA', 'TModB']
    # Calculate Z-scores for selected columns
    z_scores = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()

    # Define threshold for outlier detection
    threshold = 3

    # Identify outliers
    outliers = data[(np.abs(z_scores) > threshold).any(axis=1)]

    # Calculate the total number of observations
    total_observations = len(data)
    number_of_outliers = len(outliers)

    # Calculate the percentage of outliers
    percentage_outliers = (number_of_outliers / total_observations)
    percentage_outliers_formatted = "{:.3%}".format(percentage_outliers)
    # Return the formatted percentage_outliers
    return "Percentage of outliers:", percentage_outliers_formatted


def TimeStamp(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Resample the DataFrame to a lower frequency (e.g., daily)
    df_resampled = df.set_index('Timestamp').resample('M').mean()  # Resample to monthly frequency and compute mean
    
    # Plot time series data using Streamlit's line_chart function
    st.line_chart(df_resampled)


def Correlation_matrix(df):
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Plot correlation matrix heatmap for visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    
    # Display the plot using Streamlit
    st.pyplot()


def wind_speed_and_wind_direction_over_time(df):
    # Convert 'Timestamp' column to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Set 'Timestamp' column as index
    df.set_index('Timestamp', inplace=True)
    
    # Downsample the data to reduce the number of data points
    df_resampled = df.resample('10H').mean()  # Downsample to 1 hour intervals
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled.index, df_resampled['WS'], label='Wind Speed')
    plt.plot(df_resampled.index, df_resampled['WD'], label='Wind Direction')
    plt.xlabel('Timestamp')
    plt.ylabel('Values')
    plt.title('Wind Analysis')
    plt.legend()
    return plt.gcf()  # Return the current figure object


def Temp_analysis(df):
    # Convert 'Timestamp' column to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Set 'Timestamp' column as index
    df.set_index('Timestamp', inplace=True)
    
    # Downsample the data to reduce the number of data points
    df_resampled = df.resample('1H').mean()  # Downsample to 1 hour intervals
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled.index, df_resampled['TModA'], label='Module A Temperature')
    plt.plot(df_resampled.index, df_resampled['TModB'], label='Module B Temperature')
    plt.plot(df_resampled.index, df_resampled['Tamb'], label='Ambient Temperature')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Analysis')
    plt.legend()
    return plt.gcf()  # Return the current figure object


def generate_histograms(df):
    # Generate histograms for each numeric column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(12, 10))
        df[col].hist()
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {col}')
        st.pyplot()  # Display the plot using Streamlit


def generate_box_plots(df):
    # Select columns for box plots
    columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'TModA', 'TModB']
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    df[columns].boxplot()
    plt.title('Box Plot of Solar Radiation and Temperature Data')
    plt.ylabel('Values')
    st.pyplot()  # Display the plot using Streamlit


def generate_scatter_plots(df):
    # Create scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df['GHI'], df['Tamb'], alpha=0.5)
    plt.xlabel('GHI')
    plt.ylabel('Tamb')
    plt.title('Scatter Plot: GHI vs. Ambient Temperature')
    st.pyplot()  # Display the plot using Streamlit


def Describe_data(df):
    summary_stats = df.describe()
    return summary_stats


def Data_type(df):
    return df.dtypes


def Sum_of_null_datas(df):
    return df.isnull().sum()


def negative_counts(df):
    negative_counts = df[['GHI', 'DNI', 'DHI']].lt(0).sum()
    return negative_counts
=======
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def process_data(df):
    # Replace negative values with NaN
    df[['GHI', 'DNI', 'DHI']] = df[['GHI', 'DNI', 'DHI']].applymap(lambda x: np.nan if x < 0 else x)
    # Drop "Comments" (No use)
    df.drop("Comments", axis=1, inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)

def data_info(df):
    results = {}
    results['summary_stats'] = Describe_data(df)
    results['data_types'] = Data_type(df)
    results['null_counts'] = Sum_of_null_datas(df)
    results['negative_counts'] = negative_counts(df)
    return results

def visualize_correlation(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    return plt

def Calculate_Z_Score(df):
    data = df
    numeric_columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'TModA', 'TModB']
    # Calculate Z-scores for selected columns
    z_scores = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()

    # Define threshold for outlier detection
    threshold = 3

    # Identify outliers
    outliers = data[(np.abs(z_scores) > threshold).any(axis=1)]

    # Calculate the total number of observations
    total_observations = len(data)
    number_of_outliers = len(outliers)

    # Calculate the percentage of outliers
    percentage_outliers = (number_of_outliers / total_observations)
    percentage_outliers_formatted = "{:.3%}".format(percentage_outliers)
    # Return the formatted percentage_outliers
    return "Percentage of outliers:", percentage_outliers_formatted


def TimeStamp(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Resample the DataFrame to a lower frequency (e.g., daily)
    df_resampled = df.set_index('Timestamp').resample('M').mean()  # Resample to monthly frequency and compute mean
    
    # Plot time series data using Streamlit's line_chart function
    st.line_chart(df_resampled)


def Correlation_matrix(df):
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Plot correlation matrix heatmap for visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    
    # Display the plot using Streamlit
    st.pyplot()


def wind_speed_and_wind_direction_over_time(df):
    # Convert 'Timestamp' column to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Set 'Timestamp' column as index
    df.set_index('Timestamp', inplace=True)
    
    # Downsample the data to reduce the number of data points
    df_resampled = df.resample('10H').mean()  # Downsample to 1 hour intervals
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled.index, df_resampled['WS'], label='Wind Speed')
    plt.plot(df_resampled.index, df_resampled['WD'], label='Wind Direction')
    plt.xlabel('Timestamp')
    plt.ylabel('Values')
    plt.title('Wind Analysis')
    plt.legend()
    return plt.gcf()  # Return the current figure object


def Temp_analysis(df):
    # Convert 'Timestamp' column to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Set 'Timestamp' column as index
    df.set_index('Timestamp', inplace=True)
    
    # Downsample the data to reduce the number of data points
    df_resampled = df.resample('1H').mean()  # Downsample to 1 hour intervals
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled.index, df_resampled['TModA'], label='Module A Temperature')
    plt.plot(df_resampled.index, df_resampled['TModB'], label='Module B Temperature')
    plt.plot(df_resampled.index, df_resampled['Tamb'], label='Ambient Temperature')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Analysis')
    plt.legend()
    return plt.gcf()  # Return the current figure object


def generate_histograms(df):
    # Generate histograms for each numeric column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(12, 10))
        df[col].hist()
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {col}')
        st.pyplot()  # Display the plot using Streamlit


def generate_box_plots(df):
    # Select columns for box plots
    columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'TModA', 'TModB']
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    df[columns].boxplot()
    plt.title('Box Plot of Solar Radiation and Temperature Data')
    plt.ylabel('Values')
    st.pyplot()  # Display the plot using Streamlit


def generate_scatter_plots(df):
    # Create scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df['GHI'], df['Tamb'], alpha=0.5)
    plt.xlabel('GHI')
    plt.ylabel('Tamb')
    plt.title('Scatter Plot: GHI vs. Ambient Temperature')
    st.pyplot()  # Display the plot using Streamlit


def Describe_data(df):
    summary_stats = df.describe()
    return summary_stats


def Data_type(df):
    return df.dtypes


def Sum_of_null_datas(df):
    return df.isnull().sum()


def negative_counts(df):
    negative_counts = df[['GHI', 'DNI', 'DHI']].lt(0).sum()
    return negative_counts
>>>>>>> 6e38da45511046deb08911513fa26bc263ad158d
