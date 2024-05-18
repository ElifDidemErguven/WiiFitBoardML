import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import yeojohnson, skew

# Function to correct skewness
def correct_skewness(df, columns):
    for col in columns:
        # Compute skewness
        skew_val = skew(df[col])
        
        # Apply transformations based on skewness
        if skew_val > 1:
            df[col] = np.log1p(df[col])  # log1p handles zero values, Log Transformation: for positively skewed data.
        elif skew_val > 0.5:
            df[col] = np.sqrt(df[col])  #Square Root Transformation for moderately skewed data.
        else:
            df[col], _ = yeojohnson(df[col])  # Yeo-Johnson for negatively skewed data.
    return df

def determine_threshold(data, initial_portion=0.1, multiplier=2):
    initial_data = data[:int(len(data) * initial_portion)]
    mean_initial = np.mean(initial_data)
    std_initial = np.std(initial_data)
    threshold = mean_initial + multiplier * std_initial
    return threshold

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def detect_stable_start(data, threshold, window_size):
    smoothed_data = moving_average(data, window_size)
    for i in range(len(smoothed_data)):
        if smoothed_data[i] > threshold:
            return i + window_size - 1  # Adjust for the length of the moving average window
    return 0  # Default to the start if no stable point found

def clean_data(data, start_index):
    return data[start_index:]

# Set up folder paths
folder_path = "./CSV Files"
combined_data_path = "./combined_data.csv"
standardized_data_folder = "./Standardized Csv Files"
visualization_folder_path = "./Data Visualization"

#List to hold data frames
data_frames = []

# Data Loading
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        print(f"Processing file: {file}")
        parts = file.split("-")
        if len(parts) < 2:
            print(f"Skipping csv due to unexpected filename format: {file}")
            continue

        participant = parts[0]
        correctness = parts[1].replace(".csv", "")
        label = 1 if correctness == "good" else 0

        df = pd.read_csv(os.path.join(folder_path, file))
        df.columns = df.columns.str.strip().str.replace(r'\s+', " ", regex=True)
        df.rename(columns=lambda x: x.strip().replace("  ", " "), inplace=True)

        df["label"] = label
        df["participant"] = participant
        df = df.drop("ms", axis=1)
        window_size=int(len(df["Total Force (kg)"])*(30/100))
        threshold = determine_threshold(df["Total Force (kg)"])
        start_index = detect_stable_start(df["Total Force (kg)"], threshold, window_size)
        df = clean_data(df, start_index)
        data_frames.append(df)



def plot(data,title):
    window_size=int(len(data)*(20/100))
    threshold = determine_threshold(data)
    start_index = detect_stable_start(data, threshold, window_size)
    cleaned_data = clean_data(data, start_index)

    plt.figure()
    plt.title(title)
    plt.plot(data)

    plt.plot(cleaned_data)

# Combine everything in the data frames list into one DataFrame
full_data = pd.concat(data_frames, ignore_index=True)

# Save the combined data to a CSV file
full_data.to_csv(combined_data_path, index=False)

# the combined data is now fd, checking fd's shape and missing values
fd = pd.read_csv(combined_data_path)
print(f"Data shape: {fd.shape}")
print(f"Total number of null values: {fd.isnull().sum().sum()}")

#copy the data before any process as "raw data" which will be used later for data visualization
raw_data = fd.copy()

# Apply skewness correction, (label and participant is dropped)
numeric_columns = fd.select_dtypes(include=[np.number]).columns.drop(["label"])
print("HERE ARE THE NUMERIC COLUMNS" , numeric_columns)

# Apply skewness correction
fd_corrected = correct_skewness(fd, numeric_columns)

# Standardize the skewness corrected data
scaler = MinMaxScaler(feature_range=(0, 1))
X = fd_corrected[numeric_columns]
X_scaled = scaler.fit_transform(X)
fd_standardized = pd.DataFrame(X_scaled, columns=numeric_columns)

# Add 'label' and 'participant' back to the standardized DataFrame
fd_standardized["label"] = fd_corrected["label"].values
fd_standardized["participant"] = fd_corrected["participant"].values

# Save standardized features and labels separately
features = fd_standardized.drop(columns=["label", "participant"])
labels = fd_standardized["label"]

features.to_csv(os.path.join(standardized_data_folder, "features.csv"), index=False)
labels.to_csv(os.path.join(standardized_data_folder, "labels.csv"), index=False)

# Save fully processed data
fd_standardized.to_csv(combined_data_path, index=False)

# Visualization function to save figures
def visualize_data(df_before, df_after, title_before, title_after, columns, process):
    sns.set_theme(style="whitegrid")
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.histplot(df_before[col], bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f"{title_before}: {col}")
        
        sns.histplot(df_after[col], bins=30, kde=True, ax=axes[1])
        axes[1].set_title(f"{title_after}: {col}")
        
        plt.tight_layout()
        fig.savefig(os.path.join("./", f"{process}_{col}.png"))
        plt.close(fig)

# Visualization before and after skewness correction
#visualize_data(raw_data, fd_corrected, "Before Skewness Correction", "After Skewness Correction", numeric_columns, "Skewness_Correction")

# Visualization before and after standardization
#visualize_data(fd_corrected, fd_standardized, "After Skewness Correction", "After Standardization", numeric_columns, "Standardization")

# Visualization raw vs fully processed
#visualize_data(raw_data, fd_standardized, "Raw Data", "Fully Processed Data", numeric_columns, "Raw_vs_Fully_Processed")
