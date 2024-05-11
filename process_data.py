import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import skew, yeojohnson, probplot


folder_path = './WiiFitBoardML/CSV Files'  
data_frames = []

##Data Loading

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        print("Processing file:", file)
        parts = file.split('-')
        if len(parts) < 2:
            print("Skipping file due to unexpected filename format:", file)
            continue
        participant = parts[0]
        correctness = parts[1].replace('.csv', '')
        
        df = pd.read_csv(os.path.join(folder_path, file))
        
        ##Data Cleaning
        
        df = pd.read_csv(os.path.join(folder_path, file), on_bad_lines='skip')
        df = pd.read_csv(os.path.join(folder_path, file), delimiter=',')
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
        df.rename(columns=lambda x: x.strip().replace("  ", " "), inplace=True)
        
        if correctness == "good":
            label = 1
        else:
            label = 0

        df['label'] = label
        df['participant'] = participant
        data_frames.append(df)

full_data = pd.concat(data_frames, ignore_index=True)
full_data.to_csv('./WiiFitBoardML/combined_data.csv', index=False)

##Missing Data 

fd = pd.read_csv('./WiiFitBoardML/combined_data.csv')
print(fd.shape)

#print(fd.isnull().sum())
#print("Total number of null values ", fd.isnull().sum().sum())

#Skewness
numeric_columns = fd.select_dtypes(include=[np.number]).columns.drop("label", "participant")
initial_skewness = fd[numeric_columns].apply(skew)
print("Initial Skewness:\n", initial_skewness)

# Check skewness and apply transformations
for col in numeric_columns:
    fd[col] += 0.001  # Shift all values slightly to handle zeros in log transformations
    skew_val = skew(fd[col])
    # For positive skewness > 1
    if skew_val > 1:
        # Initially apply log transformation
        transformed = np.log1p(fd[col])
        # Recheck skewness to see if further adjustment is needed
        if skew(transformed) > 1:  # Check if still highly skewed
            fd[col] = np.sqrt(transformed + 1)  # Apply a milder transformation
        else:
            fd[col] = transformed  # Use log-transformed data

    # For negative skewness < -1
    elif skew_val < -1:
        fd[col], _ = yeojohnson(fd[col])  # Apply Yeo-Johnson transformation

# Visualizing the effect of the transformation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(fd[col], bins=30, alpha=0.7)
    plt.title(f'Histogram of {col} After Transformation')

    plt.subplot(1, 2, 2)
    probplot(fd[col], dist="norm", plot=plt)
    plt.title(f'Q-Q plot for {col}')
    plt.show()

post_skewness = fd[numeric_columns].apply(skew)
print("Post-Transformation Skewness:\n", post_skewness)

# Save or continue processing as needed
fd.to_csv('./WiiFitBoardML/combined_data.csv', index=False)

##Data Standardization

# Separate label and participant
X = fd.drop(['label', 'participant'], axis=1)  
y = fd['label']  # Target Variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling to features

# Save the standardized features and labels to CSVs
pd.DataFrame(X_scaled, columns=X.columns).to_csv('./WiiFitBoardML/Standardized Csv Files/features.csv', index=False)
y.to_csv('./WiiFitBoardML/Standardized Csv Files/labels.csv', index=False)

# Checking summary statistics
pre_standardization_mean = X.mean()
pre_standardization_std = X.std()

print("Mean before standardization:\n", pre_standardization_mean)
print("Standard deviation before standardization:\n", pre_standardization_std)

# After standardization
standardized_mean = X_scaled.mean(axis=0)
standardized_std = X_scaled.std(axis=0)

print("Mean after standardization:\n", standardized_mean)
print("Standard deviation after standardization:\n", standardized_std)

# Visualizing distributions
feature_name = 'Sensor 1 (kg)'  

plt.figure(figsize=(10, 5))

# Histogram before standardization
plt.subplot(1, 2, 1)
plt.hist(X[feature_name], bins=30, alpha=0.7)
plt.title(f"Distribution of {feature_name} Before")

# Histogram after standardization
plt.subplot(1, 2, 2)
plt.hist(X_scaled[:, X.columns.get_loc(feature_name)], bins=30, alpha=0.7)
plt.title(f"Distribution of {feature_name} After")

plt.tight_layout()
plt.show()

