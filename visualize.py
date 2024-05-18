import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


'''def visualize_random_csv(csv_path):
    for file in csv_path:
        df2 = pd.read_csv(file)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    plt.tight_layout()
    fig.savefig(os.path.join("./", f"{process}_{col}.png"))
    plt.close(fig)
'''

csv_path= "./CSV Files"
csv_files_good = [f for f in os.listdir(csv_path) if f.endswith('good.csv')]
csv_files_bad = [f for f in os.listdir(csv_path) if f.endswith('bad.csv')]

random_csv_file1 = random.choice(csv_files_good)
file_path1 = os.path.join(csv_path, random_csv_file1)

random_csv_file0 = random.choice(csv_files_bad)
file_path0 = os.path.join(csv_path, random_csv_file0)

df1 = pd.read_csv(file_path1)
df0 = pd.read_csv(file_path0)
combined_df = pd.read_csv("combined_data.csv")

columns_tobe_plot = ["Sensor 1 (kg)","Sensor 2 (kg)","Sensor 3 (kg)","Sensor 4 (kg)"]

plt.figure(figsize=(10,6))

def plot_line(df, columns, filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for col in columns:
        ax1.plot(df.index, df[col], label=col)

    ax1.bar(df.index, df[df.columns[7]], color='gray', alpha=0.3, label=df.columns[7])
    ax1.set_xlabel('Index')
    ax1.set_ylabel('KG Values')
    ax1.set_title(f'Line Plot of 4 Columns with Bar Plot of 1 Column from {filename}')
    
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')
    plt.show()

plot_line(df1, columns_tobe_plot, random_csv_file1)
plot_line(df0, columns_tobe_plot, random_csv_file0)
#plot_line(combined_df, columns_tobe_plot)


