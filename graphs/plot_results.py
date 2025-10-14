import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Load Factor Analysis
df_lf = pd.read_csv('load_factor_analysis.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Insert Time vs Load Factor
for impl in df_lf['Implementation'].unique():
    data = df_lf[df_lf['Implementation'] == impl]
    axes[0, 0].plot(data['LoadFactor'], data['AvgInsertTime'], marker='o', label=impl)
axes[0, 0].set_xlabel('Load Factor')
axes[0, 0].set_ylabel('Avg Insert Time (ns)')
axes[0, 0].set_title('Insert Time vs Load Factor')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Find Time vs Load Factor
for impl in df_lf['Implementation'].unique():
    data = df_lf[df_lf['Implementation'] == impl]
    axes[0, 1].plot(data['LoadFactor'], data['AvgFindTime'], marker='o', label=impl)
axes[0, 1].set_xlabel('Load Factor')
axes[0, 1].set_ylabel('Avg Find Time (ns)')
axes[0, 1].set_title('Find Time vs Load Factor')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Average Probes vs Load Factor
for impl in df_lf['Implementation'].unique():
    data = df_lf[df_lf['Implementation'] == impl]
    axes[1, 0].plot(data['LoadFactor'], data['AvgProbes'], marker='o', label=impl)
axes[1, 0].set_xlabel('Load Factor')
axes[1, 0].set_ylabel('Average Probes')
axes[1, 0].set_title('Average Probes vs Load Factor')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Max Length vs Load Factor
for impl in df_lf['Implementation'].unique():
    data = df_lf[df_lf['Implementation'] == impl]
    axes[1, 1].plot(data['LoadFactor'], data['MaxLength'], marker='o', label=impl)
axes[1, 1].set_xlabel('Load Factor')
axes[1, 1].set_ylabel('Max Chain/Probe Length')
axes[1, 1].set_title('Max Length vs Load Factor')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('load_factor_analysis.png', dpi=300)
plt.show()

# Scalability Analysis
df_scale = pd.read_csv('scalability_analysis.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for impl in df_scale['Implementation'].unique():
    data = df_scale[df_scale['Implementation'] == impl]
    axes[0].plot(data['DataSize'], data['TotalInsertTime'], marker='o', label=impl)
axes[0].set_xlabel('Data Size')
axes[0].set_ylabel('Total Insert Time (ms)')
axes[0].set_title('Scalability: Total Insert Time')
axes[0].legend()
axes[0].grid(True)

for impl in df_scale['Implementation'].unique():
    data = df_scale[df_scale['Implementation'] == impl]
    axes[1].plot(data['DataSize'], data['AvgInsertTime'], marker='o', label=impl)
axes[1].set_xlabel('Data Size')
axes[1].set_ylabel('Avg Insert Time (ns)')
axes[1].set_title('Scalability: Average Insert Time')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('scalability_analysis.png', dpi=300)
plt.show()
