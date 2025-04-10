from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors

datas_up = pd.read_pickle(r'data/predictions/pred_pred_num_peak_up.pickle')
true_values_up = datas_up['test']['TRUE_NUM_PEAK'].astype(int)
predicted_values_up = datas_up['test']['PRED_NUM_PEAK'].astype(int)

datas_down = pd.read_pickle(r'data/predictions/pred_pred_num_peak_down.pickle')
true_values_down = datas_down['test']['TRUE_NUM_PEAK'].astype(int)
predicted_values_down = datas_down['test']['PRED_NUM_PEAK'].astype(int)

# RMSE
rmse_up = np.sqrt(mean_squared_error(true_values_up, predicted_values_up))
print("RMSE:", rmse_up)
rmse_down = np.sqrt(mean_squared_error(true_values_down, predicted_values_down))
print("RMSE:", rmse_down)

# R2
r2_up = r2_score(true_values_up, predicted_values_up)
print("R^2:", r2_up)
r2_down = r2_score(true_values_down, predicted_values_down)
print("R^2:", r2_down)

# Create the scatter plot
points = list(zip(true_values_up, predicted_values_up))
counts = Counter(points)  # Dictionary {(x, y): count}

# Convert data into lists with corresponding colors
colors = np.array([counts[point] for point in points])

# Scatter plot with count-based color
plt.figure(figsize=(6, 6), dpi=150)
norm = mcolors.Normalize(vmin=0, vmax=np.percentile(colors, 55))
sc = plt.scatter(true_values_up, predicted_values_up, c=colors, cmap='Blues', s=80, alpha=1, norm=norm)

# Add colorbar showing the number of overlapping points
cb = plt.colorbar(sc)
cb.set_label('Number of Overlapping Points')

min_val = min(min(true_values_up), min(predicted_values_up))
max_val = max(max(true_values_up), max(predicted_values_up))

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
plt.grid()
plt.show()

plt.figure(figsize=(10, 6), dpi=150)
plt.scatter(true_values_down, predicted_values_down, c='blue', alpha=0.01, edgecolor='w', s=200)
min_val = min(min(true_values_down), min(predicted_values_down))
max_val = max(max(true_values_down), max(predicted_values_down))
# plt.title('Fingerprint Region', fontsize=100 / 3.75)
plt.xlabel('True Values', fontsize=50 / 3.75)
plt.ylabel('Predicted Values', fontsize=50 / 3.75)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.5)
plt.grid()
