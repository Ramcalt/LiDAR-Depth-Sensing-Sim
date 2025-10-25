# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ----------------------------
# # Step 1: Load the CSV file
# # ----------------------------
# # Save your provided data as, e.g., "zones.csv"
# csv_file = "zone_heatmap.csv"
# df = pd.read_csv(csv_file)
#
# # Set 'truth' column as index
# df.set_index('truth', inplace=True)
#
# # ----------------------------
# # Step 2: Function to reshape 1D zone data into 8×8 grid
# # ----------------------------
# def zones_to_grid(zone_values):
#     """
#     Converts a flat list of 64 zone values into an 8×8 grid.
#     The mapping matches the figure:
#       - Zone 0 is bottom-left
#       - Zone 7 is bottom-right
#       - Zone 56 is top-left
#       - Zone 63 is top-right
#     """
#     grid = np.zeros((8, 8))
#     for i in range(8):
#         for j in range(8):
#             grid[7 - i, j] = zone_values[i * 8 + j]
#     return grid
#
# # ----------------------------
# # Step 3: Compute difference from "ground truth" (mean or reference)
# # ----------------------------
# def compute_closeness(zone_values):
#     """
#     Calculates absolute difference from mean (ground truth) for each zone.
#     """
#     ground_truth = np.mean(zone_values)
#     diff = np.abs(zone_values - ground_truth)
#     return diff, ground_truth
#
# # ----------------------------
# # Step 4: Plot heatmaps for each truth level
# # ----------------------------
# plt.figure(figsize=(12, 10))
#
# for idx, truth_value in enumerate(df.index):
#     zone_values = df.loc[truth_value].values.astype(float)
#     diff, ref = compute_closeness(zone_values)
#     grid = zones_to_grid(diff)
#
#     plt.subplot(2, 2, idx + 1)
#     sns.heatmap(grid, cmap='coolwarm', square=True, cbar_kws={'label': 'Abs. Difference from Mean'})
#     plt.title(f"Truth = {truth_value}  (Ref ≈ {ref:.4f})", fontsize=12)
#     plt.xlabel("X (Columns)")
#     plt.ylabel("Y (Rows)")
#
# plt.tight_layout()
# plt.show()
#
#
#
#
# # ======================== LINE OF BEST FIT COMPARISON ========================== #
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ----------------------------
# # Step 1: Load CSV data
# # ----------------------------
# csv_file = "zone_heatmap.csv"  # replace with your actual path
# df = pd.read_csv(csv_file)
#
# # Convert column names (except 'truth') to integers
# df.columns = [int(c) if c != 'truth' else c for c in df.columns]
#
# # Set 'truth' as index
# df.set_index('truth', inplace=True)
#
# # ----------------------------
# # Step 2: Fit linear regression for each zone across truths
# # ----------------------------
# truth_levels = df.index.values.astype(float)
# zone_columns = [c for c in df.columns if c != 'truth']  # already ints
#
# offsets = []  # store intercept (offset) of best-fit line for each zone
#
# for zone in zone_columns:
#     y = df[str(zone)].values if str(zone) in df.columns else df[zone].values
#     x = truth_levels
#     m, b = np.polyfit(x, y, 1)  # slope, intercept
#     offsets.append(b)
#
# offsets = np.array(offsets)
#
# # ----------------------------
# # Step 3: Map offsets into 8×8 spatial grid
# # ----------------------------
# def zones_to_grid(values):
#     """Maps 64 zone values (0–63) into an 8×8 grid."""
#     grid = np.zeros((8, 8))
#     for i in range(8):
#         for j in range(8):
#             grid[7 - i, j] = values[i * 8 + j]
#     return grid
#
# offset_grid = zones_to_grid(offsets)
#
# # ----------------------------
# # Step 4: Compute deviation from ground truth offset (0)
# # ----------------------------
# offset_diff = offset_grid  # since ground truth = 0
#
# # ----------------------------
# # Step 5: Plot the heatmap
# # ----------------------------
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     offset_diff,
#     cmap='coolwarm',
#     center=0,
#     square=True,
#     cbar_kws={'label': 'Offset from Ground Truth (Intercept)'}
# )
# plt.title("Zone Offset Map: Line of Best Fit vs Ground Truth (0)")
# plt.xlabel("X (columns)")
# plt.ylabel("Y (rows)")
# plt.tight_layout()
# plt.show()
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: LOAD THE CSV DATA ===
# (Assumes the file is named 'zone_truth_data.csv')
data = pd.read_csv("zone_heatmap.csv")

# Extract the ground truth values (first column)
truth_values = data['truth'].values

# Drop the 'truth' column for comparison data
zone_data = data.drop(columns=['truth'])

# === STEP 2: COMPUTE ABSOLUTE DIFFERENCES ===
# For each row (truth value), compute the difference to each zone
diff_matrices = []
for i, truth in enumerate(truth_values):
    diff = np.abs(zone_data.iloc[i] - truth)
    diff_matrices.append(diff.values.reshape(8, 8))  # reshape to match 8x8 layout

# === STEP 3: PLOT HEATMAPS FOR EACH TRUTH VALUE ===
for i, truth in enumerate(truth_values):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        diff_matrices[i],
        annot=True,
        fmt=".3f",
        cmap="coolwarm_r",
        cbar_kws={'label': 'Absolute Difference from Truth'},
        square=True
    )
    plt.title(f"Heatmap of Zone Differences (Truth = {truth})")
    plt.xlabel("X-axis (Zone Columns)")
    plt.ylabel("Y-axis (Zone Rows)")
    plt.gca().invert_yaxis()  # to align with your diagram numbering
    plt.tight_layout()
    plt.show()
