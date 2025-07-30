# visualize the coverage cost over time

import pandas as pd
import matplotlib.pyplot as plt 
model_path = "./model_saved_3hour_flood3"
coverage_file_name1 = "coverage_cost_default_map"
coverage_file_name2 = "coverage_cost_no_default_map"

coverage_data_path1 = f"{model_path}/{coverage_file_name1}.csv"
coverage_data1 = pd.read_csv(coverage_data_path1, header=None)
coverage_data_path2 = f"{model_path}/{coverage_file_name2}.csv"
coverage_data2 = pd.read_csv(coverage_data_path2, header=None)

# Plot the coverage cost for both scenarios
plt.figure(figsize=(10, 6))
plt.plot(coverage_data1, marker='o', linestyle='-', color='b', label='Coverage Cost with Default Map')
plt.plot(coverage_data2, marker='x', linestyle='--', color='r', label='Coverage Cost without Default Map')
plt.title('Coverage Cost Over Time')
plt.xlabel('Timestep')
plt.ylabel('Coverage Cost')
plt.grid()
# max_value2 = 1000
# plt.ylim(0, max_value2 )
plt.xticks(range(len(coverage_data1)), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{model_path}/fig_eval/coverage_cost_comparison.png")



