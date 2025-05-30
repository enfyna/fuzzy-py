import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from skfuzzy.cluster import cmeans
from matplotlib import pyplot as plt

# 1. Data Loading
df = pd.read_csv("./data/mcs_ds_edited_iter_shuffled.csv")

input_vars = ['anchor_ratio', 'trans_range(m)', 'node_density', 'iterations']
output_var = 'ale(m)'

input_specs = {}

# 2. Determine Optimum Cluster Count (Elbow Method Data)
print("--- Inertia values for K-Means (Elbow Method Data) ---")
for var_name in input_vars + [output_var]:
    data_column = df[var_name].values.reshape(-1, 1)
    # Optional: scale data for K-Means if ranges are very different
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_column)

    count = 0
    inertia = {}
    inertias = []
    print(f"\nVariable: {var_name}")
    for k in range(2, 7):  # Test K from 2 to 6
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)\
            .fit(scaled_data)
        print(f"  K={k}, Inertia: {kmeans.inertia_:.2f}")
        if kmeans.inertia_ <= 0.0001:
            break
        inertia[k] = kmeans.inertia_
        inertias.append(kmeans.inertia_)
        count += 1

    n_samples = scaled_data.shape[0]
    max_clusters = min(count + 2, n_samples - 1)

    fpc_scores = []
    cluster_range = range(2, max_clusters)

    print(f"Testing {len(cluster_range)} different cluster configurations...")

    for n_clusters in cluster_range:
        cntr, u, u0, d, jm, p, fpc = cmeans(
            scaled_data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        fpc_scores.append(fpc)
        print(f"  {n_clusters} clusters: FPC = {fpc:.3f} | run for: {p}")

    input_specs[var_name] = {}
    input_specs[var_name]['fpc_scores'] = fpc_scores
    input_specs[var_name]['inertias'] = inertias
    input_specs[var_name]['count'] = count

    ypoints = np.array(inertias)
    xpoints = np.array([i + 2 for i in range(count)])

    plt.plot(xpoints, ypoints, 'go', label='Inertias')
    plt.plot(xpoints, ypoints, 'g')

    ypoints = np.array([fpc_scores[i] for i in range(count)])
    xpoints = np.array([i + 2 for i in range(count)])

    plt.plot(xpoints, ypoints, 'ro', label='FPC Scores')
    plt.plot(xpoints, ypoints, 'r')

    plt.legend()
    # plt.show()

input_specs['anchor_ratio']['count'] = 3
input_specs['trans_range(m)']['count'] = 4
input_specs['node_density']['count'] = 2
input_specs['iterations']['count'] = 3
input_specs['ale(m)']['count'] = 3

# 3. Find Center Points (using K=3 for rule consistency)
cluster_centers_for_rules = {}
mf_labels = ['low', 'medium', 'high', 'very high']  # Linguistic labels for K=3

# print(
# f"\n--- Cluster Centers for K={N_CLUSTERS_FOR_RULES} \
# (for Rule Extraction) ---")

for var_name in input_vars + [output_var]:
    data_column = df[var_name].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=input_specs[var_name]['count'],
                    random_state=0, n_init='auto').fit(data_column)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    cluster_centers_for_rules[var_name] = centers
    print(f"\nVariable: {var_name}")
    print(f"=> Cluster Centers: {centers}")

# 4. Discretize Data for Rule Extraction (based on K=3 clusters)
df_fuzzy_labels = pd.DataFrame()

for var_name in input_vars + [output_var]:
    data_column = df[var_name].values
    centers = cluster_centers_for_rules[var_name]

    # Assign data points to the closest center's label
    discretized_labels = []
    for val in data_column:
        distances = [abs(val - center) for center in centers]
        closest_cluster_index = np.argmin(distances)
        discretized_labels.append(mf_labels[closest_cluster_index])
    df_fuzzy_labels[var_name] = discretized_labels

print("\n--- Sample Discretized Data (first 5 rows) ---")
print(df_fuzzy_labels.head(-1))

# 5. Extract and Print Fuzzy Rules
print("\n--- Extracted Fuzzy Rules ---")
rules_generated = []

# Helper to add rule if not a duplicate string-wise


def add_rule_if_new(rule_text, rule_list_texts):
    if rule_text not in rule_list_texts:
        rule_list_texts.add(rule_text)
        print(rule_text)
        return True
    return False


# Rule generation based on frequent patterns in df_fuzzy_labels
rule_texts_set = set()

# 4-input rules
rule_patterns_4d = df_fuzzy_labels.groupby(input_vars + [output_var]).size(
).reset_index(name='count').sort_values(by='count', ascending=False)
print("\n# 4-Input Rules:")
for _, row in rule_patterns_4d.head(5).iterrows():  # Aim for ~5
    rule_text = (f"IF {input_vars[0]} IS {row[input_vars[0]]} AND "
                 f"{input_vars[1]} IS {row[input_vars[1]]} AND "
                 f"{input_vars[2]} IS {row[input_vars[2]]} AND "
                 f"{input_vars[3]} IS {row[input_vars[3]]} "
                 f"THEN {output_var} IS {row[output_var]}")
    add_rule_if_new(rule_text, rule_texts_set)

# 3-input rules
print("\n# 3-Input Rules:")
target_3_input_rules = 10
current_3_input_count = 0
for combo in combinations(input_vars, 3):
    if current_3_input_count >= target_3_input_rules:
        break
    input_subset = list(combo)
    rule_patterns_3d = df_fuzzy_labels.groupby(input_subset + [output_var]) .size(
    ) .reset_index(name='count') .sort_values(by='count', ascending=False)
    # Take top 2 from each combination
    for _, row in rule_patterns_3d.head(1).iterrows():
        if current_3_input_count >= target_3_input_rules:
            break
        rule_text = (f"IF {input_subset[0]} IS {row[input_subset[0]]} AND "
                     f"{input_subset[1]} IS {row[input_subset[1]]} AND "
                     f"{input_subset[2]} IS {row[input_subset[2]]} "
                     f"THEN {output_var} IS {row[output_var]}")
        if add_rule_if_new(rule_text, rule_texts_set):
            current_3_input_count += 1

# 2-input rules
print("\n# 2-Input Rules:")
target_2_input_rules = 10
current_2_input_count = 0
for combo in combinations(input_vars, 2):
    if current_2_input_count >= target_2_input_rules:
        break
    input_subset = list(combo)
    rule_patterns_2d = df_fuzzy_labels .groupby(input_subset + [output_var]).size(
    ) .reset_index(name='count').sort_values(by='count', ascending=False)
    # Take top 1 from each combination
    for _, row in rule_patterns_2d.head(1).iterrows():
        if current_2_input_count >= target_2_input_rules:
            break
        rule_text = (f"IF {input_subset[0]} IS {row[input_subset[0]]} AND "
                     f"{input_subset[1]} IS {row[input_subset[1]]} "
                     f"THEN {output_var} IS {row[output_var]}")
        if add_rule_if_new(rule_text, rule_texts_set):
            current_2_input_count += 1

# 1-input rules
print("\n# 1-Input Rules:")
target_1_input_rules = 5  # Aim for a few more to exceed 20 total if needed
current_1_input_count = 0
if len(rule_texts_set) < 22:  # Add more if total is low
    for var_name_1_input in input_vars:
        # Max rules cap
        if current_1_input_count >= target_1_input_rules or len(rule_texts_set) >= 25:
            break
        rule_patterns_1d = df_fuzzy_labels .groupby([var_name_1_input, output_var]) .size(
        ) .reset_index(name='count') .sort_values(by='count', ascending=False)
        for _, row in rule_patterns_1d.head(2).iterrows():
            if current_1_input_count >= target_1_input_rules or len(rule_texts_set) >= 25:
                break
            rule_text = f"IF {var_name_1_input} IS {row[var_name_1_input]} THEN {output_var} IS {row[output_var]}"
            if add_rule_if_new(rule_text, rule_texts_set):
                current_1_input_count += 1

print(f"\nTotal unique rules generated: {len(rule_texts_set)}")
