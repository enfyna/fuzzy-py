import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from skfuzzy.cluster import cmeans
from matplotlib import pyplot as plt
from kneed import KneeLocator  # Added for elbow detection

# --- Constants ---
# Multiplier for standard deviation when calculating trimf widths.
K_STD_MULTIPLIER_FOR_TRIMF = 2.0


# 1. Data Loading
df = pd.read_csv("./data/mcs_ds_edited_iter_shuffled.csv")

input_vars = ['anchor_ratio', 'trans_range(m)', 'node_density', 'iterations']
output_var = 'ale(m)'

input_specs = {}

# 2. Determine Optimum Cluster Count & Calculate Metrics
print("--- Determining Optimum Cluster Count & Calculating Metrics ---")
for var_name in input_vars + [output_var]:
    data_column = df[var_name].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_column)

    inertias = []
    # k_values_tested will store the actual k values (e.g., 2, 3, 4...)
    k_values_tested_for_metrics = []
    print(f"\nVariable: {var_name}")

    # Calculate K-Means Inertias
    print("  Calculating K-Means inertias:")
    for k_val in range(2, 7):  # Test K from 2 to 6
        kmeans_elbow = KMeans(
            n_clusters=k_val, random_state=0, n_init=10).fit(scaled_data)
        print(f"    K={k_val}, Inertia: {kmeans_elbow.inertia_:.2f}")
        # Stop if inertia is very low or doesn't change much (optional heuristic)
        if k_val > 2 and inertias[-1] - kmeans_elbow.inertia_ < 0.0001 * inertias[0]:
            print(
                f"    Inertia decrease minimal, stopping inertia calculation for K > {k_val-1}")
            break
        if kmeans_elbow.inertia_ <= 0.0001:
            print(
                f"    Inertia very low, stopping inertia calculation for K > {k_val-1}")
            break
        inertias.append(kmeans_elbow.inertia_)
        k_values_tested_for_metrics.append(k_val)

    input_specs[var_name] = {
        'fpc_scores': [],
        'inertias': inertias,
        # Store the k's for which metrics were computed
        'k_values_for_metrics': k_values_tested_for_metrics,
        'optimal_k': 2  # Default optimal k
    }

    # Calculate Fuzzy Partition Coefficient (FPC)
    fpc_scores = []
    n_samples = scaled_data.shape[0]
    if k_values_tested_for_metrics:
        print(
            f"  Calculating FPC scores for K values: {k_values_tested_for_metrics}")
        for n_clusters_fpc in k_values_tested_for_metrics:
            if n_clusters_fpc < 2 or n_clusters_fpc > n_samples - 1:
                fpc_scores.append(np.nan)
                # print(f"    Skipping FPC for K={n_clusters_fpc} (constraint not met)")
                continue
            try:
                # Note: cmeans returns cntr, u, u0, d, jm, p, fpc
                _, _, _, _, _, _, fpc = cmeans(
                    scaled_data.T, n_clusters_fpc, 2, error=0.005, maxiter=1000, init=None
                )
                fpc_scores.append(fpc)
                print(f"    K={n_clusters_fpc}, FPC = {fpc:.3f}")
            except Exception as e:
                print(f"    Error calculating FPC for K={n_clusters_fpc}: {e}")
                fpc_scores.append(np.nan)
        input_specs[var_name]['fpc_scores'] = fpc_scores

    # --- Select Optimal K ---
    k_elbow = None
    if len(k_values_tested_for_metrics) >= 2 and len(inertias) >= 2:  # Need at least 2 points for kneed
        try:
            # Adjust S (sensitivity) if needed. Higher S = less sensitive (finds more significant elbows).
            kn = KneeLocator(k_values_tested_for_metrics[:len(inertias)], inertias,  # Ensure x and y have same length
                             curve='convex', direction='decreasing', S=1.0)
            k_elbow = kn.elbow
            # If kneed fails with >=3 points
            if k_elbow is None and len(k_values_tested_for_metrics) >= 3 and len(inertias) >= 3:
                print(
                    f"    Kneed couldn't find elbow. Fallback: trying point of max 2nd derivative.")
                second_diff = np.diff(inertias, n=2)
                # k corresponding to max change in slope
                k_elbow = k_values_tested_for_metrics[np.argmax(
                    second_diff) + 1]
            elif k_elbow is None:  # kneed failed, and not enough points for 2nd diff fallback or it also failed
                # Default to smallest k if kneed fails completely
                k_elbow = k_values_tested_for_metrics[0]
                print(
                    f"    Kneed couldn't find elbow & fallback failed or too few points. Defaulting k_elbow to {k_elbow}.")

        except Exception as e:
            print(
                f"    Error during k-elbow detection: {e}. Defaulting k_elbow.")
            if k_values_tested_for_metrics:
                k_elbow = k_values_tested_for_metrics[0]
            else:
                k_elbow = 2

    k_fpc = None
    valid_fpc_values = np.array([s for s in fpc_scores if not np.isnan(s)])
    k_for_valid_fpc = np.array([k for k, s in zip(
        k_values_tested_for_metrics, fpc_scores) if not np.isnan(s)])

    if len(valid_fpc_values) > 0:
        max_fpc_idx = np.argmax(valid_fpc_values)
        k_fpc = k_for_valid_fpc[max_fpc_idx]

    print(
        f"    Calculated k_elbow: {k_elbow}, k_fpc: {k_fpc} (Max FPC: {np.max(valid_fpc_values) if len(valid_fpc_values) > 0 else 'N/A'})")

    if k_fpc is not None and len(valid_fpc_values) > 0:
        # Prefer FPC if it gives a good peak and value
        if np.max(valid_fpc_values) > 0.55 and (len(valid_fpc_values) == 1 or np.std(valid_fpc_values) > 0.05 or len(k_values_tested_for_metrics) <= 3):
            optimal_k_final = k_fpc
            print(f"    Selected optimal_k based on FPC: {optimal_k_final}")
        elif k_elbow is not None:
            optimal_k_final = k_elbow
            print(
                f"    FPC peak not strong or FPC flat/low. Selected optimal_k based on Elbow: {optimal_k_final}")
        else:  # FPC not good, elbow also None
            optimal_k_final = k_fpc  # Fallback to k_fpc if elbow is None but k_fpc exists
            print(
                f"    Elbow method inconclusive, FPC exists. Using k_fpc: {optimal_k_final}")
    elif k_elbow is not None:
        optimal_k_final = k_elbow
        print(
            f"    No valid/strong FPC scores. Selected optimal_k based on Elbow: {optimal_k_final}")
    else:
        print(
            f"    Both FPC and Elbow methods inconclusive for {var_name}. Defaulting optimal_k to: {optimal_k_final}")

    # Ensure optimal_k is within the actually tested range and at least 2
    if k_values_tested_for_metrics:
        min_k_tested = min(k_values_tested_for_metrics)
        max_k_tested = max(k_values_tested_for_metrics)
        optimal_k_final = max(min_k_tested, min(optimal_k_final, max_k_tested))
    optimal_k_final = max(2, int(optimal_k_final)
                          if optimal_k_final is not None else 2)

    input_specs[var_name]['optimal_k'] = optimal_k_final
    print(
        f"    ==> Final selected optimal_k for {var_name}: {optimal_k_final}")

    # Plotting metrics
    plt.figure(figsize=(8, 5))
    title_str = f"Metrics for {var_name} (Optimal K chosen: {optimal_k_final})"
    if k_values_tested_for_metrics and inertias:
        plt.plot(k_values_tested_for_metrics[:len(
            inertias)], inertias, 'go-', label='Inertias (K-Means)')
        if k_elbow:  # Mark the elbow if found
            plt.vlines(k_elbow, plt.ylim()[0], plt.ylim()[
                       1], linestyles='--', color='darkgreen', label=f'Elbow K={k_elbow}')

    if k_for_valid_fpc.size > 0 and valid_fpc_values.size > 0:
        plt.plot(k_for_valid_fpc, valid_fpc_values,
                 'ro-', label='FPC Scores (C-Means)')
        if k_fpc:  # Mark the max FPC
            plt.vlines(k_fpc, plt.ylim()[0], plt.ylim()[
                       1], linestyles=':', color='darkred', label=f'Max FPC K={k_fpc}')

    plt.title(title_str)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show() # Uncomment to show plots one by one during script execution

# 3. Find Center Points and Cluster Labels (USING OPTIMAL_K)
cluster_centers_for_rules = {}
cluster_labels_for_rules = {}

print(f"\n--- Cluster Centers and Labels (for Rule Extraction & MF Params) ---")
for var_name in input_vars + [output_var]:
    data_column_unscaled = df[var_name].values.reshape(-1, 1)

    # USE THE OPTIMAL_K determined in section 2
    num_clusters_for_var = input_specs[var_name]['optimal_k']

    print(f"\nVariable: {var_name} (using optimal_k = {num_clusters_for_var})")
    kmeans = KMeans(n_clusters=num_clusters_for_var,
                    random_state=0, n_init='auto').fit(data_column_unscaled)

    sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())
    centers = kmeans.cluster_centers_.flatten()[sorted_indices]

    original_labels = kmeans.labels_
    labels_remapped = np.zeros_like(original_labels)
    for i, original_idx in enumerate(sorted_indices):
        labels_remapped[original_labels == original_idx] = i

    cluster_centers_for_rules[var_name] = centers
    cluster_labels_for_rules[var_name] = labels_remapped

    print(
        f"=> Sorted Cluster Centers ({num_clusters_for_var} clusters): {centers}")


# 4. Calculate Membership Function Parameters (This section remains largely the same as previous response)
mf_params_trimf_overlap = {}
mf_params_trimf_stddev = {}
mf_params_gauss = {}

print("\n--- Calculated Membership Function Parameters ---")

for var_name in input_vars + [output_var]:
    centers = cluster_centers_for_rules[var_name]
    labels = cluster_labels_for_rules[var_name]
    num_mf = len(centers)  # This is now based on optimal_k

    var_data_values = df[var_name].values
    data_min = var_data_values.min()
    data_max = var_data_values.max()

    current_trimf_overlap_params = []
    current_trimf_stddev_params = []
    current_gauss_params = []

    print(
        f"\nVariable: {var_name} (Data range: [{data_min:.3f}, {data_max:.3f}], MFs: {num_mf})")
    # print(f"  Sorted Centers ({num_mf}): {np.array2string(centers, precision=3, floatmode='fixed')}") # Redundant if printed in sec 3

    if num_mf == 0:  # Should not happen if optimal_k is at least 2
        print(f"  No centers found for {var_name}, cannot define MFs.")
        mf_params_trimf_overlap[var_name] = []
        mf_params_trimf_stddev[var_name] = []
        mf_params_gauss[var_name] = []
        continue

    for i in range(num_mf):
        b_center = centers[i]
        points_in_cluster = var_data_values[labels == i]
        cluster_std_dev = 0
        if len(points_in_cluster) > 1:
            cluster_std_dev = np.std(points_in_cluster)

        if cluster_std_dev <= 1e-5:
            fallback_std = (data_max - data_min) * 0.05
            if fallback_std <= 1e-5:
                fallback_std = abs(
                    b_center * 0.1) if abs(b_center) > 1e-5 else 0.01
            if fallback_std <= 1e-5:
                fallback_std = 0.01
            cluster_std_dev = fallback_std

        # A. trimf (overlap-based)
        a_overlap, c_overlap = 0, 0
        if num_mf == 1:
            a_overlap, c_overlap = data_min, data_max
        else:
            if i == 0:
                a_overlap, c_overlap = data_min, centers[i+1]
            elif i == num_mf - 1:
                a_overlap, c_overlap = centers[i-1], data_max
            else:
                a_overlap, c_overlap = centers[i-1], centers[i+1]
        current_trimf_overlap_params.append([a_overlap, b_center, c_overlap])

        # B. trimf (std_dev based)
        a_std = max(data_min, b_center -
                    K_STD_MULTIPLIER_FOR_TRIMF * cluster_std_dev)
        c_std = min(data_max, b_center +
                    K_STD_MULTIPLIER_FOR_TRIMF * cluster_std_dev)
        if a_std > b_center:
            a_std = b_center
        if c_std < b_center:
            c_std = b_center
        if a_std == b_center and b_center == c_std and data_min < data_max:
            eps = (data_max-data_min)*0.001
            if b_center == data_min:
                c_std = min(data_max, b_center + eps)
            elif b_center == data_max:
                a_std = max(data_min, b_center - eps)
            else:
                a_std, c_std = max(data_min, b_center - eps /
                                   2), min(data_max, b_center + eps/2)
        current_trimf_stddev_params.append([a_std, b_center, c_std])

        # C. Gaussian MF
        current_gauss_params.append([b_center, cluster_std_dev])

    mf_params_trimf_overlap[var_name] = current_trimf_overlap_params
    mf_params_trimf_stddev[var_name] = current_trimf_stddev_params
    mf_params_gauss[var_name] = current_gauss_params

    # Print parameters (optional, can be verbose)
    print("  Method 1: Triangular MFs (Overlap-based)")
    for i, p in enumerate(current_trimf_overlap_params):
        print(f"    MF {i+1}: a={p[0]:.3f}, b={p[1]:.3f}, c={p[2]:.3f}")
    print(
        f"  Method 2: Triangular MFs (StdDev-based, k={K_STD_MULTIPLIER_FOR_TRIMF})")
    for i, p in enumerate(current_trimf_stddev_params):
        print(f"    MF {i+1}: a={p[0]:.3f}, b={p[1]:.3f}, c={p[2]:.3f}")
    print("  Method 3: Gaussian MFs")
    for i, p in enumerate(current_gauss_params):
        print(f"    MF {i+1}: mean={p[0]:.3f}, sigma={p[1]:.3f}")


# 5. Discretize Data for Rule Extraction (based on K clusters)
# This section should now work more robustly as num_actual_clusters will be optimal_k
df_fuzzy_labels = pd.DataFrame()
mf_labels_fixed = ['low', 'medium', 'high', 'very high',
                   'extra_high', 'super_high']  # Extended for up to 6 clusters

print("\n--- Sample Discretized Data (first 5 rows) ---")
for var_name in input_vars + [output_var]:
    data_column = df[var_name].values
    # These centers are from optimal_k clusters
    centers = cluster_centers_for_rules[var_name]
    num_actual_clusters = len(centers)
    discretized_labels = []
    for val in data_column:
        distances = [abs(val - center) for center in centers]
        closest_cluster_index = np.argmin(distances)

        if closest_cluster_index < len(mf_labels_fixed):
            discretized_labels.append(mf_labels_fixed[closest_cluster_index])
        # Should happen less if mf_labels_fixed covers the max k (e.g. 6)
        else:
            label_key = f"{var_name}_warned_labeling"
            if label_key not in input_specs[var_name]:
                print(f"Warning: For '{var_name}', num clusters ({num_actual_clusters}) "
                      f"> predefined mf_labels ({len(mf_labels_fixed)}). Using generic 'cluster_X'.")
                input_specs[var_name][label_key] = True
            discretized_labels.append(f"cluster_{closest_cluster_index}")
    df_fuzzy_labels[var_name] = discretized_labels
print(df_fuzzy_labels.head(-1))


# 6. Extract and Print Fuzzy Rules (Content from original user code)
# ... (rest of your rule extraction code remains the same) ...
print("\n--- Extracted Fuzzy Rules ---")
rules_generated = []


def add_rule_if_new(rule_text, rule_list_texts):
    if rule_text not in rule_list_texts:
        rule_list_texts.add(rule_text)
        print(rule_text)
        return True
    return False


rule_texts_set = set()
# 4-input rules
rule_patterns_4d = df_fuzzy_labels.groupby(input_vars + [output_var]).size(
).reset_index(name='count').sort_values(by='count', ascending=False)
print("\n# 4-Input Rules:")
for _, row in rule_patterns_4d.head(5).iterrows():
    rule_text = (f"IF {input_vars[0]} IS {row[input_vars[0]]} AND "
                 f"{input_vars[1]} IS {row[input_vars[1]]} AND "
                 f"{input_vars[2]} IS {row[input_vars[2]]} AND "
                 f"{input_vars[3]} IS {row[input_vars[3]]} "
                 f"THEN {output_var} IS {row[output_var]}")
    add_rule_if_new(rule_text, rule_texts_set)

# (The rest of your 3-input, 2-input, 1-input rule generation code follows here)
# 3-input rules
print("\n# 3-Input Rules:")
target_3_input_rules = 10
current_3_input_count = 0
for combo in combinations(input_vars, 3):
    if current_3_input_count >= target_3_input_rules:
        break
    input_subset = list(combo)
    rule_patterns_3d = df_fuzzy_labels.groupby(input_subset + [output_var]).size(
    ).reset_index(name='count').sort_values(by='count', ascending=False)
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
target_2_input_rules = 20
current_2_input_count = 0
for combo in combinations(input_vars, 2):
    if current_2_input_count >= target_2_input_rules:
        break
    input_subset = list(combo)
    rule_patterns_2d = df_fuzzy_labels.groupby(input_subset + [output_var]).size(
    ).reset_index(name='count').sort_values(by='count', ascending=False)
    for _, row in rule_patterns_2d.head(2).iterrows():
        if current_2_input_count >= target_2_input_rules:
            break
        rule_text = (f"IF {input_subset[0]} IS {row[input_subset[0]]} AND "
                     f"{input_subset[1]} IS {row[input_subset[1]]} "
                     f"THEN {output_var} IS {row[output_var]}")
        if add_rule_if_new(rule_text, rule_texts_set):
            current_2_input_count += 1

# 1-input rules
print("\n# 1-Input Rules:")
for var_name_1_input in input_vars:
    rule_patterns_1d = df_fuzzy_labels.groupby([var_name_1_input, output_var]).size(
    ).reset_index(name='count').sort_values(by='count', ascending=False)
    # Taking 1 rule per variable to ensure diversity
    for _, row in rule_patterns_1d.head(2).iterrows():
        rule_text = f"IF {var_name_1_input} IS {row[var_name_1_input]} THEN {output_var} IS {row[output_var]}"
        add_rule_if_new(rule_text, rule_texts_set)

print(f"\nTotal unique rules generated: {len(rule_texts_set)}")

# Show all plots at the end
# plt.show()
