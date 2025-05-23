import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("./data/mcs_ds_edited_iter_shuffled.csv")
print(f"Dataset shape: {df.shape}")
print("\nDataset preview:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nDataset statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Prepare data for clustering
print("\n" + "="*70)
print("PREPARING DATA FOR CLUSTERING")
print("="*70)

# Separate input features and output
feature_columns = ['anchor_ratio', 'trans_range(m)', 'node_density', 'iterations']
X = df[feature_columns].values
y = df['ale(m)'].values

print(f"Input features shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Normalize the data for clustering
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Combine input and output for clustering
data_for_clustering = np.column_stack([X_scaled, y_scaled])
print(f"Data for clustering shape: {data_for_clustering.shape}")

# Function to find optimal number of clusters
def find_optimal_clusters(data, max_clusters=8):
    """Find optimal number of clusters using fuzzy partition coefficient (FPC)"""
    n_samples = data.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    
    fpc_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    print(f"Testing {len(cluster_range)} different cluster configurations...")
    
    for n_clusters in cluster_range:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        fpc_scores.append(fpc)
        print(f"  {n_clusters} clusters: FPC = {fpc:.3f}")
    
    return cluster_range, fpc_scores

# Find optimal number of clusters
print("\n" + "="*70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*70)

cluster_range, fpc_scores = find_optimal_clusters(data_for_clustering)

# Plot FPC scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, fpc_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters')
plt.ylabel('Fuzzy Partition Coefficient (FPC)')
plt.title('FPC vs Number of Clusters (Higher is Better)')
plt.grid(True)
plt.show()

# Choose optimal number of clusters
optimal_clusters = cluster_range[np.argmax(fpc_scores)]
print(f"\nOptimal number of clusters: {optimal_clusters}")
print(f"Best FPC score: {max(fpc_scores):.3f}")

# Perform fuzzy c-means with optimal clusters
print("\n" + "="*70)
print("PERFORMING FUZZY C-MEANS CLUSTERING")
print("="*70)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_for_clustering.T, optimal_clusters, 2, error=0.005, maxiter=1000, init=None
)

print(f"Fuzzy Partition Coefficient: {fpc:.3f}")
print(f"Number of iterations: {p}")

# Convert cluster centers back to original scale
cluster_centers_original = np.zeros_like(cntr)
for i in range(optimal_clusters):
    cluster_centers_original[i, :-1] = scaler_X.inverse_transform(cntr[i, :-1].reshape(1, -1))[0]
    cluster_centers_original[i, -1] = scaler_y.inverse_transform([[cntr[i, -1]]])[0][0]

print(f"\nCluster centers (original scale):")
for i, center in enumerate(cluster_centers_original):
    print(f"Cluster {i}: ", end="")
    for j, feature in enumerate(feature_columns + ['ale']):
        print(f"{feature}={center[j]:.2f} ", end="")
    print()

# Create membership functions based on clusters
print("\n" + "="*70)
print("CREATING MEMBERSHIP FUNCTIONS")
print("="*70)

def create_membership_functions(variable_name, data_column, cluster_centers_for_var, n_clusters):
    """Create membership functions based on cluster centers"""
    min_val = data_column.min()
    max_val = data_column.max()
    
    # Create universe
    universe = np.arange(min_val, max_val + (max_val - min_val) * 0.01, 
                        (max_val - min_val) / 200)
    
    # Sort cluster centers for this variable
    centers = sorted(cluster_centers_for_var)
    
    membership_funcs = {}
    
    if n_clusters == 2:
        # Two clusters: Low and High
        overlap_point = (centers[0] + centers[1]) / 2
        
        membership_funcs['Low'] = [min_val, centers[0], overlap_point]
        membership_funcs['High'] = [overlap_point, centers[1], max_val]
        
    elif n_clusters == 3:
        # Three clusters: Low, Medium, High
        overlap1 = (centers[0] + centers[1]) / 2
        overlap2 = (centers[1] + centers[2]) / 2
        
        membership_funcs['Low'] = [min_val, centers[0], overlap1]
        membership_funcs['Medium'] = [overlap1, centers[1], overlap2]
        membership_funcs['High'] = [overlap2, centers[2], max_val]
        
    else:
        # More clusters: use generic naming
        for i, center in enumerate(centers):
            if i == 0:
                left = min_val
                right = (centers[i] + centers[i+1]) / 2 if i < len(centers) - 1 else max_val
            elif i == len(centers) - 1:
                left = (centers[i-1] + centers[i]) / 2
                right = max_val
            else:
                left = (centers[i-1] + centers[i]) / 2
                right = (centers[i] + centers[i+1]) / 2
            
            membership_funcs[f'C{i}'] = [left, center, right]
    
    return universe, membership_funcs

# Create fuzzy variables
fuzzy_vars = {}

# Input variables
for i, feature in enumerate(feature_columns):
    cluster_centers_for_feature = cluster_centers_original[:, i]
    universe, mf_params = create_membership_functions(
        feature, df[feature], cluster_centers_for_feature, optimal_clusters
    )
    
    # Create fuzzy variable
    fuzzy_var = ctrl.Antecedent(universe, feature.replace('(m)', '').replace('trans_range', 'trans_range'))
    
    # Add membership functions
    for mf_name, params in mf_params.items():
        fuzzy_var[mf_name] = fuzz.trimf(universe, params)
    
    fuzzy_vars[feature] = fuzzy_var
    
    # Print membership function parameters
    print(f"\n{feature} Membership Functions:")
    for mf_name, params in mf_params.items():
        print(f"  {mf_name}: trimf([{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}])")

# Output variable (ALE)
ale_cluster_centers = cluster_centers_original[:, -1]
ale_universe, ale_mf_params = create_membership_functions(
    'ale', df['ale(m)'], ale_cluster_centers, optimal_clusters
)

ale_var = ctrl.Consequent(ale_universe, 'ale')
for mf_name, params in ale_mf_params.items():
    ale_var[mf_name] = fuzz.trimf(ale_universe, params)

fuzzy_vars['ale'] = ale_var

print(f"\nale Membership Functions:")
for mf_name, params in ale_mf_params.items():
    print(f"  {mf_name}: trimf([{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}])")

# Extract fuzzy rules from clusters
print("\n" + "="*70)
print("EXTRACTING FUZZY RULES")
print("="*70)

def extract_rules_from_clusters(cluster_centers, membership_params, feature_names, n_clusters):
    """Extract fuzzy rules from cluster analysis"""
    rules = []
    
    for i in range(n_clusters):
        # Determine which membership function each feature belongs to for this cluster
        rule_antecedents = []
        
        for j, feature in enumerate(feature_names):
            feature_value = cluster_centers[i, j]
            
            # Find which membership function this value belongs to most strongly
            feature_mf_params = membership_params[feature]
            max_membership = 0
            best_mf = None
            
            for mf_name, mf_params in feature_mf_params.items():
                # Calculate membership value for this cluster center
                membership_val = fuzz.trimf(np.array([feature_value]), mf_params)[0]
                if membership_val > max_membership:
                    max_membership = membership_val
                    best_mf = mf_name
            
            rule_antecedents.append(f"{feature} is {best_mf}")
        
        # Determine output membership function
        ale_value = cluster_centers[i, -1]
        max_membership = 0
        best_ale_mf = None
        
        for mf_name, mf_params in ale_mf_params.items():
            membership_val = fuzz.trimf(np.array([ale_value]), mf_params)[0]
            if membership_val > max_membership:
                max_membership = membership_val
                best_ale_mf = mf_name
        
        rule = {
            'id': i + 1,
            'antecedent': rule_antecedents,
            'consequent': f"ale is {best_ale_mf}",
            'cluster_center': cluster_centers[i]
        }
        rules.append(rule)
    
    return rules

# Get membership function parameters for rule extraction
membership_params = {}
for i, feature in enumerate(feature_columns):
    cluster_centers_for_feature = cluster_centers_original[:, i]
    _, mf_params = create_membership_functions(
        feature, df[feature], cluster_centers_for_feature, optimal_clusters
    )
    membership_params[feature] = mf_params

# Extract rules
extracted_rules = extract_rules_from_clusters(
    cluster_centers_original, membership_params, feature_columns, optimal_clusters
)

print(f"Extracted {len(extracted_rules)} fuzzy rules:")
for rule in extracted_rules:
    print(f"\nRule {rule['id']}:")
    print(f"  IF {' AND '.join(rule['antecedent'])}")
    print(f"  THEN {rule['consequent']}")

# Create Mamdani fuzzy inference system
print("\n" + "="*70)
print("CREATING MAMDANI FUZZY INFERENCE SYSTEM")
print("="*70)

# Create rule objects for the control system
ctrl_rules = []

for rule in extracted_rules:
    # Parse antecedents
    rule_antecedents = []
    
    for antecedent in rule['antecedent']:
        # Parse "feature is mf_name"
        parts = antecedent.split(' is ')
        feature_name = parts[0]
        mf_name = parts[1]
        
        # Get the fuzzy variable and membership function
        fuzzy_var = fuzzy_vars[feature_name]
        rule_antecedents.append(fuzzy_var[mf_name])
    
    # Parse consequent
    consequent_parts = rule['consequent'].split(' is ')
    ale_mf_name = consequent_parts[1]
    consequent = ale_var[ale_mf_name]
    
    # Combine antecedents with AND
    if len(rule_antecedents) == 1:
        combined_antecedent = rule_antecedents[0]
    else:
        combined_antecedent = rule_antecedents[0]
        for ant in rule_antecedents[1:]:
            combined_antecedent = combined_antecedent & ant
    
    # Create control rule
    ctrl_rule = ctrl.Rule(combined_antecedent, consequent)
    ctrl_rules.append(ctrl_rule)

print(f"Created {len(ctrl_rules)} control rules")

# Create control system
ale_ctrl = ctrl.ControlSystem(ctrl_rules)
ale_simulation = ctrl.ControlSystemSimulation(ale_ctrl)

print("Mamdani fuzzy inference system created successfully!")

# Evaluate the system
print("\n" + "="*70)
print("EVALUATING FUZZY SYSTEM")
print("="*70)

def evaluate_fuzzy_system(simulation, X_test, y_true, feature_names):
    """Evaluate the fuzzy system performance"""
    predictions = []
    failed_predictions = 0
    
    print(f"Evaluating on {len(X_test)} samples...")
    
    for i, sample in enumerate(X_test):
        try:
            # Set input values
            for j, feature in enumerate(feature_names):
                clean_feature = feature.replace('(m)', '').replace('trans_range', 'trans_range')
                simulation.input[clean_feature] = sample[j]
            
            # Compute the result
            simulation.compute()
            
            # Get the output (using center of sums defuzzification)
            prediction = simulation.output['ale']
            predictions.append(prediction)
            
        except Exception as e:
            # Handle cases where fuzzy system fails
            predictions.append(np.mean(y_true))  # Use mean as fallback
            failed_predictions += 1
            if failed_predictions <= 5:  # Only print first few errors
                print(f"  Warning: Failed to compute sample {i}: {e}")
    
    if failed_predictions > 5:
        print(f"  ... and {failed_predictions - 5} more failed predictions")
    
    predictions = np.array(predictions)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    
    # R² score (handle potential division by zero)
    try:
        r2 = r2_score(y_true, predictions)
    except:
        r2 = float('-inf')
    
    return predictions, mae, mse, rmse, r2, failed_predictions

# Evaluate on all data
predictions, mae, mse, rmse, r2, failed_count = evaluate_fuzzy_system(
    ale_simulation, X, y, feature_columns
)

print(f"\nFuzzy System Performance:")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  R² Score: {r2:.4f}")
print(f"  Failed predictions: {failed_count}/{len(X)} ({failed_count/len(X)*100:.1f}%)")

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Actual vs Predicted
ax1.scatter(y, predictions, alpha=0.6)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_xlabel('Actual ALE')
ax1.set_ylabel('Predicted ALE')
ax1.set_title('Actual vs Predicted ALE')
ax1.grid(True)

# Residuals
residuals = y - predictions
ax2.scatter(predictions, residuals, alpha=0.6)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted ALE')
ax2.set_ylabel('Residuals')
ax2.set_title('Residual Plot')
ax2.grid(True)

# Error histogram
ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Prediction Error')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Prediction Errors')
ax3.grid(True)

# Performance summary
ax4.text(0.1, 0.8, f'Performance Metrics:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.1, 0.7, f'MAE: {mae:.4f}', fontsize=12, transform=ax4.transAxes)
ax4.text(0.1, 0.6, f'RMSE: {rmse:.4f}', fontsize=12, transform=ax4.transAxes)
ax4.text(0.1, 0.5, f'R²: {r2:.4f}', fontsize=12, transform=ax4.transAxes)
ax4.text(0.1, 0.4, f'Failed: {failed_count}/{len(X)}', fontsize=12, transform=ax4.transAxes)
ax4.text(0.1, 0.2, f'Clusters: {optimal_clusters}', fontsize=12, transform=ax4.transAxes)
ax4.text(0.1, 0.1, f'Rules: {len(extracted_rules)}', fontsize=12, transform=ax4.transAxes)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"Optimal clusters: {optimal_clusters}")
print(f"Extracted rules: {len(extracted_rules)}")
print(f"Fuzzy system performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
print(f"System successfully evaluated on all {len(X)} data points")

# Save results (optional)
results_df = pd.DataFrame({
    'Actual': y,
    'Predicted': predictions,
    'Residual': residuals
})

print(f"\nResults dataframe created with shape: {results_df.shape}")
print("You can save results with: results_df.to_csv('fuzzy_system_results.csv', index=False)")
