#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:36:43 2026

@author: nikoslamprou
"""


# =============================================================================
# Exercise 1: Causal inference with autoencoders (Matrix Completion)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------------------------------------------
# 1. Load and sort data (adjust file path as needed)
# -----------------------------------------------------------------------------
df = pd.read_csv(r"C:\\Users\\megss\\Downloads\\causal_panel_data.csv")
df = df.sort_values(['unit_id', 'time']).reset_index(drop=True)

# Verify each unit has 24 observations
units = df['unit_id'].unique()
time_per_unit = df.groupby('unit_id')['time'].nunique()
assert (time_per_unit == 24).all(), "Not all units have 24 time periods"
N_units = len(units)
T_periods = 24

# FIX 1: proportion ever-treated at unit level
prop_ever_treated = df[['unit_id', 'ever_treated']].drop_duplicates()['ever_treated'].mean()

print(f"Number of units: {N_units}")
print(f"Number of time periods: {T_periods}")
print(f"Proportion ever-treated: {prop_ever_treated:.3f}")

# -----------------------------------------------------------------------------
# 2. Create wide outcome matrix Y (units x time)
# -----------------------------------------------------------------------------
Y_wide = df.pivot(index='unit_id', columns='time', values='outcome')
Y_wide = Y_wide.sort_index()
Y = Y_wide.values  # shape (N, T)

print(f"Wide outcome matrix dimension: {Y.shape}")

# -----------------------------------------------------------------------------
# 3. Create mask M (1 = observed, 0 = masked for treated post-treatment)
# -----------------------------------------------------------------------------
treat_time_df = df[['unit_id', 'treat_time']].drop_duplicates().set_index('unit_id')
treat_time_dict = treat_time_df['treat_time'].to_dict()

M = np.ones_like(Y, dtype=float)
for i, unit in enumerate(Y_wide.index):
    t_treat = treat_time_dict.get(unit, np.inf)
    if not np.isnan(t_treat):
        times = Y_wide.columns.values
        M[i, times >= t_treat] = 0.0

# Number of treated observations in post-treatment periods
treated_post_mask = (M == 0)
n_treated_post = np.sum(treated_post_mask)
print(f"Number of treated observations in post-treatment periods: {n_treated_post}")

# -----------------------------------------------------------------------------
# 4. Fill masked entries temporarily with 0 (for input)
# -----------------------------------------------------------------------------
Y_fill = Y.copy()
Y_fill[M == 0] = 0.0

# -----------------------------------------------------------------------------
# 5. Add covariates (region, sector, size, digital_index, productivity_index, credit_score)
# -----------------------------------------------------------------------------
unit_covars = df[['unit_id', 'region', 'sector', 'size', 'digital_index',
                  'productivity_index', 'credit_score']].drop_duplicates().set_index('unit_id')
unit_covars = unit_covars.loc[Y_wide.index]

# One-hot encode region and sector
region_dummies = pd.get_dummies(unit_covars['region'], prefix='region')
sector_dummies = pd.get_dummies(unit_covars['sector'], prefix='sector')

# Standardize continuous covariates
continuous_vars = ['size', 'digital_index', 'productivity_index', 'credit_score']
scaler_cont = StandardScaler()
cont_scaled = scaler_cont.fit_transform(unit_covars[continuous_vars])
cont_scaled_df = pd.DataFrame(cont_scaled, columns=continuous_vars, index=unit_covars.index)

# Combine all covariates
covariates = pd.concat([region_dummies, sector_dummies, cont_scaled_df], axis=1)
covariates_matrix = covariates.values  # (N, n_cov)

# -----------------------------------------------------------------------------
# Build input matrix X: [masked outcomes (T), mask indicators (T), covariates]
# -----------------------------------------------------------------------------
X_masked = Y_fill
X_mask = M
X = np.hstack([X_masked, X_mask, covariates_matrix])
input_dim = X.shape[1]
output_dim = T_periods

# -----------------------------------------------------------------------------
# 6. Standardize outcome columns using only observed (mask=1) values
# -----------------------------------------------------------------------------
Y_std = np.zeros_like(Y)
Y_mean = np.zeros(T_periods)
Y_stddev = np.ones(T_periods)

for t in range(T_periods):
    observed_vals = Y[:, t][M[:, t] == 1]
    if len(observed_vals) > 0:
        mean_t = observed_vals.mean()
        std_t = observed_vals.std()
        if std_t < 1e-6:
            std_t = 1.0
        Y_mean[t] = mean_t
        Y_stddev[t] = std_t
        Y_std[:, t] = (Y[:, t] - mean_t) / std_t
    else:
        Y_std[:, t] = Y[:, t]

# Standardize also the masked outcome part in X
X_masked_std = np.zeros_like(Y_fill)
for t in range(T_periods):
    X_masked_std[:, t] = (Y_fill[:, t] - Y_mean[t]) / Y_stddev[t]

# FIX 2: re-zero masked entries after standardization
X_masked_std[M == 0] = 0.0

# Rebuild X_std
X_std = np.hstack([X_masked_std, X_mask, covariates_matrix])
X_std = X_std.astype(np.float32)
Y_std = Y_std.astype(np.float32)
M = M.astype(np.float32)

# -----------------------------------------------------------------------------
# 7. Build autoencoder (simple fully connected)
# -----------------------------------------------------------------------------
def build_autoencoder(input_dim, output_dim):
    input_layer = Input(shape=(input_dim,))
    dense1 = Dense(64, activation='relu')(input_layer)
    bottleneck = Dense(16, activation='relu')(dense1)
    dense2 = Dense(64, activation='relu')(bottleneck)
    output_layer = Dense(output_dim, activation='linear')(dense2)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# -----------------------------------------------------------------------------
# 8. Custom training loop with masked loss
# -----------------------------------------------------------------------------
def masked_mse(y_true, y_pred, mask):
    sq_diff = tf.square(y_true - y_pred)
    masked_sq_diff = sq_diff * mask
    loss = tf.reduce_sum(masked_sq_diff) / (tf.reduce_sum(mask) + 1e-8)
    return loss

def train_autoencoder(X_std, Y_std, M, input_dim, output_dim,
                      batch_size=32, epochs=100, patience=10, verbose=True):
    autoencoder = build_autoencoder(input_dim, output_dim)
    optimizer = Adam(learning_rate=0.001)

    best_loss = np.inf
    wait = 0
    best_weights = None

    if verbose:
        print("\nTraining autoencoder with masked loss...")

    for epoch in range(epochs):
        indices = np.random.permutation(X_std.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, X_std.shape[0], batch_size):
            end = min(start + batch_size, X_std.shape[0])
            batch_idx = indices[start:end]
            X_batch = X_std[batch_idx]
            Y_batch = Y_std[batch_idx]
            M_batch = M[batch_idx]

            with tf.GradientTape() as tape:
                Y_pred = autoencoder(X_batch, training=True)
                loss = masked_mse(Y_batch, Y_pred, M_batch)

            grads = tape.gradient(loss, autoencoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
            epoch_loss += loss.numpy()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_weights = autoencoder.get_weights()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    if best_weights is not None:
        autoencoder.set_weights(best_weights)

    return autoencoder, best_loss

autoencoder, best_loss = train_autoencoder(
    X_std, Y_std, M, input_dim, output_dim,
    batch_size=32, epochs=100, patience=10, verbose=True
)

print(f"Final training loss (masked MSE): {best_loss:.6f}")

# -----------------------------------------------------------------------------
# 9. Predict missing no-treatment outcomes
# -----------------------------------------------------------------------------
Y_pred_std = autoencoder.predict(X_std, verbose=0)

# Convert back to original scale
Y_pred = np.zeros_like(Y_pred_std)
for t in range(T_periods):
    Y_pred[:, t] = Y_pred_std[:, t] * Y_stddev[t] + Y_mean[t]

Yhat0 = Y_pred.copy()

# -----------------------------------------------------------------------------
# 10. Estimate treatment effects for treated post observations
# -----------------------------------------------------------------------------
units_list = Y_wide.index.tolist()
times_list = Y_wide.columns.tolist()

long_results = []
for i, unit in enumerate(units_list):
    for t_idx, time in enumerate(times_list):
        pred0 = Yhat0[i, t_idx]
        mask_val = M[i, t_idx]
        treated_post = (mask_val == 0)
        long_results.append({
            'unit_id': unit,
            'time': time,
            'pred0': pred0,
            'treated_post': treated_post
        })

df_long = pd.DataFrame(long_results)

# FIX 3: clean merge to avoid duplicate columns / KeyError
df = df.drop(columns=['pred0', 'treated_post'], errors='ignore')
df = df.merge(df_long[['unit_id', 'time', 'pred0', 'treated_post']],
              on=['unit_id', 'time'], how='left')

# Use original observed outcome
df['tau'] = np.where(df['treated_post'], df['outcome'] - df['pred0'], np.nan)

# -----------------------------------------------------------------------------
# 11. Compute ATT estimates
# -----------------------------------------------------------------------------
overall_att = df[df['treated_post']]['tau'].mean()
print(f"\nOverall ATT: {overall_att:.6f}")

# FIX 4: event-time ATT only for event_time >= 0
event_att = df[df['event_time'] >= 0].groupby('event_time')['tau'].mean().dropna()
print("\nEvent-time ATT:")
print(event_att)

region_att = df[df['treated_post']].groupby('region')['tau'].mean().dropna()
print("\nRegion-specific ATT:")
print(region_att)

sector_att = df[df['treated_post']].groupby('sector')['tau'].mean().dropna()
print("\nSector-specific ATT:")
print(sector_att)

# FIX 5: median digital index at unit level
med_digital = df[['unit_id', 'digital_index']].drop_duplicates()['digital_index'].median()
high_digital_att = df[(df['treated_post']) & (df['digital_index'] > med_digital)]['tau'].mean()
low_digital_att = df[(df['treated_post']) & (df['digital_index'] <= med_digital)]['tau'].mean()

print(f"\nATT for high digital (> median): {high_digital_att:.6f}")
print(f"ATT for low digital (<= median): {low_digital_att:.6f}")

# -----------------------------------------------------------------------------
# 12. Bootstrap confidence intervals (correct bootstrap with full refit)
# -----------------------------------------------------------------------------
B = 50
np.random.seed(42)

boot_overall = []
boot_region = {r: [] for r in df['region'].dropna().unique()}
boot_high_digital = []
boot_low_digital = []

print(f"\nRunning bootstrap with B={B} replications...")

for b in range(B):
    # 1. Sample units with replacement
    boot_units = np.random.choice(units_list, size=len(units_list), replace=True)

    # 2. Keep all time periods for each selected unit
    boot_dfs = []
    for new_id, old_unit in enumerate(boot_units):
        temp = df[df['unit_id'] == old_unit].copy()
        temp['boot_unit_id'] = new_id
        boot_dfs.append(temp)

    df_b = pd.concat(boot_dfs, ignore_index=True)

    # 3. Rebuild Y, M, covariates
    Yb_wide = df_b.pivot(index='boot_unit_id', columns='time', values='outcome').sort_index()
    Yb = Yb_wide.values.astype(float)

    treat_time_b = df_b[['boot_unit_id', 'treat_time']].drop_duplicates().set_index('boot_unit_id')['treat_time'].to_dict()

    Mb = np.ones_like(Yb, dtype=float)
    for i, unit in enumerate(Yb_wide.index):
        t_treat = treat_time_b.get(unit, np.inf)
        if not np.isnan(t_treat):
            times = Yb_wide.columns.values
            Mb[i, times >= t_treat] = 0.0

    Yb_fill = Yb.copy()
    Yb_fill[Mb == 0] = 0.0

    unit_covars_b = df_b[['boot_unit_id', 'region', 'sector', 'size', 'digital_index',
                          'productivity_index', 'credit_score']].drop_duplicates().set_index('boot_unit_id')
    unit_covars_b = unit_covars_b.loc[Yb_wide.index]

    region_dummies_b = pd.get_dummies(unit_covars_b['region'], prefix='region')
    sector_dummies_b = pd.get_dummies(unit_covars_b['sector'], prefix='sector')

    region_dummies_b = region_dummies_b.reindex(columns=region_dummies.columns, fill_value=0)
    sector_dummies_b = sector_dummies_b.reindex(columns=sector_dummies.columns, fill_value=0)

    cont_scaled_b = scaler_cont.transform(unit_covars_b[continuous_vars])
    cont_scaled_df_b = pd.DataFrame(cont_scaled_b, columns=continuous_vars, index=unit_covars_b.index)

    covariates_b = pd.concat([region_dummies_b, sector_dummies_b, cont_scaled_df_b], axis=1)
    covariates_matrix_b = covariates_b.values

    Y_std_b = np.zeros_like(Yb)
    Y_mean_b = np.zeros(T_periods)
    Y_stddev_b = np.ones(T_periods)

    for t in range(T_periods):
        observed_vals_b = Yb[:, t][Mb[:, t] == 1]
        if len(observed_vals_b) > 0:
            mean_t = observed_vals_b.mean()
            std_t = observed_vals_b.std()
            if std_t < 1e-6:
                std_t = 1.0
            Y_mean_b[t] = mean_t
            Y_stddev_b[t] = std_t
            Y_std_b[:, t] = (Yb[:, t] - mean_t) / std_t
        else:
            Y_std_b[:, t] = Yb[:, t]

    X_masked_std_b = np.zeros_like(Yb_fill)
    for t in range(T_periods):
        X_masked_std_b[:, t] = (Yb_fill[:, t] - Y_mean_b[t]) / Y_stddev_b[t]

    X_masked_std_b[Mb == 0] = 0.0

    X_std_b = np.hstack([X_masked_std_b, Mb, covariates_matrix_b]).astype(np.float32)
    Y_std_b = Y_std_b.astype(np.float32)
    Mb = Mb.astype(np.float32)

    # 4. Refit the autoencoder from scratch
    autoencoder_b, best_loss_b = train_autoencoder(
        X_std_b, Y_std_b, Mb,
        input_dim=X_std_b.shape[1],
        output_dim=T_periods,
        batch_size=32,
        epochs=100,
        patience=10,
        verbose=False
    )

    # 5. Recompute ATT quantities
    Y_pred_std_b = autoencoder_b.predict(X_std_b, verbose=0)
    Y_pred_b = np.zeros_like(Y_pred_std_b)
    for t in range(T_periods):
        Y_pred_b[:, t] = Y_pred_std_b[:, t] * Y_stddev_b[t] + Y_mean_b[t]

    long_results_b = []
    units_list_b = Yb_wide.index.tolist()
    times_list_b = Yb_wide.columns.tolist()

    for i, unit in enumerate(units_list_b):
        for t_idx, time in enumerate(times_list_b):
            pred0 = Y_pred_b[i, t_idx]
            treated_post = (Mb[i, t_idx] == 0)
            long_results_b.append({
                'boot_unit_id': unit,
                'time': time,
                'pred0': pred0,
                'treated_post': treated_post
            })

    df_long_b = pd.DataFrame(long_results_b)

    df_b = df_b.drop(columns=['pred0', 'treated_post'], errors='ignore')
    df_b = df_b.merge(df_long_b, on=['boot_unit_id', 'time'], how='left')
    df_b['tau'] = np.where(df_b['treated_post'], df_b['outcome'] - df_b['pred0'], np.nan)

    overall_b = df_b[df_b['treated_post']]['tau'].mean()
    boot_overall.append(overall_b)

    for reg in boot_region.keys():
        reg_taus = df_b[(df_b['treated_post']) & (df_b['region'] == reg)]['tau']
        boot_region[reg].append(reg_taus.mean() if len(reg_taus) > 0 else np.nan)

    med_digital_b = df_b[['boot_unit_id', 'digital_index']].drop_duplicates()['digital_index'].median()
    high_taus = df_b[(df_b['treated_post']) & (df_b['digital_index'] > med_digital_b)]['tau']
    low_taus = df_b[(df_b['treated_post']) & (df_b['digital_index'] <= med_digital_b)]['tau']

    boot_high_digital.append(high_taus.mean() if len(high_taus) > 0 else np.nan)
    boot_low_digital.append(low_taus.mean() if len(low_taus) > 0 else np.nan)

    if (b + 1) % 10 == 0:
        print(f"Bootstrap replication {b+1}/{B} completed")

# Remove NaNs
boot_overall = [x for x in boot_overall if not np.isnan(x)]
for reg in boot_region:
    boot_region[reg] = [x for x in boot_region[reg] if not np.isnan(x)]
boot_high_digital = [x for x in boot_high_digital if not np.isnan(x)]
boot_low_digital = [x for x in boot_low_digital if not np.isnan(x)]

ci_overall = np.percentile(boot_overall, [2.5, 97.5])
ci_high = np.percentile(boot_high_digital, [2.5, 97.5]) if boot_high_digital else [np.nan, np.nan]
ci_low = np.percentile(boot_low_digital, [2.5, 97.5]) if boot_low_digital else [np.nan, np.nan]
ci_region = {r: np.percentile(vals, [2.5, 97.5]) for r, vals in boot_region.items() if len(vals) > 0}

print("\nBootstrap 95% Confidence Intervals (B=50, full refit):")
print(f"Overall ATT: {ci_overall}")
print(f"High digital ATT: {ci_high}")
print(f"Low digital ATT: {ci_low}")
for r, ci in ci_region.items():
    print(f"Region {r}: {ci}")

# -----------------------------------------------------------------------------
# 13. Required plots
# -----------------------------------------------------------------------------
# 1.3 Mean observed outcome over time (ever-treated vs never-treated)
plot_df = df.groupby(['time', 'ever_treated'])['outcome'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(data=plot_df, x='time', y='outcome', hue='ever_treated', marker='o')
plt.title('Mean Observed Outcome over Time')
plt.xlabel('Time')
plt.ylabel('Mean Outcome')
plt.legend(title='Ever Treated')
plt.grid(True)
plt.show()

# 1.7 Plot for at least 4 treated units
treated_units = df[df['ever_treated'] == 1]['unit_id'].unique()
selected_units = treated_units[:4]

for unit in selected_units:
    unit_df = df[df['unit_id'] == unit].sort_values('time')
    treat_t = unit_df['treat_time'].iloc[0]
    plt.figure(figsize=(8,5))
    plt.plot(unit_df['time'], unit_df['outcome'], 'o-', label='Observed')
    plt.plot(unit_df['time'], unit_df['pred0'], 's--', label='Predicted No-Treatment')
    plt.axvline(x=treat_t, color='red', linestyle='--', label='Treatment Start')
    plt.title(f'Unit {unit}')
    plt.xlabel('Time')
    plt.ylabel('Outcome')
    plt.legend()
    plt.grid(True)
    plt.show()

# 1.9 Event-time ATT line plot
event_att_df = event_att.reset_index()
event_att_df.columns = ['event_time', 'ATT']
plt.figure(figsize=(10,6))
plt.plot(event_att_df['event_time'], event_att_df['ATT'], marker='o', linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Event-Time Average Treatment Effect on Treated (ATT)')
plt.xlabel('Time relative to treatment')
plt.ylabel('ATT')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# Export labels for Exercise 2
# -----------------------------------------------------------------------------
unit_tau_ex2 = df[df['treated_post']].groupby('unit_id')['tau'].mean().reset_index()
unit_tau_ex2.columns = ['unit_id', 'tau_i']
median_tau = unit_tau_ex2['tau_i'].median()
unit_tau_ex2['H_i'] = (unit_tau_ex2['tau_i'] > median_tau).astype(int)

unit_tau_ex2.to_csv(r"C:\\Users\\Bagel\\Desktop\\Python_Task_2\unit_effect_labels.csv", index=False)

print("\n--- Exercise 1 completed successfully ---")
