#%%
import os
import re
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Matching alpha values
MATCHING_ALPHA_VALUES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Load CSV
df = pd.read_csv('acrobot_results.csv', header=[0,1,2])

# Parse into long format
records = []
for (env, model, alpha_str) in df.columns:
    rewards = df[(env, model, alpha_str)]
    match = re.match(r'pi([\d.]+)_v([\d.]+)', alpha_str)
    if match:
        pi_alpha = float(match.group(1))
        v_alpha = float(match.group(2))
        
        # Only process if pi_alpha == v_alpha AND in our matching list
        if pi_alpha == v_alpha and pi_alpha in MATCHING_ALPHA_VALUES:
            for i, reward in enumerate(rewards):
                if pd.notna(reward):
                    records.append({
                        "Environment": env,
                        "Model": model,
                        "Episode": i,
                        "Reward": reward,
                        "alpha": pi_alpha
                    })

long_df = pd.DataFrame.from_records(records)

print(f"Total records after filtering: {len(long_df)}")
print(f"Unique alpha values: {sorted(long_df['alpha'].unique())}")
print(f"Unique environments: {long_df['Environment'].unique()}")

def create_strip_with_heatbar(model_name):
    df_model = long_df[long_df['Model'] == model_name]

    if df_model.empty:
        print(f"⚠️ No data found for model: {model_name}")
        return go.Figure()

    fig = go.Figure()
    
    # Calculate color scale range based on Acrobot rewards (negative values)
    all_rewards = df_model['Reward'].values
    cmin = all_rewards.min()
    cmax = all_rewards.max()
    print(f"Reward range for {model_name}: [{cmin:.1f}, {cmax:.1f}]")
    
    # Create bars for each alpha - data is already cumulative rewards
    for i, alpha in enumerate(MATCHING_ALPHA_VALUES):
        subset = df_model[df_model['alpha'] == alpha].sort_values('Episode')
        if subset.empty:
            continue
            
        print(f"Alpha {alpha}: Max reward = {subset['Reward'].max():.1f}, Min reward = {subset['Reward'].min():.1f}, Episodes = {len(subset)}")
        
        # Create horizontal bar for this alpha
        fig.add_trace(go.Scatter(
            x=subset['Episode'],
            y=[i] * len(subset),  # Constant y position for this alpha
            mode='markers',
            marker=dict(
                size=8,
                color=subset['Reward'],  # Use reward values directly (already cumulative)
                colorscale='RdYlGn',  # Better for negative to positive values
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title="Cumulative Reward",
                    x=1.02,
                    thickness=20,
                    len=0.8
                ) if i == 0 else None,  # Only show colorbar for first trace
                showscale=(i == 0),  # Only show scale for first trace
                line=dict(width=0),
                opacity=0.8
            ),
            hovertemplate=f'α={alpha}<br>Episode: %{{x}}<br>Cumulative Reward: %{{marker.color:.1f}}<extra></extra>',
            showlegend=False,
            name=f'α={alpha}'
        ))

    # Update layout
    fig.update_layout(
        title=f"Acrobot-v1 {model_name} – Strip Plot (Matching α_π = α_V)",
        xaxis_title="Episode",
        yaxis=dict(
            title="α (π = V)",
            tickmode='array',
            tickvals=list(range(len(MATCHING_ALPHA_VALUES))),
            ticktext=[str(a) for a in MATCHING_ALPHA_VALUES],
            range=[-0.5, len(MATCHING_ALPHA_VALUES) - 0.5]
        ),
        height=600,
        width=1000,
        margin=dict(r=150)
    )

    return fig

# Generate figures
print("\nCreating PolicyGradient figure...")
fig_pg = create_strip_with_heatbar('PolicyGradient')

print("\nCreating PPOClip figure...")
fig_ppo = create_strip_with_heatbar('PPOClip')

# Show figures
fig_pg.show()
fig_ppo.show()
# %%