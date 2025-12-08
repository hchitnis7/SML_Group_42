import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    df = pd.read_csv('training_data_ht2025.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found.")
    exit()

# Preprocessing: Map target to 0 and 1 for calculations
df['target_num'] = df['increase_stock'].map({'high_bike_demand': 1, 'low_bike_demand': 0})
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # 5=Sat, 6=Sun (assuming 0=Mon)

# Set global style
sns.set_theme(style="whitegrid")

# --- PLOT 1: Target Distribution ---
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='increase_stock', data=df, palette='viridis')
plt.title('1. Class Balance (The Baseline)', fontsize=15)
plt.ylabel('Count')
# Add counts on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# --- PLOT 2: The "Commuter Pattern" (Hour x Weekend Interaction) ---
# This is arguably the most important plot for bike data
plt.figure(figsize=(12, 6))
sns.pointplot(x='hour_of_day', y='target_num', hue='is_weekend', data=df, palette='coolwarm')
plt.title('2. Hourly Demand: Weekday (0) vs. Weekend (1)', fontsize=15)
plt.ylabel('Probability of High Demand')
plt.xlabel('Hour of Day')
plt.legend(title='Is Weekend?')
print("Plot 2: Notice the 'M' shape on weekdays (commute) vs the smooth curve on weekends.")
plt.show()

# --- PLOT 3: Weather Boxplots (Detailed) ---
# We use Boxplots as requested. 
# We will create a 2x2 grid for the main weather variables.
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
weather_vars = ['temp', 'humidity', 'windspeed', 'visibility']
colors = ['Oranges', 'Blues', 'Greens', 'Greys']

for i, var in enumerate(weather_vars):
    row = i // 2
    col = i % 2
    sns.boxplot(x='increase_stock', y=var, data=df, ax=axes[row, col], palette=colors[i])
    axes[row, col].set_title(f'Impact of {var.capitalize()}', fontsize=12)

plt.suptitle('3. Weather Variables vs. Bike Demand (Boxplots)', fontsize=16)
plt.show()

# --- PLOT 4: "Rare" Events (Snow & Precip) ---
# Boxplots don't work well for data that is mostly zero (like snow), 
# so we look at the average demand when these events happen.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Impact of Rain (Precipitation > 0)
df['is_raining'] = df['precip'] > 0
sns.barplot(x='is_raining', y='target_num', data=df, ax=axes[0], palette='Blues_d')
axes[0].set_title('Does Rain Kill Demand?')
axes[0].set_ylabel('Prob. of High Demand')

# Impact of Snow (Snowdepth > 0)
df['has_snow'] = df['snowdepth'] > 0
sns.barplot(x='has_snow', y='target_num', data=df, ax=axes[1], palette='cool')
axes[1].set_title('Does Snow Depth Kill Demand?')
axes[1].set_ylabel('Prob. of High Demand')

plt.suptitle('4. Impact of Precipitation and Snow', fontsize=16)
plt.show()

# --- PLOT 5: Categorical Features (Holiday & Summertime) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x='holiday', y='target_num', data=df, ax=axes[0], palette='autumn')
axes[0].set_title('Impact of Holidays')
axes[0].set_ylabel('Prob. of High Demand')

sns.barplot(x='summertime', y='target_num', data=df, ax=axes[1], palette='spring')
axes[1].set_title('Impact of Summertime')
axes[1].set_ylabel('Prob. of High Demand')

plt.suptitle('5. Categorical Feature Impact', fontsize=16)
plt.show()

# --- PLOT 6: Histograms (Feature Distributions) ---
# It's important to see if features are Normal (bell curve) or Skewed
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['temp'], kde=True, ax=axes[0], color='orange')
axes[0].set_title('Temperature Distribution')

sns.histplot(df['windspeed'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Windspeed Distribution')

sns.histplot(df['humidity'], kde=True, ax=axes[2], color='blue')
axes[2].set_title('Humidity Distribution')

plt.suptitle('6. Feature Distributions (Histograms)', fontsize=16)
plt.show()

# --- PLOT 7: Correlation Heatmap (The Summary) ---
plt.figure(figsize=(12, 10))
# Drop the helper columns we made for plotting
cols_to_corr = ['hour_of_day', 'temp', 'humidity', 'windspeed', 'visibility', 
                'dew', 'precip', 'snowdepth', 'cloudcover', 'target_num']
corr = df[cols_to_corr].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('7. Correlation Matrix', fontsize=16)
plt.show()