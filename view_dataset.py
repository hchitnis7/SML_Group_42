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

# Preprocessing
df['target_num'] = df['increase_stock'].map({'high_bike_demand': 1, 'low_bike_demand': 0})
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) 

# Set global style
sns.set_theme(style="whitegrid")

# --- PLOT 1: Class Balance ---
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='increase_stock', data=df, palette='viridis')
plt.title('Class Balance', fontsize=15)
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

# SAVE PLOT 1
plt.savefig('1_class_balance.pdf', bbox_inches='tight')
print("Saved 1_class_balance.pdf")
#plt.show()

# --- PLOT 2: Hourly Demand ---
plt.figure(figsize=(12, 6))
sns.pointplot(x='hour_of_day', y='target_num', hue='is_weekend', data=df, palette='coolwarm')
plt.title(' Hourly Demand on Weekdays (0) and Weekends (1)', fontsize=15)
plt.ylabel('Probability of High Demand')
plt.xlabel('Hour of Day')
plt.legend(title='Is Weekend?')

# SAVE PLOT 2
plt.savefig('2_hourly_demand.pdf', bbox_inches='tight')
print("Saved 2_hourly_demand.pdf")
#plt.show()

# --- PLOT 3: Weather Boxplots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Map column names (keys) to nice display names (values)
variable_map = {
    'temp': 'Temperature', 
    'humidity': 'Humidity'
}
colors = ['Oranges', 'Blues']

for i, (col_name, display_name) in enumerate(variable_map.items()):
    sns.boxplot(x='increase_stock', y=col_name, data=df, ax=axes[i], palette=colors[i])
    
    # Set the Title 
    axes[i].set_title(f'Impact of {display_name}', fontsize=12)
    
    # Set the Y-Axis Label 
    axes[i].set_ylabel(display_name)
    
    axes[i].set_xlabel('')

plt.suptitle('Impact of temperature and humidity on bike demand', fontsize=16)

# Save and show
plt.savefig('3_weather_boxplots.pdf', bbox_inches='tight')
#plt.show()

# --- PLOT 4: Rare Events (Rain/Snow) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Rain
df['is_raining'] = df['precip'] > 0
sns.barplot(x='is_raining', y='target_num', data=df, ax=axes[0], palette='Blues_d')
axes[0].set_title('Bike demand during rain')
axes[0].set_ylabel('Prob. of High Demand')
# Snow
df['has_snow'] = df['snowdepth'] > 0
sns.barplot(x='has_snow', y='target_num', data=df, ax=axes[1], palette='cool')
axes[1].set_title('Bike demand during snow')
axes[1].set_ylabel('Prob. of High Demand')

plt.suptitle('Impact of precipitation and snow on bike demand', fontsize=16)

# SAVE PLOT 4
plt.savefig('4_precip_snow_impact.pdf', bbox_inches='tight')
print("Saved 4_precip_snow_impact.pdf")
#plt.show()

# --- PLOT 5: Categorical Features ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x='holiday', y='target_num', data=df, ax=axes[0], palette='autumn')
axes[0].set_title('Impact of Holidays')
axes[0].set_ylabel('Prob. of High Demand')

sns.barplot(x='summertime', y='target_num', data=df, ax=axes[1], palette='spring')
axes[1].set_title('Impact of Summertime')
axes[1].set_ylabel('Prob. of High Demand')

plt.suptitle('Impact of holidays and summertime on bike demand', fontsize=16)

# SAVE PLOT 5
plt.savefig('5_categorical_impact.pdf', bbox_inches='tight')
print("Saved 5_categorical_impact.pdf")
#plt.show()


# "Bad Visibility" flag (less than 10km)
bad_vis_mask = df['visibility'] < 10

# Calculate probabilities
prob_low_vis = df[bad_vis_mask]['target_num'].mean()
prob_high_vis = df[~bad_vis_mask]['target_num'].mean()

print(f"Probability of High Demand when visibility is GOOD (>=10km): {prob_high_vis:.2%}")
print(f"Probability of High Demand when visibility is BAD (<10km):  {prob_low_vis:.2%}")

# --- PLOT 7: Correlation Heatmap ---
plt.figure(figsize=(12, 10))
cols_to_corr = ['hour_of_day', 'temp', 'humidity', 'windspeed', 'visibility', 
                'dew', 'precip', 'snowdepth', 'cloudcover', 'target_num']
corr = df[cols_to_corr].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('7. Correlation Matrix', fontsize=16)

# SAVE PLOT 7
plt.savefig('7_correlation_matrix.pdf', bbox_inches='tight')
print("Saved 7_correlation_matrix.pdf")
#plt.show()

plt.figure(figsize=(10, 6))
# Group by month to get the average probability
monthly_data = df.groupby('month')['target_num'].mean().reset_index()
sns.lineplot(x='month', y='target_num', data=monthly_data, marker='o', color='green', linewidth=2.5)
plt.title('Bike Demand by Month', fontsize=15)
plt.ylabel('Probability of High Demand')
plt.xlabel('Month (1=Jan, 12=Dec)')
plt.xticks(range(1, 13)) # Ensure all 12 months are shown
plt.grid(True, linestyle='--')

plt.savefig('8_monthly_trend.pdf', bbox_inches='tight')
print("Saved 8_monthly_trend.pdf")

# --- PLOT 9: Day of Week Analysis ---
plt.figure(figsize=(10, 6))
# Map 0-6 to Mon-Sun for readability
day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['day_name'] = df['day_of_week'].map(day_map)

sns.barplot(x='day_name', y='target_num', data=df, palette='viridis', 
            order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.title('Bike Demand by Day of Week', fontsize=15)
plt.ylabel('Prob. of High Demand')
plt.xlabel('Day of Week')

plt.savefig('9_weekly_trend.pdf', bbox_inches='tight')
print("Saved 9_weekly_trend.pdf")


print("All plots have been saved individually.")