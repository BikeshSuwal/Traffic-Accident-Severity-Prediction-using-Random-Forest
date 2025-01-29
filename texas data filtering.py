import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


US_accidents = 'US_Accidents_March23.csv'
df = pd.read_csv(US_accidents)
tx_data = df[df['State'] == 'TX']
#tx_data.to_csv('Texas_Accidents.csv', index=False)
#print(tx_data.head())
#print(f"Number of rows in Texas data: {len(tx_data)}")
#drop 'End_Lat', 'End_Lng', 'Wind_Chill(F)'
reduced_tx_data = tx_data.drop(['End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Description'], axis=1)

# Replace missing values in 'Precipitation(in)' with 0
reduced_tx_data['Precipitation(in)'] = reduced_tx_data['Precipitation(in)'].fillna(0)

# Verify the replacement
#print(reduced_tx_data['Precipitation(in)'].isnull().sum())  # Should print 0
#print(reduced_tx_data['Precipitation(in)'].value_counts())  # Check the distribution

# Replace empty cells in 'Weather_Condition' with 'Clear'
reduced_tx_data['Weather_Condition'] = reduced_tx_data['Weather_Condition'].fillna('Clear')


# Calculate the average of non-empty cells in 'Wind_Speed(mph)'
average_wind_speed = reduced_tx_data['Wind_Speed(mph)'].mean()

# Replace empty cells in 'Wind_Speed(mph)' with the average
reduced_tx_data['Wind_Speed(mph)'] = reduced_tx_data['Wind_Speed(mph)'].fillna(average_wind_speed)

# #Verify the replacement
# print("Average Wind Speed:", average_wind_speed)
# print(reduced_tx_data['Wind_Speed(mph)'].isnull().sum())  # Should print 0
# print(reduced_tx_data['Wind_Speed(mph)'].describe())  # Check the distribution


# Remove rows where 'Wind_Direction', 'Street', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight' has empty cells
tx_data_cleaned = reduced_tx_data.dropna(subset=['Wind_Direction', 'Street', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'])

# Verify the removal
print("Number of rows before removal:", len(reduced_tx_data))
print("Number of rows after removal:", len(tx_data_cleaned))
#print(tx_data_cleaned['Wind_Direction', 'Street', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'].isnull().sum())  # Should print 0


print(tx_data_cleaned.count())
#print(tx_data.describe())
#tx_data_cleaned.to_csv('Texas_Accidents_Cleaned.csv', index=False)

# Step 1: Drop unnecessary columns
columns_to_drop = ['ID', 'State', 'Country', 'Timezone']
tx_data_cleaned = tx_data_cleaned.drop(columns=columns_to_drop)

print(tx_data_cleaned.head())

# Step 2: One-hot encode the 'Source' column
tx_data_cleaned = pd.get_dummies(tx_data_cleaned, columns=['Source'], prefix='Source')



#----------------------------------------------------
# List of columns to frequency encode
columns_to_encode = ['Street', 'City', 'County', 'Zipcode', 'Airport_Code', 'Wind_Direction', 'Weather_Condition']
# 

# Step 1: Group rare values into 'Others'
threshold = 10  # Values with 10 or fewer occurrences will be grouped

for column in columns_to_encode:
    # Calculate the frequency of each unique value
    value_counts = tx_data_cleaned[column].value_counts()
    
    # Identify rare values
    rare_values = value_counts[value_counts <= threshold].index
    
    # Replace rare values with 'Others'
    tx_data_cleaned[column] = tx_data_cleaned[column].replace(rare_values, 'Others')

# Step 2: Perform Frequency Encoding
for column in columns_to_encode:
    # Calculate the frequency of each unique value
    value_counts = tx_data_cleaned[column].value_counts()

    # Map frequencies to the column
    tx_data_cleaned[f'{column}_Frequency'] = tx_data_cleaned[column].map(value_counts)

    # Drop the original column (optional)
    tx_data_cleaned = tx_data_cleaned.drop(column, axis=1)

# Verify the changes
print(tx_data_cleaned.head())
# Verify the columns have been dropped
print(len(tx_data_cleaned.columns))

tx_data_cleaned['Start_Time'] = pd.to_datetime(tx_data_cleaned['Start_Time'], format='ISO8601')

# Extract features
tx_data_cleaned['Start_Hour'] = tx_data_cleaned['Start_Time'].dt.hour
tx_data_cleaned['Start_Day_of_Week'] = tx_data_cleaned['Start_Time'].dt.dayofweek  # Monday=0, Sunday=6
tx_data_cleaned['Start_Month'] = tx_data_cleaned['Start_Time'].dt.month
tx_data_cleaned['Start_Year'] = tx_data_cleaned['Start_Time'].dt.year
tx_data_cleaned['Start_Is_Weekend'] = tx_data_cleaned['Start_Time'].dt.dayofweek >= 5  # Weekend=True

# Categorize time of day
def categorize_time(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

tx_data_cleaned['Start_Time_of_Day'] = tx_data_cleaned['Hour'].apply(categorize_time)

# One-hot encode 'Time_of_Day'
tx_data_cleaned = pd.get_dummies(tx_data_cleaned, columns=['Start_Time_of_Day'], prefix='Time')

tx_data_cleaned = tx_data_cleaned.drop('Start_Time', axis=1)

tx_data_cleaned['End_Time'] = pd.to_datetime(tx_data_cleaned['End_Time'], format='ISO8601')
# Extract features
tx_data_cleaned['End_Hour'] = tx_data_cleaned['End_Time'].dt.hour
tx_data_cleaned['End_Day_of_Week'] = tx_data_cleaned['End_Time'].dt.dayofweek  # Monday=0, Sunday=6
tx_data_cleaned['End_Month'] = tx_data_cleaned['End_Time'].dt.month
tx_data_cleaned['End_Year'] = tx_data_cleaned['End_Time'].dt.year
tx_data_cleaned['End_Is_Weekend'] = tx_data_cleaned['End_Time'].dt.dayofweek >= 5  # Weekend=True

# Categorize time of day
def categorize_time(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

tx_data_cleaned['End_Time_of_Day'] = tx_data_cleaned['End_Hour'].apply(categorize_time)

# One-hot encode 'Time_of_Day'
tx_data_cleaned = pd.get_dummies(tx_data_cleaned, columns=['End_Time_of_Day'], prefix='Time')

tx_data_cleaned = tx_data_cleaned.drop('End_Time', axis=1)

tx_data_cleaned['End_Time_of_Day'] = tx_data_cleaned['End_Hour'].apply(categorize_time)

# One-hot encode 'Time_of_Day'
tx_data_cleaned = pd.get_dummies(tx_data_cleaned, columns=['End_Time_of_Day'], prefix='Time')

tx_data_cleaned['Weather_Timestamp'] = pd.to_datetime(tx_data_cleaned['Weather_Timestamp'], format='ISO8601')
# Extract features
tx_data_cleaned['Weather_Hour'] = tx_data_cleaned['Weather_Timestamp'].dt.hour
tx_data_cleaned['Weather_Day_of_Week'] = tx_data_cleaned['Weather_Timestamp'].dt.dayofweek  # Monday=0, Sunday=6
tx_data_cleaned['Weather_Month'] = tx_data_cleaned['Weather_Timestamp'].dt.month
tx_data_cleaned['Weather_Year'] = tx_data_cleaned['Weather_Timestamp'].dt.year
tx_data_cleaned['Weather_Is_Weekend'] = tx_data_cleaned['Weather_Timestamp'].dt.dayofweek >= 5  # WeekWeather=True

# Categorize time of day
def categorize_time(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

tx_data_cleaned['Weather_Time_of_Day'] = tx_data_cleaned['Weather_Hour'].apply(categorize_time)

# One-hot encode 'Time_of_Day'
tx_data_cleaned = pd.get_dummies(tx_data_cleaned, columns=['Weather_Time_of_Day'], prefix='Time')

tx_data_cleaned = tx_data_cleaned.drop('Weather_Timestamp', axis=1)

###############################################

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply binary encoding to each column
columns_to_encode = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

for col in columns_to_encode:
    tx_data_cleaned[col] = label_encoder.fit_transform(tx_data_cleaned[col])

# Features (X) - All columns except 'Severity'
X = tx_data_cleaned.drop('Severity', axis=1)

# Target (y) - The 'Severity' column
y = tx_data_cleaned['Severity']

print(X.columns)
print(y.describe())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# Initialize the Random Forest model with default hyperparameters
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Baseline Accuracy:", accuracy_score(y_test, y_pred))