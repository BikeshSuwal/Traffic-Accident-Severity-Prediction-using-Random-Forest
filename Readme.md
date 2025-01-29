# Traffic Accident Severity Prediction using Random Forest

## Overview

This project applies a **Random Forest** machine learning model to predict accident severity based on accident records in Texas. The dataset was sourced from [Kaggle: US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

## Dataset Preprocessing

The following steps were performed to clean and preprocess the dataset:

### Step 1: Filtering Data

- Extracted Texas accident records from the US dataset and saved them as **Texas\_Accidents.csv**.

### Step 2: Data Cleaning

- **Dropped Columns:**
  - `End_Lat` (223,465 missing values out of 582,837 rows)
  - `End_Lng` (223,465 missing values)
  - `Wind_Chill(F)` (356,841 missing values)
  - `Description` (considered unreliable)
- **Handled Missing Values:**
  - Replaced missing values in `Precipitation(in)` with `0`.
  - Replaced empty cells in `Weather_Condition` with `Clear`.
  - Filled empty cells in `Wind_Speed(mph)` with the column average.
  - Removed rows where the following columns had missing values:
    - `Wind_Direction`, `Street`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`
- Saved the cleaned file as **Texas\_Accidents\_Cleaned.csv**.

### Step 3: Feature Engineering

- **Dropped Columns:**
  - `ID`, `State`, `Country`, `Timezone` (contained only one unique value)
- **Encoding:**
  - **One-Hot Encoding:** `Source` (3 unique values)
  - **Frequency Encoding:** Applied to categorical variables with high cardinality:
    - `Street`, `City`, `County`, `Zipcode`, `Airport_Code`, `Wind_Direction`, `Weather_Condition`
    - Grouped rare values (≤10 occurrences) into an "Others" category
  - **DateTime Feature Extraction:**
    - Converted `Start_Time`, `End_Time`, and `Weather_Timestamp` into `datetime` format
    - Extracted `Hour`, `Day_of_Week`, `Month`, `Year`, and `Is_Weekend`
    - Categorized `Hour` into `Morning`, `Afternoon`, `Evening`, and `Night`
  - **Binary Encoding:**
    - Encoded `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, and `Astronomical_Twilight` (Day/Night values)

All preprocessing was performed using **pandas**.

## Model Training

- The cleaned dataset was split into training and testing sets (`test_size=0.2`).
- A **Random Forest** model with default hyperparameters was trained using **sklearn**.
- **Baseline Accuracy:** `91.44%`

## Installation & Usage

To reproduce the results, follow these steps:

### Prerequisites

- Python 3.x
- Required libraries:
  ```sh
  pip install pandas scikit-learn numpy matplotlib seaborn
  ```

### Run the Project

1. Clone the repository:
   ```sh
   git clone https://github.com/BikeshSuwal/Traffic-Accident-Severity-Prediction-using-Random-Forest.git
   ```
2. Run the preprocessing and model training script:
   ```sh
   texas data filtering.py
   ```

## Results & Future Work

- The model achieved an accuracy of **91.44%**.
- Future improvements may include hyperparameter tuning and trying different models such as XGBoost or Neural Networks.

## License

This project is licensed under the MIT License.

---

Feel free to contribute by opening issues or submitting pull requests!

