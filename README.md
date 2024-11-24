# Diarrhea-Prediction-Model

## Project Overview

This project analyzes and predicts diarrhea cases using historical data from four divisions in Bangladesh: Rajshahi, Khulna, Dhaka, and Chattogram. The goal is to explore trends, identify significant correlations with weather variables, and create predictive models for better disease management.

## Features

- **Data Exploration**: Examines trends in diarrhea cases across regions and their relationship with weather variables like temperature, humidity, and precipitation.
- **Data Cleaning**: Implements robust methods to handle outliers while preserving seasonal and trend components.
- **Time Series Analysis**: Decomposes diarrhea cases into seasonal, trend, and residual components for better understanding.
- **Machine Learning Models**: Predicts diarrhea cases using Linear Regression, Random Forest, Support Vector Regression (SVR), and Decision Tree.
- **Visualization**: Provides visual insights through line plots, heatmaps, and decomposition graphs.

### Data Sources

- **Regions**: Rajshahi, Khulna, Dhaka, and Chattogram.
- **Variables**: Diarrhea cases, minimum/maximum temperature, humidity, and precipitation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/diarrhea-prediction.git
   cd diarrhea-prediction
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load datasets:
   - Update data paths in `diarrhea_analysis.py` or use the linked GitHub source.
2. Run the script to perform:
   - Data preprocessing and cleaning.
   - Exploratory data analysis (EDA).
   - Time series decomposition.
   - Machine learning model training and evaluation.
3. Visualizations will be generated and displayed in interactive plots.

## Workflow

1. **Data Preprocessing**:
   - Combines datasets for all divisions.
   - Handles missing values and performs outlier detection.

2. **Exploratory Data Analysis (EDA)**:
   - Identifies trends in diarrhea cases across regions.
   - Examines correlations between weather variables and diarrhea cases.

3. **Time Series Analysis**:
   - Decomposes time series into seasonal, trend, and residual components.

4. **Machine Learning**:
   - Trains predictive models using weather variables and trends.

5. **Visualization**:
   - Generates plots to compare actual vs. predicted values.

## Results

- Minimum temperature showed the strongest correlation with diarrhea cases.
- Random Forest outperformed other models in accuracy and predictive power.

## Models

| Model                     | Performance (R²) |
|---------------------------|-------------------|
| Linear Regression         | *0.8*    |
| Random Forest             | *0.8*    |
| Support Vector Regression | *0.7*    |
| Decision Tree             | *0.6*    |

## Contributions

Feel free to contribute by submitting pull requests for enhancements or bug fixes. Suggestions for new features and improvements are welcome!
