# Cab Fare Prediction

Predicting Uber/Lyft cab ride prices using ride and weather data, comparing Linear
Regression, Random Forest, and XGBoost.

## Overview

This project merges a ride-history dataset (~693,000 rides) with weather data by
pickup/dropoff location, then trains and evaluates three regression models to predict
ride price from features like distance, cab type, and weather conditions.

## Results

| Model | RMSE | R2 Score |
|---|---|---|
| Linear Regression | 2.49 | 0.929 |
| Random Forest | 1.81 | 0.962 |
| XGBoost | 1.66 | 0.968 |

XGBoost achieved the best RMSE and R2 score of the three models. 5-fold cross-validation
was also used to validate the Linear Regression model's stability.

## Pipeline

1. Load ride and weather datasets
2. Clean data (drop nulls, duplicates, fill missing rain values with 0)
3. Merge weather data onto rides by source and destination location
4. One-hot encode categorical features (cab type, source, destination, product ID, name)
5. Standard-scale numeric features
6. Train/test split (70/30)
7. Train and evaluate Linear Regression, Random Forest, and XGBoost

## Project Structure

```text
cab-fare-prediction/
|-- README.md
|-- cab_fare_prediction.ipynb
|-- data/
|   |-- cab_rides.csv
|   |-- weather.csv
```

## Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Contributor

**Kaniz Reza Mithila**
Department of Computer Science, East West University, Dhaka, Bangladesh

