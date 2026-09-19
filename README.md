# Cab Fare Prediction

Predicting Uber/Lyft cab ride prices using ride history and weather data, comparing
Linear Regression, Random Forest, and XGBoost regression models.

## Overview

This project builds a price prediction pipeline for ride-hailing trips by combining a
ride-history dataset with weather conditions at the pickup and dropoff locations. The
goal is to see how well ride price can be predicted from trip and environmental features,
and to compare a simple linear baseline against two more powerful tree-based ensemble
models.

## Dataset

- **cab_rides.csv** - approximately 693,000 individual ride records, including distance,
  cab type (Uber/Lyft), source and destination location, product type, surge multiplier,
  timestamp, and price.
- **weather.csv** - weather readings (temperature, clouds, pressure, rain, humidity, wind)
  by location and timestamp, used to enrich each ride with source and destination weather
  conditions at the time of the trip.

Both files are Boston-area ride and weather data (commonly known as the Uber and Lyft
Cab Prices dataset). They are included directly in this repository under `data/`.

## Results

| Model | RMSE | R2 Score |
|---|---|---|
| Linear Regression | 2.49 | 0.929 |
| Random Forest | 1.81 | 0.962 |
| XGBoost | 1.66 | 0.968 |

XGBoost achieved the best RMSE and R2 score of the three models. 5-fold cross-validation
was also run on the Linear Regression model (Average R2: 0.929, Average RMSE: 2.49),
confirming the result is stable and not due to a lucky train/test split.

## Methodology

1. **Load data** - read `cab_rides.csv` and `weather.csv`
2. **Clean rides data** - drop rows with missing `price` (55,095 rows removed); confirmed
   no duplicate rows
3. **Clean weather data** - fill missing `rain` values with 0 (no rain recorded)
4. **Aggregate weather** - average weather readings by location, since each location has
   multiple readings over time
5. **Merge datasets** - join averaged weather onto rides twice: once for the source
   location, once for the destination location
6. **Encode categorical features** - one-hot encode `destination`, `source`, `product_id`,
   and `name`; map `cab_type` to 0 (Lyft) / 1 (Uber)
7. **Split features and target** - separate `price` as the target variable (Y) from all
   other features (X)
8. **Train/test split** - 70% train, 30% test, shuffled with a fixed random seed
9. **Scale numeric features** - StandardScaler fit on training data, applied to both
   train and test sets
10. **Train models** - Linear Regression, Random Forest, and XGBoost
11. **Evaluate** - RMSE and R2 score on the held-out test set; 5-fold cross-validation
    for Linear Regression

## Models

- **Linear Regression** - scikit-learn default parameters, used as a baseline
- **Random Forest Regressor** - 100 trees (`n_estimators=100`), fixed random seed for
  reproducibility
- **XGBoost Regressor** - 300 estimators, learning rate 0.05, max depth 6, squared error
  objective

## Project Structure

```text
cab-fare-prediction/
|-- README.md
|-- cab_fare_prediction.ipynb
|-- data/
|   |-- cab_rides.csv
|   |-- weather.csv
```

## Getting Started

This notebook was built and run in Google Colab. To run it yourself:

1. Open `cab_fare_prediction.ipynb` in Google Colab.
2. Go to **Runtime > Run all**.
3. When prompted, upload `cab_rides.csv` and `weather.csv` from the `data/` folder in
   this repository (a built-in upload cell handles this automatically - no manual setup
   needed).
4. Note that Colab sessions are temporary: if your session disconnects, you will need to
   re-upload the data files the next time you run the notebook.

To run locally instead (Jupyter, VS Code, etc.), keep `data/cab_rides.csv` and
`data/weather.csv` in place relative to the notebook and skip the upload cell.

## Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Known Limitations

- The raw ride `time_stamp` (Unix epoch) is used directly as a numeric feature rather
  than being broken down into more meaningful features like hour-of-day or day-of-week.
  This did not prevent strong model performance, but extracting time-based features could
  further improve results.

