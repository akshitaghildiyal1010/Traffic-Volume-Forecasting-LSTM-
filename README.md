# **Traffic Volume Forecasting Using LSTM**

## **🔍 Project Overview**
- Forecasts traffic volume using LSTM (Long Short-Term Memory) neural networks.
- Utilizes historical traffic data along with weather and holiday indicators.
- Captures temporal patterns for accurate time series prediction.

## **📁 Dataset**
- **Train.csv** and **Test.csv** used.
- Key features:
  - **date_time**: Timestamp of observation.
  - **traffic_volume**: Recorded traffic.
  - **is_holiday**: Holiday indicator.
  - **weather_type**: General weather condition.
  - **weather_description**: Detailed weather condition.

## **🧪 Methodology**

**Data Preprocessing**

- Parsed and indexed by `date_time`.  
- One-hot encoded categorical features (`is_holiday`, `weather_type`, `weather_description`).  
- Scaled numerical features using `StandardScaler`.  

**Exploratory Data Analysis (EDA)**

- Visualized hourly traffic, holiday vs. non-holiday traffic, weather impact, and day-of-week patterns.  

**Sequence Creation**

- Created sequences with 14 past steps to predict 1 future step.  
- Split data into input features (`trainX`) and target variable (`trainY`).  

**LSTM Model Building**

- Built sequential model with two LSTM layers (64 & 32 units), dropout, and output dense layer.  
- Compiled with Adam optimizer and MSE loss.  

**Model Training**

- Trained for 5 epochs, batch size 16, with 10% validation split.  
- Monitored training and validation loss.  

**Evaluation**

- Predicted on test set, inverse transformed outputs.  
- Calculated MAE and RMSE.  
- Achieved **29.8% improvement in MAE** and **28.8% improvement in RMSE** over the baseline model.  
- Plotted predicted vs. actual traffic.  

**Future Prediction**

- Performed rolling forecasts using model outputs for subsequent inputs.  


## **📊 Results**
- Visuals show close match between predictions and actuals.
- Evaluation metrics indicate good model accuracy:
  - **MAE** and **RMSE** values reported.
- Model effectively captures impact of holidays, weather, and time.

## **🧩 Dependencies**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` / `keras`

## **📌 Notes**
- Model is sensitive to scaling—ensure proper transformation.
- Tuning epochs, sequence length (`n_past`), and LSTM units may improve accuracy.
