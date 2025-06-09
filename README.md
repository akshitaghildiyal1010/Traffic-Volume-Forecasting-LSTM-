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

### **1. Data Preprocessing**
- Parsed `date_time` and set it as index.
- One-hot encoded:
  - `is_holiday`
  - `weather_type`
  - `weather_description`
- Scaled all numerical features using `StandardScaler`.

### **2. Exploratory Data Analysis (EDA)**
- Plotted:
  - **Hourly traffic volume**
  - **Traffic on holidays vs. non-holidays**
  - **Traffic by weather conditions**
  - **Traffic by day of the week**

### **3. Sequence Creation**
- Created sequences using:
  - **n_past = 14** (previous 14 time steps)
  - **n_future = 1** (predicting next time step)
- Split into:
  - **trainX** (input features)
  - **trainY** (target variable)

### **4. LSTM Model Building**
- Sequential LSTM model:
  - Two LSTM layers (64 and 32 units).
  - Dropout for regularization.
  - Dense layer for output.
- Compiled using:
  - **Optimizer**: Adam
  - **Loss**: Mean Squared Error (MSE)

### **5. Model Training**
- Trained with:
  - **Epochs**: 5
  - **Batch Size**: 16
  - **Validation Split**: 10%
- Plotted training and validation loss.

### **6. Evaluation**
- Predicted on test data (scaled and inverse transformed).
- Calculated:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
- Plotted predictions vs. actual values.

### **7. Future Prediction**
- Forecasted traffic on test dataset.
- Used rolling prediction with model outputs as inputs for next step.

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
