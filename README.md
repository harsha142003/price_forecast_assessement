# 🚀 LSTM Time Series Price Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?style=for-the-badge&logo=pandas)
![Keras](https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras)

</div>

<div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px;">
A robust deep learning solution for time series forecasting using LSTM (Long Short-Term Memory) neural networks. The model is designed to predict future price trends based on historical data patterns.
</div>

## 📊 Project Structure

```bash
project1/
├── main.ipynb           # Main Jupyter notebook with LSTM implementation
├── price_data.csv       # Input historical price data
├── LSTM_future_forecast.csv  # Generated future predictions
└── README.md           # Project documentation
```

## 🎯 Features

- **Robust Column Detection**: Automatically identifies date and price columns
- **Data Preprocessing**: Handles missing values and performs normalization
- **LSTM Architecture**: Deep learning model with sequential layers
- **Time Series Analysis**: 12-month rolling window for predictions
- **Performance Metrics**: MAE, RMSE, and accuracy calculations
- **Future Forecasting**: Generates 12-month future price predictions

## 🏗️ Model Architecture

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">

### Model Summary

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(12, 1)),  # First LSTM Layer
    LSTM(64, return_sequences=False),                      # Second LSTM Layer
    Dense(32, activation='relu'),                          # Dense Hidden Layer
    Dense(1)                                              # Output Layer
])
```

### 🔄 Detailed Layer Architecture

```plaintext
Input Shape (12, 1)
     ↓
┌────────────────┐
│ LSTM Layer 1   │
│ Units: 64      │ ═══► Return Sequences: True
│ Input: (12, 1) │      Output Shape: (12, 64)
└────────────────┘
     ↓
┌────────────────┐
│ LSTM Layer 2   │
│ Units: 64      │ ═══► Return Sequences: False
│ Input: (12,64) │      Output Shape: (64,)
└────────────────┘
     ↓
┌────────────────┐
│ Dense Layer    │
│ Units: 32      │ ═══► Activation: ReLU
│ Input: (64,)   │      Output Shape: (32,)
└────────────────┘
     ↓
┌────────────────┐
│ Output Layer   │
│ Units: 1       │ ═══► Linear Activation
│ Input: (32,)   │      Output Shape: (1,)
└────────────────┘
```

### 🔍 Layer-by-Layer Details

<div style="background-color: #e7f3fe; padding: 15px; border-radius: 8px; margin: 10px 0;">

#### 1️⃣ Input Layer

- Shape: `(12, 1)` - 12 time steps with 1 feature
- Represents 12 months of historical price data
- Each time step contains a normalized price value

#### 2️⃣ First LSTM Layer

- Units: 64 memory cells
- `return_sequences=True`: Outputs full sequence
- Maintains temporal information for second LSTM
- Input shape: `(12, 1)`
- Output shape: `(12, 64)`

#### 3️⃣ Second LSTM Layer

- Units: 64 memory cells
- `return_sequences=False`: Returns only last output
- Further processes temporal patterns
- Input shape: `(12, 64)`
- Output shape: `(64,)`

#### 4️⃣ Dense Hidden Layer

- Units: 32 neurons
- ReLU activation function
- Learns non-linear relationships
- Input shape: `(64,)`
- Output shape: `(32,)`

#### 5️⃣ Output Layer

- Units: 1 neuron
- Linear activation (default)
- Predicts single price value
- Input shape: `(32,)`
- Output shape: `(1,)`

</div>

### 🔧 Model Parameters

- Total parameters: 41,569
- Trainable parameters: 41,569
- Non-trainable parameters: 0

</div>

## 📈 Key Features

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
<h3>Data Processing</h3>
<ul>
<li>Automatic column detection</li>
<li>Date parsing with error handling</li>
<li>MinMax scaling (0-1 range)</li>
<li>80-20 train-test split</li>
</ul>
</div>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
<h3>Model Features</h3>
<ul>
<li>12-month rolling window</li>
<li>Adam optimizer</li>
<li>MSE loss function</li>
<li>Batch size of 8</li>
</ul>
</div>
</div>

## 🚀 Getting Started

1. **Setup Environment**

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

2. **Prepare Data**

- Place your price data CSV file in the project directory
- Ensure it contains date and price columns (flexible naming)

3. **Run the Model**

- Open `main.ipynb` in Jupyter Notebook
- Run all cells sequentially
- Find predictions in `LSTM_future_forecast.csv`

## 📊 Performance Metrics

<div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;">
The model's performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Percentage Accuracy
- Validation Loss Tracking
</div>

## 📈 Visualization

The project includes several visualizations:

- Historical price trends
- Training & validation loss curves
- Actual vs predicted price comparison
- Future price forecasts

## 🛠️ Technical Requirements

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 👥 Author

- **Harsha** - _Initial work_ - [harsha142003](https://github.com/harsha142003)

<div style="background-color: #d4edda; padding: 15px; border-radius: 8px; margin-top: 20px;">
<h2>💡 Future Improvements</h2>
<ul>
<li>Hyperparameter optimization</li>
<li>Additional feature engineering</li>
<li>Extended forecast horizon</li>
<li>Ensemble modeling approaches</li>
</ul>
</div>

---

<div align="center">
Made with ❤️ and TensorFlow
</div>
