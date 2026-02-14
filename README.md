# Load Demand Forecasting with LSTM Networks
<img width="720" height="363" alt="image" src="https://github.com/user-attachments/assets/50203089-a264-4f6d-b3e2-7c93c440e9a1" />

## Overview

The objective of this project is to build Deep Learning models to forecast electricity load demand using LSTM architectures. Accurate load forecasting is essential for power grid management and energy optimization.

The project involves:

- **Dataset**: https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama?select=continuous+dataset.csv

- **Data Preprocessing**: Feature Engineering, Normalization, Sequence Generation, Train-Test Split.

- **Modeling**: Implemented three LSTM architectures (**Uni-directional LSTM**, **Bi-directional LSTM**, and **Stacked Bi-directional LSTM**) for **time series forecasting** of electricity load.

- **Evaluation and Visualization**: Model performance is evaluated using **Mean Absolute Error (MAE)** and **validation loss**.

## Dataset

The dataset contains **48,048 hourly records** of electricity load consumption spanning from **January 3, 2015 to June 27, 2020** (approximately 5.5 years).

The dataset includes the following features:

- **Time**: Timestamp for each hourly record
- **Load**: Electricity consumption in MWh (target variable)
- **Temperature**: Ambient temperature in degrees Celsius
- **Humidity**: Relative humidity percentage
- **Liquid**: Precipitation levels
- **Wind**: Wind speed
- **Holiday**: Binary indicator (0 = working day, 1 = holiday)
- **School**: Binary indicator (0 = school holiday, 1 = school day)

## Data Preprocessing

To ensure high-quality input data for the models, the following preprocessing steps were performed:

- **Feature Engineering**:
  - Extracted temporal features from timestamps (hour, day of week, month).
  - Used 7 input features: Load, Temperature, Humidity, Liquid, Wind, Holiday, and School.

- **Normalization**:
  - Applied **Min-Max Scaling** to normalize all features to the range **[0,1]**.

- **Sequence Generation**:
  - Created sliding window sequences with a **lookback period of 12 time steps**.
  - Each input sequence has shape **(12 time steps × 9 features)** including engineered temporal features.

- **Dataset Splitting**:
  - Split into **Train (80%)** and **Test (20%)** sets.
  - Further divided training set into **Train (90%)** and **Validation (10%)** for model tuning.

## Model Overview

This implementation compares three LSTM architectures for time series forecasting:

### Uni-directional LSTM

<img width="460" height="840" alt="Uni-LSTM" src="https://github.com/user-attachments/assets/83084d36-2ae3-4eac-ba1f-795f6993da15" />

- **Input Layer**: Shape (12 time steps × 9 features)
- **LSTM Layer**: 512 units with recurrent dropout = 0.2
- **Dropout Layer**: Rate = 0.5
- **Dense Layer**: 16 units with ReLU activation
- **Output Dense Layer**: 1 unit

### Bi-directional LSTM

<img width="460" height="840" alt="Bi-LSTM" src="https://github.com/user-attachments/assets/630241c8-4e02-4240-ac97-3e70f972f2ea" />

- **Input Layer**: Shape (12 time steps × 9 features)
- **Bi-LSTM Layer**: 512 units with recurrent dropout = 0.2
- **Dropout Layer**: Rate = 0.5
- **Dense Layer**: 16 units with ReLU activation
- **Output Dense Layer**: 1 unit

### Stacked Bi-directional LSTM

<img width="460" height="1160" alt="Stacked Bi-LSTM" src="https://github.com/user-attachments/assets/87dbc942-da0c-4a89-a651-3fb05ae06bb4" />

- **Input Layer**: Shape (12 time steps × 9 features)
- **Bi-LSTM Layer 1**: 512 units, return_sequences = True, recurrent_dropout = 0.2
- **Dropout Layer**: Rate = 0.5
- **Bi-LSTM Layer 2**: 256 units, return_sequences = False, recurrent_dropout = 0.2
- **Dropout Layer**: Rate = 0.5
- **Dense Layer**: 16 units with ReLU activation
- **Output Dense Layer**: 1 unit

### Hyperparameters

- Learning Rate: 1e-3 (Dynamic with ReduceLROnPlateau)
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 32
- Epochs: 200 (with Early Stopping, patience = 25)

## Results & Visualization

The models achieved the following performance on the validation set:

| Model | MAPE (%) | R² Score |
|-------|---------------------|----------------------|
| Uni-directional LSTM | 4.320 | 0.870 |
| Bi-directional LSTM | 4.386 | 0.927 |
| Stacked Bi-directional LSTM | **4.414** | **0.963** |

The **Stacked Bi-directional LSTM** achieved the best performance with a R² Score of **0.963**.

The notebook includes visualizations for each model:
- Actual vs Predicted load values (sample and full test set)
- Scatter plot showing prediction accuracy
- Distribution of prediction errors
- Training and validation loss curves

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or contributions, feel free to reach out to:

- **Ariharasudhan A** - [Email](mailto:ariadaikalam1234@gmail.com)
- **Harish R** - [Email](mailto:harishsekar2004@gmail.com)
