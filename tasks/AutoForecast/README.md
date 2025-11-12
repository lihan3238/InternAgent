# AutoForecast: Building Energy Consumption Forecasting

This task implements AI-powered forecasting for building energy consumption using time series analysis and deep learning.

## Overview

The AutoForecast task uses the UCI Individual Household Electric Power Consumption dataset to predict future energy consumption based on historical patterns, weather conditions, and temporal features.

## Dataset

We use the [UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), which contains:
- 4 years of electrical consumption data
- Measurements collected every minute
- Features include active/reactive power, voltage, current intensity, and sub-metering data
- Time-based features (hour, day of week, month)

## Model Architecture

The system uses a Transformer-based architecture for time series forecasting:
- **Input Embedding**: Linear projection of multivariate time series data
- **Positional Encoding**: Adds temporal position information
- **Transformer Encoder**: Multi-head self-attention mechanism
- **Output Layer**: Predicts future energy consumption

## Key Features

- **Multi-scale Temporal Modeling**: Captures hourly, daily, and weekly patterns
- **Attention Mechanism**: Focuses on relevant historical time steps
- **Uncertainty Estimation**: Provides prediction confidence intervals
- **Physics-informed Constraints**: Ensures realistic energy predictions

## Files Structure

```
AutoForecast/
├── experiment.py          # Main training script
├── metrics.py             # Evaluation metrics (MAE, RMSE, MAPE, etc.)
├── prompt.json           # Task description and research background
├── launcher.sh           # Training launcher script
├── download_data.py      # Data download utility
├── README.md            # This file
└── run_0/               # Output directory for experiments
```

## Quick Start

### 1. Download Dataset

```bash
cd tasks/AutoForecast
python download_data.py
```

### 2. Run Training

```bash
# Using launcher script
./launcher.sh run_0

# Or directly with python
python experiment.py --out_dir run_0
```

### 3. Monitor Training

The training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir ./tensorboard_logs/
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_length` | 24 | Input sequence length (hours) |
| `pred_length` | 1 | Prediction horizon (hours) |
| `batch_size` | 64 | Training batch size |
| `d_model` | 128 | Transformer model dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 4 | Number of transformer layers |
| `learning_rate` | 1e-4 | Learning rate |
| `max_epochs` | 50 | Maximum training epochs |

## Evaluation Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **RMSE** (Root Mean Square Error): Root of mean squared error
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **SMAPE** (Symmetric MAPE): Symmetric percentage error
- **R²** (Coefficient of Determination): Explained variance ratio

## Expected Performance

On the UCI energy consumption dataset:
- MAE: < 0.5 (normalized scale)
- MAPE: < 15%
- Training time: ~10-30 minutes on GPU

## Advanced Usage

### Custom Dataset

To use your own energy consumption dataset:

1. Format your data as CSV with columns: `datetime`, power measurements, and optional weather data
2. Modify `load_energy_data()` function in `experiment.py`
3. Update feature columns in `create_data_loaders()`

### Model Customization

The `EnergyTransformer` class can be extended with:
- Additional attention mechanisms
- Weather feature integration
- Multi-step prediction horizons
- Uncertainty quantification modules

## Research Background

This implementation is based on the methodology described in `prompt.json`, which includes:
- Multi-Scale Temporal Attention (MSTA)
- Physics-informed loss functions
- Uncertainty estimation via Monte Carlo dropout
- Comprehensive ablation studies

## Citation

If you use this code in your research, please cite the UCI dataset:

```
@misc{misc_individual_household_electric_power_consumption_235,
  title={Individual Household Electric Power Consumption},
  url={https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption},
  journal={UCI Machine Learning Repository}
}
```

## Troubleshooting

**Data Download Issues**: If automatic download fails, manually download from the UCI website and place `household_power_consumption.txt` in the `datasets/` directory.

**Memory Issues**: Reduce `batch_size` or `seq_length` for smaller GPUs.

**Convergence Issues**: Adjust `learning_rate` or increase `num_layers` for better performance.
