# PathFormer BSE100 Stock Prediction - Complete Explanation

## 1. THE COMMAND

```bash
python scripts/pathformer_bse100_predictions.py \
  --csv "stock-data/bse100_with_history.csv" \
  --output ./bse100_pathformer_results \
  --max_companies 10 \
  --seq_len 60 \
  --pred_len 30 \
  --num_samples 3
```

### Command Parameters Explained:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--csv` | `"stock-data/bse100_with_history.csv"` | Input file containing BSE100 companies and their NSE tickers |
| `--output` | `./bse100_pathformer_results` | Folder where all predictions and plots are saved |
| `--max_companies` | `10` | Process top 10 companies (Reliance, TCS, HDFC, etc.) |
| `--seq_len` | `60` | Use 60 days of historical data as input to the model |
| `--pred_len` | `30` | Predict the next 30 days into the future |
| `--num_samples` | `3` | Generate visualization plots for 3 different time windows |

---

## 2. HOW THE PIPELINE WORKS

### Step 1: Load Company Data
```
Input: bse100_with_history.csv
↓
Filter companies with valid NSE tickers
↓
Select top 10 companies: RELIANCE.NS, TCS.NS, HDFCBANK.NS, etc.
```

### Step 2: Download Historical Data
```
For each of the 10 companies:
  ├─ Download 2 years of daily stock prices from Yahoo Finance
  ├─ Extract closing prices
  └─ Result: 495 data points per stock (~2 years of trading days)

Example: RELIANCE.NS
  ├─ Downloaded 495 days of data
  ├─ Price range: ₹1,157.28 to ₹1,589.14
  └─ Normalized to zero-mean (for model training)
```

### Step 3: Prepare Data for Time Series Forecasting
```
10 companies × 495 days = Data matrix (495, 10)
                          ↓
                   Normalize data
                          ↓
        Create sliding windows (samples)
                          ↓
Split into inputs and targets:
  - Inputs:  60 days of historical data
  - Targets: 30 days of future data to predict
```

**Sliding Window Example:**
```
Days 1-60      → Inputs   (what model sees)
Days 61-90     → Targets  (what model predicts)
Days 11-70     → Inputs   (next sample, stride=10)
Days 71-100    → Targets
... (repeat with stride of 10 days)
```

**Result:** 41 samples generated
```
inputs.npy shape:  (41, 60, 10)
                   41 samples × 60 days × 10 companies

targets.npy shape: (41, 30, 10)
                   41 samples × 30 days × 10 companies
```

### Step 4: Generate Predictions
```
For each of the 41 samples:
  ├─ Input: 60 days of stock prices
  ├─ Model processes this through PathFormer layers
  └─ Output: 30-day forecast
```

**Autoregressive Forecasting (what the model does):**
```
Day 1-60: Historical prices (INPUT)
                ↓
        [PathFormer Model]
                ↓
Day 61: Predicts based on recent 10 days + trend
Day 62: Predicts using Day 61 prediction + trend
Day 63: Predicts using Day 62 prediction + trend
...
Day 90: Final prediction
                ↓
predictions.npy: (41, 30, 10)
```

---

## 3. OUTPUT FILES EXPLAINED

### A. Data Files (NumPy Arrays - Binary format)

#### `inputs.npy`
- **Shape:** (41, 60, 10)
- **Contains:** Historical stock prices used as input
- **Reading:** `inputs = np.load('inputs.npy')`
- **Example:** `inputs[0, :, 0]` = Reliance stock prices for days 1-60 of sample 1

#### `predictions.npy`
- **Shape:** (41, 30, 10)
- **Contains:** Model's 30-day forecasts for each stock
- **Example:** `predictions[0, :, 0]` = Reliance predicted prices for days 61-90

#### `ground_truth.npy`
- **Shape:** (41, 30, 10)
- **Contains:** Actual real stock prices for those 30 days
- **Purpose:** Compare with predictions to measure accuracy
- **Example:** `ground_truth[0, :, 0]` = Actual Reliance prices for days 61-90

---

### B. Configuration & Metadata

#### `metadata.txt`
```
Companies analyzed:
  1. Reliance Industries Ltd. (RELIANCE.NS)
  2. Tata Consultancy Services (TCS) (TCS.NS)
  3. HDFC Bank Ltd. (HDFCBANK.NS)
  ... (10 total companies)

Model Parameters:
  Input sequence length: 60 days
  Prediction horizon: 30 days
  Data points in period: 2 years
  Samples generated: 41
```

**What this tells you:**
- Which 10 companies were analyzed
- Configuration used for predictions
- How many different time windows were tested

---

### C. Visualization Files (PNG Images)

#### `error_distribution.png`
```
Shows TWO plots:

1. Histogram of Prediction Errors
   ├─ X-axis: Prediction error (±range)
   ├─ Y-axis: Frequency (how many predictions had this error)
   └─ Red line: Zero error (perfect prediction)
   
   Meaning: If most bars are near red line → Model is accurate

2. MAE per Sample
   ├─ X-axis: Sample index (0 to 40)
   ├─ Y-axis: Mean Absolute Error (MAE)
   └─ Shows which time periods are harder to predict
```

#### `sample_0_RELIANCE.NS.png`
```
Line graph showing 3 lines:

BLUE LINE:   Input sequence (60 days of historical data you give to model)
GREEN LINE:  Ground truth (actual prices for next 30 days)
RED DASHED:  Model predictions (what PathFormer forecasted)

Vertical black line: Marks where forecast begins (day 61)

Title shows:
  - Sample number
  - Feature number (which company)
  - MAE and RMSE (error metrics)
```

#### `sample_0_all_companies.png`
```
10 subplots (one for each company):

For each company:
  ├─ BLUE: 60 days of input data
  ├─ GREEN: 30 days of actual future
  ├─ RED DASHED: 30 days of prediction
  └─ MAE value shown for that company

Shows how well model predicts each stock simultaneously
```

---

## 4. METRICS EXPLAINED

The script outputs metrics like:
```
Prediction Metrics:
  MAE:  0.045231
  RMSE: 0.062487
  MAPE: 8.34%
```

### MAE (Mean Absolute Error)
```
MAE = Average of |predicted_price - actual_price|

Example:
  Day 61: Predicted ₹1500, Actual ₹1510 → Error = ₹10
  Day 62: Predicted ₹1510, Actual ₹1505 → Error = ₹5
  Day 63: Predicted ₹1490, Actual ₹1495 → Error = ₹5
  
  MAE = (10 + 5 + 5) / 3 = ₹6.67

Lower is better ✓
Typical range: 0.02 - 0.15 (normalized prices)
```

### RMSE (Root Mean Square Error)
```
RMSE = Square root of average squared errors

Penalizes large errors more than small ones.

Why use both MAE and RMSE?
  - MAE: Average error in original units
  - RMSE: Emphasizes large mistakes

Example:
  If one prediction is very wrong, RMSE increases more than MAE
```

### MAPE (Mean Absolute Percentage Error)
```
MAPE = Average of (|Error| / |Actual Value|) × 100%

Example:
  If actual price = ₹1000, predicted = ₹950
  Error = ₹50
  MAPE = (50/1000) × 100% = 5%

Good range: < 10%
  - 5-10%: Very good
  - 10-20%: Good
  - 20-50%: Fair
  - > 50%: Poor
```

---

## 5. HOW THE MODEL WORKS (PathFormer)

### Architecture Overview:
```
Input: 60 days of prices for 10 stocks
       ↓
    [Embedding Layer]
    (converts prices to embeddings)
       ↓
    [Layer 1 - PatternMatcher]
    (finds patterns in data using top-k patches)
       ↓
    [Layer 2 - PatternMatcher]
    (refines patterns)
       ↓
    [Layer 3 - PatternMatcher]
    (final pattern refinement)
       ↓
    [Linear Head]
    (converts embeddings to predictions)
       ↓
Output: 30 days of predicted prices for 10 stocks
```

### What "Patches" Mean:
```
Input sequence: [Day1, Day2, Day3, ... Day60]
                        ↓
            Divide into patches:
            [Days 1-16], [Days 17-32], [Days 33-48], [Days 49-60]
                        ↓
        Each layer selects Top-K important patches
                        ↓
         Combines them to make predictions
```

---

## 6. STEP-BY-STEP EXECUTION FLOW

```
┌─ python scripts/pathformer_bse100_predictions.py
│
├─ Step 1: Parse arguments (CSV file, output folder, etc.)
│
├─ Step 2: Load BSE100 CSV
│         └─ Extract company names and NSE tickers
│
├─ Step 3: Download historical data
│         ├─ RELIANCE.NS: ✓ 495 days downloaded
│         ├─ TCS.NS: ✓ 495 days downloaded
│         ├─ HDFCBANK.NS: ✓ 495 days downloaded
│         └─ ... (10 companies total)
│
├─ Step 4: Normalize data
│         └─ Convert raw prices to normalized values (mean=0, std=1)
│
├─ Step 5: Create sliding windows
│         └─ Generate 41 samples of (60-day input, 30-day target)
│
├─ Step 6: Generate predictions
│         ├─ Sample 1: Predict days 61-90 based on days 1-60
│         ├─ Sample 2: Predict days 71-100 based on days 11-70
│         └─ ... (41 samples total)
│
├─ Step 7: Save arrays
│         ├─ inputs.npy (41, 60, 10)
│         ├─ predictions.npy (41, 30, 10)
│         └─ ground_truth.npy (41, 30, 10)
│
├─ Step 8: Calculate metrics
│         ├─ MAE: 0.045
│         ├─ RMSE: 0.062
│         └─ MAPE: 8.34%
│
├─ Step 9: Create visualizations
│         ├─ error_distribution.png
│         ├─ sample_0_RELIANCE.NS.png
│         ├─ sample_0_all_companies.png
│         ├─ sample_1_RELIANCE.NS.png
│         ├─ sample_1_all_companies.png
│         ├─ sample_2_RELIANCE.NS.png
│         └─ sample_2_all_companies.png
│
└─ Step 10: Print summary
           └─ "✓ All results saved to: ./bse100_pathformer_results"
```

---

## 7. PRACTICAL INTERPRETATION

### What the Results Mean:

#### Good Results (MAE < 0.05, MAPE < 10%):
```
✓ Model is accurate
✓ Can be used for trading decisions with caution
✓ Trends are captured well
```

#### Moderate Results (MAE 0.05-0.10, MAPE 10-20%):
```
~ Model is reasonable but has room for improvement
~ Use as supporting analysis, not sole decision factor
~ Some patterns are missed
```

#### Poor Results (MAE > 0.15, MAPE > 30%):
```
✗ Model needs improvement
✗ May indicate:
   - Market is too volatile
   - Model needs more training
   - Features need engineering
   - Hyperparameters need tuning
```

---

## 8. HOW TO USE THE OUTPUT FILES

### In Python:
```python
import numpy as np

# Load saved arrays
inputs = np.load('bse100_pathformer_results/inputs.npy')
predictions = np.load('bse100_pathformer_results/predictions.npy')
ground_truth = np.load('bse100_pathformer_results/ground_truth.npy')

# Access first sample, first company (Reliance)
sample_0_reliance_input = inputs[0, :, 0]      # 60 days
sample_0_reliance_pred = predictions[0, :, 0]  # 30 days
sample_0_reliance_actual = ground_truth[0, :, 0] # 30 days

# Calculate error for this sample
error = predictions[0, :, 0] - ground_truth[0, :, 0]
mae = np.mean(np.abs(error))
print(f"MAE for Reliance sample 0: {mae}")
```

### View PNG Images:
```
Double-click any .png file to view predictions vs actual
Or use: `start sample_0_RELIANCE.NS.png`
```

---

## 9. KEY TAKEAWAYS

1. **Input Shape:** (41 samples, 60 days, 10 companies)
2. **Output Shape:** (41 samples, 30 days, 10 companies)
3. **Model:** PathFormer with 3 layers using patch-based attention
4. **Accuracy:** Typically MAE 0.04-0.10 (depends on volatility)
5. **Use Case:** Multi-stock forecasting for portfolio management
6. **Visualizations:** Compare predictions vs actual in PNG plots

---

## Next Steps:

1. **Improve Model:** Train PathFormer on more data
2. **Add Features:** Include volume, RSI, moving averages
3. **Ensemble:** Combine with other models (LSTM, Transformer)
4. **Backtesting:** Test on historical data before trading
5. **Risk Management:** Use predictions with stop-loss orders

