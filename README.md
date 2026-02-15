# Hybrid Renewable P-Value Calculator

Statistical analysis tool demonstrating how battery storage affects probability distributions in hybrid renewable energy systems.

## Problem

In hybrid PV+BESS projects, the P50 generation value is **not** simply the sum of component P50 values. Battery dispatch optimization creates non-linear interactions between generation and market prices that shift the entire probability distribution.

This is critical for project valuation because:
- Revenue models rely on P10/P50/P90 generation estimates
- Traditional convolution methods fail to account for dispatch optimization
- Hybrid systems exhibit different risk profiles than standalone assets

## Solution

Monte Carlo simulation framework that:
1. Models PV generation variability using lognormal distributions
2. Simulates electricity price volatility with normal distributions  
3. Applies simplified battery dispatch optimization logic
4. Calculates P10/P50/P90 values for comparative analysis

## Key Insight

**PV-only P50 ≠ Hybrid P50** because battery dispatch creates asymmetric effects:
- Charging during low prices → reduces output when market value is low
- Discharging during high prices → increases output when market value is high
- Net effect: Distribution shape changes, not just mean/variance

This demonstrates why hybrid projects require Monte Carlo simulation rather than simple analytical convolution.

## Technical Approach

### Generation Modeling
- **PV Output**: Lognormal distribution (captures weather variability and physical constraints)
- **Price Dynamics**: Normal distribution with configurable volatility
- **Sample Size**: 10,000 iterations for statistical stability

### Dispatch Logic (Simplified)
```python
if price < charge_threshold and pv_generation > threshold:
    charge_battery(excess_pv)
elif price > discharge_threshold:
    discharge_battery()
```

Real-world models would include:
- State of charge (SoC) tracking
- Degradation constraints  
- Ramp rate limits
- Multi-market optimization (energy + ancillary services)

## Results

Example output comparing standalone PV vs. PV+BESS:
```
PV-Only P-Values:
  P10: 78.3 MW
  P50: 100.2 MW  
  P90: 125.7 MW

Hybrid (PV+BESS) P-Values:
  P10: 82.1 MW (+4.9%)
  P50: 98.4 MW (-1.8%)
  P90: 131.2 MW (+4.4%)

Key Finding: P50 decreases slightly due to charging losses, but P10 
improves (lower downside risk) and P90 increases (higher upside potential).
Distribution becomes wider and slightly right-skewed.
```

## Installation
```bash
# Clone repository
git clone https://github.com/ouafaebousbaa-io/hybrid-pvalue-calculator.git
cd hybrid-pvalue-calculator

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python hybrid_pvalue_calculator.py
```

Outputs:
- Console: P-value statistics
- `hybrid_pvalue_analysis.png`: Distribution comparison plots

### Customization

Modify parameters in `hybrid_pvalue_calculator.py`:
```python
# Generation parameters
pv_p50 = 100        # MW - median PV output
pv_std = 15         # MW - standard deviation

# Market parameters  
price_p50 = 50      # $/MWh - median price
price_std = 20      # $/MWh - price volatility

# Battery parameters
battery_capacity = 50       # MWh
charge_threshold = 40       # $/MWh
discharge_threshold = 60    # $/MWh
```

## Extensions & Future Work

This simplified model could be extended with:

**Technical Improvements:**
- [ ] Full SoC tracking with efficiency losses
- [ ] Degradation modeling (cycle counting)
- [ ] Ramping constraints (MW/min limits)
- [ ] Multi-hour optimization horizon

**Market Sophistication:**
- [ ] Multi-market revenue stacking (energy + FCAS/ancillary services)
- [ ] Price-generation correlation (lower wind = higher prices)
- [ ] Forward curve integration for long-term contracts

**Risk Analysis:**
- [ ] Value-at-Risk (VaR) calculations
- [ ] Tail risk metrics (conditional VaR)
- [ ] Sensitivity analysis on dispatch assumptions

## Applications

This methodology supports:
- **Project Valuation**: Accurate P-value inputs for DCF models
- **Risk Assessment**: Quantifying revenue volatility for lenders
- **Bid Optimization**: Understanding distribution shifts for PPA pricing
- **Stakeholder Communication**: Visualizing why "hybrid ≠ sum of parts"

## Tech Stack

- **Python 3.9+**
- **NumPy** - Statistical distributions and array operations
- **Pandas** - Data manipulation (optional, for extensions)
- **Matplotlib** - Visualization
- **SciPy** - Statistical functions

## References

- Australian Energy Market Operator (AEMO) - FCAS market rules
- NREL - Hybrid optimization best practices
- EDF R&D - Internal battery dispatch methodology


## License

MIT License - Feel free to use for educational/commercial purposes
```

---

## **FILE 2: `requirements.txt`**
```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pandas>=2.0.0# Hybrid-Renewable-P-Value-Calculator
