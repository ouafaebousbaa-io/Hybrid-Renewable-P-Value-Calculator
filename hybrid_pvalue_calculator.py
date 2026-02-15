import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Generation parameters
PV_P50 = 100          # MW - median PV output
PV_SIGMA = 0.15       # Lognormal shape parameter (controls variability)
N_SAMPLES = 10000     # Monte Carlo iterations

# Market parameters
PRICE_P50 = 50        # $/MWh - median electricity price
PRICE_STD = 20        # $/MWh - price volatility

# Battery parameters
BATTERY_CAPACITY = 50     # MWh - storage capacity
CHARGE_THRESHOLD = 40     # $/MWh - price below which to charge
DISCHARGE_THRESHOLD = 60  # $/MWh - price above which to discharge
CHARGE_RATE = 25          # MW - maximum charging power
DISCHARGE_RATE = 25       # MW - maximum discharging power

# ============================================================================
# PROBABILITY DISTRIBUTION GENERATION
# ============================================================================

def generate_pv_samples(p50, sigma, n_samples):
    """
    Generate PV output samples using lognormal distribution.
    
    Lognormal is appropriate for generation because:
    - Bounded below by zero (can't have negative generation)
    - Right-skewed (weather variability creates occasional high output)
    - Physically realistic for solar irradiance patterns
    """
    mu = np.log(p50)  # Mean of underlying normal distribution
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
    return samples

def generate_price_samples(p50, std, n_samples):
    """
    Generate electricity price samples using normal distribution.
    
    Simplified model - real markets show:
    - Fat tails (extreme prices more common than normal distribution)
    - Mean reversion
    - Seasonality and time-of-day patterns
    """
    samples = np.random.normal(loc=p50, scale=std, size=n_samples)
    # Clip negative prices (rare but possible in some markets)
    samples = np.maximum(samples, 0)
    return samples

# ============================================================================
# BATTERY DISPATCH LOGIC
# ============================================================================

def simple_dispatch(pv_output, price, battery_capacity, charge_threshold, 
                   discharge_threshold, charge_rate, discharge_rate):
    """
    Simplified battery dispatch optimization.
    
    Real-world implementations would include:
    - State of charge (SoC) tracking across time periods
    - Round-trip efficiency (typically 85-90%)
    - Degradation constraints (cycle counting)
    - Ramp rate limits
    - Multi-period optimization horizon
    
    This version: Single-period decision based on price thresholds
    """
    net_output = pv_output
    
    # Charging logic: Low prices + excess PV generation
    if price < charge_threshold and pv_output > 50:
        # Charge battery with excess PV (simplified)
        excess_pv = pv_output - 50
        charge_amount = min(battery_capacity, excess_pv, charge_rate)
        net_output = pv_output - charge_amount
        
    # Discharging logic: High prices
    elif price > discharge_threshold:
        # Discharge battery to maximize revenue
        discharge_amount = min(battery_capacity, discharge_rate)
        net_output = pv_output + discharge_amount
    
    return net_output

# ============================================================================
# P-VALUE CALCULATION
# ============================================================================

def calculate_pvalues(samples):
    """
    Calculate P10, P50, P90 from sample distribution.
    
    P10: 10th percentile (pessimistic case - only 10% chance of lower output)
    P50: 50th percentile (median - most likely outcome)
    P90: 90th percentile (optimistic case - only 10% chance of higher output)
    """
    return {
        'P10': np.percentile(samples, 10),
        'P50': np.percentile(samples, 50),
        'P90': np.percentile(samples, 90),
        'Mean': np.mean(samples),
        'Std': np.std(samples)
    }

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation():
    """Execute Monte Carlo simulation and generate results."""
    
    print("=" * 70)
    print("HYBRID RENEWABLE P-VALUE CALCULATOR")
    print("=" * 70)
    print(f"\nSimulation Parameters:")
    print(f"  PV P50: {PV_P50} MW")
    print(f"  Battery Capacity: {BATTERY_CAPACITY} MWh")
    print(f"  Charge Threshold: ${CHARGE_THRESHOLD}/MWh")
    print(f"  Discharge Threshold: ${DISCHARGE_THRESHOLD}/MWh")
    print(f"  Monte Carlo Samples: {N_SAMPLES:,}")
    print("\n" + "-" * 70)
    
    # Generate samples
    print("\nGenerating probability distributions...")
    pv_samples = generate_pv_samples(PV_P50, PV_SIGMA, N_SAMPLES)
    price_samples = generate_price_samples(PRICE_P50, PRICE_STD, N_SAMPLES)
    
    # Apply battery dispatch to each scenario
    print("Simulating hybrid dispatch optimization...")
    hybrid_samples = np.array([
        simple_dispatch(pv, price, BATTERY_CAPACITY, CHARGE_THRESHOLD,
                       DISCHARGE_THRESHOLD, CHARGE_RATE, DISCHARGE_RATE)
        for pv, price in zip(pv_samples, price_samples)
    ])
    
    # Calculate P-values
    pv_pvalues = calculate_pvalues(pv_samples)
    hybrid_pvalues = calculate_pvalues(hybrid_samples)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nPV-Only Statistics:")
    for key, value in pv_pvalues.items():
        print(f"  {key}: {value:.2f} MW")
    
    print("\nHybrid (PV+BESS) Statistics:")
    for key, value in hybrid_pvalues.items():
        print(f"  {key}: {value:.2f} MW")
    
    print("\n" + "-" * 70)
    print("P-Value Comparison:")
    for key in ['P10', 'P50', 'P90']:
        delta = hybrid_pvalues[key] - pv_pvalues[key]
        pct_change = (delta / pv_pvalues[key]) * 100
        print(f"  {key}: {delta:+.2f} MW ({pct_change:+.1f}%)")
    
    # Generate visualization
    create_visualization(pv_samples, hybrid_samples, pv_pvalues, hybrid_pvalues)
    
    print("\n" + "=" * 70)
    print("Visualization saved: hybrid_pvalue_analysis.png")
    print("=" * 70)
    
    return pv_samples, hybrid_samples, pv_pvalues, hybrid_pvalues

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(pv_samples, hybrid_samples, pv_pvalues, hybrid_pvalues):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hybrid Renewable P-Value Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Probability distributions
    ax1 = axes[0, 0]
    ax1.hist(pv_samples, bins=60, alpha=0.6, label='PV Only', 
             density=True, color='orange', edgecolor='black', linewidth=0.5)
    ax1.hist(hybrid_samples, bins=60, alpha=0.6, label='PV+BESS', 
             density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax1.axvline(pv_pvalues['P50'], color='orange', linestyle='--', 
                linewidth=2, label=f"PV P50: {pv_pvalues['P50']:.1f} MW")
    ax1.axvline(hybrid_pvalues['P50'], color='blue', linestyle='--', 
                linewidth=2, label=f"Hybrid P50: {hybrid_pvalues['P50']:.1f} MW")
    ax1.set_xlabel('Output (MW)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('Generation Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P-value comparison
    ax2 = axes[0, 1]
    p_levels = ['P10', 'P50', 'P90']
    pv_vals = [pv_pvalues[p] for p in p_levels]
    hybrid_vals = [hybrid_pvalues[p] for p in p_levels]
    
    x = np.arange(len(p_levels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, pv_vals, width, label='PV Only', 
                    color='orange', edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, hybrid_vals, width, label='PV+BESS', 
                    color='blue', edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Output (MW)', fontsize=11)
    ax2.set_title('P-Value Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(p_levels, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Cumulative distribution function
    ax3 = axes[1, 0]
    pv_sorted = np.sort(pv_samples)
    hybrid_sorted = np.sort(hybrid_samples)
    cumulative = np.arange(1, len(pv_samples) + 1) / len(pv_samples)
    
    ax3.plot(pv_sorted, cumulative, label='PV Only', color='orange', linewidth=2)
    ax3.plot(hybrid_sorted, cumulative, label='PV+BESS', color='blue', linewidth=2)
    ax3.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='P10')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='P50')
    ax3.axhline(0.9, color='gray', linestyle=':', alpha=0.7, label='P90')
    ax3.set_xlabel('Output (MW)', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Delta analysis (percentage change)
    ax4 = axes[1, 1]
    deltas = [(hybrid_pvalues[p] - pv_pvalues[p]) / pv_pvalues[p] * 100 
              for p in p_levels]
    colors = ['green' if d > 0 else 'red' for d in deltas]
    
    bars = ax4.bar(p_levels, deltas, color=colors, edgecolor='black', 
                   linewidth=1, alpha=0.7)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Change (%)', fontsize=11)
    ax4.set_title('Hybrid Impact on P-Values', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{delta:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('hybrid_pvalue_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization created successfully")

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    pv_samples, hybrid_samples, pv_pvalues, hybrid_pvalues = run_simulation()
