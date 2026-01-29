# Theoretical Foundations

SYMBIONT is built on rigorous theoretical foundations with 17 theorems and 2 propositions providing performance guarantees.

## SMIS Theorems (1-5)

### Theorem 1: Mutual Information Through Structure

The mutual information between source event and received signal through structural waveguide:

```
I(X; Y) = ∫ log₂(1 + |H(f)|² · Sₓ(f) / N₀) df
```

where H(f) is the transmissibility function, Sₓ(f) is input PSD, and N₀ is noise variance.

**Implementation:**
```python
from symbiont.core.theorems import mutual_information_through_structure

mi = mutual_information_through_structure(
    transmissibility_mag=transmissibility,
    input_psd=psd,
    noise_variance=1e-12,
    frequencies=freq_axis
)
```

### Theorem 2: SMIS Synchronization Bound

For n sensors, synchronization error scales as O(1/√n):

```
ε_sync ≤ (σ_noise · √2) / (c_eff · √n)
```

**Implementation:**
```python
from symbiont.core.theorems import smis_synchronization_bound

bound = smis_synchronization_bound(
    noise_std=1e-6,
    effective_velocity=4500.0,
    n_samples=100
)
```

### Theorem 3: Multimodal Synchronization Error

Using both L and T waves reduces error by factor of √2:

```
SE_multimodal = σ / (d · √(1/c_L² + 1/c_T²))
```

**Implementation:**
```python
from symbiont.core.theorems import multimodal_synchronization_error

error = multimodal_synchronization_error(
    noise_std=1e-6,
    c_L=5960.0,
    c_T=3240.0,
    num_events=10
)
```

### Theorem 4: Cramér-Rao Lower Bound

Fisher information matrix for joint timestamp-velocity estimation:

```
CRLB(θ) = I(θ)⁻¹
```

**Implementation:**
```python
from symbiont.core.theorems import cramer_rao_fisher_information

fisher, crlb = cramer_rao_fisher_information(
    distances=sensor_distances,
    noise_variance=1e-12,
    velocity=5960.0
)
```

### Theorem 5: Optimal Dispersion Fusion

Minimum variance estimator for multi-modal fusion:

```
σ²_fused = 1 / Σᵢ (1/σᵢ²)
```

---

## ECTI Theorems (6-7) and Proposition 1

### Proposition 1: Velocity-Damage Sensitivity

Wave velocity change indicates damage before transmissibility:

```
Δc/c₀ = -α · D
```

where D is damage parameter and α is sensitivity coefficient.

**Implementation:**
```python
from symbiont.core.theorems import velocity_damage_sensitivity

delta_c = velocity_damage_sensitivity(
    baseline_velocity=5960.0,
    damage_parameter=0.1,
    sensitivity=0.5
)
```

### Theorem 6: False Alarm Probability Bound

For threshold τ, false alarm probability is bounded:

```
P_FA ≤ P(score > τ | healthy) · P(uncertainty < τ_u | healthy)
```

**Implementation:**
```python
from symbiont.core.theorems import false_alarm_probability_bound

p_fa = false_alarm_probability_bound(
    threshold=0.5,
    score_std_healthy=0.1,
    uncertainty_mean_healthy=0.05
)
```

### Theorem 7: Damage Localization

Damage location estimated from transmissibility gradient:

```
x_damage = argmax_x |∇T(x)|
```

---

## ECCP Theorems (8-10) and Proposition 2

### Proposition 2: Harvested Power Model

Piezoelectric power from structural vibration:

```
P = (d₃₁² · V · ω²) / R_int · A²
```

**Implementation:**
```python
from symbiont.core.theorems import harvested_power

power = harvested_power(
    d31=190e-12,
    volume=1e-6,
    frequency=50.0,
    amplitude=1e-3,
    internal_resistance=1e6
)
```

### Theorem 8: Nash Equilibrium Probability

Optimal transmission probability in game-theoretic setting:

```
p* = (1 - 1/e) · E[energy] / (n · E_tx)
```

**Implementation:**
```python
from symbiont.core.theorems import nash_equilibrium_probability

p_nash = nash_equilibrium_probability(
    expected_energy=1e-6,
    num_nodes=10,
    tx_energy=1e-7
)
```

### Theorem 9: ECCP Coverage Probability

Network coverage probability:

```
P_coverage = 1 - exp(-λ · π · r²)
```

### Theorem 10: ECCP Throughput Factor

Throughput optimality factor:

```
η = 1 / (1 + CV²)
```

where CV is coefficient of variation of energy.

---

## TEFL Theorems (11-17)

### Theorem 11: TEFL Convergence Bound

Federated learning convergence rate:

```
||w_T - w*||² ≤ O(1/T)
```

**Implementation:**
```python
from symbiont.core.theorems import tefl_convergence_bound

bound = tefl_convergence_bound(
    num_rounds=100,
    learning_rate=0.01,
    num_nodes=10
)
```

### Theorem 12: Differential Privacy Noise Scale

Required noise for (ε, δ)-DP:

```
σ = Δf · √(2 ln(1.25/δ)) / ε
```

**Implementation:**
```python
from symbiont.core.theorems import dp_noise_scale

sigma = dp_noise_scale(
    epsilon=1.0,
    delta=1e-5,
    sensitivity=1.0
)
```

### Theorem 13: Personalized Convergence

Non-IID convergence with local adaptation:

```
||f_i - f_i*||² ≤ (1-μη)^T · ||f_i⁰ - f_i*||²
```

### Theorem 14: Digital Twin Convergence

Twin synchronization error bound:

```
||twin - physical||² ≤ O(exp(-λt))
```

### Theorem 15: Hierarchical Communication Cost

Communication savings with hierarchical aggregation:

```
Cost_hierarchical = O(n · log(n)) vs O(n²) flat
```

**Implementation:**
```python
from symbiont.core.theorems import hierarchical_communication_cost

cost = hierarchical_communication_cost(num_nodes=100)
# Returns log-scale cost vs linear
```

### Theorem 16: Asynchronous Convergence

Convergence with stale gradients:

```
||w_T - w*||² ≤ O((1 + τ_max)/T)
```

where τ_max is maximum staleness.

### Theorem 17: Optimal Training Time

Optimal number of FL rounds:

```
T* = √(n · B / (η · ε))
```

**Implementation:**
```python
from symbiont.core.theorems import tefl_optimal_time

optimal_rounds = tefl_optimal_time(
    num_edges=50,
    bytes_per_round=1024,
    target_accuracy=0.01
)
```

---

## Using Theorems in Practice

All theorems are implemented with input validation:

```python
from symbiont.core import theorems

# Theorems raise ValueError for invalid inputs
try:
    theorems.smis_synchronization_bound(
        noise_std=-1.0,  # Invalid: negative noise
        effective_velocity=5960.0,
        n_samples=100
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

Theorems can be chained for end-to-end analysis:

```python
# SMIS → ECTI → ECCP chain
sync_bound = theorems.smis_synchronization_bound(1e-6, 5960, 100)
fa_bound = theorems.false_alarm_probability_bound(0.5, 0.1, 0.05)
throughput = theorems.eccp_throughput_factor(0.1, 1.0)

print(f"System guarantees: sync={sync_bound:.2e}s, FA={fa_bound:.2%}, η={throughput:.2%}")
```
