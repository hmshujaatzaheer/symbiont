# Examples

## Basic Usage

### Clock Synchronization with SMIS

```python
import numpy as np
from symbiont import SMIS

# Define sensor positions (meters)
positions = np.array([
    [0, 0],    # Sensor 0 (reference)
    [5, 0],    # Sensor 1
    [10, 0],   # Sensor 2
    [5, 5]     # Sensor 3
])

# Initialize SMIS
smis = SMIS(
    sensor_positions=positions,
    c_L=5960.0,  # Steel longitudinal velocity
    c_T=3240.0,  # Steel transverse velocity
    sample_rate=100000
)

# Raw timestamps from sensors (with clock drift)
raw_timestamps = {
    0: 0.0,
    1: 0.000839 + 0.0001,  # True arrival + 100µs drift
    2: 0.001678 + 0.0002,  # True arrival + 200µs drift
    3: 0.001186 + 0.00015  # True arrival + 150µs drift
}

# Synchronize
result = smis.synchronize_timestamps(raw_timestamps, reference_sensor=0)

print(f"Corrected timestamps: {result.corrected_timestamps}")
print(f"Clock offsets: {result.clock_offsets}")
print(f"Sync error bound: {result.sync_error_bound * 1e6:.2f} µs")
```

### Damage Detection with ECTI

```python
import numpy as np
from symbiont import ECTI
from symbiont.ecti import ECTIFlag

# Initialize ECTI for 4-sensor network
ecti = ECTI(n_sensors=4, frequency_bins=64)

# Simulate sensor signals (1024 samples at 100kHz)
signals = {
    i: np.random.randn(1024) * 0.01  # Baseline noise
    for i in range(4)
}

# Run inference between sensor pair
result = ecti.infer(signals, sensor_i=0, sensor_j=1)

print(f"Anomaly score: {result.anomaly_score:.3f}")
print(f"Uncertainty: {result.uncertainty:.3f}")
print(f"Detection flag: {result.flag.value}")

# Check against thresholds
if result.flag == ECTIFlag.STRUCTURAL_ANOMALY:
    print("⚠️  Structural anomaly detected!")
elif result.flag == ECTIFlag.SENSOR_FAULT:
    print("🔧 Sensor fault detected")
elif result.flag == ECTIFlag.NORMAL:
    print("✅ Structure healthy")
```

### Energy-Aware Protocol with ECCP

```python
import numpy as np
from symbiont import ECCP

# Initialize ECCP
eccp = ECCP(
    n_sensors=10,
    transmission_energy=1e-6,
    sense_energy=1e-7
)

# Simulate wave event
signal = np.sin(2 * np.pi * 50 * np.linspace(0, 0.1, 1000))  # 50Hz wave

# Process event at node 0
result = eccp.on_wave_event(
    node_id=0,
    timestamp=0.0,
    amplitude=0.001,
    signal=signal
)

print(f"Energy harvested: {result.energy_harvested * 1e6:.2f} µJ")
print(f"Should transmit: {result.transmission_decision.should_transmit}")
print(f"Transmission probability: {result.transmission_decision.probability:.2%}")
print(f"Scheduled time: {result.transmission_decision.scheduled_time:.3f}s")
```

### Federated Learning with TEFL

```python
import numpy as np
from symbiont import TEFL

# Network topology (adjacency matrix)
adjacency = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

# Initialize TEFL with privacy guarantees
tefl = TEFL(
    n_sensors=4,
    adjacency=adjacency,
    embedding_dim=16,
    epsilon=1.0,  # Privacy budget
    delta=1e-5
)

# Local training at each node
embeddings = []
for node_id in range(4):
    signals = {i: np.random.randn(1024) for i in range(4)}
    labels = np.array([0])  # Healthy label
    
    embedding = tefl.local_update(node_id, signals, labels)
    embeddings.append(embedding)

# Aggregate with differential privacy
global_embedding = tefl.aggregate(embeddings)

# Check privacy guarantee
eps, delta = tefl.get_privacy_guarantee()
print(f"Privacy guarantee: ({eps}, {delta})-DP")
```

---

## Advanced Examples

### End-to-End Pipeline

```python
import numpy as np
from symbiont import SMIS, ECTI, ECCP, TEFL
from symbiont.core import theorems

# 1. Setup network
positions = np.array([[0,0], [5,0], [10,0], [5,5]])
adjacency = np.ones((4, 4)) - np.eye(4)

smis = SMIS(sensor_positions=positions)
ecti = ECTI(n_sensors=4)
eccp = ECCP(n_sensors=4)
tefl = TEFL(n_sensors=4, adjacency=adjacency)

# 2. Simulate impact event
def simulate_event(positions, impact_location, c_L=5960):
    """Simulate wave arrival times from impact."""
    distances = np.linalg.norm(positions - impact_location, axis=1)
    return distances / c_L

impact = np.array([7, 2])
true_arrivals = simulate_event(positions, impact)
print(f"True arrivals: {true_arrivals * 1e6} µs")

# 3. Add clock drift and synchronize
drifts = np.random.randn(4) * 1e-4  # 100µs std drift
raw_timestamps = dict(enumerate(true_arrivals + drifts))

sync_result = smis.synchronize_timestamps(raw_timestamps, reference_sensor=0)
print(f"Sync error: {sync_result.sync_error_bound * 1e6:.2f} µs")

# 4. Run damage detection
signals = {i: np.random.randn(1024) for i in range(4)}
ecti_result = ecti.infer(signals, sensor_i=0, sensor_j=1)
print(f"Anomaly score: {ecti_result.anomaly_score:.3f}")

# 5. Energy-aware transmission decision
eccp_result = eccp.on_wave_event(
    node_id=0,
    timestamp=sync_result.corrected_timestamps[0],
    amplitude=0.001,
    signal=signals[0]
)
print(f"Transmit: {eccp_result.transmission_decision.should_transmit}")

# 6. Federated learning round
embeddings = [tefl.local_update(i, signals, np.array([0])) for i in range(4)]
global_model = tefl.aggregate(embeddings)
print(f"Global embedding norm: {np.linalg.norm(global_model):.3f}")

# 7. Verify theoretical guarantees
sync_bound = theorems.smis_synchronization_bound(1e-6, 5960, 4)
fa_bound = theorems.false_alarm_probability_bound(0.5, 0.1, 0.05)
throughput = theorems.eccp_throughput_factor(0.1, 1.0)

print(f"\nTheoretical guarantees:")
print(f"  Sync bound: {sync_bound * 1e6:.2f} µs")
print(f"  FA rate: {fa_bound:.2%}")
print(f"  Throughput factor: {throughput:.2%}")
```

### Monitoring Dashboard Simulation

```python
import numpy as np
from symbiont import SMIS, ECTI
from symbiont.ecti import ECTIFlag

class SHMDashboard:
    """Simple SHM monitoring dashboard."""
    
    def __init__(self, positions):
        self.smis = SMIS(sensor_positions=positions)
        self.ecti = ECTI(n_sensors=len(positions))
        self.history = []
    
    def process_event(self, signals, timestamps):
        """Process a monitoring event."""
        # Synchronize
        sync = self.smis.synchronize_timestamps(
            dict(enumerate(timestamps)), 
            reference_sensor=0
        )
        
        # Detect anomalies for all pairs
        anomalies = []
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                result = self.ecti.infer(signals, sensor_i=i, sensor_j=j)
                anomalies.append({
                    'pair': (i, j),
                    'score': result.anomaly_score,
                    'flag': result.flag
                })
        
        # Record history
        event = {
            'sync_error': sync.sync_error_bound,
            'anomalies': anomalies,
            'max_score': max(a['score'] for a in anomalies)
        }
        self.history.append(event)
        
        return event
    
    def get_status(self):
        """Get current system status."""
        if not self.history:
            return "NO DATA"
        
        last = self.history[-1]
        if last['max_score'] > 0.8:
            return "🔴 CRITICAL"
        elif last['max_score'] > 0.5:
            return "🟡 WARNING"
        else:
            return "🟢 HEALTHY"

# Usage
positions = np.array([[0,0], [10,0], [10,10], [0,10]])
dashboard = SHMDashboard(positions)

# Simulate 10 monitoring events
for t in range(10):
    signals = {i: np.random.randn(1024) * (1 + 0.1*t) for i in range(4)}
    timestamps = np.random.rand(4) * 1e-3
    
    event = dashboard.process_event(signals, timestamps)
    print(f"Event {t}: {dashboard.get_status()} (max_score={event['max_score']:.3f})")
```

### Benchmarking Theoretical Bounds

```python
import numpy as np
from symbiont.core import theorems
import matplotlib.pyplot as plt

# Benchmark sync bound vs number of sensors
n_values = np.arange(4, 101)
bounds = [theorems.smis_synchronization_bound(1e-6, 5960, n) for n in n_values]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(n_values, np.array(bounds) * 1e6)
plt.xlabel('Number of sensors')
plt.ylabel('Sync bound (µs)')
plt.title('Theorem 2: √n scaling')

# Benchmark privacy-utility tradeoff
epsilon_values = np.logspace(-1, 1, 50)
noise_scales = [theorems.dp_noise_scale(eps, 1e-5, 1.0) for eps in epsilon_values]

plt.subplot(1, 2, 2)
plt.loglog(epsilon_values, noise_scales)
plt.xlabel('Privacy budget ε')
plt.ylabel('Noise scale σ')
plt.title('Theorem 12: Privacy-utility tradeoff')

plt.tight_layout()
plt.savefig('theoretical_bounds.png', dpi=150)
print("Saved theoretical_bounds.png")
```
