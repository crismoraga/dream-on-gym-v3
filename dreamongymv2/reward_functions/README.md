# DREAM-ON-GYM-V2: Módulo de Funciones de Recompensa

## 📋 Resumen del Proyecto

Este módulo implementa **5 funciones de recompensa** para entrenamiento de agentes de Deep Reinforcement Learning (DRL) en redes ópticas elásticas (EON), incluyendo una **propuesta novedosa** basada en entropía espectral.

## 🎯 Funciones Implementadas

| # | Función | Tipo | Descripción |
|---|---------|------|-------------|
| 1 | `BaselineReward` | Binaria | +1 éxito, -1 bloqueo (referencia) |
| 2 | `QoTAwareReward` | QoT | Considera OSNR y distancia de transmisión |
| 3 | `MultiObjectiveReward` | Multi-objetivo | Combinación ponderada de métricas |
| 4 | `FragmentationAwareReward` | Fragmentación | Penaliza fragmentación espectral |
| 5 | `SpectralEntropyAdaptiveReward` | **NOVEL** | Basada en entropía de Shannon |

## 📁 Estructura del Módulo

```
dreamongymv2/reward_functions/
├── __init__.py              # Exports del módulo
├── reward_functions.py      # Clases de recompensa (5 implementaciones)
├── metrics.py               # Cálculos de fragmentación, utilización, QoT
├── examples.py              # Ejemplos de uso (4 ejemplos)
├── demo.py                  # Demo interactiva
├── quick_evaluation.py      # Evaluación rápida
├── full_evaluation.py       # Evaluación con simulador completo
├── run_experiments.py       # Script de experimentos
├── evaluate_rewards.py      # Evaluación con estadísticas
├── dashboard.py             # Dashboard interactivo (Plotly)
├── DOCUMENTATION.md         # Documentación matemática completa
└── plots/                   # Visualizaciones generadas
    ├── blocking_probability.png
    ├── rewards.png
    ├── fragmentation.png
    ├── evolution.png
    ├── radar.png
    ├── heatmap.png
    ├── GermanNet_*.png
    └── ItalianNet_*.png
```

## 🚀 Uso Rápido

```python
from dreamongymv2.reward_functions import (
    BaselineReward,
    SpectralEntropyAdaptiveReward,
    RewardFactory
)

# Crear función de recompensa
reward_fn = SpectralEntropyAdaptiveReward()

# Calcular recompensa
reward = reward_fn.calculate(
    allocated=True,
    network=network,  # Objeto Network del simulador
)

# O usar factory
reward_fn = RewardFactory.create('spectral_entropy')
```

## 📐 Formulaciones Matemáticas

### 1. Baseline Reward
```
R = +1  si conexión asignada
R = -1  si conexión bloqueada
```

### 2. QoT-Aware Reward
```
R = w_base × R_base + w_qot × (OSNR_est / OSNR_thresh) + w_dist × (1 - d/d_max)
```

### 3. Multi-Objective Reward
```
R = Σ(w_i × R_i)  donde i ∈ {blocking, fragmentation, throughput}
Pesos default: w_block=0.5, w_frag=0.2, w_tput=0.3
```

### 4. Fragmentation-Aware Reward
```
R = R_base - α × F_external - β × F_internal + γ × (1 - F_total)
Donde F_external = bloques_libres/total_slots
```

### 5. Spectral-Entropy Adaptive Reward (NOVEL)
```
H(S) = -Σ p_i × log₂(p_i)  (Entropía de Shannon)

Zonas adaptativas:
- Baja (H < 0.3):    R = R_base + 0.15  (red casi vacía)
- Media (0.3-0.6):   R = R_base + 0.05  (operación normal)
- Alta (0.6-0.8):    R = R_base - 0.10  (precaución)
- Crítica (H > 0.8): R = R_base - 0.25  (saturación)
```

## 📊 Ejecución de Evaluaciones

```bash
# Demo interactiva
python -m dreamongymv2.reward_functions.demo

# Evaluación rápida con gráficos
python -m dreamongymv2.reward_functions.quick_evaluation

# Evaluación completa con simulador
python -m dreamongymv2.reward_functions.full_evaluation

# Ejemplos de uso
python -m dreamongymv2.reward_functions.examples
```

## 📈 Visualizaciones Generadas

El módulo genera automáticamente:
- Gráficos de Blocking Probability vs Carga
- Comparativas de recompensa promedio
- Distribución de recompensas (boxplots)
- Heatmaps de rendimiento
- Radar charts multidimensionales
- Curvas de evolución temporal

## 🔬 Referencias Bibliográficas

1. **DeepRMSA**: Chen et al., "DeepRMSA: A Deep RL Framework for EON", JLT 2019
2. **QoT-Aware DRL**: Salami et al., "QoT-Aware Resource Allocation", JOCN 2020
3. **Multi-Band DRL**: Etezadi et al., "Multi-Band EON with DRL", ECOC 2021
4. **Fragmentación**: Wright et al., "Fragmentation-Aware RMSA", OFC 2015

## ✅ Estado del Proyecto

- [x] Implementación de 5 funciones de recompensa
- [x] Módulo de métricas (fragmentación, utilización, QoT)
- [x] Ejemplos de uso documentados
- [x] Integración con simulador Flex-Net-Sim
- [x] Generación de visualizaciones
- [x] Documentación matemática completa
- [x] Demo interactiva
- [ ] Entrenamiento comparativo con PPO/DQN
- [ ] Dashboard web interactivo

## 👨‍💻 Autor

DREAM-ON-GYM-V2 Team - 2024
