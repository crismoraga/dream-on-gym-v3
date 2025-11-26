# DREAM-ON-GYM-V3 ULTRA: Módulo de Funciones de Recompensa

[![Docs](https://img.shields.io/badge/Docs-View-blue)](DOCUMENTATION.md)
[![Examples](https://img.shields.io/badge/Examples-Run-green)](examples.py)
[![Try Quick Eval](https://img.shields.io/badge/Try%20Quick%20Eval-run-orange)](quick_evaluation.py)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../licenses/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/crismoraga/dream-on-gym-v3)

## 📋 Resumen del Proyecto

Este módulo es la evolución de DREAM-ON-GYM-V2 y ahora forma parte de la **versión DREAM-ON-GYM-V3 (Ultra, optimizada y mejorada)**. Está diseñado para facilitar experimentos reproducibles y comparaciones rigurosas de estrategias de reward engineering en redes ópticas elásticas (EON).

En v3 nos centramos en estabilidad, reusabilidad y en un conjunto de utilidades para benchmarking de RL en EON: implementamos 5 reward functions, métricas avanzadas, scripts de evaluación y dashboards interactivos para comparar enfoques.

> Nota: Este README describe la versión del paquete de `reward_functions` dentro del release DREAM-ON-GYM-V3. Si necesitas la integración completa con el paquete raíz, revisa la documentación en `docs/`.

---

## 📚 Índice (Table of Contents)

- Resumen del Proyecto
- Qué hay de nuevo (v3)
- Funciones Implementadas
- Estructura del Módulo
- Instalación & Quickstart
- Ejemplos & Demostraciones
- Evaluaciones y Scripts
- Visualizaciones y Reportes
- Arquitectura (Mermaid)
- Métricas disponibles
- Migración V2 → V3
- Contribuir
- Licencia & Contacto

---

## 🎯 Funciones Implementadas

| # | Función | Tipo | Descripción |
|---|---------|------|-------------|
| 1 | `BaselineReward` | Binaria | +1 éxito, -1 bloqueo (referencia) |
| 2 | `QoTAwareReward` | QoT | Considera OSNR y distancia de transmisión |
| 3 | `MultiObjectiveReward` | Multi-objetivo | Combinación ponderada de métricas |
| 4 | `FragmentationAwareReward` | Fragmentación | Penaliza fragmentación espectral |
| 5 | `SpectralEntropyAdaptiveReward` | **NOVEL** | Basada en entropía de Shannon; adaptive reward que ajusta bonificaciones y penalizaciones según la entropía espectral de la red |

---


🎉 Pro-tip: Si vienes de v2, en la sección "Qué hay de nuevo (v3)" más abajo verás las ventajas principales y la guía de migración.

---

<a name="que-hay-de-nuevo-v3"></a>
## 🔄 Qué hay de nuevo (v3)

DREAM-ON-GYM-V3 es una versión de reingeniería: no es solo más funciones, es una **evolución de arquitectura** con soporte mejorado para benchmarking y reproducibilidad.

| Categoría | v2 | v3 (ULTRA) |
|-----------|----|------------|
| Reward Engineering | 1-2 reward functions | 5 funciones (QoT, Fragmentation, Multi-objective, SpectralEntropy NOVEL, Baseline) |
| Evaluaciones | manual / ejemplos | `quick_evaluation`, `full_evaluation`, `run_experiments` pipelines |
| Visualizaciones | estáticas | Radar, Heatmaps, Boxplots, Plots automáticos y dashboard interactivo |
| Integraciones | Gym/basic | Gymnasium, Stable-Baselines3 + sb3-contrib compatibles |
| Reproductibilidad | limitada | JSON export, reproducible reports, deterministic sim settings |

### Migración rápida: puntos clave

- Cambios de API: `calculate()` ahora acepta `allocated` y `network`. Revisa cómo tu wrapper inyecta el `reward_fn` al crear el env.
- Revisa `DOCUMENTATION.md` para los parámetros y la configuración por defecto de cada reward.

---

<a name="arquitectura-mermaid"></a>
## 🏗️ Arquitectura (Visión general)

```mermaid
flowchart LR
        subgraph Simulation
            Sim[Simulator (simNetPy)] -->|Events & State| Network[Network (links, slots)]
            Network -->|spectrum state| Metrics[metrics.py]
        end
        subgraph RL
            Env[Gym Env (RlOnEnv)] -->|obs| Agent[RL Agent]
            Agent -->|actions| Env
            Env -->|invokes| RewardFns[Reward Functions]
        end
        Metrics -->|features/entropy| RewardFns
        RewardFns -->|reward| Env
        Agent -->|training| Logger[Training & Logger]
        Logger -->|plots/stats| Dashboard[Plotly Dashboard]
```


<a name="estructura-del-modulo"></a>
## 📁 Estructura del Módulo

```text
dreamongymv2/reward_functions/         # Paquete principal con lógica de reward
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

<a name="uso-rapido"></a>
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

```text
R = +1  si conexión asignada
R = -1  si conexión bloqueada
```

### 2. QoT-Aware Reward

```text
R = w_base × R_base + w_qot × (OSNR_est / OSNR_thresh) + w_dist × (1 - d/d_max)
```

### 3. Multi-Objective Reward

```text
R = Σ(w_i × R_i)  donde i ∈ {blocking, fragmentation, throughput}
Pesos default: w_block=0.5, w_frag=0.2, w_tput=0.3
```

### 4. Fragmentation-Aware Reward

```text
R = R_base - α × F_external - β × F_internal + γ × (1 - F_total)
Donde F_external = bloques_libres/total_slots
```

### 5. Spectral-Entropy Adaptive Reward (NOVEL)

```text
H(S) = -Σ p_i × log₂(p_i)  (Entropía de Shannon)

Zonas adaptativas:
- Baja (H < 0.3):    R = R_base + 0.15  (red casi vacía)
- Media (0.3-0.6):   R = R_base + 0.05  (operación normal)
- Alta (0.6-0.8):    R = R_base - 0.10  (precaución)
- Crítica (H > 0.8): R = R_base - 0.25  (saturación)
```

<a name="ejecucion-de-evaluaciones"></a>
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

<a name="visualizaciones-y-reportes"></a>
## 📈 Visualizaciones Generadas

El módulo genera automáticamente:

- Gráficos de Blocking Probability vs Carga
- Comparativas de recompensa promedio
- Distribución de recompensas (boxplots)
- Heatmaps de rendimiento
- Radar charts multidimensionales
- Curvas de evolución temporal

### 🎨 Galería (previews)

| Blocking Probability | Rewards | Radar |
|----------------------|---------|-------|
| ![BP](/dreamongymv2/reward_functions/plots/blocking_probability.png) | ![Rewards](/dreamongymv2/reward_functions/plots/rewards.png) | ![Radar](/dreamongymv2/reward_functions/plots/radar.png) |

---

<a name="reproducibilidad"></a>
## 🔒 Reproducibilidad y Seeds

Para ejecutar experimentos reproducibles, define seeds en el simulador **antes** de llamar `init()`:

```python
from dreamongymv2.simNetPy.simulator_finite import Simulator

sim = Simulator(network_file, routes_file, "")
sim.setSeedArrive(42)
sim.setSeedDeparture(43)
sim.setSeedSrc(44)
sim.setSeedDst(45)
sim.init()
```

Adicionalmente, fija la semilla para Python y NumPy para reproducibilidad de experimentos RL:

```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
```

---

<a name="metricas-disponibles"></a>
## 🧮 Métricas disponibles

El módulo `metrics.py` ofrece las principales métricas ya implementadas, entre otras:

- `calculate_fragmentation_ratio(link_slots, method='external|internal|average')`
- `get_network_spectrum_state(network)` → retorna dict con `avg_fragmentation`, `avg_utilization`, `entropy`, etc.
- QoT estimators (OSNR-based)

Usa estas funciones para instrumentar recompensas personalizadas y dashboards.

---

<a name="optimización-produccion"></a>
## ⚡ Optimización y producción (enterprise)

- Usa `run_experiments.py` con `--parallel` (si lo habilitas) para ejecutar múltiples configuraciones en paralelo.
- Configura entornos con GPU para entrenamiento (PyTorch/TF) si usas enfoques basados en NN intensivos.
- Para pipelines de CI/CD: añade tests con `pytest` y crea artefactos (JSON + PNG) por cada release.

### 🎞️ Crear GIFs/Animaciones a partir de gráficos

Si deseas convertir una secuencia de PNG en un GIF animado (para presentaciones o dashboard), usa ImageMagick o `convert`:

```bash
# Ejemplo con ImageMagick (Windows o macOS con brew)
magick convert -delay 20 -loop 0 plots/radar_*.png plots/radar.gif
```

O usa `ffmpeg`:

```bash
ffmpeg -framerate 10 -pattern_type glob -i 'plots/radar_*.png' -vf "scale=800:-1" plots/radar.gif
```


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

---

## 🤝 Cómo contribuir

Nos encantan las contribuciones. Para colaborar con DREAM-ON-GYM-V3:

1. Fork del repositorio
2. Crear una rama con tu feature: `git checkout -b feature/my-feature`
3. Añadir pruebas o un ejemplo reproducible
4. Abrir PR con descripción, benchmarks y gráficos (si aplican)

Si trabajas en algoritmos RL, por favor incluye seed/fixed-config para reproducibilidad.

---

## 📜 Licencia & Contacto

El proyecto contiene múltiples licencias, revisa `licenses/`. Las implementaciones nuevas están bajo MIT salvo que se indique lo contrario.

Si necesitas soporte o quieres colaborar en integraciones enterprise, abre un `issue` o contacta al equipo en `support@dreamongym.org`.

---

## 📦 Changelog (Resumen rápido)

- **v3.0.0** — (Hoy): Reorganización del paquete, nuevo conjunto de 5 reward functions (incl. SpectralEntropyAdaptiveReward), pipelines reproducibles, dashboards y documentación extendida.
- **v2.0.0** — Implementación original: ejemplos y la conexión inicial con Flex-Net-Sim.

---

Gracias por usar DREAM-ON-GYM-V3 — si encuentras un bug o limitación, cuenta con nosotros para resolverlo.
