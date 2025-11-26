# -*- coding: utf-8 -*-
"""
DREAM-ON-GYM-V2: Advanced Reward Functions Module
=================================================

Este modulo proporciona funciones de recompensa avanzadas para el entrenamiento
de agentes de Deep Reinforcement Learning en redes opticas elasticas (EON).

Funciones de Recompensa Implementadas:
-------------------------------------
1. BaselineReward: Recompensa binaria simple (+1/-1)
2. QoTAwareReward: Considera la calidad de transmision (OSNR, distancia)
3. MultiObjectiveReward: Combinacion ponderada de multiples objetivos
4. FragmentationAwareReward: Minimiza la fragmentacion espectral
5. SpectralEntropyAdaptiveReward: NOVEL - Entropia espectral adaptativa

@author: Generado con AI para DREAM-ON-GYM-V2
@version: 2.0.0
@date: 2024

Uso:
----
>>> from dreamongymv2.reward_functions import RewardFactory
>>> reward_fn = RewardFactory.create('spectral_entropy', entropy_weight=0.4)
>>> reward = reward_fn.calculate(allocated=True, network=network)
"""

# Importaciones lazy para evitar errores de dependencia circular
def _import_reward_functions():
    from .reward_functions import (
        BaselineReward,
        QoTAwareReward,
        MultiObjectiveReward,
        FragmentationAwareReward,
        SpectralEntropyAdaptiveReward,
        RewardFactory,
        RewardMetrics,
        BaseRewardFunction,
        create_reward_wrapper
    )
    return {
        'BaselineReward': BaselineReward,
        'QoTAwareReward': QoTAwareReward,
        'MultiObjectiveReward': MultiObjectiveReward,
        'FragmentationAwareReward': FragmentationAwareReward,
        'SpectralEntropyAdaptiveReward': SpectralEntropyAdaptiveReward,
        'RewardFactory': RewardFactory,
        'RewardMetrics': RewardMetrics,
        'BaseRewardFunction': BaseRewardFunction,
        'create_reward_wrapper': create_reward_wrapper
    }

def _import_metrics():
    from .metrics import (
        calculate_fragmentation_ratio,
        calculate_entropy,
        calculate_spectral_compactness,
        calculate_link_utilization,
        calculate_network_utilization,
        calculate_path_length_penalty,
        calculate_osnr_quality,
        calculate_load_balance_factor,
        get_network_spectrum_state
    )
    return {
        'calculate_fragmentation_ratio': calculate_fragmentation_ratio,
        'calculate_entropy': calculate_entropy,
        'calculate_spectral_compactness': calculate_spectral_compactness,
        'calculate_link_utilization': calculate_link_utilization,
        'calculate_network_utilization': calculate_network_utilization,
        'calculate_path_length_penalty': calculate_path_length_penalty,
        'calculate_osnr_quality': calculate_osnr_quality,
        'calculate_load_balance_factor': calculate_load_balance_factor,
        'get_network_spectrum_state': get_network_spectrum_state
    }

# Cargar al importar el modulo
try:
    _rewards = _import_reward_functions()
    _metrics = _import_metrics()
    
    # Exportar clases de recompensa
    BaselineReward = _rewards['BaselineReward']
    QoTAwareReward = _rewards['QoTAwareReward']
    MultiObjectiveReward = _rewards['MultiObjectiveReward']
    FragmentationAwareReward = _rewards['FragmentationAwareReward']
    SpectralEntropyAdaptiveReward = _rewards['SpectralEntropyAdaptiveReward']
    RewardFactory = _rewards['RewardFactory']
    RewardMetrics = _rewards['RewardMetrics']
    BaseRewardFunction = _rewards['BaseRewardFunction']
    create_reward_wrapper = _rewards['create_reward_wrapper']
    
    # Exportar funciones de metricas
    calculate_fragmentation_ratio = _metrics['calculate_fragmentation_ratio']
    calculate_entropy = _metrics['calculate_entropy']
    calculate_spectral_compactness = _metrics['calculate_spectral_compactness']
    calculate_link_utilization = _metrics['calculate_link_utilization']
    calculate_network_utilization = _metrics['calculate_network_utilization']
    calculate_path_length_penalty = _metrics['calculate_path_length_penalty']
    calculate_osnr_quality = _metrics['calculate_osnr_quality']
    calculate_load_balance_factor = _metrics['calculate_load_balance_factor']
    get_network_spectrum_state = _metrics['get_network_spectrum_state']
    
except ImportError as e:
    import warnings
    warnings.warn(f"Error importando modulos de reward_functions: {e}")

__all__ = [
    # Clases de recompensa
    'BaselineReward',
    'QoTAwareReward',
    'MultiObjectiveReward',
    'FragmentationAwareReward',
    'SpectralEntropyAdaptiveReward',
    'RewardFactory',
    'RewardMetrics',
    'BaseRewardFunction',
    'create_reward_wrapper',
    
    # Funciones de metricas
    'calculate_fragmentation_ratio',
    'calculate_entropy',
    'calculate_spectral_compactness',
    'calculate_link_utilization',
    'calculate_network_utilization',
    'calculate_path_length_penalty',
    'calculate_osnr_quality',
    'calculate_load_balance_factor',
    'get_network_spectrum_state'
]

__version__ = '2.0.0'
