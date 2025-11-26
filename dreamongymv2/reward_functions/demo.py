#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
DREAM-ON-GYM-V2: Demo de Funciones de Recompensa
=============================================================================

Script de demostración que muestra el comportamiento de las 5 funciones
de recompensa implementadas con visualización en tiempo real.

Autor: DREAM-ON-GYM-V2 Team
Fecha: 2024
=============================================================================
"""

import os
import sys
import numpy as np

# Configurar paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Imports de funciones de recompensa
from dreamongymv2.reward_functions import (
    BaselineReward,
    QoTAwareReward,
    MultiObjectiveReward,
    FragmentationAwareReward,
    SpectralEntropyAdaptiveReward,
)


def demo_reward_functions():
    """
    Demostración de las funciones de recompensa con escenarios simulados.
    """
    
    print("\n" + "="*75)
    print("  🎮 DEMO: Funciones de Recompensa para Redes Ópticas Elásticas")
    print("="*75)
    
    # Crear instancias
    reward_functions = {
        '1. Baseline': BaselineReward(),
        '2. QoT-Aware': QoTAwareReward(),
        '3. Multi-Objective': MultiObjectiveReward(),
        '4. Fragmentation-Aware': FragmentationAwareReward(),
        '5. Spectral-Entropy (NOVEL)': SpectralEntropyAdaptiveReward(),
    }
    
    # Escenarios de prueba
    scenarios = [
        {
            'name': '✅ Conexión EXITOSA, Red VACÍA',
            'allocated': True,
            'utilization': 0.1,
            'fragmentation': 0.1,
            'description': 'Caso ideal: asignación exitosa con red casi vacía'
        },
        {
            'name': '✅ Conexión EXITOSA, Red MODERADA',
            'allocated': True,
            'utilization': 0.5,
            'fragmentation': 0.3,
            'description': 'Asignación exitosa con carga media'
        },
        {
            'name': '✅ Conexión EXITOSA, Red CONGESTIONADA',
            'allocated': True,
            'utilization': 0.85,
            'fragmentation': 0.6,
            'description': 'Asignación exitosa bajo alta carga (valioso)'
        },
        {
            'name': '❌ Conexión BLOQUEADA, Red MODERADA',
            'allocated': False,
            'utilization': 0.5,
            'fragmentation': 0.4,
            'description': 'Bloqueo con carga media (penalización moderada)'
        },
        {
            'name': '❌ Conexión BLOQUEADA, Alta FRAGMENTACIÓN',
            'allocated': False,
            'utilization': 0.4,
            'fragmentation': 0.8,
            'description': 'Bloqueo por fragmentación (penalización adicional)'
        },
    ]
    
    # Evaluar cada escenario
    for scenario in scenarios:
        print(f"\n{'─'*75}")
        print(f"  📋 Escenario: {scenario['name']}")
        print(f"     {scenario['description']}")
        print(f"     Utilización: {scenario['utilization']:.0%} | "
              f"Fragmentación: {scenario['fragmentation']:.0%}")
        print(f"{'─'*75}")
        
        results = []
        
        for name, reward_fn in reward_functions.items():
            if hasattr(reward_fn, 'reset_episode'):
                reward_fn.reset_episode()
            
            r = reward_fn.calculate(
                allocated=scenario['allocated'],
                utilization=scenario['utilization'],
                fragmentation=scenario['fragmentation'],
                network=None
            )
            results.append((name, r))
            
            # Barra visual
            bar_len = int(abs(r) * 20)
            if r >= 0:
                bar = '█' * bar_len + '░' * (20 - bar_len)
                color_indicator = '🟢'
            else:
                bar = '░' * (20 - bar_len) + '█' * bar_len
                color_indicator = '🔴'
            
            print(f"     {name:<30} {color_indicator} R = {r:+.4f} [{bar}]")
        
        # Ranking
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        print(f"\n     🏆 Ranking: ", end="")
        for i, (name, r) in enumerate(results_sorted[:3]):
            medals = ['🥇', '🥈', '🥉']
            print(f"{medals[i]} {name.split('.')[1].strip()[:15]} ({r:+.2f}) ", end="")
        print()
    
    # Tabla comparativa final
    print("\n" + "="*75)
    print("  📊 TABLA COMPARATIVA FINAL")
    print("="*75)
    
    print(f"\n{'Función':<25} │ {'Exitosa (baja carga)':^18} │ {'Exitosa (alta carga)':^18} │ {'Bloqueada':^18}")
    print("─"*85)
    
    for name, reward_fn in reward_functions.items():
        # Caso exitoso, baja carga
        reward_fn.reset_episode() if hasattr(reward_fn, 'reset_episode') else None
        r1 = reward_fn.calculate(allocated=True, utilization=0.1, fragmentation=0.1, network=None)
        
        # Caso exitoso, alta carga
        reward_fn.reset_episode() if hasattr(reward_fn, 'reset_episode') else None
        r2 = reward_fn.calculate(allocated=True, utilization=0.8, fragmentation=0.5, network=None)
        
        # Caso bloqueado
        reward_fn.reset_episode() if hasattr(reward_fn, 'reset_episode') else None
        r3 = reward_fn.calculate(allocated=False, utilization=0.5, fragmentation=0.5, network=None)
        
        short_name = name.split('.')[1].strip()
        print(f"{short_name:<25} │ {r1:^+18.4f} │ {r2:^+18.4f} │ {r3:^+18.4f}")
    
    # Características de cada función
    print("\n" + "="*75)
    print("  📝 CARACTERÍSTICAS DE CADA FUNCIÓN")
    print("="*75)
    
    characteristics = {
        'Baseline': [
            '• Recompensa binaria simple: +1 (éxito) / -1 (bloqueo)',
            '• Fácil de interpretar y depurar',
            '• No considera el estado de la red',
            '• Ideal como punto de referencia (baseline)',
        ],
        'QoT-Aware': [
            '• Considera calidad de transmisión (OSNR)',
            '• Penaliza distancias largas',
            '• Bonifica modulaciones eficientes',
            '• Ideal para redes sensibles a calidad de señal',
        ],
        'Multi-Objective': [
            '• Combina múltiples objetivos con pesos',
            '• Balance: Blocking (0.5) + Fragmentación (0.2) + Throughput (0.3)',
            '• Personalizable según prioridades',
            '• Recomendado para entrenamiento estable',
        ],
        'Fragmentation-Aware': [
            '• Enfocado en minimizar fragmentación espectral',
            '• Usa tanto fragmentación externa como interna',
            '• Incentiva uso contiguo del espectro',
            '• Mejora eficiencia a largo plazo',
        ],
        'Spectral-Entropy (NOVEL)': [
            '• 🆕 INNOVACIÓN: Basada en entropía de Shannon',
            '• Sistema de zonas adaptativas (baja/media/alta/crítica)',
            '• Bonificaciones dinámicas según estado de red',
            '• Comportamiento emergente sofisticado',
        ],
    }
    
    for name, chars in characteristics.items():
        print(f"\n  📌 {name}:")
        for char in chars:
            print(f"     {char}")
    
    # Fórmulas matemáticas
    print("\n" + "="*75)
    print("  📐 FORMULACIONES MATEMÁTICAS")
    print("="*75)
    
    print("""
  1️⃣ Baseline:
     R = +1  si asignada
     R = -1  si bloqueada

  2️⃣ QoT-Aware:
     R = w_base × R_base + w_qot × R_qot + w_dist × penalty_dist

  3️⃣ Multi-Objective:
     R = Σ(w_i × R_i)  donde i ∈ {blocking, fragmentation, throughput}

  4️⃣ Fragmentation-Aware:
     R = R_base - α × F_ext - β × F_int + γ × (1 - F_total)

  5️⃣ Spectral-Entropy (NOVEL):
     H = -Σ p_i × log₂(p_i)          (Entropía de Shannon)
     zona = clasificar(H)             (baja/media/alta/crítica)
     R = R_base + bonus(zona) - penalty(zona)
    """)
    
    print("="*75)
    print("  ✅ DEMO COMPLETADA")
    print("="*75)
    print(f"\n  📁 Gráficos disponibles en: {os.path.join(script_dir, 'plots')}")
    print(f"  📄 Documentación completa en: {os.path.join(script_dir, 'DOCUMENTATION.md')}")
    print()


if __name__ == '__main__':
    demo_reward_functions()
