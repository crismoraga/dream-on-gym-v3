# -*- coding: utf-8 -*-
"""
DREAM-ON-GYM-V2: Ejemplo de Uso de Funciones de Recompensa
==========================================================

Este script demuestra como usar las diferentes funciones de recompensa
con el entorno de Gymnasium para entrenar agentes de RL.

Ejemplos incluidos:
------------------
1. Uso basico de cada funcion de recompensa
2. Integracion con RlOnEnv
3. Entrenamiento con Stable-Baselines3
4. Comparacion de resultados

@author: Generado con AI para DREAM-ON-GYM-V2
@version: 2.0.0
"""

import os
import sys
from pathlib import Path

# Configurar paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importar gymnasium
import gymnasium
sys.modules["gym"] = gymnasium

# Importar componentes del framework
from dreamongymv2.simNetPy.simulator_finite import Simulator
from dreamongymv2.simNetPy.network import Network
from dreamongymv2.simNetPy.controller import Controller
from dreamongymv2.simNetPy.connection import Connection
from dreamongymv2.simNetPy.bitRate import BitRate

# Importar funciones de recompensa
from dreamongymv2.reward_functions.reward_functions import (
    BaselineReward,
    QoTAwareReward,
    MultiObjectiveReward,
    FragmentationAwareReward,
    SpectralEntropyAdaptiveReward,
    RewardFactory,
    create_reward_wrapper
)
from dreamongymv2.reward_functions.metrics import (
    get_network_spectrum_state,
    calculate_fragmentation_ratio
)


# =============================================================================
# ALGORITMO FIRST-FIT
# =============================================================================

def first_fit_algorithm(src, dst, b, c, n, path, action):
    """Algoritmo First-Fit para asignacion de espectro."""
    numberOfSlots = b.getNumberofSlots(0)
    actionSpace = len(path[src][dst])
    
    if action is not None:
        if action >= actionSpace:
            action = actionSpace - 1
        link_ids = path[src][dst][action]
    else:
        link_ids = path[src][dst][0]
    
    general_link = [False] * n.getLink(0).getSlots()
    
    for link in link_ids:
        link = n.getLink(link.id)
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(slot)
    
    currentNumberSlots = 0
    currentSlotIndex = 0
    
    for j in range(len(general_link)):
        if not general_link[j]:
            currentNumberSlots += 1
        else:
            currentNumberSlots = 0
            currentSlotIndex = j + 1
        
        if currentNumberSlots == numberOfSlots:
            for k in link_ids:
                c.addLink(k, fromSlot=currentSlotIndex, 
                         toSlot=currentSlotIndex + currentNumberSlots)
            return Controller.Status.Allocated, c
    
    return Controller.Status.Not_Allocated, c


# =============================================================================
# EJEMPLO 1: USO BASICO DE CADA FUNCION DE RECOMPENSA
# =============================================================================

def example_basic_usage():
    """
    Demuestra el uso basico de cada funcion de recompensa.
    """
    print("\n" + "="*70)
    print("EJEMPLO 1: Uso Basico de Funciones de Recompensa")
    print("="*70)
    
    # 1. Baseline Reward
    print("\n1. BaselineReward:")
    baseline = BaselineReward(success_reward=1.0, failure_penalty=1.0)
    print(f"   Conexion asignada: r = {baseline.calculate(allocated=True)}")
    print(f"   Conexion bloqueada: r = {baseline.calculate(allocated=False)}")
    
    # 2. QoT-Aware Reward
    print("\n2. QoTAwareReward:")
    qot_reward = QoTAwareReward(qot_weight=0.5, base_reward=1.0)
    # Simular una ruta (sin enlaces reales)
    print(f"   Conexion asignada (ruta corta): r = {qot_reward.calculate(allocated=True, path_links=[]):.3f}")
    print(f"   Conexion bloqueada: r = {qot_reward.calculate(allocated=False, path_links=[]):.3f}")
    
    # 3. Multi-Objective Reward
    print("\n3. MultiObjectiveReward:")
    multi = MultiObjectiveReward(weights={
        'blocking': 1.0,
        'fragmentation': 0.3,
        'utilization': 0.2,
        'balance': 0.2,
        'path_length': 0.1
    })
    print(f"   Conexion asignada (sin red): r = {multi.calculate(allocated=True):.3f}")
    print(f"   Conexion bloqueada: r = {multi.calculate(allocated=False):.3f}")
    
    # 4. Fragmentation-Aware Reward
    print("\n4. FragmentationAwareReward:")
    frag = FragmentationAwareReward(local_weight=0.4, global_weight=0.3)
    print(f"   Conexion asignada: r = {frag.calculate(allocated=True):.3f}")
    print(f"   Conexion bloqueada: r = {frag.calculate(allocated=False):.3f}")
    
    # 5. Spectral-Entropy Adaptive Reward (NOVEL)
    print("\n5. SpectralEntropyAdaptiveReward (NOVEL):")
    entropy = SpectralEntropyAdaptiveReward(
        entropy_weight=0.4,
        temporal_discount=0.9,
        memory_window=50
    )
    print(f"   Conexion asignada: r = {entropy.calculate(allocated=True):.3f}")
    print(f"   Conexion bloqueada: r = {entropy.calculate(allocated=False):.3f}")
    
    # Usando RewardFactory
    print("\n\nUsando RewardFactory:")
    for name in RewardFactory.list_available():
        reward_fn = RewardFactory.create(name)
        print(f"   {name}: r_success = {reward_fn.calculate(allocated=True):.3f}")


# =============================================================================
# EJEMPLO 2: INTEGRACION CON EL SIMULADOR
# =============================================================================

def example_simulator_integration():
    """
    Demuestra la integracion con el simulador.
    """
    print("\n" + "="*70)
    print("EJEMPLO 2: Integracion con el Simulador")
    print("="*70)
    
    # Obtener paths de archivos de red
    base_dir = Path(__file__).parent.parent.parent
    network_dir = base_dir / "network"
    
    network_file = str(network_dir / "NSFNet.json")
    routes_file = str(network_dir / "NSFNet_routes.json")
    
    # Verificar que existan los archivos
    if not Path(network_file).exists():
        print(f"\nArchivo no encontrado: {network_file}")
        print("Usando ejemplo simulado...")
        return
    
    # Crear simulador
    print("\nCreando simulador con NSFNet...")
    simulator = Simulator(network_file, routes_file, "")
    simulator.setGoalConnections(1000)  # 1000 conexiones para prueba
    simulator.setRho(0.5)
    simulator.setAllocator(first_fit_algorithm)
    simulator.init()
    
    # Crear funciones de recompensa
    reward_functions = {
        'Baseline': BaselineReward(),
        'QoT-Aware': QoTAwareReward(qot_weight=0.5),
        'Multi-Objective': MultiObjectiveReward(),
        'Fragmentation': FragmentationAwareReward(),
        'Spectral-Entropy': SpectralEntropyAdaptiveReward()
    }
    
    # Evaluar cada funcion en los primeros 100 pasos
    print("\nEvaluando funciones de recompensa (100 pasos):")
    
    for name, reward_fn in reward_functions.items():
        reward_fn.set_simulator(simulator)
    
    results = {name: [] for name in reward_functions}
    
    # Reiniciar simulador
    simulator = Simulator(network_file, routes_file, "")
    simulator.setGoalConnections(100)
    simulator.setRho(0.5)
    simulator.setAllocator(first_fit_algorithm)
    simulator.init()
    
    for step in range(100):
        # Ejecutar paso - guardar conexiones asignadas antes del paso
        prev_allocated = simulator.allocatedConnections
        simulator.step(0)
        simulator.forwardDepartures()
        
        # Obtener estado - determinar si esta conexion fue asignada
        curr_allocated = simulator.allocatedConnections
        allocated = curr_allocated > prev_allocated
        network = simulator._Simulator__controller.network
        state = get_network_spectrum_state(network)
        
        # Calcular recompensa con cada funcion
        for name, reward_fn in reward_functions.items():
            r = reward_fn.calculate(
                allocated=allocated,
                network=network
            )
            results[name].append(r)
        
        # Siguiente evento
        if step < 99:
            simulator.createEventConnection()
    
    # Mostrar resultados
    print("\nResultados (100 pasos):")
    print("-" * 60)
    print(f"{'Funcion':<25} {'R_promedio':>12} {'R_total':>12} {'Std':>10}")
    print("-" * 60)
    
    import numpy as np
    for name, rewards in results.items():
        avg = np.mean(rewards)
        total = np.sum(rewards)
        std = np.std(rewards)
        print(f"{name:<25} {avg:>12.3f} {total:>12.1f} {std:>10.3f}")
    
    print("-" * 60)
    print(f"\nBlocking Probability: {simulator.getBlockingProbability():.4f}")


# =============================================================================
# EJEMPLO 3: ENTRENAMIENTO CON STABLE-BASELINES3
# =============================================================================

def example_sb3_training():
    """
    Demuestra entrenamiento con Stable-Baselines3.
    """
    print("\n" + "="*70)
    print("EJEMPLO 3: Entrenamiento con Stable-Baselines3")
    print("="*70)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("\nstable-baselines3 no disponible.")
        print("Instale con: pip install stable-baselines3")
        return
    
    # Importar entorno
    try:
        from dreamongymv2.gym_basic.envs.rl_on_env import RlOnEnv
    except ImportError:
        print("\nEntorno RlOnEnv no disponible.")
        return
    
    # Obtener paths
    base_dir = Path(__file__).parent.parent.parent
    network_dir = base_dir / "network"
    
    network_file = str(network_dir / "NSFNet.json")
    routes_file = str(network_dir / "NSFNet_routes.json")
    
    if not Path(network_file).exists():
        print(f"\nArchivo no encontrado: {network_file}")
        return
    
    print("\nCreando entorno...")
    
    # Crear entorno con Gymnasium
    env = gymnasium.make(
        'rlonenv-v0',
        action_space=3,
        observation_space=100,
        start_training=1000,
        disable_env_checker=True
    )
    
    # Configurar entorno
    env_unwrapped = env.unwrapped
    env_unwrapped.initEnviroment(network_file, routes_file, "")
    
    # Obtener simulador
    simulator = env_unwrapped.getSimulator()
    simulator.setAllocator(first_fit_algorithm)
    simulator.setRho(0.5)
    
    # Crear funcion de recompensa (usando Spectral-Entropy)
    print("Configurando funcion de recompensa Spectral-Entropy...")
    reward_fn = SpectralEntropyAdaptiveReward(entropy_weight=0.4)
    
    # Variable para rastrear conexiones
    last_allocated = [0]  # Usar lista para mantener referencia mutable
    
    # Crear wrapper para el entorno
    def custom_reward():
        network = simulator._Simulator__controller.network
        curr_allocated = simulator.allocatedConnections
        allocated = curr_allocated > last_allocated[0]
        last_allocated[0] = curr_allocated
        return reward_fn.calculate(allocated=allocated, network=network)
    
    env_unwrapped.setRewardFunc(custom_reward)
    
    # Funcion de estado simple
    def custom_state():
        network = simulator._Simulator__controller.network
        state = get_network_spectrum_state(network)
        return int(state['avg_utilization'] * 99)
    
    env_unwrapped.setStateFunc(custom_state)
    
    # Iniciar entorno
    env_unwrapped.start(verbose=False)
    
    print("Entrenando agente PPO (5000 pasos de prueba)...")
    
    # Crear y entrenar modelo
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64
    )
    
    model.learn(total_timesteps=5000)
    
    print("Entrenamiento completado!")
    
    # Evaluar
    print("\nEvaluando agente...")
    obs, info = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"Recompensa total en 100 pasos: {total_reward:.2f}")
    
    env.close()


# =============================================================================
# EJEMPLO 4: COMPARACION DE CONVERGENCIA
# =============================================================================

def example_convergence_comparison():
    """
    Compara la convergencia de diferentes funciones de recompensa.
    """
    print("\n" + "="*70)
    print("EJEMPLO 4: Comparacion de Convergencia")
    print("="*70)
    
    # Obtener paths
    base_dir = Path(__file__).parent.parent.parent
    network_dir = base_dir / "network"
    
    network_file = str(network_dir / "NSFNet.json")
    routes_file = str(network_dir / "NSFNet_routes.json")
    
    if not Path(network_file).exists():
        print(f"\nArchivo no encontrado: {network_file}")
        return
    
    # Configuracion
    n_episodes = 5
    steps_per_episode = 200
    
    reward_functions = [
        ('Baseline', BaselineReward()),
        ('Multi-Objective', MultiObjectiveReward()),
        ('Spectral-Entropy', SpectralEntropyAdaptiveReward())
    ]
    
    print(f"\nEjecutando {n_episodes} episodios de {steps_per_episode} pasos cada uno...")
    
    import numpy as np
    all_results = {}
    
    for name, reward_fn in reward_functions:
        print(f"\nEvaluando {name}...")
        episode_rewards = []
        
        for ep in range(n_episodes):
            # Crear simulador fresco
            simulator = Simulator(network_file, routes_file, "")
            simulator.setGoalConnections(steps_per_episode)
            simulator.setRho(0.5)
            simulator.setAllocator(first_fit_algorithm)
            simulator.init()
            
            # Reiniciar estado de la funcion de recompensa
            if hasattr(reward_fn, 'reset_episode'):
                reward_fn.reset_episode()
            if hasattr(reward_fn, 'reset_state'):
                reward_fn.reset_state()
            
            cumulative_reward = 0
            
            for step in range(steps_per_episode):
                # Guardar conexiones antes del paso
                prev_allocated = simulator.allocatedConnections
                simulator.step(0)
                simulator.forwardDepartures()
                
                # Determinar si esta conexion fue asignada
                curr_allocated = simulator.allocatedConnections
                allocated = curr_allocated > prev_allocated
                network = simulator._Simulator__controller.network
                
                r = reward_fn.calculate(allocated=allocated, network=network)
                cumulative_reward += r
                
                if step < steps_per_episode - 1:
                    simulator.createEventConnection()
            
            episode_rewards.append(cumulative_reward)
            print(f"   Episodio {ep+1}: R_total = {cumulative_reward:.2f}")
        
        all_results[name] = episode_rewards
    
    # Resumen
    print("\n" + "-"*60)
    print("RESUMEN DE CONVERGENCIA:")
    print("-"*60)
    print(f"{'Funcion':<25} {'R_promedio':>12} {'R_std':>10} {'Mejor':>12}")
    print("-"*60)
    
    for name, rewards in all_results.items():
        avg = np.mean(rewards)
        std = np.std(rewards)
        best = np.max(rewards)
        print(f"{name:<25} {avg:>12.2f} {std:>10.2f} {best:>12.2f}")
    
    print("-"*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Funcion principal que ejecuta todos los ejemplos."""
    print("\n" + "="*70)
    print("DREAM-ON-GYM-V2: Ejemplos de Funciones de Recompensa")
    print("="*70)
    
    # Ejemplo 1: Uso basico
    example_basic_usage()
    
    # Ejemplo 2: Integracion con simulador
    example_simulator_integration()
    
    # Ejemplo 3: Entrenamiento con SB3 (opcional)
    try:
        import stable_baselines3
        example_sb3_training()
    except ImportError:
        print("\nEjemplo 3 omitido: stable-baselines3 no disponible")
    
    # Ejemplo 4: Comparacion de convergencia
    example_convergence_comparison()
    
    print("\n" + "="*70)
    print("EJEMPLOS COMPLETADOS")
    print("="*70)


if __name__ == "__main__":
    main()
