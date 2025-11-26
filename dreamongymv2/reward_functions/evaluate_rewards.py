# -*- coding: utf-8 -*-
"""
DREAM-ON-GYM-V2: Script de Evaluacion de Funciones de Recompensa
================================================================

Este script ejecuta experimentos comparativos para evaluar
diferentes funciones de recompensa en el entorno de redes opticas.

Experimentos:
------------
1. Entrenamiento con cada funcion de recompensa
2. Evaluacion de blocking probability
3. Metricas de fragmentacion
4. Comparativas de convergencia
5. Generacion de visualizaciones

@author: Generado con AI para DREAM-ON-GYM-V2
@version: 2.0.0
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Agregar paths necesarios
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importar gymnasium antes de otros modulos
import gymnasium
sys.modules["gym"] = gymnasium

# Importar componentes del framework
from dreamongymv2.simNetPy.simulator_finite import Simulator
from dreamongymv2.simNetPy.bitRate import BitRate
from dreamongymv2.simNetPy.connection import Connection
from dreamongymv2.simNetPy.network import Network
from dreamongymv2.simNetPy.controller import Controller

# Importar funciones de recompensa
from dreamongymv2.reward_functions.reward_functions import (
    BaselineReward,
    QoTAwareReward,
    MultiObjectiveReward,
    FragmentationAwareReward,
    SpectralEntropyAdaptiveReward,
    RewardFactory
)
from dreamongymv2.reward_functions.metrics import (
    calculate_fragmentation_ratio,
    calculate_network_utilization,
    get_network_spectrum_state
)

# Intentar importar bibliotecas de ML
try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 no disponible. Usando evaluacion sin RL.")

# Intentar importar matplotlib y seaborn
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn no disponible. Visualizaciones deshabilitadas.")


# =============================================================================
# CONFIGURACION DEL EXPERIMENTO
# =============================================================================

class ExperimentConfig:
    """Configuracion centralizada del experimento."""
    
    def __init__(self):
        # Directorios
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.results_dir / "plots"
        self.logs_dir = self.results_dir / "logs"
        
        # Crear directorios
        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Parametros del experimento
        self.n_connections = 100000  # Conexiones por experimento
        self.n_runs = 3  # Repeticiones por configuracion
        self.rho_values = [0.3, 0.5, 0.7, 0.9]  # Valores de carga
        
        # Topologias a evaluar (relativas al directorio network/)
        self.network_dir = self.base_dir.parent.parent / "network"
        self.topologies = [
            ("NSFNet", "NSFNet.json", "NSFNet_routes.json"),
            ("USNet", "USNet.json", "USNet_routes.json"),
            ("Eurocore", "Eurocore.json", "Eurocore_routes.json")
        ]
        
        # Funciones de recompensa a evaluar
        self.reward_functions = [
            ("Baseline", BaselineReward, {}),
            ("QoT-Aware", QoTAwareReward, {"qot_weight": 0.5}),
            ("Multi-Objective", MultiObjectiveReward, {"adaptive_weights": True}),
            ("Fragmentation-Aware", FragmentationAwareReward, {"local_weight": 0.4}),
            ("Spectral-Entropy", SpectralEntropyAdaptiveReward, {"entropy_weight": 0.4})
        ]
        
        # Parametros de RL (si disponible)
        self.rl_timesteps = 50000
        self.rl_algorithm = "PPO"


# =============================================================================
# ALGORITMO DE ASIGNACION FIRST-FIT
# =============================================================================

def first_fit_algorithm(src: int, dst: int, b: BitRate, c: Connection, 
                        n: Network, path, action):
    """Algoritmo First-Fit para asignacion de espectro."""
    numberOfSlots = b.getNumberofSlots(0)
    actionSpace = len(path[src][dst])
    
    if action is not None:
        if action >= actionSpace:
            action = actionSpace - 1
        link_ids = path[src][dst][action]
    else:
        link_ids = path[src][dst][0]
    
    # Construir vista general de slots
    general_link = []
    for _ in range(n.getLink(0).getSlots()):
        general_link.append(False)
    
    for link in link_ids:
        link = n.getLink(link.id)
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(slot)
    
    # Buscar bloque contiguo
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
# CLASE PRINCIPAL DE EXPERIMENTO
# =============================================================================

class RewardExperiment:
    """
    Clase para ejecutar experimentos de evaluacion de funciones de recompensa.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = defaultdict(list)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_simulation(self, 
                       topology_name: str,
                       network_file: str,
                       routes_file: str,
                       reward_fn,
                       rho: float,
                       n_connections: int) -> Dict:
        """
        Ejecuta una simulacion individual.
        
        Returns:
        --------
        Dict con metricas de la simulacion
        """
        # Crear simulador
        network_path = str(self.config.network_dir / network_file)
        routes_path = str(self.config.network_dir / routes_file)
        
        simulator = Simulator(network_path, routes_path, "")
        simulator.setGoalConnections(n_connections)
        simulator.setRho(rho)
        simulator.setAllocator(first_fit_algorithm)
        
        # Configurar funcion de recompensa
        reward_fn.set_simulator(simulator)
        
        # Iniciar simulacion
        start_time = time.time()
        simulator.init()
        
        # Ejecutar y recolectar metricas
        rewards_collected = []
        fragmentations = []
        
        for step in range(n_connections):
            # Ejecutar paso - guardar conexiones asignadas antes
            prev_allocated = simulator.allocatedConnections
            simulator.step(0)  # Action 0 = primera ruta
            simulator.forwardDepartures()
            
            # Obtener estado - determinar si esta conexion fue asignada
            curr_allocated = simulator.allocatedConnections
            allocated = curr_allocated > prev_allocated
            network = simulator._Simulator__controller.network
            state = get_network_spectrum_state(network)
            
            # Calcular recompensa
            reward = reward_fn.calculate(
                allocated=allocated,
                network=network
            )
            
            rewards_collected.append(reward)
            fragmentations.append(state['avg_fragmentation'])
            
            # Crear siguiente evento
            if step < n_connections - 1:
                simulator.createEventConnection()
        
        elapsed_time = time.time() - start_time
        
        # Recopilar metricas finales
        blocking_prob = simulator.getBlockingProbability()
        
        # Calcular estadisticas
        results = {
            'topology': topology_name,
            'reward_function': reward_fn.name,
            'rho': rho,
            'n_connections': n_connections,
            'blocking_probability': blocking_prob,
            'avg_reward': np.mean(rewards_collected),
            'std_reward': np.std(rewards_collected),
            'cumulative_reward': np.sum(rewards_collected),
            'avg_fragmentation': np.mean(fragmentations),
            'final_fragmentation': fragmentations[-1] if fragmentations else 0,
            'elapsed_time': elapsed_time,
            'allocated_connections': simulator.getAllocatedConnections(),
            'rejected_connections': simulator.getRejectConnections()
        }
        
        return results
    
    def run_experiments(self, verbose: bool = True) -> pd.DataFrame:
        """
        Ejecuta todos los experimentos configurados.
        
        Returns:
        --------
        DataFrame con todos los resultados
        """
        all_results = []
        total_experiments = (
            len(self.config.topologies) * 
            len(self.config.reward_functions) * 
            len(self.config.rho_values) *
            self.config.n_runs
        )
        
        current = 0
        
        for topo_name, net_file, routes_file in self.config.topologies:
            # Verificar que existan los archivos
            net_path = self.config.network_dir / net_file
            routes_path = self.config.network_dir / routes_file
            
            if not net_path.exists() or not routes_path.exists():
                if verbose:
                    print(f"Saltando {topo_name}: archivos no encontrados")
                continue
            
            for rf_name, rf_class, rf_params in self.config.reward_functions:
                for rho in self.config.rho_values:
                    for run in range(self.config.n_runs):
                        current += 1
                        
                        if verbose:
                            print(f"\n[{current}/{total_experiments}] "
                                  f"{topo_name} | {rf_name} | rho={rho} | run={run+1}")
                        
                        try:
                            # Crear instancia de funcion de recompensa
                            reward_fn = rf_class(**rf_params)
                            
                            # Ejecutar simulacion
                            result = self.run_simulation(
                                topology_name=topo_name,
                                network_file=net_file,
                                routes_file=routes_file,
                                reward_fn=reward_fn,
                                rho=rho,
                                n_connections=self.config.n_connections
                            )
                            
                            result['run'] = run + 1
                            all_results.append(result)
                            
                            if verbose:
                                print(f"   BP: {result['blocking_probability']:.4f} | "
                                      f"Reward: {result['avg_reward']:.3f} | "
                                      f"Frag: {result['avg_fragmentation']:.3f}")
                        
                        except Exception as e:
                            print(f"   ERROR: {str(e)}")
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_results)
        
        # Guardar resultados
        results_file = self.config.results_dir / f"results_{self.experiment_id}.csv"
        df.to_csv(results_file, index=False)
        
        if verbose:
            print(f"\nResultados guardados en: {results_file}")
        
        return df
    
    def generate_plots(self, df: pd.DataFrame):
        """
        Genera visualizaciones de los resultados.
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib no disponible. Saltando generacion de plots.")
            return
        
        print("\nGenerando visualizaciones...")
        
        # 1. Blocking Probability vs Rho por Reward Function
        self._plot_blocking_vs_rho(df)
        
        # 2. Comparativa de Recompensas Promedio
        self._plot_reward_comparison(df)
        
        # 3. Fragmentacion por Funcion de Recompensa
        self._plot_fragmentation_comparison(df)
        
        # 4. Heatmap de Rendimiento
        self._plot_performance_heatmap(df)
        
        # 5. Boxplots de Variabilidad
        self._plot_variability_boxplots(df)
        
        print(f"Plots guardados en: {self.config.plots_dir}")
    
    def _plot_blocking_vs_rho(self, df: pd.DataFrame):
        """Grafico de Blocking Probability vs Carga (Rho)."""
        fig, axes = plt.subplots(1, len(df['topology'].unique()), 
                                  figsize=(15, 5), sharey=True)
        
        if len(df['topology'].unique()) == 1:
            axes = [axes]
        
        for ax, topo in zip(axes, df['topology'].unique()):
            topo_df = df[df['topology'] == topo]
            
            for rf in topo_df['reward_function'].unique():
                rf_df = topo_df[topo_df['reward_function'] == rf]
                means = rf_df.groupby('rho')['blocking_probability'].mean()
                stds = rf_df.groupby('rho')['blocking_probability'].std()
                
                ax.errorbar(means.index, means.values, yerr=stds.values,
                           marker='o', capsize=3, label=rf)
            
            ax.set_xlabel('Carga (rho)')
            ax.set_ylabel('Blocking Probability')
            ax.set_title(f'{topo}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Blocking Probability vs Carga de Red', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.config.plots_dir / 'blocking_vs_rho.png', dpi=150)
        plt.close()
    
    def _plot_reward_comparison(self, df: pd.DataFrame):
        """Comparativa de recompensas promedio."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Agrupar por funcion de recompensa
        grouped = df.groupby('reward_function').agg({
            'avg_reward': ['mean', 'std'],
            'cumulative_reward': ['mean', 'std']
        }).round(4)
        
        x = np.arange(len(grouped))
        width = 0.35
        
        ax.bar(x - width/2, grouped['avg_reward']['mean'], width,
               yerr=grouped['avg_reward']['std'],
               label='Recompensa Promedio', capsize=3)
        
        ax.set_xlabel('Funcion de Recompensa')
        ax.set_ylabel('Recompensa')
        ax.set_title('Comparativa de Recompensas por Funcion')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.config.plots_dir / 'reward_comparison.png', dpi=150)
        plt.close()
    
    def _plot_fragmentation_comparison(self, df: pd.DataFrame):
        """Comparativa de fragmentacion."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Fragmentacion promedio
        ax1 = axes[0]
        df_frag = df.groupby(['reward_function', 'rho'])['avg_fragmentation'].mean().unstack()
        df_frag.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xlabel('Funcion de Recompensa')
        ax1.set_ylabel('Fragmentacion Promedio')
        ax1.set_title('Fragmentacion Promedio por Funcion y Carga')
        ax1.legend(title='Rho')
        ax1.tick_params(axis='x', rotation=45)
        
        # Fragmentacion final
        ax2 = axes[1]
        df_final = df.groupby(['reward_function', 'rho'])['final_fragmentation'].mean().unstack()
        df_final.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xlabel('Funcion de Recompensa')
        ax2.set_ylabel('Fragmentacion Final')
        ax2.set_title('Fragmentacion al Final de Simulacion')
        ax2.legend(title='Rho')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.config.plots_dir / 'fragmentation_comparison.png', dpi=150)
        plt.close()
    
    def _plot_performance_heatmap(self, df: pd.DataFrame):
        """Heatmap de rendimiento."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap de Blocking Probability
        ax1 = axes[0]
        pivot_bp = df.pivot_table(
            values='blocking_probability',
            index='reward_function',
            columns='rho',
            aggfunc='mean'
        )
        sns.heatmap(pivot_bp, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   ax=ax1, cbar_kws={'label': 'Blocking Probability'})
        ax1.set_title('Blocking Probability por Configuracion')
        
        # Heatmap de Recompensa
        ax2 = axes[1]
        pivot_reward = df.pivot_table(
            values='avg_reward',
            index='reward_function',
            columns='rho',
            aggfunc='mean'
        )
        sns.heatmap(pivot_reward, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax2, cbar_kws={'label': 'Recompensa Promedio'})
        ax2.set_title('Recompensa Promedio por Configuracion')
        
        plt.tight_layout()
        plt.savefig(self.config.plots_dir / 'performance_heatmap.png', dpi=150)
        plt.close()
    
    def _plot_variability_boxplots(self, df: pd.DataFrame):
        """Boxplots de variabilidad entre runs."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = [
            ('blocking_probability', 'Blocking Probability'),
            ('avg_reward', 'Recompensa Promedio'),
            ('avg_fragmentation', 'Fragmentacion Promedio')
        ]
        
        for ax, (metric, title) in zip(axes, metrics):
            sns.boxplot(data=df, x='reward_function', y=metric, ax=ax)
            ax.set_xlabel('Funcion de Recompensa')
            ax.set_ylabel(title)
            ax.set_title(f'Variabilidad de {title}')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.config.plots_dir / 'variability_boxplots.png', dpi=150)
        plt.close()
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Genera un reporte en formato Markdown.
        """
        report = []
        report.append("# Reporte de Evaluacion de Funciones de Recompensa")
        report.append(f"\n**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**ID Experimento:** {self.experiment_id}")
        
        report.append("\n## Configuracion del Experimento")
        report.append(f"- **Conexiones por simulacion:** {self.config.n_connections:,}")
        report.append(f"- **Repeticiones por config:** {self.config.n_runs}")
        report.append(f"- **Valores de carga (rho):** {self.config.rho_values}")
        
        report.append("\n## Funciones de Recompensa Evaluadas")
        for name, _, _ in self.config.reward_functions:
            report.append(f"- {name}")
        
        report.append("\n## Resumen de Resultados")
        
        # Tabla de blocking probability
        report.append("\n### Blocking Probability por Funcion de Recompensa")
        bp_summary = df.groupby('reward_function')['blocking_probability'].agg(['mean', 'std', 'min', 'max'])
        report.append(bp_summary.to_markdown())
        
        # Tabla de recompensas
        report.append("\n### Recompensa Promedio por Funcion")
        reward_summary = df.groupby('reward_function')['avg_reward'].agg(['mean', 'std'])
        report.append(reward_summary.to_markdown())
        
        # Mejor configuracion
        report.append("\n### Mejor Configuracion")
        best_idx = df['blocking_probability'].idxmin()
        best = df.loc[best_idx]
        report.append(f"- **Funcion:** {best['reward_function']}")
        report.append(f"- **Topologia:** {best['topology']}")
        report.append(f"- **Rho:** {best['rho']}")
        report.append(f"- **Blocking Probability:** {best['blocking_probability']:.4f}")
        
        # Conclusiones
        report.append("\n## Conclusiones")
        
        # Comparar funciones
        bp_means = df.groupby('reward_function')['blocking_probability'].mean()
        best_fn = bp_means.idxmin()
        worst_fn = bp_means.idxmax()
        improvement = ((bp_means[worst_fn] - bp_means[best_fn]) / bp_means[worst_fn]) * 100
        
        report.append(f"- La funcion **{best_fn}** obtuvo el menor blocking probability promedio.")
        report.append(f"- Mejora respecto a {worst_fn}: **{improvement:.1f}%**")
        
        # Guardar reporte
        report_text = "\n".join(report)
        report_file = self.config.results_dir / f"report_{self.experiment_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Reporte guardado en: {report_file}")
        
        return report_text


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """Punto de entrada principal."""
    print("=" * 70)
    print("DREAM-ON-GYM-V2: Evaluacion de Funciones de Recompensa")
    print("=" * 70)
    
    # Crear configuracion
    config = ExperimentConfig()
    
    # Modificar para ejecucion rapida de prueba
    config.n_connections = 10000  # Reducir para prueba rapida
    config.n_runs = 1
    config.rho_values = [0.5, 0.7]  # Solo dos valores de carga
    
    print(f"\nConfiguracion:")
    print(f"  - Conexiones: {config.n_connections:,}")
    print(f"  - Runs: {config.n_runs}")
    print(f"  - Rho: {config.rho_values}")
    print(f"  - Topologias: {[t[0] for t in config.topologies]}")
    print(f"  - Funciones de Recompensa: {[rf[0] for rf in config.reward_functions]}")
    
    # Crear y ejecutar experimento
    experiment = RewardExperiment(config)
    
    print("\n" + "-" * 70)
    print("Iniciando experimentos...")
    print("-" * 70)
    
    df = experiment.run_experiments(verbose=True)
    
    if not df.empty:
        # Generar visualizaciones
        experiment.generate_plots(df)
        
        # Generar reporte
        report = experiment.generate_report(df)
        
        print("\n" + "=" * 70)
        print("EXPERIMENTO COMPLETADO")
        print("=" * 70)
        print(f"\nResultados en: {config.results_dir}")
    else:
        print("\nNo se generaron resultados. Verifique la configuracion.")


if __name__ == "__main__":
    main()
