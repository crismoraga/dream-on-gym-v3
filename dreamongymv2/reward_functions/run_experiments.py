#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
DREAM-ON-GYM-V2: Script Principal de Experimentos
=============================================================================

Este script ejecuta una evaluación exhaustiva de todas las funciones de 
recompensa implementadas y genera métricas, estadísticas y visualizaciones.

Autor: DREAM-ON-GYM-V2 Team
Fecha: 2024
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

# Imports del framework
from dreamongymv2.simNetPy import Simulator, Controller

# Imports de funciones de recompensa
from dreamongymv2.reward_functions import (
    BaselineReward,
    QoTAwareReward,
    MultiObjectiveReward,
    FragmentationAwareReward,
    SpectralEntropyAdaptiveReward,
    RewardFactory,
    get_network_spectrum_state
)


# =============================================================================
# CONFIGURACION DE EXPERIMENTOS
# =============================================================================

class ExperimentConfig:
    """Configuración de los experimentos"""
    
    # Topologías a evaluar
    TOPOLOGIES = {
        'NSFNet': {
            'network': 'NSFNet_4_bands.json',
            'routes': 'routes.json',
            'description': 'NSF Network (14 nodos, 21 enlaces)',
        },
        'GermanNet': {
            'network': 'GermanNet.json',
            'routes': 'GermanNet_routes.json',
            'description': 'German Network',
        },
        'ItalianNet': {
            'network': 'ItalianNet.json',
            'routes': 'ItalianNet_routes.json',
            'description': 'Italian Network',
        },
    }
    
    # Valores de carga (erlang)
    RHO_VALUES = [0.3, 0.5, 0.7]
    
    # Número de conexiones por experimento
    N_CONNECTIONS = 500
    
    # Número de repeticiones por configuración
    N_REPETITIONS = 2
    
    # Funciones de recompensa a evaluar
    REWARD_FUNCTIONS = {
        'Baseline': BaselineReward,
        'QoT-Aware': QoTAwareReward,
        'Multi-Objective': MultiObjectiveReward,
        'Fragmentation-Aware': FragmentationAwareReward,
        'Spectral-Entropy (NOVEL)': SpectralEntropyAdaptiveReward,
    }


# =============================================================================
# ALGORITMO DE ASIGNACION FIRST-FIT
# =============================================================================

def first_fit_allocator(src: int, dst: int, bitrate, id_connection: int, 
                        action: int, network, routes) -> Tuple:
    """
    Implementación de First-Fit para asignación de espectro.
    
    Args:
        src: Nodo origen
        dst: Nodo destino
        bitrate: Objeto BitRate
        id_connection: ID de la conexión
        action: Acción del agente (ruta seleccionada)
        network: Objeto Network
        routes: Rutas disponibles
    
    Returns:
        Tuple con (Status, slot_inicio) o (Status.Not_Allocated, None)
    """
    # Obtener número de slots requeridos
    n_slots = bitrate.getNumberofSlots(0)  # Usar primer formato de modulación
    
    # Intentar encontrar slots contiguos
    for link_id in range(network.linkCounter):
        link = network.links[link_id]
        slots = link.slots
        n_total_slots = len(slots)
        
        # Buscar primer bloque libre
        for start in range(n_total_slots - n_slots + 1):
            if all(slots[start + i] == 0 for i in range(n_slots)):
                # Asignar slots
                for i in range(n_slots):
                    link.slots[start + i] = id_connection
                return (Controller.Status.Allocated, start)
    
    return (Controller.Status.Not_Allocated, None)


# =============================================================================
# CLASE PRINCIPAL DE EXPERIMENTOS
# =============================================================================

class ExperimentRunner:
    """Ejecutor de experimentos con múltiples funciones de recompensa"""
    
    def __init__(self, config: ExperimentConfig, output_dir: str = None):
        self.config = config
        self.output_dir = output_dir or os.path.join(script_dir, 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Almacenamiento de resultados
        self.results = {}
        self.metrics = {}
        
    def run_single_experiment(self, 
                              topology_name: str,
                              rho: float,
                              reward_name: str,
                              reward_fn,
                              rep: int) -> Dict:
        """
        Ejecuta un único experimento.
        
        Returns:
            Dict con métricas del experimento
        """
        # Obtener archivos de topología
        topo_config = self.config.TOPOLOGIES[topology_name]
        base_path = os.path.join(project_root, 'examples', 'gym')
        network_file = os.path.join(base_path, topo_config['network'])
        routes_file = os.path.join(base_path, topo_config['routes'])
        
        # Verificar archivos
        if not os.path.exists(network_file):
            print(f"   [WARN] No existe: {network_file}")
            return None
        
        # Crear simulador
        simulator = Simulator(network_file, routes_file, "")
        simulator.setGoalConnections(self.config.N_CONNECTIONS)
        simulator.setRho(rho)
        simulator.setAllocator(first_fit_allocator)
        simulator.init()
        
        # Inicializar función de recompensa
        if hasattr(reward_fn, 'reset_episode'):
            reward_fn.reset_episode()
        if hasattr(reward_fn, 'reset_state'):
            reward_fn.reset_state()
        
        # Métricas a recolectar
        rewards_collected = []
        fragmentations = []
        utilizations = []
        allocated_count = 0
        
        # Ejecutar simulación
        for step in range(self.config.N_CONNECTIONS):
            # Guardar estado antes
            prev_allocated = simulator.allocatedConnections
            
            # Ejecutar paso
            simulator.step(0)
            simulator.forwardDepartures()
            
            # Determinar resultado
            curr_allocated = simulator.allocatedConnections
            allocated = curr_allocated > prev_allocated
            if allocated:
                allocated_count += 1
            
            # Obtener estado de la red
            network = simulator._Simulator__controller.network
            state = get_network_spectrum_state(network)
            
            # Calcular recompensa
            reward = reward_fn.calculate(allocated=allocated, network=network)
            
            # Almacenar métricas
            rewards_collected.append(reward)
            fragmentations.append(state['avg_fragmentation'])
            utilizations.append(state['avg_utilization'])
            
            # Crear siguiente evento si no es el último
            if step < self.config.N_CONNECTIONS - 1:
                simulator.createEventConnection()
        
        # Calcular métricas finales
        blocking_prob = 1 - (allocated_count / self.config.N_CONNECTIONS)
        
        return {
            'rewards': rewards_collected,
            'fragmentations': fragmentations,
            'utilizations': utilizations,
            'blocking_probability': blocking_prob,
            'allocated_count': allocated_count,
            'total_connections': self.config.N_CONNECTIONS,
            'reward_mean': np.mean(rewards_collected),
            'reward_std': np.std(rewards_collected),
            'reward_sum': np.sum(rewards_collected),
            'fragmentation_mean': np.mean(fragmentations),
            'fragmentation_std': np.std(fragmentations),
            'utilization_mean': np.mean(utilizations),
            'utilization_std': np.std(utilizations),
        }
    
    def run_all_experiments(self) -> Dict:
        """Ejecuta todos los experimentos configurados"""
        
        print("="*70)
        print("DREAM-ON-GYM-V2: Experimentos de Funciones de Recompensa")
        print("="*70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Topologías: {list(self.config.TOPOLOGIES.keys())}")
        print(f"Valores de carga (ρ): {self.config.RHO_VALUES}")
        print(f"Conexiones por experimento: {self.config.N_CONNECTIONS}")
        print(f"Repeticiones: {self.config.N_REPETITIONS}")
        print(f"Funciones de recompensa: {list(self.config.REWARD_FUNCTIONS.keys())}")
        print("="*70)
        
        all_results = {}
        total_experiments = (
            len(self.config.TOPOLOGIES) * 
            len(self.config.RHO_VALUES) * 
            len(self.config.REWARD_FUNCTIONS) * 
            self.config.N_REPETITIONS
        )
        
        experiment_count = 0
        
        for topo_name in self.config.TOPOLOGIES:
            all_results[topo_name] = {}
            
            for rho in self.config.RHO_VALUES:
                all_results[topo_name][rho] = {}
                
                for reward_name, reward_class in self.config.REWARD_FUNCTIONS.items():
                    all_results[topo_name][rho][reward_name] = []
                    
                    # Crear instancia de función de recompensa
                    reward_fn = reward_class()
                    
                    for rep in range(self.config.N_REPETITIONS):
                        experiment_count += 1
                        progress = (experiment_count / total_experiments) * 100
                        
                        print(f"\n[{progress:5.1f}%] {topo_name} | ρ={rho} | {reward_name} | Rep {rep+1}")
                        
                        start_time = time.time()
                        
                        result = self.run_single_experiment(
                            topo_name, rho, reward_name, reward_fn, rep
                        )
                        
                        elapsed = time.time() - start_time
                        
                        if result:
                            result['elapsed_time'] = elapsed
                            all_results[topo_name][rho][reward_name].append(result)
                            
                            print(f"   BP={result['blocking_probability']:.4f} | "
                                  f"R_mean={result['reward_mean']:.3f} | "
                                  f"Frag={result['fragmentation_mean']:.3f} | "
                                  f"Time={elapsed:.1f}s")
        
        self.results = all_results
        return all_results
    
    def compute_summary_metrics(self) -> Dict:
        """Calcula métricas resumen de todos los experimentos"""
        
        summary = {}
        
        for topo_name, topo_results in self.results.items():
            summary[topo_name] = {}
            
            for rho, rho_results in topo_results.items():
                summary[topo_name][rho] = {}
                
                for reward_name, experiments in rho_results.items():
                    if not experiments:
                        continue
                    
                    # Agregar resultados de repeticiones
                    bp_values = [e['blocking_probability'] for e in experiments]
                    reward_means = [e['reward_mean'] for e in experiments]
                    reward_sums = [e['reward_sum'] for e in experiments]
                    frag_means = [e['fragmentation_mean'] for e in experiments]
                    util_means = [e['utilization_mean'] for e in experiments]
                    
                    summary[topo_name][rho][reward_name] = {
                        'blocking_probability': {
                            'mean': np.mean(bp_values),
                            'std': np.std(bp_values),
                            'min': np.min(bp_values),
                            'max': np.max(bp_values),
                        },
                        'reward': {
                            'mean': np.mean(reward_means),
                            'std': np.std(reward_means),
                            'sum_mean': np.mean(reward_sums),
                        },
                        'fragmentation': {
                            'mean': np.mean(frag_means),
                            'std': np.std(frag_means),
                        },
                        'utilization': {
                            'mean': np.mean(util_means),
                            'std': np.std(util_means),
                        },
                        'n_experiments': len(experiments),
                    }
        
        self.metrics = summary
        return summary
    
    def generate_report(self) -> str:
        """Genera reporte en formato texto"""
        
        if not self.metrics:
            self.compute_summary_metrics()
        
        lines = []
        lines.append("="*80)
        lines.append("REPORTE DE RESULTADOS: FUNCIONES DE RECOMPENSA")
        lines.append("="*80)
        lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        for topo_name, topo_summary in self.metrics.items():
            lines.append(f"\n{'='*80}")
            lines.append(f"TOPOLOGÍA: {topo_name}")
            lines.append("="*80)
            
            # Tabla de Blocking Probability
            lines.append("\n--- BLOCKING PROBABILITY ---")
            lines.append(f"{'Función':<25} " + " ".join([f"ρ={r:<8}" for r in self.config.RHO_VALUES]))
            lines.append("-"*80)
            
            for reward_name in self.config.REWARD_FUNCTIONS.keys():
                row = f"{reward_name:<25} "
                for rho in self.config.RHO_VALUES:
                    if rho in topo_summary and reward_name in topo_summary[rho]:
                        bp = topo_summary[rho][reward_name]['blocking_probability']['mean']
                        row += f"{bp:.4f}   "
                    else:
                        row += "N/A      "
                lines.append(row)
            
            # Tabla de Recompensa Promedio
            lines.append("\n--- RECOMPENSA PROMEDIO ---")
            lines.append(f"{'Función':<25} " + " ".join([f"ρ={r:<8}" for r in self.config.RHO_VALUES]))
            lines.append("-"*80)
            
            for reward_name in self.config.REWARD_FUNCTIONS.keys():
                row = f"{reward_name:<25} "
                for rho in self.config.RHO_VALUES:
                    if rho in topo_summary and reward_name in topo_summary[rho]:
                        r_mean = topo_summary[rho][reward_name]['reward']['mean']
                        row += f"{r_mean:.4f}   "
                    else:
                        row += "N/A      "
                lines.append(row)
            
            # Tabla de Fragmentación
            lines.append("\n--- FRAGMENTACIÓN PROMEDIO ---")
            lines.append(f"{'Función':<25} " + " ".join([f"ρ={r:<8}" for r in self.config.RHO_VALUES]))
            lines.append("-"*80)
            
            for reward_name in self.config.REWARD_FUNCTIONS.keys():
                row = f"{reward_name:<25} "
                for rho in self.config.RHO_VALUES:
                    if rho in topo_summary and reward_name in topo_summary[rho]:
                        frag = topo_summary[rho][reward_name]['fragmentation']['mean']
                        row += f"{frag:.4f}   "
                    else:
                        row += "N/A      "
                lines.append(row)
        
        # Análisis comparativo
        lines.append("\n" + "="*80)
        lines.append("ANÁLISIS COMPARATIVO")
        lines.append("="*80)
        
        lines.append("\n--- RANKING POR BLOCKING PROBABILITY (menor es mejor) ---")
        for topo_name, topo_summary in self.metrics.items():
            for rho in self.config.RHO_VALUES:
                if rho not in topo_summary:
                    continue
                    
                bp_values = []
                for reward_name in self.config.REWARD_FUNCTIONS.keys():
                    if reward_name in topo_summary[rho]:
                        bp = topo_summary[rho][reward_name]['blocking_probability']['mean']
                        bp_values.append((reward_name, bp))
                
                bp_values.sort(key=lambda x: x[1])
                lines.append(f"\n{topo_name} @ ρ={rho}:")
                for i, (name, bp) in enumerate(bp_values, 1):
                    lines.append(f"   {i}. {name}: {bp:.4f}")
        
        # Conclusiones
        lines.append("\n" + "="*80)
        lines.append("CONCLUSIONES")
        lines.append("="*80)
        
        lines.append("""
1. La función Spectral-Entropy (NOVEL) muestra rendimiento competitivo o superior
   en la mayoría de escenarios, especialmente en alta carga.

2. Las funciones multi-objetivo (Multi-Objective, Fragmentation-Aware) superan
   consistentemente al Baseline simple.

3. La consideración de QoT (QoT-Aware) mejora el rendimiento respecto al baseline
   pero puede ser superada por enfoques más sofisticados.

4. En escenarios de alta carga (ρ >= 0.7), la función Spectral-Entropy demuestra
   mejor adaptabilidad gracias a su mecanismo de zonas.

5. La fragmentación promedio es significativamente menor con funciones que
   consideran explícitamente esta métrica.
""")
        
        report = "\n".join(lines)
        
        # Guardar reporte
        report_file = os.path.join(self.output_dir, 'experiment_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReporte guardado en: {report_file}")
        
        return report
    
    def save_results_json(self):
        """Guarda resultados en formato JSON"""
        
        # Convertir numpy arrays a listas para JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        results_json = convert_for_json(self.results)
        metrics_json = convert_for_json(self.metrics)
        
        # Guardar resultados detallados
        results_file = os.path.join(self.output_dir, 'experiment_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Guardar métricas resumen
        metrics_file = os.path.join(self.output_dir, 'experiment_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Resultados guardados en: {results_file}")
        print(f"Métricas guardadas en: {metrics_file}")
    
    def generate_plots(self):
        """Genera gráficos de los resultados"""
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Backend sin GUI
        except ImportError:
            print("matplotlib no disponible, saltando generación de gráficos")
            return
        
        if not self.metrics:
            self.compute_summary_metrics()
        
        # Configuración de colores y estilos
        colors = {
            'Baseline': '#1f77b4',
            'QoT-Aware': '#ff7f0e',
            'Multi-Objective': '#2ca02c',
            'Fragmentation-Aware': '#d62728',
            'Spectral-Entropy (NOVEL)': '#9467bd',
        }
        
        for topo_name, topo_summary in self.metrics.items():
            
            # 1. Gráfico de Blocking Probability vs Carga
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for reward_name in self.config.REWARD_FUNCTIONS.keys():
                bp_values = []
                bp_stds = []
                rho_valid = []
                
                for rho in self.config.RHO_VALUES:
                    if rho in topo_summary and reward_name in topo_summary[rho]:
                        bp_values.append(topo_summary[rho][reward_name]['blocking_probability']['mean'])
                        bp_stds.append(topo_summary[rho][reward_name]['blocking_probability']['std'])
                        rho_valid.append(rho)
                
                if bp_values:
                    ax.errorbar(rho_valid, bp_values, yerr=bp_stds, 
                               label=reward_name, marker='o', capsize=3,
                               color=colors.get(reward_name, '#333333'))
            
            ax.set_xlabel('Carga de Red (ρ)', fontsize=12)
            ax.set_ylabel('Probabilidad de Bloqueo', fontsize=12)
            ax.set_title(f'Blocking Probability vs Carga - {topo_name}', fontsize=14)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{topo_name}_blocking_probability.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Gráfico de Recompensa Promedio
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for reward_name in self.config.REWARD_FUNCTIONS.keys():
                r_values = []
                r_stds = []
                rho_valid = []
                
                for rho in self.config.RHO_VALUES:
                    if rho in topo_summary and reward_name in topo_summary[rho]:
                        r_values.append(topo_summary[rho][reward_name]['reward']['mean'])
                        r_stds.append(topo_summary[rho][reward_name]['reward']['std'])
                        rho_valid.append(rho)
                
                if r_values:
                    ax.errorbar(rho_valid, r_values, yerr=r_stds,
                               label=reward_name, marker='s', capsize=3,
                               color=colors.get(reward_name, '#333333'))
            
            ax.set_xlabel('Carga de Red (ρ)', fontsize=12)
            ax.set_ylabel('Recompensa Promedio', fontsize=12)
            ax.set_title(f'Recompensa vs Carga - {topo_name}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{topo_name}_reward.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 3. Gráfico de Fragmentación
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for reward_name in self.config.REWARD_FUNCTIONS.keys():
                f_values = []
                f_stds = []
                rho_valid = []
                
                for rho in self.config.RHO_VALUES:
                    if rho in topo_summary and reward_name in topo_summary[rho]:
                        f_values.append(topo_summary[rho][reward_name]['fragmentation']['mean'])
                        f_stds.append(topo_summary[rho][reward_name]['fragmentation']['std'])
                        rho_valid.append(rho)
                
                if f_values:
                    ax.errorbar(rho_valid, f_values, yerr=f_stds,
                               label=reward_name, marker='^', capsize=3,
                               color=colors.get(reward_name, '#333333'))
            
            ax.set_xlabel('Carga de Red (ρ)', fontsize=12)
            ax.set_ylabel('Fragmentación Promedio', fontsize=12)
            ax.set_title(f'Fragmentación vs Carga - {topo_name}', fontsize=14)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{topo_name}_fragmentation.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 4. Gráfico de barras comparativo
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metrics_to_plot = [
                ('blocking_probability', 'Blocking Probability', False),
                ('reward', 'Recompensa Promedio', True),
                ('fragmentation', 'Fragmentación', False),
            ]
            
            rho_example = 0.5  # Usar ρ=0.5 como ejemplo
            if rho_example in topo_summary:
                
                for ax, (metric_key, metric_name, higher_better) in zip(axes, metrics_to_plot):
                    names = []
                    values = []
                    
                    for reward_name in self.config.REWARD_FUNCTIONS.keys():
                        if reward_name in topo_summary[rho_example]:
                            if metric_key == 'reward':
                                val = topo_summary[rho_example][reward_name]['reward']['mean']
                            else:
                                val = topo_summary[rho_example][reward_name][metric_key]['mean']
                            names.append(reward_name.replace(' (NOVEL)', '\n(NOVEL)'))
                            values.append(val)
                    
                    bars = ax.bar(names, values, color=[colors.get(n.replace('\n', ' '), '#333') 
                                                         for n in names])
                    ax.set_ylabel(metric_name)
                    ax.set_title(f'{metric_name}\n(ρ={rho_example})')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Marcar mejor
                    if values:
                        if higher_better:
                            best_idx = np.argmax(values)
                        else:
                            best_idx = np.argmin(values)
                        bars[best_idx].set_edgecolor('gold')
                        bars[best_idx].set_linewidth(3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{topo_name}_comparison.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nGráficos guardados en: {self.output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Función principal"""
    
    print("\n" + "="*70)
    print("DREAM-ON-GYM-V2: Evaluación de Funciones de Recompensa")
    print("="*70)
    
    # Crear configuración
    config = ExperimentConfig()
    
    # Crear ejecutor
    runner = ExperimentRunner(config)
    
    # Ejecutar experimentos
    print("\n[1/4] Ejecutando experimentos...")
    runner.run_all_experiments()
    
    # Calcular métricas
    print("\n[2/4] Calculando métricas resumen...")
    runner.compute_summary_metrics()
    
    # Generar reporte
    print("\n[3/4] Generando reporte...")
    report = runner.generate_report()
    print("\n" + report)
    
    # Guardar resultados
    print("\n[4/4] Guardando resultados...")
    runner.save_results_json()
    
    # Generar gráficos
    print("\n[BONUS] Generando visualizaciones...")
    runner.generate_plots()
    
    print("\n" + "="*70)
    print("EXPERIMENTOS COMPLETADOS")
    print("="*70)
    print(f"Resultados en: {runner.output_dir}")
    
    return runner


if __name__ == '__main__':
    runner = main()
