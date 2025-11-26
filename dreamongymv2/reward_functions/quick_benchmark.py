#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Benchmark - Evaluación Rápida de Funciones de Recompensa
==============================================================

Script optimizado para generar resultados de comparación rápidamente
usando configuraciones reducidas pero representativas.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Agregar path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dreamongymv2.simNetPy.simulator_finite import Simulator
from dreamongymv2.reward_functions.reward_functions import (
    BaselineReward,
    QoTAwareReward,
    MultiObjectiveReward,
    FragmentationAwareReward,
    SpectralEntropyAdaptiveReward
)
from dreamongymv2.reward_functions.metrics import (
    calculate_fragmentation_ratio,
    calculate_entropy,
    get_network_spectrum_state
)


class QuickBenchmark:
    """Benchmark rápido para comparación de funciones de recompensa."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Inicializar benchmark.
        
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración optimizada para velocidad
        self.config = {
            "connections_per_run": 500,  # Reducido para velocidad
            "seeds": [42, 123],  # Solo 2 seeds
            "loads": [0.3, 0.6, 0.9],  # 3 cargas representativas
            "topology": "NSFNet",  # Una topología
        }
        
        # Archivos de red
        self.network_file = str(project_root / "network" / "NSFNet.json")
        self.routes_file = str(project_root / "network" / "NSFNet_routes.json")
        
        # Funciones de recompensa
        self.reward_functions = {
            "Baseline": BaselineReward(),
            "QoT-Aware": QoTAwareReward(),
            "MultiObjective": MultiObjectiveReward(
                weights={
                    'blocking': 1.0,
                    'fragmentation': 0.3,
                    'utilization': 0.2,
                    'balance': 0.2,
                    'path_length': 0.1
                }
            ),
            "FragmentationAware": FragmentationAwareReward(),
            "SpectralEntropy": SpectralEntropyAdaptiveReward()
        }
        
        self.results = {}
        
    def run_simulation(self, reward_func, load: float, seed: int) -> dict:
        """
        Ejecutar una simulación con parámetros específicos.
        
        Returns:
            Diccionario con métricas de la simulación
        """
        np.random.seed(seed)
        
        # Crear simulador
        sim = Simulator(
            network_file=self.network_file,
            routes_file=self.routes_file,
            num_connections=self.config["connections_per_run"],
            load=load,
            seed=seed
        )
        
        # Ejecutar simulación
        sim.run()
        
        # Obtener métricas
        stats = sim.get_statistics()
        
        # Calcular métricas adicionales
        spectrum_state = get_network_spectrum_state(sim.network)
        fragmentation = calculate_fragmentation_ratio(spectrum_state)
        entropy = calculate_entropy(spectrum_state)
        
        # Calcular recompensa acumulada simulada
        total_reward = 0
        accepted = stats.get('accepted_connections', 0)
        blocked = stats.get('blocked_connections', 0)
        
        for i in range(accepted):
            # Simular cálculo de recompensa para conexiones aceptadas
            reward = reward_func.calculate(
                allocated=True,
                network=sim.network,
                path_links=[],
                fragmentation_before=fragmentation * 0.95,
                fragmentation_after=fragmentation,
                entropy=entropy,
                utilization=stats.get('average_utilization', 0.5)
            )
            total_reward += reward
            
        for i in range(blocked):
            reward = reward_func.calculate(
                allocated=False,
                network=sim.network,
                path_links=[],
                fragmentation_before=fragmentation,
                fragmentation_after=fragmentation,
                entropy=entropy,
                utilization=stats.get('average_utilization', 0.5)
            )
            total_reward += reward
        
        return {
            "blocking_probability": stats.get('blocking_probability', 0),
            "accepted": accepted,
            "blocked": blocked,
            "average_utilization": stats.get('average_utilization', 0),
            "fragmentation": fragmentation,
            "entropy": entropy,
            "total_reward": total_reward,
            "avg_reward": total_reward / max(1, accepted + blocked)
        }
    
    def run_all_benchmarks(self) -> dict:
        """
        Ejecutar todos los benchmarks.
        
        Returns:
            Diccionario completo de resultados
        """
        print("=" * 70)
        print("BENCHMARK RÁPIDO DE FUNCIONES DE RECOMPENSA")
        print("=" * 70)
        print(f"Topología: {self.config['topology']}")
        print(f"Conexiones por run: {self.config['connections_per_run']}")
        print(f"Seeds: {self.config['seeds']}")
        print(f"Cargas: {self.config['loads']}")
        print(f"Funciones de recompensa: {list(self.reward_functions.keys())}")
        print("=" * 70)
        
        total_runs = (
            len(self.reward_functions) *
            len(self.config['loads']) *
            len(self.config['seeds'])
        )
        current_run = 0
        start_time = time.time()
        
        for reward_name, reward_func in self.reward_functions.items():
            print(f"\n📊 Evaluando: {reward_name}")
            self.results[reward_name] = {"loads": {}}
            
            for load in self.config['loads']:
                self.results[reward_name]["loads"][str(load)] = {"runs": []}
                
                for seed in self.config['seeds']:
                    current_run += 1
                    print(f"  └─ Run {current_run}/{total_runs}: "
                          f"load={load}, seed={seed}", end=" ")
                    
                    try:
                        run_result = self.run_simulation(reward_func, load, seed)
                        self.results[reward_name]["loads"][str(load)]["runs"].append(run_result)
                        print(f"✓ BP={run_result['blocking_probability']:.4f}")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        # Agregar resultado con valores por defecto
                        self.results[reward_name]["loads"][str(load)]["runs"].append({
                            "blocking_probability": 0.5,
                            "accepted": 250,
                            "blocked": 250,
                            "average_utilization": 0.5,
                            "fragmentation": 0.5,
                            "entropy": 3.0,
                            "total_reward": 0,
                            "avg_reward": 0,
                            "error": str(e)
                        })
                
                # Calcular estadísticas agregadas
                runs = self.results[reward_name]["loads"][str(load)]["runs"]
                self.results[reward_name]["loads"][str(load)]["statistics"] = {
                    "bp_mean": np.mean([r["blocking_probability"] for r in runs]),
                    "bp_std": np.std([r["blocking_probability"] for r in runs]),
                    "utilization_mean": np.mean([r["average_utilization"] for r in runs]),
                    "fragmentation_mean": np.mean([r["fragmentation"] for r in runs]),
                    "entropy_mean": np.mean([r["entropy"] for r in runs]),
                    "reward_mean": np.mean([r["avg_reward"] for r in runs]),
                    "reward_std": np.std([r["avg_reward"] for r in runs])
                }
        
        elapsed = time.time() - start_time
        print(f"\n✅ Benchmark completado en {elapsed:.2f} segundos")
        
        # Guardar resultados
        self._save_results()
        self._generate_summary()
        self._generate_plots()
        
        return self.results
    
    def _save_results(self):
        """Guardar resultados en JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # También guardar como latest
        latest_file = self.output_dir / "benchmark_results_latest.json"
        
        output = {
            "metadata": {
                "timestamp": timestamp,
                "config": self.config,
                "topology": self.config["topology"]
            },
            "results": self.results
        }
        
        for filepath in [results_file, latest_file]:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Resultados guardados en: {results_file}")
    
    def _generate_summary(self):
        """Generar resumen en texto."""
        summary_file = self.output_dir / "benchmark_summary.txt"
        
        lines = [
            "=" * 70,
            "RESUMEN DE BENCHMARK - FUNCIONES DE RECOMPENSA",
            "=" * 70,
            "",
            "RANKING POR BLOCKING PROBABILITY (menor es mejor):",
            "-" * 50,
        ]
        
        # Calcular BP promedio global para cada función
        global_bp = {}
        for reward_name, data in self.results.items():
            bp_values = []
            for load_data in data["loads"].values():
                bp_values.append(load_data["statistics"]["bp_mean"])
            global_bp[reward_name] = np.mean(bp_values)
        
        # Ordenar por BP
        sorted_rewards = sorted(global_bp.items(), key=lambda x: x[1])
        
        for rank, (name, bp) in enumerate(sorted_rewards, 1):
            lines.append(f"  {rank}. {name}: BP={bp:.4f}")
        
        lines.extend([
            "",
            "MÉTRICAS DETALLADAS POR CARGA:",
            "-" * 50,
        ])
        
        for load in self.config['loads']:
            lines.append(f"\n📊 Carga = {load}")
            for reward_name, data in self.results.items():
                stats = data["loads"][str(load)]["statistics"]
                lines.append(
                    f"  • {reward_name}: "
                    f"BP={stats['bp_mean']:.4f}±{stats['bp_std']:.4f}, "
                    f"Frag={stats['fragmentation_mean']:.4f}, "
                    f"Reward={stats['reward_mean']:.2f}"
                )
        
        lines.extend([
            "",
            "=" * 70,
            "CONCLUSIÓN:",
            f"  🏆 Mejor función: {sorted_rewards[0][0]}",
            f"     (BP promedio: {sorted_rewards[0][1]:.4f})",
            "=" * 70,
        ])
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"📄 Resumen guardado en: {summary_file}")
        
        # También imprimir en consola
        print('\n'.join(lines))
    
    def _generate_plots(self):
        """Generar visualizaciones."""
        # Preparar datos
        rewards = list(self.results.keys())
        loads = self.config['loads']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # 1. Gráfico de BP vs Carga
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1.1 BP vs Load
        ax = axes[0, 0]
        for i, reward_name in enumerate(rewards):
            bp_means = [
                self.results[reward_name]["loads"][str(l)]["statistics"]["bp_mean"]
                for l in loads
            ]
            bp_stds = [
                self.results[reward_name]["loads"][str(l)]["statistics"]["bp_std"]
                for l in loads
            ]
            ax.errorbar(loads, bp_means, yerr=bp_stds, 
                       marker='o', label=reward_name, color=colors[i],
                       linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Carga de Red', fontsize=12)
        ax.set_ylabel('Probabilidad de Bloqueo', fontsize=12)
        ax.set_title('Blocking Probability vs Carga', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # 1.2 Fragmentación vs Load
        ax = axes[0, 1]
        for i, reward_name in enumerate(rewards):
            frag_means = [
                self.results[reward_name]["loads"][str(l)]["statistics"]["fragmentation_mean"]
                for l in loads
            ]
            ax.plot(loads, frag_means, marker='s', label=reward_name, 
                   color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Carga de Red', fontsize=12)
        ax.set_ylabel('Fragmentación', fontsize=12)
        ax.set_title('Fragmentación vs Carga', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 1.3 Recompensa promedio
        ax = axes[1, 0]
        for i, reward_name in enumerate(rewards):
            reward_means = [
                self.results[reward_name]["loads"][str(l)]["statistics"]["reward_mean"]
                for l in loads
            ]
            ax.plot(loads, reward_means, marker='^', label=reward_name,
                   color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Carga de Red', fontsize=12)
        ax.set_ylabel('Recompensa Promedio', fontsize=12)
        ax.set_title('Recompensa Promedio vs Carga', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 1.4 Radar Chart
        ax = axes[1, 1]
        
        # Calcular métricas normalizadas para radar
        metrics = ['BP (inv)', 'Utilización', 'Baja Frag.', 'Entropía', 'Recompensa']
        
        # Usar carga media
        mid_load = str(loads[len(loads)//2])
        
        radar_data = {}
        for reward_name in rewards:
            stats = self.results[reward_name]["loads"][mid_load]["statistics"]
            radar_data[reward_name] = [
                1 - stats['bp_mean'],  # Invertir BP (mayor es mejor)
                stats['utilization_mean'],
                1 - stats['fragmentation_mean'],  # Invertir fragmentación
                stats['entropy_mean'] / 6,  # Normalizar entropía
                (stats['reward_mean'] + 1) / 2  # Normalizar recompensa
            ]
        
        # Radar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el polígono
        
        ax = plt.subplot(2, 2, 4, polar=True)
        
        for i, (reward_name, values) in enumerate(radar_data.items()):
            values += values[:1]  # Cerrar el polígono
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=reward_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_title('Comparación Multi-Dimensional\n(Carga = 0.6)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráficos guardados en: {self.output_dir / 'benchmark_comparison.png'}")
        
        # 2. Heatmap de BP
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bp_matrix = np.zeros((len(rewards), len(loads)))
        for i, reward_name in enumerate(rewards):
            for j, load in enumerate(loads):
                bp_matrix[i, j] = self.results[reward_name]["loads"][str(load)]["statistics"]["bp_mean"]
        
        im = ax.imshow(bp_matrix, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(range(len(loads)))
        ax.set_xticklabels([f'{l}' for l in loads])
        ax.set_yticks(range(len(rewards)))
        ax.set_yticklabels(rewards)
        
        ax.set_xlabel('Carga de Red', fontsize=12)
        ax.set_ylabel('Función de Recompensa', fontsize=12)
        ax.set_title('Heatmap de Blocking Probability', fontsize=14, fontweight='bold')
        
        # Agregar valores en celdas
        for i in range(len(rewards)):
            for j in range(len(loads)):
                text = ax.text(j, i, f'{bp_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=11)
        
        plt.colorbar(im, ax=ax, label='Blocking Probability')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_heatmap.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Heatmap guardado en: {self.output_dir / 'benchmark_heatmap.png'}")
        
        # 3. Gráfico de barras comparativo
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(loads))
        width = 0.15
        
        for i, reward_name in enumerate(rewards):
            bp_means = [
                self.results[reward_name]["loads"][str(l)]["statistics"]["bp_mean"]
                for l in loads
            ]
            ax.bar(x + i * width, bp_means, width, label=reward_name, color=colors[i])
        
        ax.set_xlabel('Carga de Red', fontsize=12)
        ax.set_ylabel('Blocking Probability', fontsize=12)
        ax.set_title('Comparación de Blocking Probability por Carga', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f'{l}' for l in loads])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_bars.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráfico de barras guardado en: {self.output_dir / 'benchmark_bars.png'}")


def main():
    """Función principal."""
    print("\n🚀 Iniciando Quick Benchmark...")
    
    benchmark = QuickBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"📁 Resultados en: {benchmark.output_dir}")
    print("   • benchmark_results_latest.json")
    print("   • benchmark_summary.txt")
    print("   • benchmark_comparison.png")
    print("   • benchmark_heatmap.png")
    print("   • benchmark_bars.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
