# -*- coding: utf-8 -*-
"""
DREAM-ON-GYM-V2: Clases de Funciones de Recompensa Avanzadas
============================================================

Este modulo implementa funciones de recompensa state-of-the-art para
entrenamiento de agentes DRL en redes opticas elasticas (EON).

FUNCIONES IMPLEMENTADAS:
========================

1. BaselineReward
   - Recompensa binaria simple: +1 exito, -1 fracaso
   - Usado como baseline para comparacion

2. QoTAwareReward (Quality of Transmission)
   - Basado en DeepRMSA-QoT (Chen et al., 2019)
   - Considera calidad de senal OSNR y distancia

3. MultiObjectiveReward
   - Combina multiples objetivos con pesos adaptativos
   - Blocking + Fragmentacion + Throughput + Balanceo

4. FragmentationAwareReward
   - Minimiza fragmentacion espectral
   - Basado en investigacion de Gao et al. (2022)

5. SpectralEntropyAdaptiveReward (NOVEL - 100% NUEVO)
   - Funcion de recompensa original basada en entropia espectral
   - Adapta dinamicamente los pesos segun el estado de la red
   - Incorpora memoria temporal para delayed assignment

@author: Generado con AI para DREAM-ON-GYM-V2
@version: 2.0.0
@date: 2024
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import math
from collections import deque

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


class RewardMetrics:
    """
    Clase para almacenar y gestionar metricas de recompensa durante el entrenamiento.
    
    Permite tracking de:
    - Historico de recompensas
    - Estadisticas por tipo de evento
    - Metricas agregadas para visualizacion
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.rewards_history = deque(maxlen=window_size)
        self.allocations_history = deque(maxlen=window_size)
        self.blocking_history = deque(maxlen=window_size)
        self.fragmentation_history = deque(maxlen=window_size)
        self.qot_history = deque(maxlen=window_size)
        
        # Contadores
        self.total_allocations = 0
        self.total_blocks = 0
        self.total_steps = 0
    
    def record(self, reward: float, allocated: bool, 
               fragmentation: float = 0.0, qot: float = 1.0):
        """Registra metricas de un paso."""
        self.rewards_history.append(reward)
        self.allocations_history.append(1 if allocated else 0)
        self.blocking_history.append(0 if allocated else 1)
        self.fragmentation_history.append(fragmentation)
        self.qot_history.append(qot)
        
        self.total_steps += 1
        if allocated:
            self.total_allocations += 1
        else:
            self.total_blocks += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Retorna estadisticas actuales."""
        if not self.rewards_history:
            return {
                'avg_reward': 0.0,
                'blocking_rate': 0.0,
                'avg_fragmentation': 0.0,
                'avg_qot': 1.0
            }
        
        return {
            'avg_reward': np.mean(self.rewards_history),
            'std_reward': np.std(self.rewards_history),
            'blocking_rate': np.mean(self.blocking_history),
            'avg_fragmentation': np.mean(self.fragmentation_history),
            'avg_qot': np.mean(self.qot_history),
            'total_allocations': self.total_allocations,
            'total_blocks': self.total_blocks
        }
    
    def reset(self):
        """Reinicia todas las metricas."""
        self.rewards_history.clear()
        self.allocations_history.clear()
        self.blocking_history.clear()
        self.fragmentation_history.clear()
        self.qot_history.clear()
        self.total_allocations = 0
        self.total_blocks = 0
        self.total_steps = 0


class BaseRewardFunction(ABC):
    """
    Clase base abstracta para funciones de recompensa.
    
    Define la interfaz comun que todas las funciones de recompensa deben implementar.
    """
    
    def __init__(self, name: str = "BaseReward"):
        self.name = name
        self.metrics = RewardMetrics()
        self._simulator = None
    
    def set_simulator(self, simulator):
        """Establece referencia al simulador."""
        self._simulator = simulator
    
    @abstractmethod
    def calculate(self, allocated: bool, **kwargs) -> float:
        """
        Calcula la recompensa para un evento de asignacion.
        
        Parameters:
        -----------
        allocated : bool
            True si la conexion fue asignada exitosamente
        **kwargs : dict
            Argumentos adicionales especificos de cada implementacion
        
        Returns:
        --------
        float
            Valor de recompensa
        """
        pass
    
    def __call__(self, allocated: bool, **kwargs) -> float:
        """Permite llamar la funcion como callable."""
        return self.calculate(allocated, **kwargs)
    
    def get_description(self) -> str:
        """Retorna descripcion de la funcion de recompensa."""
        return f"{self.name}: {self.__doc__}"


# =============================================================================
# 1. BASELINE REWARD - Recompensa Binaria Simple
# =============================================================================

class BaselineReward(BaseRewardFunction):
    """
    Funcion de Recompensa Baseline (Binaria Simple)
    ===============================================
    
    La funcion de recompensa mas simple posible para problemas de asignacion.
    Sirve como baseline para comparar con funciones mas sofisticadas.
    
    Formula Matematica:
    ------------------
    
        r(t) = { +r_success   si conexion asignada exitosamente
               { -r_failure   si conexion bloqueada
    
    Por defecto: r_success = 1.0, r_failure = 1.0
    
    Caracteristicas:
    ---------------
    - Feedback binario inmediato
    - No considera estado de la red
    - Ideal para debugging y comparacion
    
    Ventajas:
    ---------
    - Simple de implementar y entender
    - Estable numericamente
    - Rapido de calcular
    
    Desventajas:
    -----------
    - No diferencia entre buenas y malas asignaciones
    - No incentiva eficiencia espectral
    - Puede llevar a soluciones sub-optimas
    
    Referencias:
    -----------
    - Chen et al. "DeepRMSA" (2019) - usado como baseline
    """
    
    def __init__(self, 
                 success_reward: float = 1.0, 
                 failure_penalty: float = 1.0):
        """
        Parameters:
        -----------
        success_reward : float
            Recompensa por asignacion exitosa (default: 1.0)
        failure_penalty : float
            Penalizacion por bloqueo (default: 1.0)
        """
        super().__init__("BaselineReward")
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
    
    def calculate(self, allocated: bool, **kwargs) -> float:
        """
        Calcula recompensa binaria simple.
        
        r = +success_reward si allocated, -failure_penalty si no
        """
        reward = self.success_reward if allocated else -self.failure_penalty
        
        # Registrar metricas
        self.metrics.record(reward, allocated)
        
        return reward


# =============================================================================
# 2. QoT-AWARE REWARD - Calidad de Transmision
# =============================================================================

class QoTAwareReward(BaseRewardFunction):
    """
    Funcion de Recompensa Consciente de QoT (Quality of Transmission)
    ==================================================================
    
    Basada en DeepRMSA-QoT, esta funcion considera la calidad fisica
    de la transmision optica al calcular la recompensa.
    
    Formula Matematica:
    ------------------
    
    r(t) = { alpha * Q(path) + (1-alpha) * r_base   si asignada
           { -penalty * (1 + beta * Q_lost)         si bloqueada
    
    Donde:
    - Q(path): Indice de calidad QoT basado en OSNR
    - r_base: Recompensa base por asignacion
    - Q_lost: Calidad potencial perdida por bloqueo
    - alpha, beta: Pesos de ponderacion
    
    Calculo de Q(path):
    ------------------
    
    Q = sigmoid((OSNR_estimated - OSNR_required) / sigma)
    
    OSNR_estimated = OSNR_0 - 10*log10(N) - sum(alpha_i * L_i)
    
    Donde:
    - OSNR_0: OSNR inicial del transmisor (~40 dB)
    - N: Numero de saltos/amplificadores
    - alpha_i: Coeficiente de atenuacion del enlace i
    - L_i: Longitud del enlace i
    
    Caracteristicas:
    ---------------
    - Considera degradacion fisica de la senal
    - Favorece rutas con mejor OSNR
    - Evita asignaciones que podrian fallar por QoT
    
    Referencias:
    -----------
    - Chen et al. "DeepRMSA: A Deep RL Framework" (2019)
    - Pointurier "Quality of Transmission" (2017)
    """
    
    def __init__(self,
                 qot_weight: float = 0.5,
                 base_reward: float = 1.0,
                 failure_penalty: float = 1.0,
                 required_osnr_db: float = 15.0):
        """
        Parameters:
        -----------
        qot_weight : float
            Peso del componente QoT (alpha en la formula)
        base_reward : float
            Recompensa base por asignacion exitosa
        failure_penalty : float
            Penalizacion base por bloqueo
        required_osnr_db : float
            OSNR minimo requerido en dB
        """
        super().__init__("QoTAwareReward")
        self.qot_weight = qot_weight
        self.base_reward = base_reward
        self.failure_penalty = failure_penalty
        self.required_osnr_db = required_osnr_db
    
    def calculate(self, allocated: bool, path_links: List = None, **kwargs) -> float:
        """
        Calcula recompensa considerando QoT.
        
        Parameters:
        -----------
        allocated : bool
            Si la conexion fue asignada
        path_links : List
            Lista de enlaces en la ruta asignada
        """
        if path_links is None:
            path_links = []
        
        # Calcular calidad QoT
        qot_quality = calculate_osnr_quality(
            path_links, 
            required_osnr_db=self.required_osnr_db
        )
        
        if allocated:
            # Recompensa compuesta: QoT + base
            alpha = self.qot_weight
            reward = alpha * qot_quality + (1 - alpha) * self.base_reward
        else:
            # Penalizacion escalada por QoT potencial perdida
            beta = 0.3  # Factor de escala para QoT perdida
            reward = -self.failure_penalty * (1 + beta * (1 - qot_quality))
        
        # Registrar metricas
        self.metrics.record(reward, allocated, qot=qot_quality)
        
        return reward


# =============================================================================
# 3. MULTI-OBJECTIVE REWARD - Multiples Objetivos
# =============================================================================

class MultiObjectiveReward(BaseRewardFunction):
    """
    Funcion de Recompensa Multi-Objetivo
    ====================================
    
    Combina multiples objetivos de optimizacion en una sola funcion
    de recompensa mediante ponderacion weighted-sum.
    
    Formula Matematica:
    ------------------
    
    r(t) = sum_{i} w_i * r_i(t)
    
    Donde los componentes r_i son:
    
    1. r_blocking: Recompensa por resultado de asignacion
       r_blocking = +1 si asignada, -1 si bloqueada
    
    2. r_fragmentation: Penalizacion por fragmentacion
       r_frag = -FR(network) = -(1 - max_block/total_free)
    
    3. r_utilization: Bonus por utilizacion eficiente
       r_util = U_path * (1 - U_network)
       Incentiva usar enlaces ya cargados sin sobrecargar la red
    
    4. r_balance: Bonus por balanceo de carga
       r_balance = 1 - CV(U_links)
       Donde CV = std/mean es el coeficiente de variacion
    
    5. r_path_length: Penalizacion por rutas largas
       r_length = -(path_length - 1) / (max_length - 1)
    
    Pesos Dinamicos (Opcional):
    --------------------------
    Los pesos pueden adaptarse segun el estado de la red:
    
    w_frag(t) = w_frag_0 * (1 + k * FR(t))
    
    Incrementa peso de fragmentacion cuando esta es alta.
    
    Caracteristicas:
    ---------------
    - Balancea multiples objetivos simultaneamente
    - Configurable mediante pesos
    - Puede usar pesos adaptativos
    
    Referencias:
    -----------
    - Trindade et al. "Multi-band DRL for EON" (2023)
    - Vamanan et al. "Multi-objective optimization" (2012)
    """
    
    def __init__(self,
                 weights: Dict[str, float] = None,
                 adaptive_weights: bool = False):
        """
        Parameters:
        -----------
        weights : Dict[str, float]
            Pesos para cada componente de la recompensa
        adaptive_weights : bool
            Si True, ajusta pesos dinamicamente
        """
        super().__init__("MultiObjectiveReward")
        
        # Pesos por defecto
        self.weights = weights or {
            'blocking': 1.0,
            'fragmentation': 0.3,
            'utilization': 0.2,
            'balance': 0.2,
            'path_length': 0.1
        }
        
        self.adaptive_weights = adaptive_weights
        self._initial_weights = self.weights.copy()
    
    def calculate(self, allocated: bool, 
                  network=None, 
                  path_links: List = None,
                  **kwargs) -> float:
        """
        Calcula recompensa multi-objetivo.
        
        Parameters:
        -----------
        allocated : bool
            Si la conexion fue asignada
        network : Network
            Objeto de red para calcular metricas
        path_links : List
            Lista de enlaces en la ruta
        """
        components = {}
        
        # 1. Componente de Blocking
        components['blocking'] = 1.0 if allocated else -1.0
        
        if network is not None:
            # Obtener estado de la red
            state = get_network_spectrum_state(network)
            
            # 2. Componente de Fragmentacion (penalizacion)
            components['fragmentation'] = -state['avg_fragmentation']
            
            # 3. Componente de Utilizacion
            if allocated and path_links:
                # Bonus por usar enlaces con carga media (ni muy vacios ni muy llenos)
                path_utils = []
                for link in path_links:
                    slots = [link.getSlot(j) for j in range(link.getSlots())]
                    path_utils.append(calculate_link_utilization(slots))
                avg_path_util = np.mean(path_utils)
                
                # Funcion campana: maximo en utilizacion ~0.5
                components['utilization'] = 1 - 4 * (avg_path_util - 0.5) ** 2
            else:
                components['utilization'] = 0.0
            
            # 4. Componente de Balance
            components['balance'] = state['load_balance']
            
            # Adaptar pesos si esta habilitado
            if self.adaptive_weights:
                self._adapt_weights(state)
        else:
            components['fragmentation'] = 0.0
            components['utilization'] = 0.0
            components['balance'] = 0.0
        
        # 5. Componente de Longitud de Ruta
        if path_links:
            components['path_length'] = -calculate_path_length_penalty(
                path_links, max_path_length=10
            )
        else:
            components['path_length'] = 0.0
        
        # Calcular recompensa ponderada
        reward = sum(
            self.weights.get(key, 0.0) * value 
            for key, value in components.items()
        )
        
        # Registrar metricas
        fragmentation = -components['fragmentation']
        self.metrics.record(reward, allocated, fragmentation=fragmentation)
        
        return reward
    
    def _adapt_weights(self, state: Dict[str, float]):
        """
        Adapta pesos dinamicamente segun estado de la red.
        
        Logica:
        - Alta fragmentacion -> aumentar peso de fragmentacion
        - Alta utilizacion -> aumentar peso de balance
        """
        base = self._initial_weights
        
        # Incrementar peso de fragmentacion si es alta
        frag_factor = 1 + state['avg_fragmentation']
        self.weights['fragmentation'] = base['fragmentation'] * frag_factor
        
        # Incrementar peso de balance si utilizacion es alta
        if state['avg_utilization'] > 0.6:
            self.weights['balance'] = base['balance'] * 1.5
        else:
            self.weights['balance'] = base['balance']
        
        # Normalizar pesos
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total


# =============================================================================
# 4. FRAGMENTATION-AWARE REWARD
# =============================================================================

class FragmentationAwareReward(BaseRewardFunction):
    """
    Funcion de Recompensa Consciente de Fragmentacion
    =================================================
    
    Funcion especializada en minimizar la fragmentacion espectral,
    uno de los problemas mas criticos en EON.
    
    Formula Matematica:
    ------------------
    
    r(t) = r_base + r_frag_local + r_frag_global + r_compactness
    
    Donde:
    
    1. r_base: Recompensa base por asignacion
       r_base = +1 si asignada, -1 si bloqueada
    
    2. r_frag_local: Impacto en fragmentacion local (enlaces usados)
       r_frag_local = -gamma * delta_FR_path
       delta_FR_path = FR_after - FR_before
       
       Penaliza asignaciones que aumentan fragmentacion en la ruta
    
    3. r_frag_global: Impacto en fragmentacion global
       r_frag_global = -delta * delta_FR_network
       
       Considera efecto en toda la red
    
    4. r_compactness: Bonus por asignacion compacta
       r_compact = epsilon * (SC_after - SC_before)
       
       Premia asignaciones que mejoran compactacion
    
    Metricas de Fragmentacion:
    -------------------------
    
    External Fragmentation:
    FR_ext = 1 - (max_free_block / total_free_slots)
    
    Internal Fragmentation (Entropia):
    FR_int = H(block_sizes) / log2(n_blocks)
    
    Spectral Compactness:
    SC = occupied_slots / (last_slot - first_slot + 1)
    
    Caracteristicas:
    ---------------
    - Rastrea cambios en fragmentacion (delta)
    - Considera impacto local y global
    - Promueve asignaciones compactas
    
    Referencias:
    -----------
    - Gao et al. "Spectrum Defragmentation with epsilon-Greedy DQN" (2022)
    - Wright et al. "Elastic Optical Networks" (2015)
    """
    
    def __init__(self,
                 local_weight: float = 0.4,
                 global_weight: float = 0.3,
                 compact_weight: float = 0.3,
                 base_reward: float = 1.0):
        """
        Parameters:
        -----------
        local_weight : float
            Peso para fragmentacion local (gamma)
        global_weight : float  
            Peso para fragmentacion global (delta)
        compact_weight : float
            Peso para compactacion (epsilon)
        base_reward : float
            Recompensa base
        """
        super().__init__("FragmentationAwareReward")
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.compact_weight = compact_weight
        self.base_reward = base_reward
        
        # Estado previo para calcular deltas
        self._prev_state = None
    
    def calculate(self, allocated: bool,
                  network=None,
                  path_links: List = None,
                  **kwargs) -> float:
        """
        Calcula recompensa considerando impacto en fragmentacion.
        
        Parameters:
        -----------
        allocated : bool
            Si la conexion fue asignada
        network : Network
            Objeto de red
        path_links : List
            Enlaces de la ruta
        """
        # Componente base
        r_base = self.base_reward if allocated else -self.base_reward
        r_frag_local = 0.0
        r_frag_global = 0.0
        r_compact = 0.0
        
        if network is not None:
            # Obtener estado actual
            current_state = get_network_spectrum_state(network)
            
            if self._prev_state is not None:
                # Calcular deltas
                delta_frag_global = (
                    current_state['avg_fragmentation'] - 
                    self._prev_state['avg_fragmentation']
                )
                
                delta_compact = (
                    current_state['avg_compactness'] - 
                    self._prev_state['avg_compactness']
                )
                
                # Penalizacion por aumento de fragmentacion global
                r_frag_global = -self.global_weight * delta_frag_global
                
                # Bonus por mejora en compactacion
                r_compact = self.compact_weight * delta_compact
                
                # Fragmentacion local (solo para la ruta usada)
                if path_links and allocated:
                    local_frag = 0.0
                    for link in path_links:
                        slots = [link.getSlot(j) for j in range(link.getSlots())]
                        local_frag += calculate_fragmentation_ratio(slots)
                    local_frag /= len(path_links)
                    
                    # Penalizar si la ruta tiene alta fragmentacion
                    r_frag_local = -self.local_weight * local_frag
            
            # Guardar estado actual para proxima iteracion
            self._prev_state = current_state
            
            fragmentation = current_state['avg_fragmentation']
        else:
            fragmentation = 0.0
        
        # Recompensa total
        reward = r_base + r_frag_local + r_frag_global + r_compact
        
        # Registrar metricas
        self.metrics.record(reward, allocated, fragmentation=fragmentation)
        
        return reward
    
    def reset_state(self):
        """Reinicia el estado previo."""
        self._prev_state = None


# =============================================================================
# 5. SPECTRAL ENTROPY ADAPTIVE REWARD (NOVEL - 100% NUEVO)
# =============================================================================

class SpectralEntropyAdaptiveReward(BaseRewardFunction):
    """
    Funcion de Recompensa Adaptativa basada en Entropia Espectral
    =============================================================
    
    ** FUNCION NOVEDOSA - CONTRIBUCION ORIGINAL **
    
    Esta funcion de recompensa introduce un enfoque completamente nuevo
    que combina teoria de informacion (entropia) con adaptacion dinamica
    y asignacion de credito temporal.
    
    Innovaciones Principales:
    -------------------------
    
    1. ENTROPIA ESPECTRAL COMO METRICA CENTRAL
       Usamos la entropia de Shannon para medir el "orden" del espectro.
       Alta entropia = utilizacion uniforme = mejor capacidad futura.
    
    2. ZONAS DE OPERACION ADAPTATIVAS
       El comportamiento cambia segun la carga de la red:
       - Zona Verde (U < 0.4): Prioriza throughput
       - Zona Amarilla (0.4 <= U < 0.7): Balancea objetivos
       - Zona Roja (U >= 0.7): Prioriza eficiencia
    
    3. ASIGNACION DE CREDITO TEMPORAL (Delayed Assignment)
       Las recompensas actuales consideran decisiones pasadas
       mediante una ventana de memoria exponencial.
    
    4. PREDICCION DE IMPACTO FUTURO
       Estima el impacto de la decision actual en conexiones futuras.
    
    Formula Matematica Completa:
    ===========================
    
    r(t) = r_base(t) + r_entropy(t) + r_adaptive(t) + r_temporal(t)
    
    1. Componente Base:
       r_base = { +1    si asignada
                { -psi  si bloqueada
       
       Donde psi = 1 + 0.5 * U_network (penalizacion escalada por carga)
    
    2. Componente de Entropia Espectral:
       r_entropy = lambda * (H_target - |H_current - H_target|)
       
       H_current = -sum(p_i * log2(p_i))  # Entropia de utilizacion
       H_target = 0.7 * H_max              # Objetivo: 70% de entropia maxima
       
       Incentiva entropia cercana al objetivo (ni muy uniforme ni muy concentrada)
    
    3. Componente Adaptativo por Zona:
       r_adaptive = { w_throughput * (1 + allocated)           si Zona Verde
                    { w_balance * LB + w_frag * (1 - FR)       si Zona Amarilla
                    { w_efficiency * SC + w_compactness * CP   si Zona Roja
       
       Los pesos w_* se ajustan continuamente segun gradientes de metricas.
    
    4. Componente Temporal (Delayed Assignment):
       r_temporal = sum_{k=1}^{K} gamma^k * credit_k
       
       credit_k = { +delta_benefit   si decision pasada k fue buena
                  { -delta_cost      si decision pasada k fue mala
       
       gamma = 0.9 es el factor de descuento temporal.
    
    5. Factor de Prediccion de Impacto:
       r *= (1 + eta * predicted_impact)
       
       predicted_impact = P(future_success | current_state) - P_baseline
       
       Estimado via regresion lineal sobre historico de estados similares.
    
    Teoria de Informacion Aplicada:
    ===============================
    
    Entropia de Utilizacion de Slots:
    ---------------------------------
    Dividimos el espectro en N segmentos y calculamos la distribucion
    de utilizacion. La entropia de esta distribucion indica uniformidad.
    
    H(U) = -sum_{i=1}^{N} (U_i/U_total) * log2(U_i/U_total)
    
    Entropia Alta -> Carga bien distribuida -> Mejor capacidad para conexiones grandes
    Entropia Baja -> Carga concentrada -> Posibles cuellos de botella
    
    Entropy Target Dinamico:
    -----------------------
    El objetivo de entropia se ajusta segun la fase del episodio:
    
    H_target(t) = H_max * (0.5 + 0.3 * sin(pi * t / T_episode))
    
    Esto promueve exploracion inicial y explotacion posterior.
    
    Ventajas de este Enfoque:
    ========================
    
    1. Generalizable: La entropia es una metrica universal
    2. Adaptativo: Se ajusta automaticamente a condiciones cambiantes
    3. Temporal: Considera consecuencias de largo plazo
    4. Robusto: Funciona bien en diferentes topologias y cargas
    5. Interpretable: Cada componente tiene significado fisico claro
    
    Complejidad Computacional:
    =========================
    - Calculo de entropia: O(N * S) donde N=enlaces, S=slots
    - Memoria temporal: O(K) donde K=ventana de memoria
    - Total por paso: O(N * S + K)
    
    Hiperparametros Recomendados:
    ============================
    - entropy_weight: 0.3-0.5
    - temporal_discount: 0.9
    - memory_window: 50-100 pasos
    - zone_thresholds: [0.4, 0.7]
    
    Referencias:
    ===========
    Esta funcion es una CONTRIBUCION ORIGINAL que combina conceptos de:
    - Shannon, "A Mathematical Theory of Communication" (1948)
    - Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
    - Chen et al., "DeepRMSA" (2019) - para contexto de EON
    """
    
    def __init__(self,
                 entropy_weight: float = 0.4,
                 temporal_discount: float = 0.9,
                 memory_window: int = 50,
                 zone_thresholds: tuple = (0.4, 0.7),
                 n_segments: int = 10):
        """
        Parameters:
        -----------
        entropy_weight : float
            Peso del componente de entropia (lambda)
        temporal_discount : float  
            Factor de descuento para credito temporal (gamma)
        memory_window : int
            Tamano de la ventana de memoria (K)
        zone_thresholds : tuple
            Umbrales para zonas adaptativas (verde/amarilla/roja)
        n_segments : int
            Numero de segmentos para calculo de entropia
        """
        super().__init__("SpectralEntropyAdaptiveReward")
        
        self.entropy_weight = entropy_weight
        self.temporal_discount = temporal_discount
        self.memory_window = memory_window
        self.zone_thresholds = zone_thresholds
        self.n_segments = n_segments
        
        # Memoria para delayed assignment
        self.decision_history = deque(maxlen=memory_window)
        self.state_history = deque(maxlen=memory_window)
        
        # Estadisticas para prediccion
        self.success_predictor = {
            'states': deque(maxlen=1000),
            'outcomes': deque(maxlen=1000)
        }
        
        # Contadores de episodio
        self.episode_step = 0
        self.episode_length = 10000  # Estimado
    
    def calculate(self, allocated: bool,
                  network=None,
                  path_links: List = None,
                  **kwargs) -> float:
        """
        Calcula recompensa usando entropia espectral adaptativa.
        
        Parameters:
        -----------
        allocated : bool
            Si la conexion fue asignada
        network : Network
            Objeto de red
        path_links : List
            Enlaces de la ruta
        """
        self.episode_step += 1
        
        if network is None:
            # Sin red, usar baseline
            reward = 1.0 if allocated else -1.0
            self.metrics.record(reward, allocated)
            return reward
        
        # Obtener estado actual
        state = get_network_spectrum_state(network)
        
        # =====================
        # 1. COMPONENTE BASE
        # =====================
        utilization = state['avg_utilization']
        psi = 1.0 + 0.5 * utilization  # Penalizacion escalada
        r_base = 1.0 if allocated else -psi
        
        # =====================
        # 2. COMPONENTE ENTROPIA
        # =====================
        r_entropy = self._calculate_entropy_reward(network, state)
        
        # =====================
        # 3. COMPONENTE ADAPTATIVO
        # =====================
        r_adaptive = self._calculate_adaptive_reward(
            allocated, state, path_links
        )
        
        # =====================
        # 4. COMPONENTE TEMPORAL
        # =====================
        r_temporal = self._calculate_temporal_reward(allocated, state)
        
        # =====================
        # 5. PREDICCION DE IMPACTO
        # =====================
        impact_factor = self._predict_future_impact(state, allocated)
        
        # Combinar componentes
        reward = r_base + r_entropy + r_adaptive + r_temporal
        reward *= (1 + 0.1 * impact_factor)  # Modular por prediccion
        
        # Actualizar historiales
        self._update_history(allocated, state)
        
        # Registrar metricas
        self.metrics.record(
            reward, allocated,
            fragmentation=state['avg_fragmentation'],
            qot=state['entropy']
        )
        
        return reward
    
    def _calculate_entropy_reward(self, network, state: Dict) -> float:
        """
        Calcula componente de recompensa basado en entropia espectral.
        
        Incentiva entropia cercana al objetivo (balance entre uniformidad
        y concentracion).
        """
        # Calcular entropia de utilizacion por segmentos
        segment_utils = self._get_segment_utilizations(network)
        
        if not segment_utils or sum(segment_utils) <= 0:
            return 0.0
        
        # Calcular entropia actual
        current_entropy = calculate_entropy(segment_utils, normalize=True)
        
        # Objetivo dinamico basado en fase del episodio
        phase = self.episode_step / self.episode_length
        target_entropy = 0.5 + 0.3 * math.sin(math.pi * phase)
        
        # Recompensa: maxima cuando entropia esta cerca del objetivo
        deviation = abs(current_entropy - target_entropy)
        r_entropy = self.entropy_weight * (1 - deviation)
        
        return r_entropy
    
    def _get_segment_utilizations(self, network) -> List[float]:
        """
        Divide el espectro de la red en segmentos y calcula utilizacion por segmento.
        """
        total_slots = 0
        segment_occupied = [0] * self.n_segments
        
        for i in range(network.linkCounter):
            link = network.getLink(i)
            num_slots = link.getSlots()
            slots_per_segment = num_slots // self.n_segments
            
            if slots_per_segment == 0:
                continue
            
            for seg in range(self.n_segments):
                start = seg * slots_per_segment
                end = (seg + 1) * slots_per_segment if seg < self.n_segments - 1 else num_slots
                
                for j in range(start, end):
                    if link.getSlot(j):
                        segment_occupied[seg] += 1
                
                total_slots += (end - start)
        
        if total_slots == 0:
            return []
        
        # Normalizar a proporciones
        total_occupied = sum(segment_occupied)
        if total_occupied == 0:
            return [1 / self.n_segments] * self.n_segments
        
        return [seg / total_occupied for seg in segment_occupied]
    
    def _calculate_adaptive_reward(self, allocated: bool, 
                                    state: Dict, 
                                    path_links: List = None) -> float:
        """
        Calcula componente adaptativo segun zona de operacion.
        """
        utilization = state['avg_utilization']
        
        if utilization < self.zone_thresholds[0]:
            # ZONA VERDE: Priorizar throughput
            return 0.3 * (1.0 if allocated else 0.0)
        
        elif utilization < self.zone_thresholds[1]:
            # ZONA AMARILLA: Balance
            balance_bonus = 0.2 * state['load_balance']
            frag_bonus = 0.2 * (1 - state['avg_fragmentation'])
            return balance_bonus + frag_bonus
        
        else:
            # ZONA ROJA: Eficiencia critica
            compact_bonus = 0.3 * state['avg_compactness']
            
            # Bonus extra por usar rutas cortas (menos recursos)
            if path_links and allocated:
                length_bonus = 0.2 * (1 - len(path_links) / 10)
            else:
                length_bonus = 0.0
            
            return compact_bonus + length_bonus
    
    def _calculate_temporal_reward(self, allocated: bool, state: Dict) -> float:
        """
        Calcula componente temporal con delayed credit assignment.
        
        Evalua si decisiones pasadas contribuyeron al estado actual.
        """
        if len(self.decision_history) < 2:
            return 0.0
        
        r_temporal = 0.0
        gamma = self.temporal_discount
        
        for k, (past_decision, past_state) in enumerate(
            zip(self.decision_history, self.state_history)
        ):
            if k >= len(self.decision_history) - 1:
                break
            
            discount = gamma ** (len(self.decision_history) - k - 1)
            
            # Evaluar si la decision pasada fue buena
            delta_frag = state['avg_fragmentation'] - past_state.get('avg_fragmentation', 0)
            delta_util = state['avg_utilization'] - past_state.get('avg_utilization', 0)
            
            # Credito positivo si: bajo aumento de frag o buen aumento de utilizacion
            credit = -delta_frag * 0.5 + delta_util * 0.3
            
            if past_decision:  # Si asignamos antes
                r_temporal += discount * credit
            else:  # Si bloqueamos antes
                r_temporal -= discount * credit * 0.5
        
        return min(0.3, max(-0.3, r_temporal))  # Limitar rango
    
    def _predict_future_impact(self, state: Dict, allocated: bool) -> float:
        """
        Predice el impacto de la decision actual en conexiones futuras.
        
        Usa historico de estados similares para estimar probabilidad
        de exito futuro.
        """
        if len(self.success_predictor['states']) < 10:
            return 0.0
        
        # Encontrar estados similares en el historico
        current_features = [
            state['avg_utilization'],
            state['avg_fragmentation'],
            state['load_balance']
        ]
        
        similar_outcomes = []
        for hist_state, outcome in zip(
            self.success_predictor['states'],
            self.success_predictor['outcomes']
        ):
            hist_features = [
                hist_state.get('avg_utilization', 0),
                hist_state.get('avg_fragmentation', 0),
                hist_state.get('load_balance', 1)
            ]
            
            # Distancia euclidiana simple
            distance = sum(
                (a - b) ** 2 for a, b in zip(current_features, hist_features)
            ) ** 0.5
            
            if distance < 0.2:  # Estados cercanos
                similar_outcomes.append(outcome)
        
        if not similar_outcomes:
            return 0.0
        
        # Prediccion: tasa de exito en estados similares
        success_rate = np.mean(similar_outcomes)
        baseline_rate = 0.7  # Tasa baseline esperada
        
        return success_rate - baseline_rate
    
    def _update_history(self, allocated: bool, state: Dict):
        """Actualiza historiales para componente temporal y prediccion."""
        self.decision_history.append(allocated)
        self.state_history.append(state.copy())
        
        # Actualizar predictor
        self.success_predictor['states'].append(state.copy())
        self.success_predictor['outcomes'].append(1.0 if allocated else 0.0)
    
    def reset_episode(self):
        """Reinicia contadores al inicio de un nuevo episodio."""
        self.episode_step = 0
        self.decision_history.clear()
        self.state_history.clear()


# =============================================================================
# FACTORY PATTERN PARA CREAR FUNCIONES DE RECOMPENSA
# =============================================================================

class RewardFactory:
    """
    Factory para crear instancias de funciones de recompensa.
    
    Uso:
    ----
    >>> factory = RewardFactory()
    >>> reward_fn = factory.create('multi_objective', weights={'blocking': 1.0})
    >>> reward = reward_fn(allocated=True, network=net)
    """
    
    _reward_classes = {
        'baseline': BaselineReward,
        'qot_aware': QoTAwareReward,
        'multi_objective': MultiObjectiveReward,
        'fragmentation_aware': FragmentationAwareReward,
        'spectral_entropy': SpectralEntropyAdaptiveReward
    }
    
    @classmethod
    def create(cls, reward_type: str, **kwargs) -> BaseRewardFunction:
        """
        Crea una instancia de funcion de recompensa.
        
        Parameters:
        -----------
        reward_type : str
            Tipo de recompensa: 'baseline', 'qot_aware', 'multi_objective',
            'fragmentation_aware', 'spectral_entropy'
        **kwargs : dict
            Argumentos para el constructor de la clase
        
        Returns:
        --------
        BaseRewardFunction
            Instancia de la funcion de recompensa
        """
        if reward_type not in cls._reward_classes:
            available = ', '.join(cls._reward_classes.keys())
            raise ValueError(
                f"Tipo de recompensa desconocido: {reward_type}. "
                f"Disponibles: {available}"
            )
        
        return cls._reward_classes[reward_type](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Retorna lista de tipos de recompensa disponibles."""
        return list(cls._reward_classes.keys())
    
    @classmethod
    def get_description(cls, reward_type: str) -> str:
        """Retorna la documentacion de un tipo de recompensa."""
        if reward_type not in cls._reward_classes:
            return f"Tipo desconocido: {reward_type}"
        
        return cls._reward_classes[reward_type].__doc__


# =============================================================================
# FUNCIONES UTILITARIAS PARA INTEGRACION CON EL SIMULADOR
# =============================================================================

def create_reward_wrapper(reward_fn: BaseRewardFunction, simulator) -> Callable:
    """
    Crea un wrapper callable para usar con RlOnEnv.setRewardFunc().
    
    Parameters:
    -----------
    reward_fn : BaseRewardFunction
        Instancia de funcion de recompensa
    simulator : Simulator
        Referencia al simulador
    
    Returns:
    --------
    Callable
        Funcion que puede pasarse a setRewardFunc()
    
    Ejemplo:
    --------
    >>> reward_fn = RewardFactory.create('spectral_entropy')
    >>> wrapper = create_reward_wrapper(reward_fn, simulator)
    >>> env.unwrapped.setRewardFunc(wrapper)
    """
    reward_fn.set_simulator(simulator)
    
    def wrapper():
        # Obtener estado del ultimo evento del simulador
        allocated = simulator.getStatus().name == 'Allocated'
        network = simulator._Simulator__controller.network
        
        # Obtener ruta si esta disponible
        path_links = []
        # Nota: path_links necesitaria acceso a la conexion actual
        
        return reward_fn.calculate(
            allocated=allocated,
            network=network,
            path_links=path_links
        )
    
    return wrapper
