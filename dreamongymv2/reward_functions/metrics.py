# -*- coding: utf-8 -*-
"""
DREAM-ON-GYM-V2: Módulo de Métricas para Funciones de Recompensa
================================================================

Este módulo contiene funciones utilitarias para calcular métricas de red
utilizadas en las funciones de recompensa avanzadas.

Métricas Implementadas:
-----------------------
1. Fragmentación Espectral (External, Internal, Shannon Entropy)
2. Compactación Espectral
3. Utilización de Enlaces y Red
4. Calidad de Transmisión (QoT) basada en OSNR
5. Factor de Balanceo de Carga
6. Penalización por Longitud de Ruta

Referencias:
-----------
- Chen et al. "DeepRMSA: A Deep Reinforcement Learning Framework for RMSA" (2019)
- Trindade et al. "Multi-band DRL for Elastic Optical Networks" (2023)
- Gao et al. "Spectrum Defragmentation with ε-Greedy DQN" (2022)

@author: Generado con AI para DREAM-ON-GYM-V2
@version: 2.0.0
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import math


def calculate_fragmentation_ratio(
    link_slots: List[bool], 
    method: str = 'external'
) -> float:
    """
    Calcula el ratio de fragmentación espectral de un enlace.
    
    La fragmentación espectral es un problema crítico en EON que reduce
    la capacidad efectiva de la red al crear "huecos" inutilizables.
    
    Métodos disponibles:
    -------------------
    
    1. EXTERNAL FRAGMENTATION (Método por defecto):
       Mide la fragmentación causada por bloques libres pequeños.
       
       FR_ext = 1 - (max_free_block / total_free_slots)
       
       Donde:
       - max_free_block: El bloque contiguo de slots libres más grande
       - total_free_slots: Número total de slots libres
    
    2. INTERNAL FRAGMENTATION:
       Basado en la entropía de Shannon de la distribución de bloques.
       
       FR_int = 1 - (1 / (1 + H(p)))
       
       Donde H(p) = -Σ p_i * log2(p_i) es la entropía de Shannon
       y p_i = block_size_i / total_free_slots
    
    3. AVERAGE BLOCK SIZE:
       Normalizado respecto al tamaño máximo posible.
       
       FR_abs = 1 - (avg_block_size / num_slots)
    
    Parameters:
    -----------
    link_slots : List[bool]
        Array de ocupación del enlace (True=ocupado, False=libre)
    method : str
        Método de cálculo: 'external', 'internal', 'average'
    
    Returns:
    --------
    float
        Ratio de fragmentación en rango [0, 1]
        0 = Sin fragmentación (todo contiguo)
        1 = Máxima fragmentación
    
    Example:
    --------
    >>> slots = [False, True, False, False, True, False]
    >>> calculate_fragmentation_ratio(slots, 'external')
    0.5  # max_block=2, total_free=4, FR=1-2/4=0.5
    """
    if not link_slots or all(link_slots):
        return 0.0  # No hay slots libres, sin fragmentación calculable
    
    if not any(link_slots):
        return 0.0  # Todos libres, sin fragmentación
    
    # Encontrar todos los bloques libres contiguos
    free_blocks = []
    current_block = 0
    
    for slot in link_slots:
        if not slot:  # Slot libre
            current_block += 1
        else:  # Slot ocupado
            if current_block > 0:
                free_blocks.append(current_block)
                current_block = 0
    
    # Agregar último bloque si existe
    if current_block > 0:
        free_blocks.append(current_block)
    
    if not free_blocks:
        return 0.0
    
    total_free = sum(free_blocks)
    max_block = max(free_blocks)
    
    if method == 'external':
        # FR_ext = 1 - (max_free_block / total_free_slots)
        return 1.0 - (max_block / total_free) if total_free > 0 else 0.0
    
    elif method == 'internal':
        # Entropía de Shannon
        if len(free_blocks) <= 1:
            return 0.0
        
        probabilities = [block / total_free for block in free_blocks]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        # Normalizar: máxima entropía cuando todos los bloques son iguales
        max_entropy = math.log2(len(free_blocks))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    elif method == 'average':
        # Basado en tamaño promedio de bloque
        avg_block = total_free / len(free_blocks)
        return 1.0 - (avg_block / len(link_slots))
    
    else:
        raise ValueError(f"Método no reconocido: {method}")


def calculate_entropy(
    distribution: List[float],
    normalize: bool = True
) -> float:
    """
    Calcula la entropía de Shannon de una distribución de probabilidad.
    
    La entropía mide el desorden o incertidumbre en un sistema.
    En el contexto de EON, se usa para evaluar la uniformidad de
    la utilización de recursos espectrales.
    
    Fórmula:
    --------
    H(X) = -Σ p(x_i) * log2(p(x_i))
    
    Entropía Normalizada:
    H_norm = H(X) / log2(n)  ∈ [0, 1]
    
    Parameters:
    -----------
    distribution : List[float]
        Distribución de probabilidad (debe sumar 1 o será normalizada)
    normalize : bool
        Si True, normaliza la entropía al rango [0, 1]
    
    Returns:
    --------
    float
        Valor de entropía (normalizada si se especifica)
    
    Example:
    --------
    >>> # Distribución uniforme (máxima entropía)
    >>> calculate_entropy([0.25, 0.25, 0.25, 0.25])
    1.0
    
    >>> # Distribución concentrada (mínima entropía)
    >>> calculate_entropy([1.0, 0.0, 0.0, 0.0])
    0.0
    """
    if not distribution:
        return 0.0
    
    # Normalizar si no suma 1
    total = sum(distribution)
    if total <= 0:
        return 0.0
    
    probs = [p / total for p in distribution]
    
    # Calcular entropía
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    
    if normalize and len(probs) > 1:
        max_entropy = math.log2(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    return entropy


def calculate_spectral_compactness(link_slots: List[bool]) -> float:
    """
    Calcula la compactación espectral de un enlace.
    
    La compactación espectral mide qué tan "juntos" están los slots
    ocupados. Un valor alto indica mejor aprovechamiento del espectro.
    
    Fórmula:
    --------
    SC = slots_ocupados / (last_occupied - first_occupied + 1)
    
    Donde:
    - slots_ocupados: Número total de slots en uso
    - first_occupied: Índice del primer slot ocupado
    - last_occupied: Índice del último slot ocupado
    
    Interpretación:
    - SC = 1.0: Todos los slots ocupados son contiguos (óptimo)
    - SC < 1.0: Hay fragmentación entre slots ocupados
    
    Parameters:
    -----------
    link_slots : List[bool]
        Array de ocupación del enlace (True=ocupado, False=libre)
    
    Returns:
    --------
    float
        Índice de compactación en rango [0, 1]
    
    Example:
    --------
    >>> slots = [True, True, False, True, True]
    >>> calculate_spectral_compactness(slots)
    0.8  # 4 ocupados / (4-0+1) = 4/5 = 0.8
    """
    if not link_slots or not any(link_slots):
        return 1.0  # Sin ocupación, consideramos compacto por defecto
    
    occupied_indices = [i for i, slot in enumerate(link_slots) if slot]
    
    if not occupied_indices:
        return 1.0
    
    first_occupied = min(occupied_indices)
    last_occupied = max(occupied_indices)
    span = last_occupied - first_occupied + 1
    
    return len(occupied_indices) / span if span > 0 else 1.0


def calculate_link_utilization(link_slots: List[bool]) -> float:
    """
    Calcula la utilización de un enlace.
    
    Fórmula:
    --------
    U_link = slots_ocupados / total_slots
    
    Parameters:
    -----------
    link_slots : List[bool]
        Array de ocupación del enlace
    
    Returns:
    --------
    float
        Utilización del enlace en rango [0, 1]
    """
    if not link_slots:
        return 0.0
    
    occupied = sum(1 for slot in link_slots if slot)
    return occupied / len(link_slots)


def calculate_network_utilization(network) -> float:
    """
    Calcula la utilización promedio de toda la red.
    
    Fórmula:
    --------
    U_network = (1/N) * Σ U_link_i
    
    Parameters:
    -----------
    network : Network
        Objeto Network del simulador
    
    Returns:
    --------
    float
        Utilización promedio de la red en rango [0, 1]
    """
    total_utilization = 0.0
    num_links = network.linkCounter
    
    if num_links == 0:
        return 0.0
    
    for i in range(num_links):
        link = network.getLink(i)
        num_slots = link.getSlots()
        occupied = sum(1 for j in range(num_slots) if link.getSlot(j))
        total_utilization += occupied / num_slots if num_slots > 0 else 0.0
    
    return total_utilization / num_links


def calculate_path_length_penalty(
    path_links: List,
    max_path_length: int = 10
) -> float:
    """
    Calcula la penalización por longitud de ruta.
    
    Rutas más largas consumen más recursos y tienen mayor probabilidad
    de degradación de señal. Esta penalización incentiva rutas cortas.
    
    Fórmula:
    --------
    P_length = (path_length - 1) / (max_path_length - 1)
    
    Normalizado a [0, 1]:
    - 0: Ruta de 1 salto (óptimo)
    - 1: Ruta de longitud máxima
    
    Parameters:
    -----------
    path_links : List
        Lista de enlaces en la ruta
    max_path_length : int
        Longitud máxima esperada de ruta
    
    Returns:
    --------
    float
        Penalización normalizada en [0, 1]
    """
    path_length = len(path_links)
    
    if path_length <= 1:
        return 0.0
    
    if max_path_length <= 1:
        return 0.0
    
    return min(1.0, (path_length - 1) / (max_path_length - 1))


def calculate_osnr_quality(
    path_links: List,
    required_osnr_db: float = 15.0,
    amplifier_noise_db: float = 5.0
) -> float:
    """
    Calcula la calidad de transmisión basada en OSNR (Optical Signal-to-Noise Ratio).
    
    El OSNR es crítico en redes ópticas para determinar la calidad
    de la señal transmitida. Esta función estima el OSNR basándose
    en la longitud de la ruta y las propiedades físicas simplificadas.
    
    Modelo Simplificado:
    -------------------
    OSNR_total = OSNR_initial - 10*log10(N_spans) - α*L_total
    
    Donde:
    - OSNR_initial: OSNR inicial del transmisor (típicamente 40 dB)
    - N_spans: Número de saltos/amplificadores
    - α: Coeficiente de atenuación (típicamente 0.2 dB/km)
    - L_total: Longitud total de la ruta
    
    Calidad Normalizada:
    -------------------
    Q = sigmoid((OSNR_estimated - OSNR_required) / 10)
    
    Parameters:
    -----------
    path_links : List
        Lista de enlaces en la ruta
    required_osnr_db : float
        OSNR mínimo requerido en dB
    amplifier_noise_db : float
        Ruido añadido por cada amplificador en dB
    
    Returns:
    --------
    float
        Índice de calidad QoT en rango [0, 1]
        - 1.0: Excelente calidad (alto margen sobre OSNR requerido)
        - 0.5: Exactamente en el umbral
        - 0.0: Mala calidad (por debajo del umbral)
    """
    if not path_links:
        return 1.0
    
    # Parámetros del modelo
    OSNR_INITIAL_DB = 40.0  # dB, típico para transmisores modernos
    ATTENUATION_DB_PER_KM = 0.2  # dB/km para fibra SMF
    
    num_spans = len(path_links)
    
    # Estimar longitud total (simplificado: asumimos 80km por span)
    # En un modelo más completo, usaríamos la distancia real de cada enlace
    SPAN_LENGTH_KM = 80.0
    total_length = num_spans * SPAN_LENGTH_KM
    
    # Calcular degradación OSNR
    span_noise_penalty = 10 * math.log10(num_spans) if num_spans > 0 else 0
    attenuation_penalty = ATTENUATION_DB_PER_KM * total_length * 0.01  # Simplificado
    amplifier_penalty = amplifier_noise_db * num_spans
    
    estimated_osnr = OSNR_INITIAL_DB - span_noise_penalty - attenuation_penalty - amplifier_penalty
    
    # Calcular margen sobre OSNR requerido
    margin = estimated_osnr - required_osnr_db
    
    # Normalizar usando función sigmoide
    quality = 1 / (1 + math.exp(-margin / 5))
    
    return quality


def calculate_load_balance_factor(
    network,
    path_links: Optional[List] = None
) -> float:
    """
    Calcula el factor de balanceo de carga de la red o ruta.
    
    El balanceo de carga mide qué tan uniformemente distribuida está
    la carga entre los enlaces. Un buen balanceo evita cuellos de botella.
    
    Fórmula (basada en coeficiente de variación inverso):
    ----------------------------------------------------
    LB = 1 - (σ / μ)  si μ > 0
    LB = 1            si μ = 0
    
    Donde:
    - σ: Desviación estándar de las utilizaciones
    - μ: Media de las utilizaciones
    
    Interpretación:
    - LB = 1: Carga perfectamente balanceada
    - LB = 0: Carga totalmente desbalanceada
    
    Parameters:
    -----------
    network : Network
        Objeto Network del simulador
    path_links : Optional[List]
        Si se proporciona, calcula balance solo para esos enlaces
    
    Returns:
    --------
    float
        Factor de balanceo en rango [0, 1]
    """
    utilizations = []
    
    if path_links is not None:
        # Calcular para enlaces específicos
        for link in path_links:
            num_slots = link.getSlots()
            if num_slots > 0:
                occupied = sum(1 for j in range(num_slots) if link.getSlot(j))
                utilizations.append(occupied / num_slots)
    else:
        # Calcular para toda la red
        for i in range(network.linkCounter):
            link = network.getLink(i)
            num_slots = link.getSlots()
            if num_slots > 0:
                occupied = sum(1 for j in range(num_slots) if link.getSlot(j))
                utilizations.append(occupied / num_slots)
    
    if not utilizations:
        return 1.0
    
    mean_util = np.mean(utilizations)
    std_util = np.std(utilizations)
    
    if mean_util <= 0:
        return 1.0
    
    # Coeficiente de variación inverso
    cv = std_util / mean_util
    balance_factor = max(0.0, 1.0 - cv)
    
    return balance_factor


def calculate_spectrum_continuity(
    path_links: List,
    slot_index: int,
    num_slots_required: int
) -> float:
    """
    Calcula el índice de continuidad espectral para una asignación propuesta.
    
    La restricción de continuidad espectral en EON requiere que los mismos
    índices de slot estén disponibles en todos los enlaces de la ruta.
    
    Fórmula:
    --------
    SC = links_con_slots_disponibles / total_links_en_ruta
    
    Parameters:
    -----------
    path_links : List
        Lista de enlaces en la ruta
    slot_index : int
        Índice inicial del slot propuesto
    num_slots_required : int
        Número de slots requeridos
    
    Returns:
    --------
    float
        Índice de continuidad en [0, 1]
        - 1.0: Todos los enlaces tienen los slots disponibles
        - < 1.0: Algunos enlaces están bloqueados
    """
    if not path_links:
        return 1.0
    
    available_count = 0
    
    for link in path_links:
        # Verificar si todos los slots requeridos están disponibles
        all_available = True
        for offset in range(num_slots_required):
            slot_idx = slot_index + offset
            if slot_idx >= link.getSlots() or link.getSlot(slot_idx):
                all_available = False
                break
        
        if all_available:
            available_count += 1
    
    return available_count / len(path_links)


def calculate_slot_distance_penalty(
    slot_index: int,
    total_slots: int,
    preferred_position: str = 'first_fit'
) -> float:
    """
    Calcula la penalización por distancia de slot según la estrategia preferida.
    
    Esta función penaliza asignaciones que no siguen la política de
    asignación de slots preferida (First-Fit, Last-Fit, etc.).
    
    Estrategias:
    -----------
    - 'first_fit': Penaliza slots alejados del inicio
    - 'last_fit': Penaliza slots alejados del final
    - 'best_fit': Penaliza slots en los extremos
    
    Parameters:
    -----------
    slot_index : int
        Índice del slot asignado
    total_slots : int
        Número total de slots en el enlace
    preferred_position : str
        Estrategia de asignación preferida
    
    Returns:
    --------
    float
        Penalización normalizada en [0, 1]
    """
    if total_slots <= 0:
        return 0.0
    
    normalized_position = slot_index / total_slots
    
    if preferred_position == 'first_fit':
        return normalized_position  # 0 al inicio, 1 al final
    elif preferred_position == 'last_fit':
        return 1.0 - normalized_position  # 1 al inicio, 0 al final
    elif preferred_position == 'best_fit':
        return abs(normalized_position - 0.5) * 2  # 0 en medio, 1 en extremos
    else:
        return 0.0


def get_network_spectrum_state(network) -> Dict[str, Any]:
    """
    Obtiene el estado completo del espectro de la red.
    
    Esta función recopila todas las métricas espectrales de la red
    para su uso en funciones de recompensa y análisis.
    
    Returns:
    --------
    Dict con:
    - 'utilizations': Lista de utilización por enlace
    - 'fragmentations': Lista de fragmentación por enlace
    - 'compactness': Lista de compactación por enlace
    - 'avg_utilization': Utilización promedio
    - 'avg_fragmentation': Fragmentación promedio
    - 'load_balance': Factor de balanceo de carga
    - 'entropy': Entropía de la distribución de utilización
    """
    utilizations = []
    fragmentations = []
    compactness_values = []
    
    for i in range(network.linkCounter):
        link = network.getLink(i)
        num_slots = link.getSlots()
        
        if num_slots > 0:
            # Construir array de slots
            slots = [link.getSlot(j) for j in range(num_slots)]
            
            utilizations.append(calculate_link_utilization(slots))
            fragmentations.append(calculate_fragmentation_ratio(slots, 'external'))
            compactness_values.append(calculate_spectral_compactness(slots))
    
    # Calcular promedios y métricas globales
    avg_utilization = np.mean(utilizations) if utilizations else 0.0
    avg_fragmentation = np.mean(fragmentations) if fragmentations else 0.0
    
    # Entropía de la distribución de utilización
    if utilizations and sum(utilizations) > 0:
        normalized_utils = [u / sum(utilizations) for u in utilizations]
        entropy = calculate_entropy(normalized_utils, normalize=True)
    else:
        entropy = 0.0
    
    # Factor de balanceo de carga
    if utilizations:
        mean_u = np.mean(utilizations)
        std_u = np.std(utilizations)
        load_balance = max(0.0, 1.0 - (std_u / mean_u)) if mean_u > 0 else 1.0
    else:
        load_balance = 1.0
    
    return {
        'utilizations': utilizations,
        'fragmentations': fragmentations,
        'compactness': compactness_values,
        'avg_utilization': avg_utilization,
        'avg_fragmentation': avg_fragmentation,
        'avg_compactness': np.mean(compactness_values) if compactness_values else 1.0,
        'load_balance': load_balance,
        'entropy': entropy
    }
