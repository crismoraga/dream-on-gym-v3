# -*- coding: utf-8 -*-
"""
DREAM-ON-GYM-V2: Dashboard Interactivo de Analisis
==================================================

Dashboard web interactivo para visualizar y analizar resultados
de experimentos con funciones de recompensa.

Funcionalidades:
---------------
1. Visualizacion de blocking probability
2. Comparativas entre funciones de recompensa
3. Analisis de fragmentacion espectral
4. Metricas en tiempo real
5. Exportacion de graficos

Requiere: streamlit, plotly, pandas

Uso:
----
streamlit run dashboard.py

@author: Generado con AI para DREAM-ON-GYM-V2
@version: 2.0.0
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Verificar dependencias
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Para usar el dashboard, instale: pip install streamlit plotly")

# Tambien intentar con matplotlib como fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_experiment_data(results_dir: Path) -> pd.DataFrame:
    """
    Carga datos de experimentos desde archivos CSV.
    """
    csv_files = list(results_dir.glob("results_*.csv"))
    
    if not csv_files:
        return pd.DataFrame()
    
    # Cargar el archivo mas reciente
    latest_file = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    return df


def create_blocking_plot(df: pd.DataFrame):
    """Crea grafico interactivo de blocking probability."""
    fig = px.line(
        df.groupby(['reward_function', 'rho'])['blocking_probability'].mean().reset_index(),
        x='rho',
        y='blocking_probability',
        color='reward_function',
        markers=True,
        title='Blocking Probability vs Carga de Red'
    )
    
    fig.update_layout(
        xaxis_title='Carga (rho)',
        yaxis_title='Blocking Probability',
        legend_title='Funcion de Recompensa',
        hovermode='x unified'
    )
    
    return fig


def create_reward_comparison(df: pd.DataFrame):
    """Crea grafico de comparacion de recompensas."""
    summary = df.groupby('reward_function').agg({
        'avg_reward': ['mean', 'std'],
        'cumulative_reward': 'mean'
    }).round(4)
    summary.columns = ['Promedio', 'Std', 'Acumulada']
    summary = summary.reset_index()
    
    fig = px.bar(
        summary,
        x='reward_function',
        y='Promedio',
        error_y='Std',
        title='Recompensa Promedio por Funcion',
        color='reward_function'
    )
    
    fig.update_layout(
        xaxis_title='Funcion de Recompensa',
        yaxis_title='Recompensa Promedio',
        showlegend=False
    )
    
    return fig


def create_fragmentation_heatmap(df: pd.DataFrame):
    """Crea heatmap de fragmentacion."""
    pivot = df.pivot_table(
        values='avg_fragmentation',
        index='reward_function',
        columns='rho',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot,
        labels=dict(x='Carga (rho)', y='Funcion', color='Fragmentacion'),
        title='Fragmentacion por Configuracion',
        color_continuous_scale='RdYlGn_r'
    )
    
    return fig


def create_performance_radar(df: pd.DataFrame):
    """Crea grafico radar de metricas multiples."""
    # Normalizar metricas
    metrics = ['blocking_probability', 'avg_fragmentation', 'avg_reward']
    normalized = df.groupby('reward_function')[metrics].mean()
    
    # Invertir blocking y fragmentacion (menor es mejor)
    normalized['blocking_score'] = 1 - normalized['blocking_probability'] / normalized['blocking_probability'].max()
    normalized['fragmentation_score'] = 1 - normalized['avg_fragmentation'] / normalized['avg_fragmentation'].max()
    normalized['reward_score'] = (normalized['avg_reward'] - normalized['avg_reward'].min()) / (normalized['avg_reward'].max() - normalized['avg_reward'].min())
    
    categories = ['Blocking', 'Fragmentacion', 'Recompensa']
    
    fig = go.Figure()
    
    for fn in normalized.index:
        fig.add_trace(go.Scatterpolar(
            r=[normalized.loc[fn, 'blocking_score'],
               normalized.loc[fn, 'fragmentation_score'],
               normalized.loc[fn, 'reward_score']],
            theta=categories,
            fill='toself',
            name=fn
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Comparativa Multi-dimensional de Rendimiento'
    )
    
    return fig


def create_time_series_analysis(df: pd.DataFrame):
    """Analisis de serie temporal (si hay datos de entrenamiento)."""
    if 'step' not in df.columns:
        return None
    
    fig = px.line(
        df,
        x='step',
        y='cumulative_reward',
        color='reward_function',
        title='Recompensa Acumulada durante Entrenamiento'
    )
    
    return fig


def main_dashboard():
    """Funcion principal del dashboard Streamlit."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit no disponible. Instale con: pip install streamlit plotly")
        return
    
    st.set_page_config(
        page_title="DREAM-ON-GYM-V2 Dashboard",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 DREAM-ON-GYM-V2: Analisis de Funciones de Recompensa")
    
    # Sidebar para configuracion
    st.sidebar.header("Configuracion")
    
    # Cargar datos
    results_dir = Path(__file__).parent / "results"
    df = load_experiment_data(results_dir)
    
    if df.empty:
        st.warning("No se encontraron datos de experimentos. Ejecute evaluate_rewards.py primero.")
        
        # Crear datos de ejemplo para demostracion
        st.info("Mostrando datos de ejemplo para demostracion...")
        
        df = pd.DataFrame({
            'reward_function': ['Baseline']*4 + ['QoT-Aware']*4 + ['Multi-Objective']*4 + 
                              ['Fragmentation-Aware']*4 + ['Spectral-Entropy']*4,
            'rho': [0.3, 0.5, 0.7, 0.9] * 5,
            'blocking_probability': [0.001, 0.015, 0.12, 0.35,
                                     0.001, 0.012, 0.10, 0.32,
                                     0.001, 0.010, 0.08, 0.28,
                                     0.001, 0.011, 0.09, 0.30,
                                     0.001, 0.008, 0.07, 0.25],
            'avg_reward': [0.95, 0.85, 0.65, 0.35,
                          0.96, 0.87, 0.70, 0.40,
                          0.97, 0.90, 0.75, 0.45,
                          0.96, 0.88, 0.72, 0.42,
                          0.98, 0.92, 0.78, 0.50],
            'avg_fragmentation': [0.05, 0.15, 0.30, 0.50,
                                  0.04, 0.13, 0.28, 0.48,
                                  0.03, 0.10, 0.22, 0.40,
                                  0.03, 0.09, 0.20, 0.38,
                                  0.02, 0.08, 0.18, 0.35],
            'topology': ['NSFNet'] * 20,
            'run': [1] * 20
        })
    
    # Filtros
    st.sidebar.subheader("Filtros")
    
    topologies = df['topology'].unique().tolist()
    selected_topologies = st.sidebar.multiselect(
        "Topologias",
        topologies,
        default=topologies
    )
    
    reward_fns = df['reward_function'].unique().tolist()
    selected_rewards = st.sidebar.multiselect(
        "Funciones de Recompensa",
        reward_fns,
        default=reward_fns
    )
    
    # Filtrar datos
    filtered_df = df[
        (df['topology'].isin(selected_topologies)) &
        (df['reward_function'].isin(selected_rewards))
    ]
    
    # Metricas principales
    st.header("📊 Metricas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_bp = filtered_df['blocking_probability'].min()
        st.metric(
            "Mejor Blocking Probability",
            f"{best_bp:.4f}",
            delta=f"-{(filtered_df['blocking_probability'].mean() - best_bp):.4f}"
        )
    
    with col2:
        avg_reward = filtered_df['avg_reward'].mean()
        st.metric(
            "Recompensa Promedio",
            f"{avg_reward:.3f}"
        )
    
    with col3:
        avg_frag = filtered_df['avg_fragmentation'].mean()
        st.metric(
            "Fragmentacion Promedio",
            f"{avg_frag:.3f}"
        )
    
    with col4:
        best_fn = filtered_df.groupby('reward_function')['blocking_probability'].mean().idxmin()
        st.metric(
            "Mejor Funcion",
            best_fn
        )
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Blocking Probability",
        "🏆 Comparativa",
        "🔬 Fragmentacion",
        "📉 Radar Multi-dimensional"
    ])
    
    with tab1:
        st.subheader("Blocking Probability vs Carga de Red")
        fig = create_blocking_plot(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos
        with st.expander("Ver datos detallados"):
            pivot = filtered_df.pivot_table(
                values='blocking_probability',
                index='reward_function',
                columns='rho',
                aggfunc='mean'
            ).round(4)
            st.dataframe(pivot, use_container_width=True)
    
    with tab2:
        st.subheader("Comparativa de Recompensas")
        fig = create_reward_comparison(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rankings
        st.subheader("Rankings")
        rankings = filtered_df.groupby('reward_function').agg({
            'blocking_probability': 'mean',
            'avg_reward': 'mean',
            'avg_fragmentation': 'mean'
        }).round(4)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Mejor Blocking (menor)**")
            st.dataframe(rankings.sort_values('blocking_probability').head())
        
        with col2:
            st.write("**Mejor Recompensa (mayor)**")
            st.dataframe(rankings.sort_values('avg_reward', ascending=False).head())
        
        with col3:
            st.write("**Menor Fragmentacion**")
            st.dataframe(rankings.sort_values('avg_fragmentation').head())
    
    with tab3:
        st.subheader("Analisis de Fragmentacion Espectral")
        fig = create_fragmentation_heatmap(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Grafico adicional
        fig2 = px.box(
            filtered_df,
            x='reward_function',
            y='avg_fragmentation',
            color='rho',
            title='Distribucion de Fragmentacion por Configuracion'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.subheader("Comparativa Multi-dimensional")
        fig = create_performance_radar(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretacion del grafico radar:**
        - Cada dimension representa una metrica normalizada (0-1)
        - Mayor area = mejor rendimiento general
        - Blocking y Fragmentacion estan invertidos (menor es mejor)
        """)
    
    # Seccion de documentacion
    st.header("📚 Documentacion de Funciones de Recompensa")
    
    with st.expander("Ver descripciones de funciones"):
        st.markdown("""
        ### 1. Baseline Reward
        Recompensa binaria simple: +1 si la conexion es asignada, -1 si es bloqueada.
        
        ### 2. QoT-Aware Reward
        Considera la calidad de transmision (OSNR) al calcular la recompensa.
        Favorece rutas con mejor calidad de senal.
        
        ### 3. Multi-Objective Reward
        Combina multiples objetivos: blocking, fragmentacion, utilizacion, 
        balance de carga y longitud de ruta.
        
        ### 4. Fragmentation-Aware Reward
        Minimiza la fragmentacion espectral considerando impacto local y global.
        
        ### 5. Spectral-Entropy Adaptive Reward (NOVEL)
        **Funcion novedosa** que usa entropia espectral con:
        - Zonas adaptativas de operacion
        - Asignacion de credito temporal
        - Prediccion de impacto futuro
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**DREAM-ON-GYM-V2** | Dashboard de Analisis de Funciones de Recompensa | "
        f"Datos cargados: {len(df)} registros"
    )


def static_report(df: pd.DataFrame, output_dir: Path):
    """
    Genera un reporte estatico HTML cuando Streamlit no esta disponible.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib no disponible para generar reporte estatico.")
        return
    
    print("Generando reporte estatico...")
    
    # Crear figuras
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Blocking vs Rho
    ax1 = axes[0, 0]
    for rf in df['reward_function'].unique():
        rf_df = df[df['reward_function'] == rf]
        means = rf_df.groupby('rho')['blocking_probability'].mean()
        ax1.plot(means.index, means.values, marker='o', label=rf)
    ax1.set_xlabel('Carga (rho)')
    ax1.set_ylabel('Blocking Probability')
    ax1.set_title('Blocking Probability vs Carga')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Barras de recompensa
    ax2 = axes[0, 1]
    means = df.groupby('reward_function')['avg_reward'].mean()
    means.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_xlabel('Funcion de Recompensa')
    ax2.set_ylabel('Recompensa Promedio')
    ax2.set_title('Recompensa por Funcion')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Heatmap de fragmentacion
    ax3 = axes[1, 0]
    pivot = df.pivot_table(
        values='avg_fragmentation',
        index='reward_function',
        columns='rho',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax3)
    ax3.set_title('Fragmentacion por Configuracion')
    
    # 4. Boxplot de variabilidad
    ax4 = axes[1, 1]
    df.boxplot(column='blocking_probability', by='reward_function', ax=ax4)
    ax4.set_xlabel('Funcion de Recompensa')
    ax4.set_ylabel('Blocking Probability')
    ax4.set_title('Variabilidad de Blocking')
    plt.suptitle('')  # Remover titulo automatico
    
    plt.tight_layout()
    output_file = output_dir / 'static_report.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reporte guardado en: {output_file}")


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        main_dashboard()
    else:
        # Fallback a reporte estatico
        results_dir = Path(__file__).parent / "results"
        df = load_experiment_data(results_dir)
        
        if not df.empty:
            static_report(df, results_dir)
        else:
            print("No hay datos para generar reporte.")
