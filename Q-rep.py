#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:35:05 2025

@author: ces
"""

import numpy as np
import plotly.graph_objects as go

# Definir parámetros arbitrarios
alpha = 1.5 + 1.5j  # Número complejo

# Función Q(beta)
def Q(beta):
    term1 = np.exp(-np.abs(alpha - beta)**2)
    term2 = np.exp(-np.abs(alpha + beta)**2)
    term3 = np.exp(- (np.abs(alpha)**2 + np.abs(beta)**2)) * (
        np.exp(beta * np.conj(alpha) - alpha * np.conj(beta)) +
        np.exp(alpha * np.conj(beta) - beta * np.conj(alpha))
    )
    return (1 / (2 * np.pi)) * (term1 + term2 + term3)

# Crear malla de valores de beta (plano complejo)
real_vals = np.linspace(-3, 3, 100)
imag_vals = np.linspace(-3, 3, 100)
Re, Im = np.meshgrid(real_vals, imag_vals)
Beta = Re + 1j * Im  # Construcción de números complejos

# Evaluar Q(beta) en la malla
Q_vals = Q(Beta).real  # Tomamos solo la parte real para graficar

# Crear la gráfica 3D interactiva
fig = go.Figure(data=[go.Surface(z=Q_vals, x=Re, y=Im, colorscale="viridis")])

# Configurar etiquetas y estilo
fig.update_layout(
    title="Función Q(β) - Representación 3D",
    scene=dict(
        xaxis_title="x",
        yaxis_title="p",
        zaxis_title="Q(β)",
    )
)

# Mostrar la gráfica interactiva
fig.show(renderer="browser")
