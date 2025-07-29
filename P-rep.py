#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:34:55 2025

@author: ces
"""

import numpy as np
import plotly.graph_objects as go

# Definir parámetros arbitrarios
epsilon = 0.01  # Parámetro para aproximar la delta bidimensional
alpha = 3 # Número complejo arbitrario

# Función delta bidimensional aproximada
def delta_2(beta, epsilon):
    return (1 / (np.pi * epsilon**2)) * np.exp(-np.abs(beta)**2 / epsilon**2)

# Aproximación de P_alpha(beta)
def P_alpha(beta):
    exp_factor = np.exp(-np.abs(beta)**2)
    delta_term = delta_2(beta - alpha, epsilon) + delta_2(beta + alpha, epsilon)
    return np.pi**-2 * exp_factor * delta_term

# Crear malla de valores de beta (plano complejo)
real_vals = np.linspace(-3, 3, 100)
imag_vals = np.linspace(-3, 3, 100)
Re, Im = np.meshgrid(real_vals, imag_vals)
Beta = Re + 1j * Im  # Construcción de números complejos

# Evaluar P_alpha(beta) en la malla
P_vals = P_alpha(Beta).real  # Tomamos la parte real para graficar

# Crear la gráfica 3D interactiva
fig = go.Figure(data=[go.Surface(z=P_vals, x=Re, y=Im, colorscale="plasma")])

# Configurar etiquetas y estilo
fig.update_layout(
    title="Función $P_{\\alpha}(\\beta)$ - Representación 3D",
    scene=dict(
        xaxis_title="x",
        yaxis_title="p",
        zaxis_title="P(β)",
    )
)

# Mostrar la gráfica interactiva
fig.show(renderer="browser")
