#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:35:17 2025

@author: ces
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parámetros del sistema
alpha = 3.50
hbar = 1
sigma = np.sqrt(hbar/2)

# Función de Wigner
def wigner_function(x, p):
    term_plus = np.exp(-(x - alpha)**2/(2*sigma**2) - p**2/(2*sigma**2)) / (np.pi*hbar)
    term_minus = np.exp(-(x + alpha)**2/(2*sigma**2) - p**2/(2*sigma**2)) / (np.pi*hbar)
    term_interf = 2*np.exp(-x**2/(2*sigma**2) - p**2/(2*sigma**2)) / (np.pi*hbar) * \
                 np.cos(2*alpha*p/hbar)
    return term_plus + term_minus + term_interf

# Crear malla de valores
x_vals = np.linspace(-5, 5, 200)
p_vals = np.linspace(-5, 5, 200)
X, P = np.meshgrid(x_vals, p_vals)
W = wigner_function(X, P)

# Cálculo de marginales
dx = x_vals[1] - x_vals[0]
dp = p_vals[1] - p_vals[0]
marginal_x = np.sum(W, axis=0) * dp
marginal_p = np.sum(W, axis=1) * dx

# Crear figura con diseño personalizado
fig = make_subplots(
    rows=1, cols=3,
    column_widths=[1, 1, 1],
    row_heights=[1],
    specs=[[{'type': 'heatmap'}, {'type': 'xy'}, {'type': 'xy'}]],
    horizontal_spacing=0.13
)

# Mapa de calor central
fig.add_trace(
    go.Heatmap(
        z=W,
        x=x_vals,
        y=p_vals,
        colorscale="Viridis",
        #showscale=True,
        colorbar=dict(title="W(x,p)", len=0.8,x=.25, y=0.5)
    ),
    row=1, col=1
)

# Marginal en X (derecha, arriba)
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=marginal_x,
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False  # Eliminar leyenda
    ),
    row=1, col=2
)

# Marginal en P (derecha, abajo)
fig.add_trace(
    go.Scatter(
        x=p_vals,
        y=marginal_p,
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False  # Eliminar leyenda
    ),
    row=1, col=3
)

# Configuración del layout
fig.update_layout(
    #title_text=f"Función de Wigner y sus Marginales (α={alpha})",
    height=500,
    width=1500,
    margin=dict(t=60, b=50, l=50, r=50),
    plot_bgcolor='white'
)

# Configuración de ejes
fig.update_xaxes(title_text="x", row=1, col=1)
fig.update_yaxes(title_text="p", row=1, col=1)

fig.update_xaxes(title_text="x", row=1, col=2)
fig.update_yaxes(title_text="ρ(x)", row=1, col=2)

fig.update_xaxes(title_text="p", row=1, col=3)
fig.update_yaxes(title_text="ρ(p)", row=1, col=3)

# Ajustar rangos para mejor visualización
fig.update_yaxes(range=[-0.1, max(marginal_x)*1.1], row=1, col=2)
fig.update_yaxes(range=[-0.1, max(marginal_p)*1.1], row=1, col=3)

fig.show(renderer="browser")   