#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:43:49 2025

@author: ces
"""

import numpy as np
import galois
from itertools import combinations
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)  # Suprime notación científica para números pequeños


def galois_trace(element, GF):
    """
    Calcula la traza de un elemento en GF(p^n) sobre GF(p)
    """
    p = GF.characteristic
    n = GF.degree
    trace = GF(0)
    current = element
    
    for _ in range(n):
        trace += current
        current = current ** p
    
    return trace

def chi(theta, GF):
    """
    Función característica aditiva χ(θ) = exp(2πi Tr(θ)/p)
    """
    if theta == 0:
        return 1
    tr = galois_trace(theta, GF)
    return np.exp(2j * np.pi * int(tr) / GF.characteristic)

def sd_basis(GF):
    """
    Encuentra una base autodual en el campo GF(2^degree) sobre el polinomio dado.
    
    Args:
        degree: Grado de la extensión del campo (n en GF(2^n))
        irreducible_poly: Polinomio irreducible como string (ej: "x^3 + x + 1").
                         Si es None, se usará el polinomio primitivo por defecto.
    
    Returns:
        Lista de elementos de la base autodual, o None si no se encuentra.
    """
    
    n = GF.degree
    elements = list(GF.elements)[1:]  # Excluir el 0


    # Construir matriz de traza
    A = np.zeros((len(elements), len(elements)), dtype=int)
    for i, x in enumerate(elements):
        for j, y in enumerate(elements):
            A[i, j] = galois_trace(x * y,GF)

    # Buscar submatriz identidad
    for indices in combinations(range(len(elements)), n):
        submatrix = A[np.ix_(indices, indices)]
        if np.array_equal(submatrix, np.eye(n, dtype=int)):
            return [elements[i] for i in indices]
    
    return None  # No se encontró base autodual



def get_power(element, GF):
    """
    Devuelve la potencia k tal que σ^k = element, donde σ es el elemento primitivo del campo.
    
    Args:
        element: Elemento del campo GF(2^n)
        GF_obj: Objeto del campo de Galois (e.g., GF8)
    
    Returns:
        int: Potencia k tal que σ^k = element, o None si element = 0

    """
    if element == GF(0):
        return None  # 0 no es generado por potencias de σ
    
    σ = GF.primitive_element
    order = GF.order - 1  # Orden del grupo multiplicativo (2^n - 1)
    
    # Usamos el algoritmo baby-step giant-step para eficiencia con campos grandes
    if order > 1000:
        return baby_step_giant_step(element, σ, order)
    
    # Para campos pequeños, búsqueda lineal es suficiente
    current = GF(1)  # σ^0
    for k in range(order):
        if current == element:
            return k
        current *= σ
    
    raise ValueError("Elemento no encontrado - esto no debería ocurrir en un campo finito")

def baby_step_giant_step(element, σ, order):
    """Algoritmo eficiente para logaritmos discretos en grupos grandes"""
    m = int(order**0.5) + 1
    
    # Baby-step: precomputar σ^j
    table = {σ**j: j for j in range(m)}
    
    # Giant-step: σ^{-m}
    σ_m = σ**(-m)
    current = element
    
    for i in range(m):
        if current in table:
            return i * m + table[current]
        current *= σ_m
    
    raise ValueError("Elemento no encontrado")


def Z_operator(beta, GF):
    d = GF.order
    Z = np.zeros((d, d), dtype=complex)
    sigma=GF.primitive_element
    elements=[GF(0)]+[sigma**i for i in range(d-1)]
    for i, alpha in enumerate(elements):
        Z[i, i] = chi(alpha * beta, GF)
    return Z

def X_operator(beta, GF):
    sigma=GF.primitive_element
    d = GF.order
    X = np.zeros((d, d), dtype=complex)
    elements=[GF(0)]+[sigma**i for i in range(d-1)]
    
    element_to_index = {str(elem): idx for idx, elem in enumerate(elements)}
    
    for i, alpha in enumerate(elements):
        new_alpha = alpha + beta
        j = element_to_index[str(new_alpha)]
        X[j, i] = 1
    return X



def c_coefs(mu,GF, signs=None,X_nu=None):
    """
    Calcula todos los coeficientes c_{α,μ} siguiendo el Apéndice B del paper.

    Args:
        GF: campo de Galois galois.GF(2**n)
        basis: base autodual [σ1, ..., σn]
        mu: elemento fijo del campo GF(2^n)
        signs: lista opcional con signos [+1, -1, ...] para elegir la raíz √χ(κ²μ)

    Returns:
        Lista [c_0, c_1, ..., c_{q-1}] donde c_i es c_{α,μ} con α = GF(i)
    """
    basis=sd_basis(GF)
    n = GF.degree
    sigma=GF.primitive_element
    q = 2**n
    c = [0] * q  # Lista de coeficientes, índice = int(α)

    # Todas las combinaciones binarias posibles
    all_combinations = list(product([0, 1], repeat=n))

    # Mapeo binario <-> elemento del campo
    bin_to_field = {
        tuple(b): sum((c * b_elem for c, b_elem in zip(b, basis)), start=GF(0))
        for b in all_combinations
    }

    # Fijar los c_{κ,μ} = ±√χ(κ²μ) para κ en la base
    fixed_c = []
    for i, κ in enumerate(basis):
        χ_val = np.exp(2j * np.pi * float(galois_trace(κ * κ * mu, GF)) / 2)
        root = np.sqrt(χ_val)

        # Aplicar el signo si se proporciona
        if signs is not None:
            root *= signs[i]
        fixed_c.append(root)
    # Establecer el valor de referencia
    c[0] = 1  # c_{0,μ} = 1

    # Calcular el resto de los coeficientes usando (B.1) y (29)
    for b in all_combinations:
        α = bin_to_field[tuple(b)]
        if α == GF(0):
            continue
        c_val = 1
        sign_term = GF(0)
        for k in range(n):
            ak = b[k]
            σk = basis[k]
            inner_sum = GF(0)
            for j in range(k + 1, n):
                aj = b[j]
                σj = basis[j]
                inner_sum += aj * σj
            sign_term += ak * σk * inner_sum

        χ_val = np.exp(2j * np.pi * int(galois_trace(mu * sign_term, GF)) / 2)
        for k in range(n):
            if b[k] == 1:
                c_val *= fixed_c[k]
        c[int(α)] = c_val * χ_val
        
    if X_nu is not None:
        if X_nu == 0:
            variable_random=1+1
        else:
            for i in range(len(c)-1):
                c[i+1]=c[i+1]*chi(-sigma**(i)*sigma**X_nu,GF)
        
    ordered_elements = [GF(0)] + [sigma**k for k in range(0, q-1)]
    c = [c[int(α)] for α in ordered_elements]
    return c


def V_coefs(mu, GF,signs=None,X_nu=None):
    """
    Devuelve dos listas:
      - elements: todos los alpha en GF(p^n)
      - c_list: los c_{alpha,mu} correspondientes

    Funciona tanto para característica par (p=2) como para impar (p!=2).
    """
    p = GF.characteristic
    elements = list(GF.elements)

    c_list = []

    if p != 2:
        for alpha in elements:
            # caso impar: c = chi(-1/2 * alpha^2 * mu)
            inv2 = GF(1) / GF(2)
            c_val = chi(-inv2 * (alpha**2) * mu, GF)
            c_list+=[c_val]
    else:
        c_list=c_coefs(mu,GF,signs,X_nu)

    return  c_list

def displacement_operator(alpha, beta, GF,signs=None,X_nus=None):
    """
    Define el operador de desplazamiento D(alpha, beta) = phi(alpha, beta) Z_alpha X_beta
    con la fase definida según la característica del campo GF.

    Args:
        alpha: elemento de GF(p^n)
        beta: elemento de GF(p^n)
        GF: campo de Galois (galois.GF(p**n))

    Returns:
        Matriz compleja de dimensión (d, d) que representa D(alpha, beta)
    """
    p = GF.characteristic
    d = GF.order

    Z = Z_operator(alpha, GF)
    X = X_operator(beta, GF)

    if alpha == GF(0) or beta == GF(0):
        phase = 1
    elif p != 2:
        # Característica impar: phi = chi(-1/2 * alpha * beta)
        inv2 = GF(1) / GF(2)
        phase = chi(-inv2 * alpha * beta, GF)
    else:
        mu = alpha ** (-1) * beta    #
        i=get_power(mu, GF)
        if X_nus is not None:
            c_list = V_coefs(mu, GF,signs,X_nu=X_nus[i])
        else:
            c_list = V_coefs(mu, GF,signs,X_nu=None)
        num=get_power(alpha, GF)
        phase = c_list[num+1]
    return phase * Z @ X


def fourier_operator(GF):
    p = GF.characteristic
    n = GF.degree
    d = p**n
    elements = list(GF.elements)

    F = np.zeros((d, d), dtype=complex)

    for i, alpha in enumerate(elements):
        for j, beta in enumerate(elements):
            # χ(αβ) = exp(2πi Tr(αβ)/p)
            trace = int(galois_trace(alpha * beta,GF)) # Traza de GF(p^n) a GF(p)
            F[i, j] = np.exp(2j * np.pi * int(trace) / p) / np.sqrt(d)

    return F


def wigner_kernel(alpha, beta, GF,signs=None,X_nus=None):
    """
    Calcula el núcleo de Wigner 
    Args:
        alpha: elemento de GF
        beta: elemento de GF
        GF: campo de Galois galois.GF(p**n)
        X_nus: lista con los operadores de desplazamiento en orden de potencias
    Returns:
        Matriz (d x d) compleja que representa el núcleo A(alpha, beta)
    """
    sigma=GF.primitive_element
    d = GF.order
    A = np.zeros((d, d), dtype=complex)
    elements=[GF(0)]+[sigma**i for i in range(d-1)]
    
    if X_nus is not None:
        for kappa  in (elements):
            for lamb in (elements):
                phase = chi(alpha * lamb - beta * kappa, GF)
                D = displacement_operator(kappa, lamb, GF,signs,X_nus)
                A += phase * D
    else:
        for kappa in elements:
            for lamb in elements:
                phase = chi(alpha * lamb - beta * kappa, GF)
                D = displacement_operator(kappa, lamb, GF,signs)
                A += phase * D
    return A / (d)


def wigner_function(rho, GF,signs=None,X_nus=None):
    """
    Calcula la función de Wigner W(alpha, beta) y devuelve una matriz (d x d),
    donde las filas corresponden a alpha y las columnas a beta.

    Args:
        rho: matriz de densidad (d x d)
        GF: campo de Galois galois.GF(p**n)

    Returns:
        Matriz numpy (d x d) con Wigner(alpha, beta)
    """
    sigma=GF.primitive_element
    s = GF.order
    elements=[GF(0)]+[sigma**i for i in range(s-1)]

    d = len(elements)
    W_matrix = np.zeros((d, d), dtype=float)

    for i, alpha in enumerate(elements):
        for j, beta in enumerate(elements):
            if X_nus is not None:
                A = wigner_kernel(alpha, beta, GF,signs,X_nus)
            else:
                A = wigner_kernel(alpha, beta, GF,signs)
            W_matrix[i, j] = np.real(np.trace(rho @ A))

    return W_matrix


def reorder_matrix_by_sigma_powers(W_matrix, GF):
    sigma = GF.primitive_element
    elements = [GF(0)] + [sigma**i for i in range(1, GF.order)]
    indices = [list(GF.elements).index(e) for e in elements]
    return W_matrix[np.ix_(indices, indices)], elements


def plot_wigner(W_matrix, GF):
    # Reordenar la matriz W por potencias de sigma
    W_reordered, ordered_elements = reorder_matrix_by_sigma_powers(W_matrix, GF)
    labels = [str(e) for e in ordered_elements]
    d = GF.order

    # Coordenadas X, Y en grilla
    X, Y = np.meshgrid(range(d), range(d))
    Z = W_reordered

    # Crear figura y ejes 3D
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie tipo barra
    dx = dy = 0.6
    xpos = X.flatten()
    ypos = Y.flatten()
    zpos = np.zeros_like(xpos)
    dz = Z.flatten()

    # Elegir colores basados en valor (mantenido igual que el original)
    colors = plt.cm.seismic((dz + 1) / 2)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    # Etiquetas de los ejes
    ax.set_xticks(range(d))
    ax.set_xticklabels(labels)
    ax.set_xlabel(r'$\beta$')

    ax.set_yticks(range(d))
    ax.set_yticklabels(labels)
    ax.set_ylabel(r'$\alpha$')

    ax.set_zlabel(r'$W(\alpha, \beta)$')
    ax.set_title(r'Wigner Function')
    ax.set_zlim(np.min(Z), np.max(Z))

    plt.tight_layout()
    plt.show()
    
    
# ==================== EJEMPLOS =========================================

#GF = galois.GF(2**2, irreducible_poly=galois.Poly([1,1,1]))
GF= galois.GF(2**3, irreducible_poly=galois.Poly([1,1,0,1]))
#GF=galois.GF(5)
sigma = GF.primitive_element

d = GF.order
elements = [GF(0)] + [sigma**i for i in range(0, GF.order-1)]


# Diccionario para mapear elemento -> índice
element_to_index = {str(e): i for i, e in enumerate(elements)}

# Estado base |0>
ket_0 = np.zeros((d, 1), dtype=complex)
ket_0[0, 0] = 1

# Estado |sigma^7>
sigma3 = sigma **7
idx_sigma3 = element_to_index[str(sigma3)]
ket_sigma3 = np.zeros((d, 1), dtype=complex)
ket_sigma3[idx_sigma3, 0] = 1

# Estado |\psi> = (|0> + |sigma^3>)/sqrt(2)
ket_psi = (ket_0 + ket_sigma3) / np.sqrt(2)

# Matriz de densidad
rho = ket_psi @ ket_psi.T.conj()


# Calcular la matriz de Wigner
"""
Selecciones de operadores de rotacion 
[0,0,0,0,0,0,0]
[0,1,0,0,0,0,0] - La misma que la base
[0,1,2,0,0,0,0]
[0,1,2,0,4,0,0]
[0,4,1,0,4,0,0]
Completamente positivo= [0,4,2,0,4,0,0]
[0,7,1,0,7,0,0]
[0,3,7,0,5,0,0]- la misma que la positiva
"""

W_matrix = wigner_function(rho, GF,X_nus=[0,7,1,0,7,0,0])  
plot_wigner(W_matrix, GF)

# Formula de inversion del paper
f_reconstructed = np.zeros((d, d), dtype=complex)

for i, alpha in enumerate(elements):
    for j, beta in enumerate(elements):
        A = wigner_kernel(alpha, beta, GF,X_nus=[0,7,1,0,7,0,0])
        f_reconstructed += W_matrix[i, j] * A
f_reconstructed=f_reconstructed/d
        

print(np.allclose(rho, f_reconstructed, atol=1e-10)) # True si obtenemos el operador 

# ========================= Rectas del campo  ===========================
GF = galois.GF(2**2, irreducible_poly=galois.Poly([1,1,1]))
sigma = GF.primitive_element
d = GF.order
elements = [GF(0)] + [sigma**i for i in range(0, GF.order-1)]
element_to_index = {str(e): i for i, e in enumerate(elements)}
eta = GF(0)
idx_eta = element_to_index[str(eta)]
ket_eta=np.zeros((d, 1), dtype=complex)
ket_eta[idx_eta, 0] = 1

xi=sigma**0
V_xi_daig=np.diag(V_coefs(xi, GF))
V_xi=fourier_operator(GF)@V_xi_daig@fourier_operator(GF).conj().T
ket_psi=V_xi @ ket_eta

rho=ket_psi @ ket_psi.conj().T

W_matrix = wigner_function(rho, GF)  
plot_wigner(W_matrix, GF)
