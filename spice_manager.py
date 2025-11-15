"""
spice_manager.py - SPICE-Kernel Verwaltung

Stellt Funktionen zum Laden und Entladen von SPICE-Kernels bereit.
SPICE-Kernels enthalten Ephemeridendaten (Positionen und Geschwindigkeiten)
von Himmelskörpern und Raumsonden.
"""

import spiceypy as sp
import os
from config import kernel_paths, LIGHT_TIME_CORRECTION

# Funktion zum Laden der SPICE-Kernels
def load_kernels():
    for path in kernel_paths:
        if kernel_paths and os.path.exists(path):
            sp.furnsh(path)
            print(f"{path} geladen...")
        else:
            print(f"WARNUNG: Fehler beim Laden von {path}...")


# Funktion zum Entladen der SPICE-Kernels
def unload_kernels():
    sp.kclear()
    print("Alle SPICE-Kernel entladen...")


# Funktion zum Abrufen des Zustandsvektors (x, y, z, v_x, v_y, v_z)
def get_initial_state(target, et, observer='SUN', reference_frame='ECLIPJ2000'):
    state, lt = sp.spkezr(target, et, reference_frame, LIGHT_TIME_CORRECTION, observer)
    return state