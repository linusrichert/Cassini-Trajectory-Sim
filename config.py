"""
config.py - Konfigurationsdatei für die Cassini-Simulation

Enthält alle wichtigen Parameter und Konstanten:
- Pfade zu SPICE-Kernels (Ephemeridendaten)
- Simulationszeitraum und Himmelskörper
- Gravitationsparameter (GM-Werte) der Planeten
- Integrator-Einstellungen und Toleranzen
- NAIF-IDs für Himmelskörper und Cassini
"""

import spiceypy as sp
import numpy as np

# Pfadvariablen für alle Kernel
kernel_paths = [
    "kernels/naif0012.tls", # Schaltsekunden-Kernel
    "kernels/pck00010.tpc", # Planetenkonstanten-Kernel
    "kernels/de440.bsp", # Planetenephemeride (Sonnensystem-Baryzentrum zu Planeten-Baryzentern)
    "kernels/co_1997319_99311_i_cru_v1.bsp", # Cassini Trajektorie - erste Periode
    "kernels/co_1999312_01066_o_cru_v1.bsp", # Cassini Trajektorie
    "kernels/041014R_SCPSE_01066_04199.bsp", # Cassini Trajektorie - letzte Periode
    "kernels/jup365.bsp", # Jupiter-Ephemeride
    "kernels/sat441.bsp", # Saturn-Ephemeride
    "kernels/ura111xl-799.bsp", # Uranus-Ephemeride
    "kernels/nep097.bsp", # Neptun-Ephemeride
    "kernels/mar099.bsp" # Mars-Ephemeride
]

# Simulationsstart- und endzeit als UTC-Strings
START_UTC = '1998-01-01 00:00:00'
END_UTC = '2004-01-01 00:00:00'

# 1. Venus GA
# START_UTC = '1998-03-26'
# END_UTC = '1998-05-26'

# 2. Venus GA
# START_UTC = '1999-05-24'
# END_UTC = '1999-07-24'

# Erde GA
# START_UTC = '1999-07-18'
# END_UTC = '1999-09-18'

# Jupiter GA
# START_UTC = '2000-11-30 00:00:00'
# END_UTC = '2001-01-30 00:00:00'

# Liste der Himmelskörper, deren Gravitation berechnet werden soll
# Baryzentren werden für Mars, Jupiter, Saturn, Uranus, Neptun verwendet (konsistent mit GM-Werten)
TARGET_BODIES = [
    'SUN',      # Sonne (NAIF-ID: 10)
    'MERCURY',  # Merkur (NAIF-ID: 199)
    'VENUS',    # Venus (NAIF-ID: 299) 
    'EARTH',    # Erde (NAIF-ID: 399)
    'MOON',     # Mond (NAIF-ID: 301)
    'MARS',     # Mars Baryzentrum (NAIF-ID: 4)
    'JUPITER',  # Jupiter Baryzentrum (NAIF-ID: 5)
    'SATURN',   # Saturn Baryzentrum (NAIF-ID: 6)
    'URANUS',   # Uranus Baryzentrum (NAIF-ID: 7)
    'NEPTUNE'   # Neptun Baryzentrum (NAIF-ID: 8)
]

TARGET_BODIES_GM = {
    'SUN'    : 1.32712440041279419e11,  # km³/s² - Sonne
    'MERCURY': 22031.868551,            # km³/s² - Merkur
    'VENUS'  : 324858.592000,           # km³/s² - Venus
    'EARTH'  : 398600.435507,           # km³/s² - Erde
    'MOON'   : 4902.800118,             # km³/s² - Mond
    'MARS'   : 42828.375816,            # km³/s² - Mars System (Baryzentrum)
    'JUPITER': 126712764.100000,        # km³/s² - Jupiter System (Baryzentrum)
    'SATURN' : 37940584.841800,         # km³/s² - Saturn System (Baryzentrum)
    'URANUS' : 5794556.400000,          # km³/s² - Uranus System (Baryzentrum)
    'NEPTUNE': 6836527.100580           # km³/s² - Neptun System (Baryzentrum)
}

TARGET_BODIES_GM_ARRAY = np.array([TARGET_BODIES_GM[body] for body in TARGET_BODIES])  # Form: (N,)

for body in TARGET_BODIES:
    print(f"GM-Wert für {body}: {TARGET_BODIES_GM[body]:.3f} km³/s²")


# Physikalische Konstanten und Cassini-spezifische Daten
PHYSICAL_CONSTANTS = {
    # Cassini Triebwerksparameter
    'cassini': {
        'initial_total_mass': 5570.0,   # Startgesamtmasse in kg (inkl. Huygens)
    }
}

# NUMERISCHE INTEGRATION
INTEGRATOR_TOLERANCES = {
    'rtol': 1e-11,  # Relative Toleranz
    'atol': 1e-14   # Absolute Toleranz
}

# Gravitationsmodell-Konfiguration - Nur Punktmassen
GRAVITY_MODEL = 'point_mass'  # Reine Punktmassen-Gravitations-Simulation

LIGHT_TIME_CORRECTION = "NONE"

MAX_STEP_INTEGRATION = 0.1 * 86400  # maximaler Zeitschritt in Sekunden
SOLVER_INTEGRATION = 'RK45'

# NAIF IDs für Himmelskörper und Cassini
# Baryzentren für Mars, Jupiter, Saturn, Uranus, Neptun (konsistent mit GM-Werten und BODY_INFO)
NAIF_IDS = {
    'CASSINI': -82,      # Cassini Raumsonde
    'SUN': 10,           # Sonne
    'MERCURY': 199,      # Merkur
    'VENUS': 299,        # Venus
    'EARTH': 399,        # Erde
    'MOON': 301,         # Mond
    'MARS': 4,           # Mars Baryzentrum
    'JUPITER': 5,        # Jupiter Baryzentrum
    'SATURN': 6,         # Saturn Baryzentrum
    'URANUS': 7,         # Uranus Baryzentrum
    'NEPTUNE': 8         # Neptun Baryzentrum
}