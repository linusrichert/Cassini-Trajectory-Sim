"""
dynamics.py - Dynamikfunktionen für die Bahnberechnung

Enthält Hilfsfunktionen zur Berechnung von Himmelskörperpositionen
mittels SPICE. Wird von der Integrationsfunktion verwendet, um die
Gravitationskräfte der Planeten zu berechnen.
"""

import spiceypy as sp
import numpy as np
from config import LIGHT_TIME_CORRECTION

# NAIF ID Zuordnung für Himmelskörper
# Baryzentren für Mars, Jupiter, Saturn, Uranus, Neptun (konsistent mit Baryzentrum-GM-Werten)
BODY_INFO = {
    'SUN': 10,
    'MERCURY': 199,
    'VENUS': 299,
    'EARTH': 399,
    'MOON': 301,
    'MARS': 4,       # Mars Baryzentrum
    'JUPITER': 5,    # Jupiter Baryzentrum
    'SATURN': 6,     # Saturn Baryzentrum
    'URANUS': 7,     # Uranus Baryzentrum
    'NEPTUNE': 8     # Neptun Baryzentrum
}
def get_body_position(body_name: str, et: float, frame: str = 'ECLIPJ2000', observer: str = 'SUN', correction: str = LIGHT_TIME_CORRECTION) -> np.ndarray:
    """
    Gibt die Position eines Himmelskörpers zurück.
    
    Args:
        body_name: Name des Körpers (muss Schlüssel in BODY_INFO sein)
        et: Ephemeridenzeit (Sekunden seit J2000)
        frame: Referenzrahmen (Standard: 'ECLIPJ2000')
        observer: Beobachter-Körper (Standard: 'SUN')
        correction: Lichtlaufzeit/Aberrationskorrektur (Standard: 'NONE')
        
    Returns:
        Positionsvektor (km)
    """
    # Hole die Körper-ID aus BODY_INFO
    body_id = BODY_INFO[body_name]
    
    # Verwende die Körper-ID direkt mit SPICE
    pos, _ = sp.spkpos(str(body_id), et, frame, correction, observer)
    return pos
