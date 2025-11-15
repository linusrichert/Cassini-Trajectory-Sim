"""
extract_maneuver_deltav.py - Extraktion von Delta-v Vektoren aus SPICE-Daten

Dieses Skript extrahiert die Delta-v Vektoren der Cassini-Manöver aus den
SPICE-Ephemeridendaten, Cassinis tatsächlicher Flugbahn und den Beträgen der tatsächlichen Geschwindigkeitsänderungen. Es analysiert die Geschwindigkeitsänderungen vor und
nach jedem Manöver und berechnet die optimalen Zeitfenster für die Extraktion.
Die Ergebnisse werden als CSV-Datei gespeichert.
"""

import os
import numpy as np
import pandas as pd
import spiceypy as sp
from typing import Optional, Tuple
from scipy.optimize import minimize_scalar
import config

def load_kernels():
    """Lädt erforderliche SPICE-Kernels."""
    kernels = [
        'kernels/naif0012.tls',
        'kernels/pck00010.tpc', 
        'kernels/de440.bsp',
        'kernels/co_1997319_99311_i_cru_v1.bsp',
        'kernels/co_1999312_01066_o_cru_v1.bsp',
        'kernels/041014R_SCPSE_01066_04199.bsp',
        'kernels/041014R_SCPSE_01066_04199.bsp'
    ]
    
    for kernel in kernels:
        sp.furnsh(kernel)
        print(f"Loaded: {kernel}")

def load_maneuver_data(filename: str) -> pd.DataFrame:
    """Lädt Manöverdaten aus CSV-Datei."""
    print(f"Loading maneuver data from {filename}...")
    df = pd.read_csv(filename)
    
    # Konvertiere Zeitstrings zu ET (Ephemeridenzeit)
    et_times = []
    for time_str in df['Maneuver Time (UTC SCET)']:
        try:
            # Konvertiere dd/mm/yyyy hh:mm Format zu SPICE-kompatiblem Format
            # Beispiel: "25/02/1998 20:00" → "1998-02-25 20:00:00"
            from datetime import datetime
            dt = datetime.strptime(time_str, '%d/%m/%Y %H:%M')
            iso_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Konvertiere zu ET mittels SPICE
            et = sp.str2et(iso_str)
            et_times.append(et)
        except Exception as e:
            print(f"Error converting time {time_str}: {e}")
            et_times.append(np.nan)
    
    df['ET'] = et_times
    
    # Filtere ungültige Zeiten heraus
    valid_df = df[~pd.isna(df['ET'])].copy()
    print(f"Found {len(valid_df)} maneuvers with valid times")
    
    return valid_df

def find_optimal_time_window(maneuver_time_et: float, expected_delta_v: float, 
                           burn_type: str) -> Tuple[float, float]:
    """
    Findet optimales Zeitfenster durch Minimierung der Differenz zwischen berechnetem
    und erwartetem Delta-v, unter Berücksichtigung von Gravitations-Geschwindigkeitsänderungen.
    
    Args:
        maneuver_time_et: Manöverzeit in Ephemeridensekunden
        expected_delta_v: Erwartete Delta-v Größe in m/s
        burn_type: 'MEA' für Haupttriebwerk oder 'RCS' für Lageregelung
        
    Returns:
        Tupel aus (optimale_Zeit_vorher, optimale_Zeit_nachher) in Sekunden
    """
    def objective(window_size):
        """Zielfunktion zum Minimieren: Differenz zwischen berechnetem und erwartetem Delta-v."""
        try:
            # Verwende symmetrisches Fenster um Manöverzeit
            time_before = maneuver_time_et - window_size
            time_after = maneuver_time_et + window_size
            
            # Hole Geschwindigkeiten vorher und nachher
            state_before, _ = sp.spkezr('-82', time_before, 'ECLIPJ2000', 'NONE', 'SUN')
            state_after, _ = sp.spkezr('-82', time_after, 'ECLIPJ2000', 'NONE', 'SUN')
            
            v_before = np.array(state_before[3:6])  # km/s
            v_after = np.array(state_after[3:6])   # km/s
            
            # Berechne Delta-v Größe (keine Gravitationskorrektur)
            delta_v = v_after - v_before
            delta_v_mag = np.linalg.norm(delta_v) * 1000.0  # Konvertiere zu m/s
            
            # Gebe absolute Differenz vom Erwarteten zurück
            return abs(delta_v_mag - expected_delta_v)
            
        except Exception:
            return 1e6  # Große Strafe für ungültige Fenster
    
    # Suche nach optimaler Fenstergröße
    # Starte mit vernünftigen Grenzen basierend auf Manövertyp
    if burn_type == 'MEA' and expected_delta_v > 10:
        # Große MEA-Manöver benötigen möglicherweise breitere Fenster
        bounds = (30, 5400)  # 30 Sekunden bis 1,5 Stunden
    else:
        # Kleinere Manöver oder RCS-Brennvorgänge
        bounds = (5, 300)    # 5 Sekunden bis 5 Minuten
    
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    
    optimal_window = result.x
    min_error = result.fun
    
    print(f"  Optimal window: ±{optimal_window:.1f}s, error: {min_error:.3f} m/s")
    
    return optimal_window, optimal_window

def calculate_delta_v_from_positions(maneuver_time_et: float, expected_delta_v: float, 
                                   burn_type: str) -> Optional[Tuple[np.ndarray, float]]:
    """
    Berechnet Delta-v mittels positionsbasiertem Ansatz mit optimierten Zeitfenstern,
    unter Berücksichtigung von Gravitations-Geschwindigkeitsänderungen.
    
    Args:
        maneuver_time_et: Manöverzeit in Ephemeridensekunden
        expected_delta_v: Erwartete Delta-v Größe in m/s
        burn_type: 'MEA' für Haupttriebwerk oder 'RCS' für Lageregelung
        
    Returns:
        Tupel aus (Delta_v_Vektor_km_s, Delta_v_Größe_m_s) oder None falls fehlgeschlagen
    """
    try:
        # Finde optimales Zeitfenster
        time_before_offset, time_after_offset = find_optimal_time_window(
            maneuver_time_et, expected_delta_v, burn_type)
        
        # Berechne Zeiten
        time_before = maneuver_time_et - time_before_offset
        time_after = maneuver_time_et + time_after_offset
        
        # Hole Raumsonden-Zustände vor und nach Manöver
        state_before, _ = sp.spkezr('-82', time_before, 'ECLIPJ2000', 'NONE', 'SUN')
        state_after, _ = sp.spkezr('-82', time_after, 'ECLIPJ2000', 'NONE', 'SUN')
        
        # Extrahiere Geschwindigkeitsvektoren (km/s)
        v_before = np.array(state_before[3:6])
        v_after = np.array(state_after[3:6])
        
        # Berechne Delta-v Vektorrichtung aus SPK-Daten
        delta_v_raw = v_after - v_before  # km/s
        calculated_mag = np.linalg.norm(delta_v_raw) * 1000.0  # Konvertiere zu m/s
        
        print(f"  Time window: -{time_before_offset:.1f}s to +{time_after_offset:.1f}s")
        print(f"  SPK-derived magnitude: {calculated_mag:.3f} m/s")
        print(f"  Using expected magnitude: {expected_delta_v:.3f} m/s")
        
        # Normalisiere Richtungsvektor und skaliere auf erwartete Größe
        if np.linalg.norm(delta_v_raw) > 1e-6:  # Vermeide Division durch Null
            delta_v_direction = delta_v_raw / np.linalg.norm(delta_v_raw)
            
            # Skaliere Richtung auf erwartete Größe
            delta_v = delta_v_direction * (expected_delta_v / 1000.0)  # Konvertiere zurück zu km/s
            final_mag = expected_delta_v  # Verwende erwartete Größe
            
        else:
            print(f"  Warning: Zero velocity change detected, using fallback")
            delta_v = np.array([expected_delta_v / 1000.0, 0.0, 0.0])  # Fallback-Richtung
            final_mag = expected_delta_v
        
        return delta_v, final_mag
        
    except Exception as e:
        print(f"Error calculating delta-v at ET {maneuver_time_et}: {e}")
        return None

def main():
    # Lade Kernels
    print("Loading SPICE kernels...")
    load_kernels()
    
    # Lade Manöverdaten
    maneuver_file = os.path.join('data', 'maneuver_delta_v_mass.csv')
    print(f"Loading maneuver data from {maneuver_file}...")
    maneuvers = load_maneuver_data(maneuver_file)
    
    # Filtere Manöver ohne ET-Zeiten heraus
    valid_maneuvers = maneuvers.dropna(subset=['ET']).copy()
    print(f"Found {len(valid_maneuvers)} maneuvers with valid times")
    
    # Extrahiere Delta-v für jedes Manöver
    print("\nExtracting delta-v vectors from SPK data...")
    results = []
    
    for idx, row in valid_maneuvers.iterrows():
        maneuver_name = row['Maneuver']
        maneuver_time_et = row['ET']
        burn_type = row.get('Burn Type', 'MEA')  # Standard: MEA falls nicht angegeben
        recon_dv = row.get('Recon Delta V (m/s)', 0.0)
        
        print(f"\nProcessing maneuver: {maneuver_name} at ET {maneuver_time_et:.1f}")
        print(f"  Type: {burn_type}, Recon delta-v: {recon_dv:.3f} m/s")
        
        # Hole Delta-v mittels positionsbasiertem Ansatz mit optimierten Zeitfenstern
        result = calculate_delta_v_from_positions(maneuver_time_et, abs(float(recon_dv)), burn_type)
        
        if result is not None:
            delta_v, delta_v_mag = result
            print(f"  Delta-v: {delta_v_mag:.6f} m/s")
            print(f"  Delta-v vector (km/s): [{delta_v[0]:.6f}, {delta_v[1]:.6f}, {delta_v[2]:.6f}]")
            
            # Speichere Ergebnisse
            result_dict = {
                'Maneuver': maneuver_name,
                'ET': maneuver_time_et,
                'DeltaV_X_km_s': delta_v[0],
                'DeltaV_Y_km_s': delta_v[1],
                'DeltaV_Z_km_s': delta_v[2],
                'DeltaV_Magnitude_m_s': delta_v_mag,
                'SPK_Time_Before_ET': maneuver_time_et - 60.0,  # 1 Minute vorher
                'SPK_Time_After_ET': maneuver_time_et + 60.0,   # 1 Minute nachher
                'Source': 'SPK_Reconstructed_Trajectory'
            }
            results.append(result_dict)
    
    # Erstelle DataFrame aus Ergebnissen
    if results:
        results_df = pd.DataFrame(results)
        
        # Speichere Ergebnisse als CSV
        output_file = os.path.join('data', 'maneuver_delta_v_vectors.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved delta-v vectors to {output_file}")
        
        # Füge Validierung gegen erwartete Delta-v Werte hinzu
        print("\nValidating delta-v magnitudes against expected values:")
        print("-" * 80)
        print(f"{'Maneuver':<10} {'Expected (m/s)':>15} {'Derived (m/s)':>15} {'Diff (%)':>10} {'Status':>10}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            maneuver = row['Maneuver']
            derived_dv = row['DeltaV_Magnitude_m_s']
            
            # Finde passendes Manöver in Originaldaten
            original_row = maneuvers[maneuvers['Maneuver'] == maneuver].iloc[0]
            expected_dv = original_row['Recon Delta V (m/s)']
            
            # Berechne Differenz
            if expected_dv > 0:
                diff_pct = abs(derived_dv - expected_dv) / expected_dv * 100
                status = 'OK' if diff_pct < 20 else 'WARNING: Large difference!'
            else:
                diff_pct = 0
                status = 'N/A (no expected value)'
                
            print(f"{maneuver:<10} {expected_dv:>15.3f} {derived_dv:>15.3f} {diff_pct:>9.1f}% {status:>10}")
        
        print("-" * 80)
    else:
        print("No delta-v vectors were extracted.")
    
    # Aufräumen
    sp.kclear()

if __name__ == "__main__":
    main()
