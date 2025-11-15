"""
integration_impulse.py - Numerische Integration mit Impuls-Manövern

Führt die numerische Integration der Cassini-Trajektorie durch.
Teilt die Integration in Segmente zwischen Manövern auf und wendet
die Geschwindigkeitsänderungen zu den exakten Zeitpunkten an.
Berechnet die Gravitationskräfte aller relevanten Himmelskörper.
"""

import numpy as np
import spiceypy as sp
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict, Any, Callable, Optional
from maneuvers_impulse import ImpulseManeuverManager
from config import MAX_STEP_INTEGRATION, SOLVER_INTEGRATION
from dynamics import get_body_position
from config import (TARGET_BODIES, TARGET_BODIES_GM_ARRAY, INTEGRATOR_TOLERANCES)


def integrate_with_impulse_maneuvers(
    dynamics_func: Callable,
    t_span: Tuple[float, float], 
    y0: np.ndarray,
    maneuver_manager: ImpulseManeuverManager,
    method: str = SOLVER_INTEGRATION,
    rtol: float = INTEGRATOR_TOLERANCES['rtol'],
    atol: float = INTEGRATOR_TOLERANCES['atol'],
    max_step: float = MAX_STEP_INTEGRATION,
    dense_output: bool = True,
) -> Dict[str, Any]:
    """
    Integriert Dynamik mit Impuls-Manövern zu exakten Zeiten.
    
    Args:
        dynamics_func: Dynamikfunktion (ohne Manövereffekte)
        t_span: (Startzeit, Endzeit) in Sekunden
        y0: Anfangszustandsvektor [x, y, z, vx, vy, vz, m]
        maneuver_manager: ImpulseManeuverManager Instanz
        method: Integrationsmethode
        rtol, atol: Integrationstoleranzen
        max_step: Maximale Schrittgröße
        dense_output: Ob dichte Ausgabe aktiviert werden soll
        
    Returns:
        Dictionary mit kombinierten Lösungsdaten
    """
    
    # Hole alle Manöverzeiten innerhalb des Integrationsbereichs
    all_maneuver_times = maneuver_manager.get_all_maneuver_times()
    maneuver_times = all_maneuver_times[
        (all_maneuver_times >= t_span[0]) & (all_maneuver_times <= t_span[1])
    ]
    maneuver_times = np.sort(maneuver_times)
    
    print(f"Integrating with {len(maneuver_times)} impulse maneuvers")
    if len(maneuver_times) > 0:
        print(f"Maneuver times: {maneuver_times}")
    
    # Setze angewendete Manöver zurück
    maneuver_manager.reset_applied_maneuvers()
    
    # Falls keine Manöver, führe reguläre Integration durch
    if len(maneuver_times) == 0:
        print("No maneuvers in time span, performing regular integration")
        solution = solve_ivp(
                    dynamics_func, t_span, y0, method=method, rtol=rtol, atol=atol, max_step = max_step, dense_output=dense_output)
      
        return {
            'success': solution.success,
            'message': solution.message,
            't': solution.t,
            'y': solution.y,
            'sol': solution.sol if dense_output else None,
            'maneuver_times': maneuver_times,
            'maneuver_states': []
        }
    
    # Erstelle Integrationssegmente zwischen Manövern
    segment_times = [t_span[0]] + list(maneuver_times) + [t_span[1]]
    
    # Speicher für kombinierte Ergebnisse
    all_times = []
    all_states = []
    maneuver_states = []  # Zustände direkt nach Manövern
    
    current_state = y0.copy()
    
    for i in range(len(segment_times) - 1):
        segment_start = segment_times[i]
        segment_end = segment_times[i + 1]
        
        print(f"\nIntegrating segment {i+1}/{len(segment_times)-1}: {segment_start:.1f} to {segment_end:.1f}")
        print(f"  UTC: {sp.et2utc(segment_start, 'ISOC', 0)} to {sp.et2utc(segment_end, 'ISOC', 0)}")
        
        # Integriere dieses Segment
        if segment_end > segment_start:  # Nur integrieren falls Zeitdifferenz vorhanden
            solution = solve_ivp(
                dynamics_func,
                (segment_start, segment_end),
                current_state,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=dense_output,
            )
            
            if not solution.success:
                print(f"Integration failed in segment {i+1}: {solution.message}")
                return {
                    'success': False,
                    'message': f"Integration failed in segment {i+1}: {solution.message}",
                    't': np.array(all_times) if all_times else np.array([]),
                    'y': np.array(all_states).T if all_states else np.array([]),
                    'sol': None,
                    'maneuver_times': maneuver_times,
                    'maneuver_states': maneuver_states
                }
            
            # Speichere Ergebnisse dieses Segments
            if i == 0:
                # Erstes Segment: alle Punkte einbeziehen
                all_times.extend(solution.t)
                all_states.extend(solution.y.T)
            else:
                # Folgende Segmente: ersten Punkt überspringen um Duplikate zu vermeiden
                all_times.extend(solution.t[1:])
                all_states.extend(solution.y.T[1:])
            
            # Aktualisiere aktuellen Zustand auf Segmentende
            current_state = solution.y[:, -1].copy()
        
        # Wende Manöver an falls wir bei einer Manöverzeit sind
        if i < len(maneuver_times):
            maneuver_time = maneuver_times[i]
            print(f"\nApplying maneuver at ET={maneuver_time:.1f}")
            
            # Wende Impuls-Manöver an
            modified_state, maneuver_applied = maneuver_manager.check_and_apply_maneuver(
                maneuver_time, current_state, tolerance=1.0
            )
            
            if maneuver_applied:
                current_state = modified_state
                maneuver_states.append({
                    'time': maneuver_time,
                    'state_before': solution.y[:, -1].copy() if 'solution' in locals() else current_state,
                    'state_after': current_state.copy()
                })
                print(f"Maneuver applied successfully")
            else:
                print(f"Warning: No maneuver applied at time {maneuver_time}")
    
    # Konvertiere zu Numpy-Arrays
    all_times = np.array(all_times)
    all_states = np.array(all_states).T
    
    print(f"\nIntegration completed successfully")
    print(f"Total time points: {len(all_times)}")
    print(f"Maneuvers applied: {len(maneuver_states)}")
    
    return {
        'success': True,
        'message': 'Integration with impulse maneuvers completed successfully',
        't': all_times,
        'y': all_states,
        'sol': None,  # Dense output not available for segmented integration
        'maneuver_times': maneuver_times,
        'maneuver_states': maneuver_states,
        'full_data': {  # Behalte vollständige Daten für Analyse
            't': all_times,
            'y': all_states
        }
    }

def dynamics_without_maneuvers(t: float, y: np.ndarray) -> np.ndarray:
    """
    Vektorisierte Dynamikfunktion ohne Manövereffekte.
    Reine Punktmassen-Gravitations-Simulation
    """
    # Entpacke Zustandsvektor
    r = y[:3]  # Position (km)
    v = y[3:6]  # Geschwindigkeit (km/s)
    m = y[6]  # Masse (kg)

    # Sammle Körperpositionen und GM-Werte als Arrays
    body_positions = np.array([
        get_body_position(body, t, 'ECLIPJ2000', 'SUN') for body in TARGET_BODIES
    ])  # Form: (N, 3)
    gms = TARGET_BODIES_GM_ARRAY

    # Relativvektoren von Körper zu Raumsonde
    r_rel = r - body_positions  # shape: (N, 3)
    r_mag = np.linalg.norm(r_rel, axis=1)  # shape: (N,)

    # Maskiere kleine Distanzen um Division durch Null zu vermeiden
    mask = r_mag >= 1.0

    # Berechne Punktmassen-Beiträge: GM * r_rel / r^3 für gültige Körper
    inv_r3 = np.zeros_like(r_mag)
    inv_r3[mask] = 1.0 / (r_mag[mask] ** 3)

    # Multipliziere GM-Werte und r_rel Vektoren, dann summiere
    # Verwende Broadcasting: (N,1) * (N,) * (N,3) → (N,3)
    a = -(gms[:, None] * inv_r3[:, None] * r_rel).sum(axis=0)

    dm_dt = 0.0  # Kein Schub (Masse konstant)

    return np.concatenate([v, a, [dm_dt]])

