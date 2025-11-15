"""
main.py - Hauptprogramm der Cassini-Trajektorien-Simulation

Koordiniert die gesamte Simulation:
1. Lädt SPICE-Kernels
2. Initialisiert Manöver-Manager mit Manöverdaten
3. Holt Anfangszustand von Cassini aus SPICE
4. Führt numerische Integration mit Manövern durch
5. Erstellt interaktive 3D-Visualisierung der Trajektorie
"""

import os
import spiceypy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from spice_manager import *
from config import *
from plotting import plot_trajectory_3d, plot_2d_trajectory, create_trajectory_video, interactive_3d_plot, plot_mission_overview
from integration_impulse import integrate_with_impulse_maneuvers
from maneuvers_impulse import ImpulseManeuverManager

# Erstelle CONFIG Dictionary mit notwendigen Parametern
CONFIG = {
    'use_impulse_maneuvers': True  # Verwende Impuls-Manöver statt schubbasierte
}

def main():
    try:
        # Lade Kernel
        load_kernels()
        
        # Initialisiere Manöver-Manager mit Datendateien
        delta_v_file = os.path.join('data', 'maneuver_delta_v_vectors.csv')
        
        print("Using SPK-derived delta-v vectors from:", os.path.abspath(delta_v_file))
        
        # Wähle Manöver-Ansatz basierend auf Konfiguration
        if CONFIG.get('use_impulse_maneuvers', True):
            print("Using IMPULSE maneuver approach (instantaneous velocity changes)")
            impulse_manager = ImpulseManeuverManager()
            impulse_manager.load_maneuvers(delta_v_file)
        else:
            print("Using THRUST-based maneuver approach (continuous acceleration)")
            initialize_maneuver_manager(delta_v_file)
        
        # Rechne Zeiten in Ephemeridenzeit um
        et_start = sp.utc2et(START_UTC)
        et_end = sp.utc2et(END_UTC)
        
        # Definiere t_span
        t_span = (et_start, et_end)  # Tupel mit Start- und Endzeit
        
        print(f"Simulationsstart (ET): {et_start}")
        print(f"Simulationsende (ET): {et_end}") 
        print(f"Simulationsdauer: {(et_end - et_start)/86400:.1f} Tage")
        
        # Hole Anfangszustand von Cassini
        # Cassinis NAIF ID ist -82
        cassini_id = -82
        
        # Verwende konsistente Lichtlaufzeitkorrektur
        state_initial, _ = sp.spkezr(
            str(cassini_id),  # Cassini ID als String
            et_start,         # Startzeit
            'ECLIPJ2000',     # Referenzrahmen
            LIGHT_TIME_CORRECTION, # Lichtlaufzeitkorrektur
            'SUN'             # Relativ zur Sonne
        )
        
        # Zur Fehlersuche: zeige Anfangsposition
        print(f"Initial position (ECLIPJ2000): [{state_initial[0]:.3f}, {state_initial[1]:.3f}, {state_initial[2]:.3f}] km")
        
        # Füge Masse zum Zustandsvektor hinzu [x, y, z, vx, vy, vz, m]
        initial_mass = PHYSICAL_CONSTANTS['cassini']['initial_total_mass']
        y0 = np.concatenate([state_initial, [initial_mass]])
        
        # Wähle Integrationsansatz
        if CONFIG.get('use_impulse_maneuvers', True):
            # Verwende Impuls-Manöver Integration
            print("Starting impulse maneuver integration...")
            
            # Hole Manöverzeiten vom Impuls-Manager
            maneuver_times = impulse_manager.get_all_maneuver_times()
            valid_maneuver_times = maneuver_times[
                (maneuver_times >= et_start) & (maneuver_times <= et_end)
            ]
            print(f"Found {len(valid_maneuver_times)} maneuvers in simulation period")
            
            # Importiere Dynamikfunktion ohne Manövereffekte für Impuls-Integration
            from integration_impulse import dynamics_without_maneuvers
            
            # Verwende Impuls-Integrations-Wrapper
            solution_data = integrate_with_impulse_maneuvers(
                dynamics_func=dynamics_without_maneuvers,
                t_span=(et_start, et_end),
                y0=y0,
                maneuver_manager=impulse_manager,
                rtol=INTEGRATOR_TOLERANCES['rtol'],
                atol=INTEGRATOR_TOLERANCES['atol'],
                max_step=MAX_STEP_INTEGRATION,
            )
           
            if solution_data['success']:
                print(f"Impulse integration completed successfully!")
                print(f"Applied {len(solution_data['maneuver_states'])} maneuvers")
                
                # Führe Positionsverifizierung durch
                print("\n=== Position Verification ===")
                
                # Definiere einfachen Lösungs-Wrapper für Kompatibilität mit Plotting
                class ImpulseSolution:
                    def __init__(self, t, y, success):
                        self.t = t
                        self.y = y
                        self.success = success
                        self.t_events = []
                
                solution = ImpulseSolution(solution_data['t'], solution_data['y'], True)
                solutions = [solution]
            else:
                raise RuntimeError("Impulse integration failed!")
        
        # Kombiniere alle Lösungen für Plotting
        if solutions:
            # Bereite Daten für Plotting vor
            plot_data = []
            for sol in solutions:
                if sol.t_events and len(sol.t_events[0]) > 0:
                    # Für Lösungen mit Events, verwende Event-Zeit als Endpunkt
                    t = sol.t[sol.t <= sol.t_events[0][0]]
                    states = sol.sol(t)
                else:
                    t = sol.t
                    states = sol.y
                plot_data.append((t, states))
            
            # Plotte 3D Trajektorie
            plot_trajectory_3d(plot_data)
            
            # Erstelle Mission Overview Plot - Fokussiert auf simulierte Trajektorie
            print("Creating focused mission overview plot...")
            plot_mission_overview(plot_data, 
                                maneuver_manager=impulse_manager,
                                show_real_cassini=True,
                                save_path='Simulation_Results/mission_overview_focused.png',
                                title='',
                                focus_on_simulated=True)
            
            # Erstelle Mission Overview Plot - Vollständige Ansicht
            print("Creating full mission overview plot...")
            plot_mission_overview(plot_data, 
                                maneuver_manager=impulse_manager,
                                show_real_cassini=True,
                                save_path='Simulation_Results/mission_overview_full.png',
                                title='',
                                focus_on_simulated=False)
            
            plt.show()
    
            # Erstelle interaktiven 3D Plot mit Zeitschieber
            print("Creating interactive 3D plot...")
            interactive_3d_plot(plot_data, 'Interactive Cassini Trajectory', 
                              save_path='Simulation_Results/interactive_trajectory.html')
            
           
            # Plotte 2D Projektionen
            plot_2d_trajectory(plot_data, 0, 1, 'Cassini Trajectory (X-Y Plane)')
            plot_2d_trajectory(plot_data, 0, 2, 'Cassini Trajectory (X-Z Plane)')
            plot_2d_trajectory(plot_data, 1, 2, 'Cassini Trajectory (Y-Z Plane)')
        
        print("Integration completed successfully!")
        
    finally:
        unload_kernels()


if __name__ == "__main__":
    main()
