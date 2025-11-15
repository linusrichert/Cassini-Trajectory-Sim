"""
plotting.py - Visualisierungs-Werkzeuge für Cassini-Trajektorien

Stellt verschiedene Plotting-Funktionen bereit:
- 3D-Trajektorien-Plots (statisch und interaktiv)
- 2D-Projektionen der Trajektorie
- Interaktiver Plot mit Zeitschieber und Vergleich zu echten SPICE-Daten
- Animations-Funktionen für Videos mit Planetenbewegungen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, Button
from typing import List, Tuple, Optional, Dict, Any
import spiceypy as sp
from datetime import datetime
from dynamics import get_body_position, BODY_INFO
from config import kernel_paths, NAIF_IDS, LIGHT_TIME_CORRECTION
import mpld3

def plot_trajectory_3d(solutions: List[Tuple[np.ndarray, np.ndarray]], 
                      title: str = 'Cassini Trajectory') -> Tuple[plt.Figure, Axes3D]:
    """Plottet die 3D Trajektorie aus mehreren Lösungssegmenten.
    
    Args:
        solutions: Liste von (Zeiten, Zustände) Tupeln vom Integrator
        title: Titel für den Plot
        
    Returns:
        Tupel aus (Figure, Axes) für weitere Anpassungen
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Kombiniere alle Lösungen und wende zeitbasiertes Sampling an
    all_times = []
    all_states = []
    
    for times, states in solutions:
        all_times.append(times)
        all_states.append(states)
    
    if all_times:
        # Verkette alle Daten
        combined_times = np.concatenate(all_times)
        combined_states = np.hstack(all_states)
        
        # Sortiere nach Zeit
        sort_idx = np.argsort(combined_times)
        combined_times = combined_times[sort_idx]
        combined_states = combined_states[:, sort_idx]
        
        # Wende zeitbasiertes Downsampling für Visualisierung an
        if len(combined_times) > 5000:
            print(f"Applying time-based downsampling from {len(combined_times)} to ~2000 points for better visualization")
            
            # Erstelle gleichmäßiges Zeitgitter
            time_span = combined_times[-1] - combined_times[0]
            uniform_times = np.linspace(combined_times[0], combined_times[-1], 2000)
            
            # Interpoliere Positionen zu gleichmäßigen Zeiten
            uniform_x = np.interp(uniform_times, combined_times, combined_states[0])
            uniform_y = np.interp(uniform_times, combined_times, combined_states[1])
            uniform_z = np.interp(uniform_times, combined_times, combined_states[2])
            
            # Plotte gleichmäßig gesampelte Trajektorie
            ax.plot(uniform_x, uniform_y, uniform_z, 
                   linewidth=1.5, alpha=0.7, label='Cassini Trajectory')
        else:
            # Plotte direkt falls nicht zu viele Punkte
            ax.plot(combined_states[0], combined_states[1], combined_states[2], 
                   linewidth=1.5, alpha=0.7, label='Cassini Trajectory')
    
    # Füge Sonne am Ursprung hinzu
    ax.scatter([0], [0], [0], color='yellow', s=100, label='Sun')
    
    # Füge Beschriftungen und Titel hinzu
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    ax.legend()
    
    # Gleiches Seitenverhältnis
    if all_states:
        all_combined_states = np.hstack(all_states)
        max_range = np.array([
            all_combined_states[0].max() - all_combined_states[0].min(),
            all_combined_states[1].max() - all_combined_states[1].min(),
            all_combined_states[2].max() - all_combined_states[2].min()
        ]).max() / 2.0
        
        mid_x = (all_combined_states[0].max() + all_combined_states[0].min()) * 0.5
        mid_y = (all_combined_states[1].max() + all_combined_states[1].min()) * 0.5
        mid_z = (all_combined_states[2].max() + all_combined_states[2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax

def plot_2d_trajectory(solutions: List[Tuple[np.ndarray, np.ndarray]], 
                      x_axis: int = 0, y_axis: int = 1,
                      title: str = 'Cassini Trajectory (2D)') -> Tuple[plt.Figure, plt.Axes]:
    """Plottet eine 2D Projektion der Trajektorie.
    
    Args:
        solutions: Liste von (Zeiten, Zustände) Tupeln
        x_axis: Index der x-Achse (0, 1, oder 2 für X, Y, Z)
        y_axis: Index der y-Achse (0, 1, oder 2 für X, Y, Z)
        title: Titel für den Plot
        
    Returns:
        Tupel aus (Figure, Axes) für weitere Anpassungen
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plotte jedes Lösungssegment
    for times, states in solutions:
        ax.plot(states[x_axis], states[y_axis], 
               linewidth=1.5, alpha=0.7, label='Cassini Trajectory')
    
    # Add the Sun at the origin
    ax.scatter([0], [0], color='yellow', s=100, label='Sun')
    
    # Add labels and title
    axis_labels = ['X', 'Y', 'Z']
    ax.set_xlabel(f'{axis_labels[x_axis]} (km)')
    ax.set_ylabel(f'{axis_labels[y_axis]} (km)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    return fig, ax

def interactive_3d_plot(solutions: List[Tuple[np.ndarray, np.ndarray]], 
                       title: str = 'Interactive Cassini Trajectory',
                       show_real_cassini: bool = True,
                       real_kernel_paths: Optional[List[str]] = None,
                       aberration_correction: str = LIGHT_TIME_CORRECTION,
                       save_path: Optional[str] = None) -> None:
    """Erstellt einen interaktiven 3D Plot der Trajektorie mit Zeitschieber.
    
    Args:
        solutions: Liste von (Zeiten, Zustände) Tupeln vom Integrator
        title: Titel für den Plot
        save_path: Optionaler Pfad zum Speichern des Plots
    """
    # Stelle sicher dass Kernel verfügbar sind falls echte Trajektorie angefordert
    if show_real_cassini:
        _load_kernels_with_leapseconds(real_kernel_paths)

    # Kombiniere alle Lösungen
    all_times = []
    all_states = []
    
    for times, states in solutions:
        if times is not None and states is not None:
            all_times.append(times)
            all_states.append(states)
    
    if not all_times:
        raise ValueError("No valid solution data found")
    
    # Kombiniere alle Daten
    times = np.concatenate(all_times)
    states = np.hstack(all_states)

    # Optional: Berechne echte Cassini Positionen und Geschwindigkeiten für dieselben Zeiten
    real_positions = None
    real_velocities = None
    if show_real_cassini:
        try:
            real_positions, real_velocities = _get_cassini_state_over_time(
                times,
                observer='SUN',
                frame='ECLIPJ2000',
                aberration=aberration_correction
            )  # shapes (N,3), (N,3)
        except Exception as e:
            print(f"Warning: failed to compute real Cassini positions: {e}")
            real_positions = None
            real_velocities = None
    
    # Erstelle Figure mit angepasstem Layout für Schieber
    fig = plt.figure(figsize=(12, 10))
    
    # Erstelle 3D Achse mit angepasster Position für Schieber
    ax = fig.add_axes([0.1, 0.25, 0.8, 0.7], projection='3d')
    
    # Plotte vollständige Trajektorie in heller Farbe
    full_traj, = ax.plot(states[0], states[1], states[2], 
                        'b-', alpha=0.3, linewidth=0.5, label='Full Trajectory')
    
    # Plotte aktuelle Position (startet am Anfang)
    current_pos, = ax.plot([states[0,0]], [states[1,0]], [states[2,0]], 
                          'ro', markersize=8, label='Current Position')

    # Falls verfügbar, plotte echte Cassini Trajektorie und aktuelle Position
    real_traj_line = None
    real_pos_marker = None
    if real_positions is not None:
        real_traj_line, = ax.plot(real_positions[:, 0], real_positions[:, 1], real_positions[:, 2],
                                  color='green', alpha=0.4, linewidth=0.8, label='Cassini (SPICE)')
        real_pos_marker, = ax.plot([real_positions[0, 0]], [real_positions[0, 1]], [real_positions[0, 2]],
                                   'mo', markersize=6, label='Cassini (SPICE) pos')
    
    # Add the Sun at the origin
    sun = ax.scatter([0], [0], [0], color='yellow', s=150, label='Sun')
    
    # Setze Beschriftungen und Titel
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    
    # Setze gleiches Seitenverhältnis
    max_range = np.array([
        states[0].max() - states[0].min(),
        states[1].max() - states[1].min(),
        states[2].max() - states[2].min()
    ]).max() / 2.0
    
    mid_x = (states[0].max() + states[0].min()) * 0.5
    mid_y = (states[1].max() + states[1].min()) * 0.5
    mid_z = (states[2].max() + states[2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Füge Zeit-Text-Annotation hinzu
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.7))

    # Füge Geschwindigkeits-Info hinzu (simuliert vs echt)
    speed_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes,
                          bbox=dict(facecolor='white', alpha=0.7))

    # Füge Distanz-zur-Sonne Verhältnis hinzu (echt/simuliert)
    dist_ratio_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes,
                                bbox=dict(facecolor='white', alpha=0.7))
    
    # Füge Schieber-Achse hinzu
    ax_slider = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    time_slider = Slider(
        ax=ax_slider,
        label='Time Step',
        valmin=0,
        valmax=len(times)-1,
        valinit=0,
        valstep=1
    )
    
    # Füge Play/Pause Button hinzu
    ax_button = fig.add_axes([0.45, 0.15, 0.1, 0.04])
    button = Button(ax_button, 'Play/Pause')
    
    # Animations-Steuerung
    is_playing = False
    current_index = 0
    
    # Vorberechnung simulierter Geschwindigkeiten (km/s) falls Geschwindigkeit vorhanden
    sim_speeds = None
    if states.shape[0] >= 6:
        sim_speeds = np.linalg.norm(states[3:6, :], axis=0)

    # Vorberechnung Distanzen zur Sonne (km)
    sim_dists = np.linalg.norm(states[0:3, :], axis=0)
    real_dists = None
    if show_real_cassini and (real_positions is not None):
        real_dists = np.linalg.norm(real_positions, axis=1)

    def update(val):
        """Aktualisiert Plot wenn Schieber geändert wird."""
        nonlocal current_index
        current_index = int(time_slider.val)
        
        # Aktualisiere Positionsmarker
        current_pos.set_data(
            [states[0, current_index]],
            [states[1, current_index]]
        )
        current_pos.set_3d_properties([states[2, current_index]])
        
        # Aktualisiere Zeit-Text
        et = times[current_index]
        utc = sp.et2utc(et, 'ISOC', 0)
        time_text.set_text(f'Time: {utc}')

        # Aktualisiere Geschwindigkeits-HUD
        speed_info = ''
        sim_speed = None
        real_speed = None
        if sim_speeds is not None:
            sim_speed = sim_speeds[current_index]
        if real_velocities is not None and 0 <= current_index < len(real_velocities):
            if not np.any(np.isnan(real_velocities[current_index])):
                real_speed = np.linalg.norm(real_velocities[current_index])
        if sim_speed is not None and real_speed is not None:
            diff = abs(sim_speed - real_speed)
            speed_info = f"Speed sim: {sim_speed:,.3f} km/s | real: {real_speed:,.3f} km/s | |Δv|: {diff:,.3f} km/s"
        elif sim_speed is not None:
            speed_info = f"Speed sim: {sim_speed:,.3f} km/s | real: N/A | |Δv|: N/A"
        elif real_speed is not None:
            speed_info = f"Speed sim: N/A | real: {real_speed:,.3f} km/s | |Δv|: N/A"
        else:
            speed_info = "Speed sim: N/A | real: N/A | |Δv|: N/A"
        speed_text.set_text(speed_info)

        # Aktualisiere Distanzverhältnis-HUD
        dist_info = ''
        sim_dist = None
        real_dist = None
        if sim_dists is not None:
            sim_dist = sim_dists[current_index]
        if real_dists is not None and 0 <= current_index < len(real_dists):
            if not np.isnan(real_dists[current_index]):
                real_dist = real_dists[current_index]
        if sim_dist is not None and real_dist is not None and sim_dist != 0 and not np.isnan(sim_dist):
            ratio = real_dist / sim_dist
            dist_info = f"|r|_real/|r|_sim: {ratio:,.6f}  (real: {real_dist:,.0f} km, sim: {sim_dist:,.0f} km)"
        elif sim_dist is not None:
            dist_info = f"|r|_real/|r|_sim: N/A  (real: N/A, sim: {sim_dist:,.0f} km)"
        elif real_dist is not None:
            dist_info = f"|r|_real/|r|_sim: N/A  (real: {real_dist:,.0f} km, sim: N/A)"
        else:
            dist_info = "|r|_real/|r|_sim: N/A"
        dist_ratio_text.set_text(dist_info)

        # Aktualisiere echten Cassini Positionsmarker
        if real_positions is not None and real_pos_marker is not None:
            if 0 <= current_index < len(real_positions):
                real_pos_marker.set_data([real_positions[current_index, 0]], [real_positions[current_index, 1]])
                real_pos_marker.set_3d_properties([real_positions[current_index, 2]])
        
        fig.canvas.draw_idle()
    
    def animate(frame):
        """Animationsfunktion für Auto-Play."""
        if is_playing:
            current_index = (frame % len(times))
            time_slider.set_val(current_index)
            return current_pos, time_text, speed_text, dist_ratio_text
    
    def toggle_play(event):
        """Schaltet Play/Pause Animation um."""
        nonlocal is_playing
        is_playing = not is_playing
        if is_playing:
            ani.event_source.start()
        else:
            ani.event_source.stop()
    
    # Verbinde Update-Funktion mit Schieber
    time_slider.on_changed(update)
    button.on_clicked(toggle_play)
    
    # Erstelle Animation für Auto-Play
    ani = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=False)
    ani.event_source.stop()  # Starte pausiert
    
    # Initiales Update
    update(0)
    
    # Füge Legende hinzu
    ax.legend()
    
    # Speichere Plot falls angefordert
    if save_path:
        _save_interactive_plot(fig, save_path)
    
    try:
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout for slider
    except:
        pass  # Skip if tight_layout fails
    
    plt.show()

def _load_kernels_with_leapseconds(extra_kernel_paths: Optional[List[str]] = None) -> None:
    """Stellt sicher dass Leapseconds und SPKs für Plotting geladen sind.
    Versucht zuerst config.kernel_paths, dann relative Pfade.
    """
    try:
        # Lade Leapseconds falls nicht geladen
        if sp.ktotal('LSK') == 0:
            candidates = [
                'kernels/naif0012.tls',
                '../kernels/naif0012.tls'
            ]
            for c in candidates:
                if os.path.exists(c):
                    sp.furnsh(c)
                    print(f"Loaded LSK: {c}")
                    break

        # Lade zuerst aus Config
        if kernel_paths:
            for path in kernel_paths:
                if os.path.exists(path):
                    try:
                        sp.furnsh(path)
                    except Exception:
                        pass

        # Lade explizit angegebene Kernel (z.B. Cassini SPKs)
        if extra_kernel_paths:
            for path in extra_kernel_paths:
                if path and os.path.exists(path):
                    try:
                        sp.furnsh(path)
                        print(f"Loaded kernel: {path}")
                    except Exception as e:
                        print(f"Warning: could not load kernel {path}: {e}")
    except Exception as e:
        print(f"Kernel loading warning: {e}")

def _get_cassini_positions_over_time(et_times: np.ndarray,
                                     observer: str = 'SUN',
                                     frame: str = 'ECLIPJ2000',
                                     aberration: str = LIGHT_TIME_CORRECTION) -> np.ndarray:
    """Fragt SPICE nach Cassini Positionen relativ zu einem Beobachter.

    Gibt Array der Form (N,3) zurück mit NaN wo Daten nicht verfügbar.
    """
    positions = np.full((len(et_times), 3), np.nan)
    target = 'CASSINI' if 'CASSINI' in NAIF_IDS else '-82'
    for i, et in enumerate(et_times):
        try:
            state, lt = sp.spkezr(target, float(et), frame, aberration, observer)
            positions[i, :] = state[:3]
        except Exception:
            # Behalte NaNs wenn außerhalb der Abdeckung
            continue
    return positions

def _get_cassini_state_over_time(et_times: np.ndarray,
                                 observer: str = 'SUN',
                                 frame: str = 'ECLIPJ2000',
                                 aberration: str = LIGHT_TIME_CORRECTION) -> Tuple[np.ndarray, np.ndarray]:
    """Fragt SPICE nach Cassini Zustand (Position, Geschwindigkeit).

    Returns:
        positions: (N,3) km
        velocities: (N,3) km/s
    """
    positions = np.full((len(et_times), 3), np.nan)
    velocities = np.full((len(et_times), 3), np.nan)
    target = 'CASSINI' if 'CASSINI' in NAIF_IDS else '-82'
    for i, et in enumerate(et_times):
        try:
            state, lt = sp.spkezr(target, float(et), frame, aberration, observer)
            positions[i, :] = state[:3]
            velocities[i, :] = state[3:6]
        except Exception:
            # Belasse NaNs wenn außerhalb der Abdeckung
            continue
    return positions, velocities

def _save_interactive_plot(fig, save_path: str) -> None:
    """Speichert Plot als hochqualitatives PNG mit Zeitstempel."""
    try:
        # Erstelle Verzeichnis falls nicht vorhanden
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Generiere Dateinamen mit Zeitstempel
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = f"{os.path.splitext(save_path)[0]}_{timestamp}.png"
        
        # Speichere als hochqualitatives PNG
        fig.savefig(png_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"High-quality PNG saved to: {os.path.abspath(png_path)}")
    except Exception as e:
        print(f"Could not save plot: {e}")

def compute_planet_positions_over_time(et_times: np.ndarray, 
                                      planets: List[str] = None) -> Dict[str, np.ndarray]:
    """Berechnet Planetenpositionen über Zeit mittels SPICE.
    
    Args:
        et_times: Array von Ephemeridenzeiten
        planets: Liste von Planetennamen. Falls None, verwendet Hauptplaneten.
        
    Returns:
        Dictionary mit Planetennamen zu Positionsarrays (N, 3)
    """
    if planets is None:
        planets = ['MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE']
    
    planet_positions = {}
    
    for planet in planets:
        if planet in BODY_INFO:
            positions = []
            for et in et_times:
                try:
                    pos, _ = get_body_position(planet, et)
                    positions.append(pos)
                except Exception as e:
                    # Falls Positionsberechnung fehlschlägt, verwende vorherige Position oder Null
                    if positions:
                        positions.append(positions[-1])
                    else:
                        positions.append(np.zeros(3))
            
            planet_positions[planet] = np.array(positions)
    
    return planet_positions

def create_trajectory_animation(solutions: List[Tuple[np.ndarray, np.ndarray]],
                               planet_positions: Dict[str, np.ndarray] = None,
                               output_file: str = 'Simulation_Results/cassini_trajectory.mp4',
                               fps: int = 30,
                               show_planets: bool = True,
                               show_trails: bool = True) -> FuncAnimation:
    """Erstellt animiertes Video der Cassini Trajektorie mit Planetenbewegungen.
    
    Args:
        solutions: Liste von (Zeiten, Zustände) Tupeln vom Integrator
        planet_positions: Dictionary von Planetenpositionen über Zeit (optional)
        output_file: Ausgabe-Videodateipfad
        fps: Frames pro Sekunde für Animation
        show_planets: Ob Planetenpositionen gezeigt werden sollen
        show_trails: Ob Pfadspuren gezeigt werden sollen
        
    Returns:
        Das matplotlib FuncAnimation Objekt
    """
    
    # Kombiniere alle Lösungsdaten
    all_times = []
    all_states = []
    
    for sol in solutions:
        if sol[0] is not None and sol[1] is not None:
            all_times.append(sol[0])
            all_states.append(sol[1])
    
    if not all_times:
        raise ValueError("No valid solution data found")
    
    # Verkette Zeit- und Zustandsarrays
    times = np.concatenate(all_times)
    states = np.hstack(all_states)
    
    # Sortiere nach Zeit
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    states = states[:, sort_idx]
    
    # Erstelle Figure und 3D Achse
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setze Plot-Grenzen basierend auf Trajektorie
    max_range = np.array([
        states[0].max() - states[0].min(),
        states[1].max() - states[1].min(),
        states[2].max() - states[2].min()
    ]).max() / 2.0
    
    mid_x = (states[0].max() + states[0].min()) * 0.5
    mid_y = (states[1].max() + states[1].min()) * 0.5
    mid_z = (states[2].max() + states[2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    
    # Füge Sonne hinzu
    sun_plot = ax.scatter([0], [0], [0], color='yellow', s=200, label='Sun', alpha=0.8)
    
    # Initialisiere Plot-Elemente
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.8, label='Cassini Trajectory')
    cassini_point, = ax.plot([], [], [], 'ro', markersize=8, label='Cassini')
    
    # Planeten-Plots
    planet_colors = {
        'MERCURY': 'gray',
        'VENUS': 'orange', 
        'EARTH': 'blue',
        'MARS': 'red',
        'JUPITER': 'brown',
        'SATURN': 'goldenrod',
        'URANUS': 'lightblue',
        'NEPTUNE': 'darkblue'
    }
    
    planet_plots = {}
    planet_trails = {}
    
    if show_planets and planet_positions:
        for planet, positions in planet_positions.items():
            color = planet_colors.get(planet, 'purple')
            planet_plots[planet] = ax.scatter([], [], [], color=color, s=50, label=planet, alpha=0.8)
            if show_trails:
                planet_trails[planet], = ax.plot([], [], [], color=color, alpha=0.3, linewidth=1)
    
    # Zeit-Text
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Erstelle größere, besser lesbare Legende
    legend = ax.legend(
        loc='upper right', 
        bbox_to_anchor=(1.02, 1), 
        borderaxespad=0.,
        fontsize=12,  # Größere Schrift
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Erhöhe die Größe der Legendenpunkte
    for handle in legend.legend_handles:
        handle.set_sizes([100])  # Größere Punkte in der Legende
    
    # Animations-Parameter
    total_frames = len(times)
    frame_skip = max(1, total_frames // (fps * 60))  # Ziel: ~60 Sekunden Video
    
    def animate(frame):
        # Aktualisiere Trajektorie bis zum aktuellen Frame
        current_idx = min(frame * frame_skip, len(times) - 1)
        
        # Aktualisiere Trajektorien-Linie
        trajectory_line.set_data(states[0, :current_idx+1], states[1, :current_idx+1])
        trajectory_line.set_3d_properties(states[2, :current_idx+1])
        
        # Aktualisiere Cassini Position
        cassini_point.set_data([states[0, current_idx]], [states[1, current_idx]])
        cassini_point.set_3d_properties([states[2, current_idx]])
        
        # Aktualisiere Planetenpositionen
        if show_planets and planet_positions:
            for planet, positions in planet_positions.items():
                if planet in planet_plots:
                    pos_idx = min(current_idx, len(positions) - 1)
                    pos = positions[pos_idx]
                    
                    # Aktualisiere Planetenposition
                    planet_plots[planet]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                    
                    # Aktualisiere Planetenspur
                    if show_trails and planet in planet_trails:
                        trail_data = positions[:pos_idx+1]
                        planet_trails[planet].set_data(trail_data[:, 0], trail_data[:, 1])
                        planet_trails[planet].set_3d_properties(trail_data[:, 2])
        
        # Aktualisiere Zeitstempel
        current_time = times[current_idx]
        utc_time = sp.et2utc(current_time, 'ISOC', 0)
        
        # Formatiere Zeit schön
        try:
            dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = utc_time
        
        time_text.set_text(f'Time: {time_str}\nFrame: {current_idx + 1}/{len(times)}')
        
        return [trajectory_line, cassini_point, time_text] + list(planet_plots.values()) + list(planet_trails.values())
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=total_frames // frame_skip, 
                        interval=1000/fps, blit=True, repeat=False)
    
    # Speichere Animation
    try:
        # Erstelle Verzeichnis falls nicht vorhanden
        output_dir = os.path.dirname(output_file) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Bestimme Ausgabedateiname und Format
        base_name = os.path.splitext(output_file)[0]
        gif_file = f"{base_name}.gif"
        
        print(f"Saving animation to: {os.path.abspath(gif_file)}")
        print(f"Using Pillow writer (GIF format)")
        
        # Speichere als GIF mittels Pillow
        anim.save(gif_file, writer='pillow', fps=fps)
        print(f"Successfully saved animation to: {os.path.abspath(gif_file)}")
        
        # Gebe tatsächlichen gespeicherten Dateipfad zurück
        return gif_file
        
    except Exception as e:
        print(f"\nFailed to save animation. Error details:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if you have write permissions in the output directory")
        print("2. Try using a different output directory")
        print("3. Check available disk space")
    
    return anim

def create_trajectory_video(solutions: List[Tuple[np.ndarray, np.ndarray]],
                           output_file: str = 'Simulation_Results/cassini_animation.gif',
                           fps: int = 30,
                           include_planets: bool = True,
                           planet_list: List[str] = None) -> str:
    """Hilfsfunktion zum Erstellen eines Trajektorien-Videos mit Planeten.
    
    Args:
        solutions: Liste von (Zeiten, Zustände) Tupeln vom Integrator
        output_file: Ausgabe-Videodateipfad
        fps: Frames pro Sekunde
        include_planets: Ob Planetenbewegungen einbezogen werden sollen
        planet_list: Liste von Planeten (Standard: Hauptplaneten)
        
    Returns:
        Pfad zur erstellten Videodatei
    """
    
    # Hole Zeitbereich aus Lösungen
    all_times = []
    for sol in solutions:
        if sol[0] is not None:
            all_times.extend(sol[0])
    
    if not all_times:
        raise ValueError("No time data found in solutions")
    
    times = np.array(sorted(set(all_times)))  # Entferne Duplikate und sortiere
    
    # Berechne Planetenpositionen falls angefordert
    planet_positions = None
    if include_planets:
        print("Computing planet positions over time...")
        planet_positions = compute_planet_positions_over_time(times, planet_list)
    
    # Erstelle Animation
    print("Creating animation...")
    anim = create_trajectory_animation(
        solutions=solutions,
        planet_positions=planet_positions,
        output_file=output_file,
        fps=fps,
        show_planets=include_planets
    )
    
    return output_file

def plot_mission_overview(solutions: List[Tuple[np.ndarray, np.ndarray]],
                         maneuver_manager=None,
                         show_real_cassini: bool = True,
                         show_simulated_cassini: bool = True,
                         real_kernel_paths: Optional[List[str]] = None,
                         aberration_correction: str = LIGHT_TIME_CORRECTION,
                         save_path: Optional[str] = None,
                         title: str = 'Simulation vs. Reale Daten',
                         focus_on_simulated: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Erstellt einen 2D Überblicksplot der Mission
    
    Zeigt:
    - Planetenbahnen (echte SPICE-Daten)
    - Cassini Trajektorie (simuliert, optional)
    - Cassini Trajektorie (SPICE, optional)
    - Sonne im Zentrum
    - Wichtige Manöver und Flyby-Ereignisse
    
    Args:
        solutions: Liste von (Zeiten, Zustände) Tupeln vom Integrator
        maneuver_manager: Manöver-Manager für Manöver-Informationen
        show_real_cassini: Ob echte SPICE-Trajektorie gezeigt werden soll
        show_simulated_cassini: Ob simulierte Trajektorie gezeigt werden soll
        real_kernel_paths: Optionale zusätzliche Kernel-Pfade
        aberration_correction: Lichtlaufzeitkorrektur
        save_path: Optionaler Pfad zum Speichern
        title: Titel für den Plot
        focus_on_simulated: Wenn True, fokussiert auf simulierte Trajektorie
        
    Returns:
        Tupel aus (Figure, Axes)
    """
    # Stelle sicher dass Kernel verfügbar sind
    _load_kernels_with_leapseconds(real_kernel_paths)
    
    # Kombiniere alle Lösungen
    all_times = []
    all_states = []
    
    for times, states in solutions:
        if times is not None and states is not None:
            all_times.append(times)
            all_states.append(states)
    
    if not all_times:
        raise ValueError("No valid solution data found")
    
    # Kombiniere alle Daten
    times = np.concatenate(all_times)
    states = np.hstack(all_states)
    
    # Sortiere nach Zeit
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    states = states[:, sort_idx]
    
    # Berechne echte Cassini Positionen falls angefordert
    real_positions = None
    if show_real_cassini:
        try:
            real_positions, _ = _get_cassini_state_over_time(
                times,
                observer='SUN',
                frame='ECLIPJ2000',
                aberration=aberration_correction
            )
        except Exception as e:
            print(f"Warning: failed to compute real Cassini positions: {e}")
            real_positions = None
    
    # Erstelle Figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Definiere Planeten und ihre Farben - hellere Farben für bessere Druckbarkeit
    planets_info = {
        'VENUS': {'color': '#FFA500', 'size': 80, 'label': 'Venus'},  # Bright Orange
        'EARTH': {'color': '#4169E1', 'size': 80, 'label': 'Erde'},   # Royal Blue
        'JUPITER': {'color': '#DAA520', 'size': 150, 'label': 'Jupiter'},  # Goldenrod
        'SATURN': {'color': '#FF4500', 'size': 130, 'label': 'Saturn'}  # OrangeRed
    }
    
    # Berechne Planetenpositionen über die gesamte Missionszeit
    # Für vollständige Bahnen: berechne über ein volles Umlaufjahr des Planeten
    planet_orbital_periods = {
        'VENUS': 225 * 86400,      # ~225 Tage in Sekunden
        'EARTH': 365.25 * 86400,   # ~365 Tage in Sekunden
        'JUPITER': 4333 * 86400,   # ~12 Jahre in Sekunden
        'SATURN': 10759 * 86400    # ~29 Jahre in Sekunden
    }
    
    planet_orbits = {}
    print("\nComputing planet orbits from SPICE data...")
    for planet in planets_info.keys():
        orbit_positions = []
        # Berechne Bahn über eine volle Umlaufperiode
        orbit_period = planet_orbital_periods[planet]  # Wir benötigen hier immer die volle Periode
        
        # Verwende genug Punkte für glatte Bahnen
        num_points = min(500, max(100, int(orbit_period / (10 * 86400))))  # Alle 10 Tage
        orbit_times = np.linspace(times[0], times[0] + orbit_period, num_points)
        
        for et in orbit_times:
            try:
                pos = get_body_position(planet, et)
                orbit_positions.append(pos[:2])  # Nur X und Y
            except Exception as e:
                orbit_positions.append([np.nan, np.nan])
        
        planet_orbits[planet] = np.array(orbit_positions)
        valid_count = np.sum(~np.any(np.isnan(planet_orbits[planet]), axis=1))
        print(f"  {planet}: {valid_count}/{num_points} valid positions")
    
    # Zeichne Planetenbahnen mit echten SPICE-Daten
    for planet, info in planets_info.items():
        if planet in planet_orbits:
            orbit_data = planet_orbits[planet]
            valid_mask = ~np.any(np.isnan(orbit_data), axis=1)
            if np.any(valid_mask):
                ax.plot(orbit_data[valid_mask, 0], orbit_data[valid_mask, 1],
                       color=info['color'], linestyle='--', linewidth=2.5, alpha=0.7,
                       label=f'{info["label"]}-Orbit')
                print(f"  Plotted {planet} orbit: X range [{orbit_data[valid_mask, 0].min():.2e}, {orbit_data[valid_mask, 0].max():.2e}] km")
    
    # Keine Planeten-Marker zeichnen - nur die Bahnen bleiben sichtbar
    
    # Zeichne Sonne im Zentrum
    ax.plot(0, 0, 'o', color='yellow', markersize=20, 
           markeredgecolor='orange', markeredgewidth=2, label='Sonne', zorder=10)
    
    # Downsampling für bessere Performance (für beide Trajektorien)
    downsample_factor = max(1, len(times) // 5000)
    
    # Zeichne simulierte Cassini Trajektorie (falls gewünscht)
    if show_simulated_cassini:
        sim_x_plot = states[0, ::downsample_factor]
        sim_y_plot = states[1, ::downsample_factor]
        
        ax.plot(sim_x_plot, sim_y_plot, 'b-', linewidth=2.5, alpha=0.7, label='Cassini (Simuliert)')
    
    # Zeichne echte Cassini Trajektorie falls verfügbar
    if real_positions is not None:
        # Filter NaN-Werte
        valid_mask = ~np.any(np.isnan(real_positions), axis=1)
        if np.any(valid_mask):
            real_x = real_positions[valid_mask, 0][::downsample_factor]
            real_y = real_positions[valid_mask, 1][::downsample_factor]
            ax.plot(real_x, real_y, 'g-', linewidth=2.5, alpha=0.7, 
                   label='Cassini (Real)', linestyle='--')
    
    # Markiere Start- und Endpunkte (Simuliert, falls gewünscht)
    if show_simulated_cassini:
        ax.plot(states[0, 0], states[1, 0], 'o', color='cyan', 
               markersize=12, label=f'Start (Simuliert)', 
               zorder=5, markeredgecolor='darkblue', markeredgewidth=2)
        ax.plot(states[0, -1], states[1, -1], 's', color='red', 
               markersize=12, label=f'Ende (Simuliert)', 
               zorder=5, markeredgecolor='darkred', markeredgewidth=2)
    
    # Markiere Start- und Endpunkte (SPICE) falls verfügbar
    if real_positions is not None:
        valid_mask = ~np.any(np.isnan(real_positions), axis=1)
        if np.any(valid_mask):
            # Finde ersten und letzten gültigen Index
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                first_idx = valid_indices[0]
                last_idx = valid_indices[-1]
                
                ax.plot(real_positions[first_idx, 0], real_positions[first_idx, 1], 
                       '^', color='cyan', markersize=12, 
                       label=f'Start (Real)', 
                       zorder=5, markeredgecolor='blue', markeredgewidth=2)
                ax.plot(real_positions[last_idx, 0], real_positions[last_idx, 1], 
                       '^', color='red', markersize=12, 
                       label=f'Ende (Real)', 
                       zorder=5, markeredgecolor='darkred', markeredgewidth=2)
    
    # Markiere die vier Gravity Assists explizit
    gravity_assists = [
        {'date': '1998-04-26', 'name': 'Venus 1 GA\n26 Apr 1998', 'color': '#FFA500', 'offset': (80, -60)},
        {'date': '1999-06-21', 'name': 'Venus 2 GA\n21 Jun 1999', 'color': '#FFA500', 'offset': (-60, -60)},
        {'date': '1999-08-18', 'name': 'Erde GA\n18 Aug 1999', 'color': '#4169E1', 'offset': (80, 40)},
        {'date': '2000-12-30', 'name': 'Jupiter GA\n30 Dez 2000', 'color': '#DAA520', 'offset': (40, -60)}
    ]
    
    print("\n=== Gravity Assist Markers ===")
    print(f"Total simulation points: {len(times)}")
    print(f"Simulation time range: {sp.et2utc(times[0], 'ISOC', 0)} to {sp.et2utc(times[-1], 'ISOC', 0)}")
    print(f"Time step statistics: min={np.min(np.diff(times)):.2f}s, max={np.max(np.diff(times)):.2f}s, mean={np.mean(np.diff(times)):.2f}s")
    
    # Check if maneuver manager has data
    if maneuver_manager is not None and hasattr(maneuver_manager, 'maneuvers'):
        print(f"\nManeuver times from manager:")
        for _, man in maneuver_manager.maneuvers.iterrows():
            man_time = man['ET']
            man_name = man['Maneuver']
            print(f"  {man_name}: ET={man_time:.2f}, UTC={sp.et2utc(man_time, 'ISOC', 0)}")
    
    for ga in gravity_assists:
        try:
            # Konvertiere Datum zu ET
            ga_et = sp.utc2et(ga['date'])
            print(f"\n{ga['name'].split(chr(10))[0]}:")
            print(f"  Target ET: {ga_et:.2f}")
            print(f"  Target UTC: {sp.et2utc(ga_et, 'ISOC', 0)}")
            
            # Prüfe ob im Zeitbereich
            if ga_et >= times[0] and ga_et <= times[-1]:
                # Verwende SPICE-Position statt simulierter Position
                if real_positions is not None:
                    # Finde nächsten Zeitindex in SPICE-Daten
                    idx = np.argmin(np.abs(times - ga_et))
                    time_diff = abs(times[idx] - ga_et)
                    
                    # Hole Position aus SPICE-Daten
                    if idx < len(real_positions) and not np.any(np.isnan(real_positions[idx])):
                        x, y = real_positions[idx, 0], real_positions[idx, 1]
                        
                        print(f"  Using SPICE position at index: {idx}/{len(times)}")
                        print(f"  Time at index: {sp.et2utc(times[idx], 'ISOC', 0)}")
                        print(f"  Time difference: {time_diff:.2f} s ({time_diff/3600:.2f} hours, {time_diff/86400:.4f} days)")
                        print(f"  SPICE Position: X={x:.3e} km, Y={y:.3e} km")
                        print(f"  Distance from Sun: {np.sqrt(x**2 + y**2):.3e} km ({np.sqrt(x**2 + y**2)/1.496e8:.3f} AU)")
                        
                        # Markiere Position mit kleinem Punkt
                        ax.plot(x, y, 'o', color=ga['color'], markersize=12, 
                               markeredgecolor='black', markeredgewidth=1.5, zorder=6, 
                               alpha=0.9)  # Slightly transparent for better visibility
                        
                        # Annotiere mit Pfeil (ohne Label für Legende)
                        ax.annotate(ga['name'], 
                                   xy=(x, y), xytext=ga['offset'],
                                   textcoords='offset points',
                                   fontsize=13,  # Increased from 9 to 13
                                   color='black', 
                                   fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.7',  # Slightly more padding
                                           facecolor='white', 
                                           edgecolor=ga['color'], 
                                           alpha=0.95, 
                                           linewidth=2.5),  # Thicker border
                                   arrowprops=dict(arrowstyle='->', 
                                                 color=ga['color'], 
                                                 lw=2.0,  # Thicker arrow
                                                 shrinkA=5,  # Arrow starts further from point
                                                 shrinkB=5),  # Arrow points closer to text
                                   zorder=7)
                    else:
                        print(f"  WARNING: No valid SPICE position at index {idx}")
                else:
                    print(f"  WARNING: No SPICE data available, cannot place marker")
            else:
                print(f"  WARNING: Outside simulation time range!")
                print(f"  Simulation: {times[0]:.2f} to {times[-1]:.2f}")
                print(f"  Flyby: {ga_et:.2f}")
        except Exception as e:
            print(f"  ERROR: Could not mark gravity assist: {e}")
            import traceback
            traceback.print_exc()
    
    # Setze Achsenbeschriftungen
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Gleiches Seitenverhältnis
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Setze Plot-Grenzen basierend auf Fokus (NACH axis('equal'))
    if focus_on_simulated:
        # Fokussiere auf simulierte Trajektorie (verwende volle Daten, nicht downsampled)
        sim_x_full = states[0, :]
        sim_y_full = states[1, :]
        x_margin = (sim_x_full.max() - sim_x_full.min()) * 0.15
        y_margin = (sim_y_full.max() - sim_y_full.min()) * 0.15
        ax.set_xlim(sim_x_full.min() - x_margin, sim_x_full.max() + x_margin)
        ax.set_ylim(sim_y_full.min() - y_margin, sim_y_full.max() + y_margin)
    else:
        # Fokussiere auf echte Trajektorie (verwende volle Daten, nicht downsampled)
        real_x_full = real_positions[:, 0]
        real_y_full = real_positions[:, 1]
        x_margin = (real_x_full.max() - real_x_full.min()) * 0.15
        y_margin = (real_y_full.max() - real_y_full.min()) * 0.15
        ax.set_xlim(real_x_full.min() - x_margin, real_x_full.max() + x_margin)
        ax.set_ylim(real_y_full.min() - y_margin, real_y_full.max() + y_margin)
    # Größere Schrift in der Legende (Original-Abstände beibehalten)
    legend = ax.legend(
        loc='upper right',
        fontsize=17,  # Größere Schrift
        ncol=2,
        frameon=True,
        title='Legende',
        title_fontsize=16  # Größere Schrift für den Titel
    )
    
    # Größere Punkte in der Legende
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([100])  # Größere Punkte
    
    # Speichere falls angefordert
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            png_path = f"{os.path.splitext(save_path)[0]}_{timestamp}.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Mission overview saved to: {os.path.abspath(png_path)}")
        except Exception as e:
            print(f"Could not save plot: {e}")
    
    plt.tight_layout()
    return fig, ax
