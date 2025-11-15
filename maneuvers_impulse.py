"""
maneuvers_impulse.py - Manöver-Verwaltung für Impuls-Manöver

Verwaltet die Bahnkorrekturmanöver von Cassini als instantane Geschwindigkeitsänderungen.
Lädt Manöverdaten aus CSV-Dateien (Delta-v Vektoren, Zeitpunkte, Masse) und
wendet sie während der Integration zu den exakten Zeitpunkten an.
"""

import os
import numpy as np
import pandas as pd
import spiceypy as sp
from typing import Tuple, Optional, Dict, Any, List


class ImpulseManeuverManager:
    """
    Manager für Manöver als instantane Geschwindigkeitsänderungen.
    """
    def __init__(self):
        self.maneuvers = None
        self.delta_v_vectors = None
        self.applied_maneuvers = set()  # Verfolgt bereits angewendete Manöver
        
    def load_maneuvers(self, delta_v_file: str) -> None:
        """Lädt Manöverdaten aus CSV-Dateien."""
        try:
            # Lade Delta-v Daten
            self.delta_v_vectors = pd.read_csv(delta_v_file)
            print(f"\nSuccessfully loaded delta-v vectors from: {delta_v_file}")
            print(self.delta_v_vectors[['Maneuver', 'DeltaV_Magnitude_m_s']].head())
            
            self.maneuvers = self.delta_v_vectors.copy()
            
            # Bereinige Spaltennamen
            self.maneuvers.columns = self.maneuvers.columns.str.strip()
            
            # Bereinige numerische Spalten
            numeric_columns = ['ET', 'DeltaV_X_km_s', 'DeltaV_Y_km_s', 'DeltaV_Z_km_s', 'DeltaV_Magnitude_m_s']
            for col in numeric_columns:
                if col in self.maneuvers.columns:
                    self.maneuvers[col] = self.maneuvers[col].astype(str)
                    self.maneuvers[col] = self.maneuvers[col].str.replace(r'^\d+(-\d)', r'\1', regex=True)
                    self.maneuvers[col] = pd.to_numeric(self.maneuvers[col], errors='coerce')
            
            # Lade Massendaten aus maneuver_delta_v_mass.csv
            
            # Lese Massendaten-Datei
            mass_data = pd.read_csv('data/maneuver_delta_v_mass.csv')
            
            # Bereinige Spaltennamen und wähle relevante Spalten
            mass_data.columns = mass_data.columns.str.strip()
            mass_data = mass_data.rename(columns={
                'Spacecraft Mass before maneuver(kg)': 'Spacecraft Mass (kg)',
                'Maneuver Time (UTC SCET)': 'Maneuver_Time_UTC'
            })
            
            # Bereinige Manövernamen für Abgleich
            mass_data['Maneuver'] = mass_data['Maneuver'].str.strip()
            
            # Konvertiere Masse zu numerischem Wert
            mass_data['Spacecraft Mass (kg)'] = pd.to_numeric(
                mass_data['Spacecraft Mass (kg)'].astype(str).str.replace(',', '.').str.replace('E+', 'E', regex=False),
                errors='coerce'
            )
            
            # Wähle nur benötigte Spalten
            mass_data = mass_data[['Maneuver', 'Spacecraft Mass (kg)']].dropna()
            
            # Füge mit Manöverdaten zusammen
            self.maneuvers = pd.merge(
                self.maneuvers, 
                mass_data,
                on='Maneuver',
                how='left'
            )
                            
            print("Successfully loaded mass data from maneuver_delta_v_mass.csv")
            
            # Sortiere nach Zeit
            self.maneuvers = self.maneuvers.sort_values('ET').reset_index(drop=True)
            print(self.maneuvers)
            
            print("\nSuccessfully loaded maneuver data:")
            print(self.maneuvers[['Maneuver', 'ET', 'Spacecraft Mass (kg)', 'DeltaV_Magnitude_m_s']].head())
            
            # Ausgabe zur Zeitverifizierung
            print("\nManeuver timing verification:")
            for idx, row in self.maneuvers.iterrows():
                utc_from_et = sp.et2utc(row['ET'], 'ISOC', 0)
                print(f"{row['Maneuver']}: ET={row['ET']:.1f}, UTC_from_ET={utc_from_et}")                
                    
        except Exception as e:
            raise RuntimeError(f"Error loading maneuver data: {str(e)}")
    
    def get_all_maneuver_times(self) -> np.ndarray:
        """Gibt alle Manöverzeiten für die Integrationsplanung zurück."""
        if self.maneuvers is None:
            return np.array([])
        return self.maneuvers['ET'].values
    
    def check_and_apply_maneuver(self, t: float, y: np.ndarray, tolerance: float = 1.0) -> Tuple[np.ndarray, bool]:
        """
        Prüft ob ein Manöver zur Zeit t vorliegt und wendet es als instantane Geschwindigkeitsänderung an.
        
        Args:
            t: Aktuelle Zeit in ET Sekunden
            y: Zustandsvektor [x, y, z, vx, vy, vz, m]
            tolerance: Zeittoleranz in Sekunden
            
        Returns:
            Tupel aus (modifizierter_Zustand, Manöver_angewendet)
        """
        if self.maneuvers is None:
            return y, False
            
        # Finde Manöver die zu dieser Zeit angewendet werden sollen
        et_values = self.maneuvers['ET'].values
        time_diffs = np.abs(et_values - t)
        
        # Finde alle Manöver innerhalb der Toleranz
        within_tolerance = time_diffs <= tolerance
        
        if not np.any(within_tolerance):
            return y, False
        
        # Hole nächstes noch nicht angewendetes Manöver
        maneuver_indices = np.where(within_tolerance)[0]
        
        modified_state = y.copy()
        maneuver_applied = False
        
        for idx in maneuver_indices:
            maneuver = self.maneuvers.iloc[idx]
            maneuver_id = f"{maneuver['Maneuver']}_{maneuver['ET']}"
            
            # Überspringe falls bereits angewendet
            if maneuver_id in self.applied_maneuvers:
                continue
            
            # Wende Manöver an
            print(f"\n*** APPLYING IMPULSE MANEUVER ***")
            print(f"Time: {sp.et2utc(t, 'ISOC', 0)} (ET: {t:.1f})")
            print(f"Maneuver: {maneuver['Maneuver']}")
            print(f"Delta-V magnitude: {maneuver['DeltaV_Magnitude_m_s']:.6f} m/s")
            
            # Prüfe ob SPK-abgeleitete Delta-v Vektordaten vorhanden sind
            has_spk_delta_v = (
                'DeltaV_X_km_s' in maneuver and 
                'DeltaV_Y_km_s' in maneuver and 
                'DeltaV_Z_km_s' in maneuver and
                not pd.isna(maneuver['DeltaV_X_km_s']) and
                not pd.isna(maneuver['DeltaV_Y_km_s']) and
                not pd.isna(maneuver['DeltaV_Z_km_s'])
            )
            
            if has_spk_delta_v:
                # Verwende SPK-abgeleiteten Delta-v Vektor
                delta_v_vector = np.array([
                    maneuver['DeltaV_X_km_s'],
                    maneuver['DeltaV_Y_km_s'], 
                    maneuver['DeltaV_Z_km_s']
                ])
                
                print(f"Delta-V vector (km/s): {delta_v_vector}")
                print(f"Delta-V magnitude check: {np.linalg.norm(delta_v_vector)*1000:.6f} m/s vs expected {maneuver['DeltaV_Magnitude_m_s']:.6f} m/s")
                
            else:
                # Fallback: Verwende Geschwindigkeitsrichtung falls keine SPK-Daten
                delta_v_mag = maneuver.get('DeltaV_Magnitude_m_s', 0.0) / 1000.0  # Konvertiere m/s zu km/s
                if delta_v_mag <= 0:
                    continue
                    
                # Verwende Geschwindigkeitsrichtung als Delta-v Richtung
                velocity = modified_state[3:6]
                velocity_norm = np.linalg.norm(velocity)
                if velocity_norm < 1e-6:
                    continue
                    
                delta_v_vector = (velocity / velocity_norm) * delta_v_mag
                print(f"Using velocity direction for delta-v: {delta_v_vector} km/s")
            
            # Wende Geschwindigkeitsänderung direkt an
            print(f"Velocity before maneuver: {modified_state[3:6]} km/s")
            modified_state[3:6] += delta_v_vector  # Addiere Delta-v zur Geschwindigkeit
            print(f"Velocity after maneuver:  {modified_state[3:6]} km/s")
            print(f"Velocity change applied:   {delta_v_vector} km/s")
            
            # Verwende Masse aus den Daten
            if 'Spacecraft Mass (kg)' in maneuver and pd.notnull(maneuver['Spacecraft Mass (kg)']):
                new_mass = maneuver['Spacecraft Mass (kg)']
                mass_consumed = modified_state[6] - new_mass
                
                print(f"Mass before maneuver: {modified_state[6]:.1f} kg")
                modified_state[6] = new_mass
                print(f"Mass after maneuver:  {modified_state[6]:.1f} kg")
                print(f"Propellant consumed:   {mass_consumed:.1f} kg")
            
            # Markiere Manöver als angewendet
            self.applied_maneuvers.add(maneuver_id)
            maneuver_applied = True
            print("*** MANEUVER APPLIED ***\n")
        
        return modified_state, maneuver_applied
    
    def reset_applied_maneuvers(self):
        """Setzt die Liste der angewendeten Manöver zurück."""
        self.applied_maneuvers.clear()


def initialize_impulse_maneuver_manager() -> ImpulseManeuverManager:
    """Initialisiert und gibt eine ImpulseManeuverManager-Instanz zurück."""
    return ImpulseManeuverManager()
