#!/usr/bin/env python3
"""
Generiert eine Planck-Kurve für 5700K im Format der Spektrometer-TXT-Dateien.
"""

import numpy as np
from scipy.constants import h, c, k

def planck_curve(wavelength_nm: np.ndarray, temperature: float, scaling_factor: float = 1.0) -> np.ndarray:
    """Berechnet die Planck-Kurve für einen schwarzen Strahler."""
    wavelength_m = wavelength_nm * 1e-9
    numerator = 2 * np.pi * h * c**2 / wavelength_m**5
    denominator = np.exp(h * c / (wavelength_m * k * temperature)) - 1
    denominator = np.where(denominator <= 0, 1e-10, denominator)
    spectral_radiance = scaling_factor * numerator / denominator
    return spectral_radiance

# Wellenlängenbereich basierend auf den Originaldaten
# Von ~177nm bis ~1100nm mit Schrittweite von ~0.6nm
wavelength_start = 177.0
wavelength_end = 1100.0
step = 0.6  # Ungefähre Schrittweite

# Erzeuge Wellenlängen-Array
wavelengths = np.arange(wavelength_start, wavelength_end + step, step)

# Berechne Planck-Kurve für 5700K
temperature = 5700.0

# Skalierung: Wähle einen Faktor, damit die Werte realistisch sind
# Der Peak der Planck-Kurve bei 5700K liegt bei ~508nm
# Skaliere so, dass der Peak bei etwa 800-1000 counts liegt (ähnlich wie Sonnenlicht-Daten)
planck_values = planck_curve(wavelengths, temperature, 1.0)
peak_idx = np.argmax(planck_values)
peak_value = planck_values[peak_idx]

# Skaliere auf etwa 900 counts am Peak (realistischer Wert für Sonnenlicht)
target_peak = 900.0
scaling_factor = target_peak / peak_value
planck_values_scaled = planck_curve(wavelengths, temperature, scaling_factor)

# Funktion zum Schreiben der Datei
def write_planck_file(filename, wavelengths, values):
    """Schreibt Planck-Kurve in TXT-Datei im Spektrometer-Format."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("\n")
        f.write("Integration time [ms]:   0,050\n")
        f.write("Averaging Nr. [scans]: 10\n")
        f.write("Smoothing Nr. [pixels]: 3\n")
        f.write("Data measured with spectrometer [name]: 7516255SP\n")
        f.write("Wave   ;Sample   ;Dark     ;Reference\n")
        f.write("[nm]   ;[counts] ;[counts] ;[counts] \n")
        f.write("\n")
        
        # Daten schreiben (Komma als Dezimaltrennzeichen)
        for wl, sample in zip(wavelengths, values):
            # Format: wavelength;sample;dark;reference
            # Dark und Reference auf 0 setzen
            # Verwende Komma als Dezimaltrennzeichen (deutsches Format)
            wl_str = f"{wl:.3f}".replace('.', ',')
            sample_str = f"{sample:.3f}".replace('.', ',')
            f.write(f"{wl_str};  {sample_str};    0,000;    0,000\n")

# Erstelle saubere Planck-Kurve
output_file_clean = "Planck_5700K.txt"
write_planck_file(output_file_clean, wavelengths, planck_values_scaled)

print(f"Planck-Kurve für {temperature}K wurde erstellt: {output_file_clean}")
print(f"Skalierungsfaktor: {scaling_factor:.6e}")
print(f"Anzahl Datenpunkte: {len(wavelengths)}")
print(f"Wellenlängenbereich: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

# Erstelle verrauschte Version
# Realistisches Rauschen: Poisson-Rauschen (Photonenzählung) + kleines Gauß-Rauschen
# Das Rauschen sollte proportional zur Signalstärke sein
np.random.seed(42)  # Für reproduzierbare Ergebnisse

# Poisson-Rauschen: Standardabweichung = sqrt(Signal)
# Für Photonenzählung ist das Rauschen Poisson-verteilt
poisson_noise = np.random.poisson(planck_values_scaled) - planck_values_scaled

# Zusätzliches kleines Gauß-Rauschen (ca. 1-2% des Signals)
# Simuliert elektronisches Rauschen und andere Quellen
gaussian_noise = np.random.normal(0, planck_values_scaled * 0.015, size=len(planck_values_scaled))

# Kombiniere Rauschen
noise = poisson_noise + gaussian_noise
planck_values_noisy = planck_values_scaled + noise

# Stelle sicher, dass keine negativen Werte entstehen
planck_values_noisy = np.maximum(planck_values_noisy, 0.1)

# Erstelle verrauschte Datei
output_file_noisy = "Planck_5700K_noisy.txt"
write_planck_file(output_file_noisy, wavelengths, planck_values_noisy)

print(f"\nVerrauschte Planck-Kurve wurde erstellt: {output_file_noisy}")
print(f"Rausch-Level: Poisson + ~1.5% Gauß-Rauschen")
print(f"Maximales Rauschen: {np.max(np.abs(noise)):.2f} counts")

