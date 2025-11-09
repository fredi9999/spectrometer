import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.optimize import minimize_scalar
from scipy.constants import h, c, k


def planck_curve(wavelength_nm: np.ndarray, temperature: float, scaling_factor: float = 1.0) -> np.ndarray:
    """
    Berechnet die Planck-Kurve für die spektrale Strahlungsverteilung eines schwarzen Strahlers.
    
    Formel: B(λ,T) = (2hc²/λ⁵) · 1/(exp(hc/(λk_B T)) - 1)
    
    Args:
        wavelength_nm: Wellenlängen in Nanometern
        temperature: Temperatur in Kelvin
        scaling_factor: Optionaler Skalierungsfaktor A zur Anpassung an die Messwerte
    
    Returns:
        Spektrale Strahldichte für jede Wellenlänge
    """
    # Umrechnung von Nanometern zu Metern (SI-Einheiten)
    wavelength_m = wavelength_nm * 1e-9
    
    # Planck'sche Strahlungsformel
    # Zähler: 2hc²/λ⁵
    numerator = 2 * h * c**2 / wavelength_m**5
    
    # Nenner: exp(hc/(λk_B T)) - 1
    denominator = np.exp(h * c / (wavelength_m * k * temperature)) - 1
    
    # Vermeide Division durch Null (kann bei sehr kleinen Wellenlängen auftreten)
    denominator = np.where(denominator <= 0, 1e-10, denominator)
    
    # Spektrale Strahldichte berechnen und mit Skalierungsfaktor multiplizieren
    spectral_radiance = scaling_factor * numerator / denominator
    
    return spectral_radiance


def fit_planck_curve(wavelength_nm: np.ndarray, intensity: np.ndarray, 
                     fit_scaling: bool = True) -> Tuple[float, float, float, np.ndarray]:
    """
    Passt die Planck-Kurve an experimentelle Daten an.
    
    1. Finde die Wellenlänge mit maximaler Intensität
    2. Berechne die Temperatur mit dem Wien'schen Verschiebungsgesetz: T = b / λ_max
    3. Passe optional nur den Skalierungsfaktor A an, um die Kurve an die Daten anzupassen
    
    Args:
        wavelength_nm: Wellenlängen in Nanometern
        intensity: Gemessene Intensitäten (Zählwerte)
        fit_scaling: Wenn True, wird der Skalierungsfaktor A angepasst
    
    Returns:
        Tuple mit (Temperatur in K, Skalierungsfaktor, R², angepasste Kurve)
    """
    # Daten bereinigen: entferne ungültige Werte
    mask = np.isfinite(wavelength_nm) & np.isfinite(intensity) & (intensity > 0)
    wl = wavelength_nm[mask]
    y = intensity[mask]
    
    if len(wl) < 5:
        return 0.0, 0.0, 0.0, np.zeros_like(wavelength_nm)
    
    # Finde die Wellenlänge mit maximaler Intensität (Peak)
    peak_idx = np.argmax(y)
    lambda_max_nm = wl[peak_idx]
    
    # erechne Temperatur mit Wien'schem Verschiebungsgesetz
    # λ_max * T = 2.898e-3 m·K
    # T = 2.898e-3 / λ_max
    wien_constant = 2.898e-3  # m·K
    lambda_max_m = lambda_max_nm * 1e-9  # Umrechnung von nm zu m
    temperature = wien_constant / lambda_max_m
    
    # Begrenze auf sinnvolle Werte
    temperature = np.clip(temperature, 500.0, 10000.0)
    
    # Passe Skalierungsfaktor an
    if fit_scaling:
        # Berechne Planck-Kurve mit geschätzter Temperatur
        planck_test = planck_curve(wl, temperature, 1.0)
        
        # Finde den Skalierungsfaktor, der die Planck-Kurve an die Daten anpasst
        # Verwende Least-Squares-Methode für den Skalierungsfaktor
        def residuals_scale(scale: float) -> float:
            """Summe der quadrierten Residuen für einen gegebenen Skalierungsfaktor"""
            model = planck_curve(wl, temperature, scale)
            return np.sum((y - model) ** 2)
        
        # Initiale Schätzung: Verhältnis der Maxima
        max_planck = np.max(planck_test)
        max_data = np.max(y)
        initial_scale = max_data / max_planck if max_planck > 0 else 1.0
        
        # Optimiere Skalierungsfaktor mit scipy.optimize.minimize_scalar
        try:
            result = minimize_scalar(
                residuals_scale,
                bounds=(1e-10, 1e12),
                method='bounded'
            )
            scaling_factor = result.x if result.success else initial_scale
        except Exception:
            scaling_factor = initial_scale
    else:
        # Keine Skalierung: verwende Skalierungsfaktor = 1.0
        scaling_factor = 1.0
    
    # Berechne R² (Güte der Anpassung)
    y_fit = planck_curve(wl, temperature, scaling_factor)
    ss_res = np.sum((y - y_fit) ** 2)  # Summe der quadrierten Residuen
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Gesamtvariation
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Erzeuge angepasste Kurve für alle Wellenlängen
    fitted_curve = planck_curve(wavelength_nm, temperature, scaling_factor)
    
    return float(temperature), float(scaling_factor), float(r_squared), fitted_curve


def polynomial_fit(wavelength_nm: np.ndarray, intensity: np.ndarray, degree: int = 3) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Führt eine Polynom-Anpassung (Least Squares) auf die Spektrum-Daten durch.
    
    Formel: y = a₀ + a₁·x + a₂·x² + ... + aₙ·xⁿ
    
    Verwendet numerisch stabile Least-Squares-Methode mit zentrierten Daten
    für bessere Konditionierung bei hohen Polynomgraden.
    
    Args:
        wavelength_nm: Wellenlängen in Nanometern
        intensity: Gemessene Intensitäten (Zählwerte)
        degree: Grad des Polynoms (z.B. 2 für quadratisch, 3 für kubisch)
    
    Returns:
        Tuple mit (Koeffizienten [aₙ, ..., a₁, a₀], R², angepasste Werte)
    """
    # Daten bereinigen: entferne ungültige Werte
    mask = np.isfinite(wavelength_nm) & np.isfinite(intensity) & (intensity >= 0)
    wl = wavelength_nm[mask]
    y = intensity[mask]
    
    if len(wl) < degree + 1:
        return np.zeros(degree + 1), 0.0, np.zeros_like(wavelength_nm)
    
    # Entferne Duplikate in x (wichtig für numerische Stabilität)
    unique_indices = np.unique(wl, return_index=True)[1]
    wl_unique = wl[unique_indices]
    y_unique = y[unique_indices]
    
    if len(wl_unique) < degree + 1:
        return np.zeros(degree + 1), 0.0, np.zeros_like(wavelength_nm)
    
    try:
        # Polynom-Anpassung mit Least Squares
        # np.polyfit verwendet QR-Zerlegung für numerisch stabile Least-Squares-Lösung
        # Die Methode minimiert: sum((y - p(x))^2) wobei p(x) das Polynom ist
        coeffs = np.polyfit(wl_unique, y_unique, deg=degree)
        
        # Berechne angepasste Werte für alle Wellenlängen
        # np.polyval wertet das Polynom aus: p(x) = coeffs[0]*x^n + ... + coeffs[n]
        fitted_values = np.polyval(coeffs, wavelength_nm)
        
        # Stelle sicher, dass keine negativen Werte entstehen (physikalisch sinnvoll)
        fitted_values = np.maximum(fitted_values, 0)
        
        # Berechne R² (Güte der Anpassung)
        y_fit = np.polyval(coeffs, wl_unique)
        ss_res = np.sum((y_unique - y_fit) ** 2)  # Summe der quadrierten Residuen
        ss_tot = np.sum((y_unique - np.mean(y_unique)) ** 2)  # Gesamtvariation
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return coeffs, float(r_squared), fitted_values
        
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: verwende einfachere Methode ohne volle Ausgabe
        try:
            coeffs = np.polyfit(wl_unique, y_unique, deg=min(degree, len(wl_unique) - 1))
            # Erweitere auf gewünschten Grad mit Nullen falls nötig
            if len(coeffs) < degree + 1:
                coeffs_full = np.zeros(degree + 1)
                coeffs_full[:len(coeffs)] = coeffs
                coeffs = coeffs_full
            
            fitted_values = np.polyval(coeffs, wavelength_nm)
            fitted_values = np.maximum(fitted_values, 0)
            
            y_fit = np.polyval(coeffs, wl_unique)
            ss_res = np.sum((y_unique - y_fit) ** 2)
            ss_tot = np.sum((y_unique - np.mean(y_unique)) ** 2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return coeffs, float(r_squared), fitted_values
        except Exception:
            # Letzter Fallback: lineare Interpolation
            fitted_values = np.interp(wavelength_nm, wl_unique, y_unique, 
                                     left=y_unique[0], right=y_unique[-1])
            return np.zeros(degree + 1), 0.0, fitted_values


def parse_spectrometer_text(file_bytes: bytes) -> pd.DataFrame:
    """
    Liest die txt Dateien des Spektrometers ein.
    - Kopfzeilen enthalten Spaltennamen (z. B. "Wave; Sample; Dark; Reference")
    - Werte sind Semikolon-getrennt und nutzen Dezimalkomma
    - Ergebnis ist ein DataFrame mit Spalten: nm, sample, dark, reference
    """
    text = file_bytes.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip() != ""]

    header_idx = None
    for idx, line in enumerate(lines):
        if line.lower().startswith("wave") and ";" in line:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Konnte keine Kopfzeile beginnend mit 'Wave' finden.")

    start_idx = header_idx + 1
    if start_idx < len(lines) and "[nm]" in lines[start_idx]:
        start_idx += 1

    def parse_row(row: str) -> Tuple[float, float, float, float]:
        parts = [p.strip() for p in row.split(";")]
        if len(parts) < 4:
            raise ValueError("Zeile hat nicht 4 Spalten: " + row)
        def to_float(s: str) -> float:
            s = s.replace(" ", "").replace(",", ".")
            return float(s)
        return (
            to_float(parts[0]),
            to_float(parts[1]),
            to_float(parts[2]),
            to_float(parts[3]),
        )

    nm_list: List[float] = []
    sample_list: List[float] = []
    dark_list: List[float] = []
    ref_list: List[float] = []

    for line in lines[start_idx:]:
        if line.lower().startswith("wave"):
            continue
        if ";" not in line:
            continue
        try:
            nm, sample, dark, ref = parse_row(line)
        except Exception:
            continue
        nm_list.append(nm)
        sample_list.append(sample)
        dark_list.append(dark)
        ref_list.append(ref)

    if not nm_list:
        raise ValueError("Keine Datenzeilen nach der Kopfzeile gefunden.")

    df = pd.DataFrame({
        "nm": np.array(nm_list, dtype=float),
        "sample": np.array(sample_list, dtype=float),
        "dark": np.array(dark_list, dtype=float),
        "reference": np.array(ref_list, dtype=float),
    })
    return df


def compute_y(df: pd.DataFrame, mode: str,
              external_dark_interp=None, external_dark_scale: float = 1.0,
              external_dark_mode: str = "Off") -> Tuple[np.ndarray, str]:
    label_local = mode
    nm_vals = df["nm"].values
    
    if mode == "sample":
        yv = df["sample"].values
        if external_dark_interp is not None and external_dark_mode == "Subtract from sample":
            try:
                d_ext = external_dark_interp(nm_vals)
                if np.isfinite(d_ext).any():
                    yv = yv - external_dark_scale * d_ext
                    label_local += f" (− ext dark×{external_dark_scale:.2f})"
            except Exception:
                pass
    elif mode == "dark":
        yv = df["dark"].values
    elif mode == "reference":
        yv = df["reference"].values
    elif mode == "sample-dark":
        if external_dark_interp is not None and external_dark_mode == "Replace file dark":
            try:
                d_ext = external_dark_interp(nm_vals)
                yv = df["sample"].values - external_dark_scale * d_ext
                label_local += f" (S − ext dark×{external_dark_scale:.2f})"
            except Exception:
                yv = df["sample"].values - df["dark"].values
        elif external_dark_interp is not None and external_dark_mode == "Subtract from sample":
            try:
                d_ext = external_dark_interp(nm_vals)
                yv = (df["sample"].values - external_dark_scale * d_ext) - df["dark"].values
                label_local += f" ((S−ext×{external_dark_scale:.2f})−D)"
            except Exception:
                yv = df["sample"].values - df["dark"].values
        else:
            yv = df["sample"].values - df["dark"].values
    elif mode == "(sample-dark)/max":
        if external_dark_interp is not None and external_dark_mode == "Replace file dark":
            try:
                d_ext = external_dark_interp(nm_vals)
                yv = df["sample"].values - external_dark_scale * d_ext
                label_local += f" (S − ext dark×{external_dark_scale:.2f})"
            except Exception:
                yv = df["sample"].values - df["dark"].values
        elif external_dark_interp is not None and external_dark_mode == "Subtract from sample":
            try:
                d_ext = external_dark_interp(nm_vals)
                yv = (df["sample"].values - external_dark_scale * d_ext) - df["dark"].values
                label_local += f" ((S−ext×{external_dark_scale:.2f})−D)"
            except Exception:
                yv = df["sample"].values - df["dark"].values
        else:
            yv = df["sample"].values - df["dark"].values
        max_val = float(np.nanmax(yv)) if np.isfinite(yv).any() else 1.0
        yv = yv / (max_val if max_val != 0.0 else 1.0)
    elif mode == "(sample-dark)/(reference-dark)":
        numerator = df["sample"].values - df["dark"].values
        denominator = df["reference"].values - df["dark"].values
        with np.errstate(divide='ignore', invalid='ignore'):
            yv = np.where(denominator != 0, numerator / denominator, np.nan)
        label_local = "relative (S-D)/(R-D)"
    else:
        yv = df["sample"].values

    return yv, label_local


def main() -> None:
    st.set_page_config(page_title="Spektrometer", layout="wide")
    
    # CSS: Checkboxen schwarz machen
    st.markdown("""
        <style>
        /* Checkbox-Hintergrund und Rahmen schwarz wenn aktiviert */
        .stCheckbox input[type="checkbox"]:checked {
            background-color: black !important;
            border-color: black !important;
            accent-color: black !important;
        }
        /* Checkbox-Rahmen schwarz */
        .stCheckbox input[type="checkbox"] {
            border-color: black !important;
            accent-color: black !important;
        }
        /* Checkbox-Label schwarz */
        .stCheckbox label {
            color: black !important;
        }
        /* Streamlit-spezifische Checkbox-Styles */
        div[data-testid="stCheckbox"] input[type="checkbox"]:checked {
            background-color: black !important;
            border-color: black !important;
            accent-color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Seitenwahl (Subpages)
    page = st.sidebar.selectbox("Seite", ["Datei-Viewer", "Live-Messung (USB)"])

    if page == "Live-Messung (USB)":
        render_live_page()
        return

    # Datei-Viewer (Standardseite)


    uploads = st.file_uploader(
        "Spektrometer-TXT-Dateien",
        type=["txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploaded_files",
    )

    if uploads:
        file_buffers: Dict[str, bytes] = {up.name: up.getvalue() for up in uploads}
        st.session_state["uploaded_files_cache"] = file_buffers
    else:
        file_buffers = st.session_state.get("uploaded_files_cache", {})

    if not file_buffers:
        st.info("Lade Spektrometer-TXT-Dateien hoch, um zu starten.")
        return

    name_to_df: Dict[str, pd.DataFrame] = {}
    parse_errors: List[str] = []
    for fname, raw_bytes in file_buffers.items():
        try:
            name_to_df[fname] = parse_spectrometer_text(raw_bytes)
        except Exception as exc:
            parse_errors.append(f"• {fname}: {exc}")
    if parse_errors:
        st.warning("Einige Dateien konnten nicht gelesen werden:\n" + "\n".join(parse_errors))
    if not name_to_df:
        st.error("Keine gültigen Dateien zum Plotten.")
        return

    # Layout: links Plot, rechts Steuerung
    left, spacer, right = st.columns([5, 0.2, 1])
    with right:
        # Optional: separates Dunkelspektrum
        st.markdown("**Optional: separates Dunkelspektrum**")
        ext_dark_file = st.file_uploader("Dunkelspektrum (gleiches Format)", type=["txt"], accept_multiple_files=False, key="ext_dark")
        ext_dark_df = None
        ext_dark_err = None
        if ext_dark_file is not None:
            try:
                ext_dark_df = parse_spectrometer_text(ext_dark_file.getvalue())
            except Exception as exc:
                ext_dark_err = str(exc)
        if ext_dark_err:
            st.warning(f"Dunkelspektrum konnte nicht gelesen werden: {ext_dark_err}")
            ext_dark_df = None

        ext_dark_scale = 1.0
        ext_dark_mode = "Off"
        if ext_dark_df is not None:
            ext_dark_scale = st.number_input(
                "Skalierungsfaktor Dunkel",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                help="Setze (t_sample / t_dark), wenn die Integrationszeiten unterschiedlich sind."
            )
            # intern auf bekannte Werte abbilden
            _dark_options_de = {"Von Sample abziehen": "Subtract from sample", "Datei-Dark ersetzen": "Replace file dark", "Aus": "Off"}
            _choice_de = st.radio(
                "Verwendung des Dunkelspektrums",
                options=list(_dark_options_de.keys()),
                index=0,
                help="Wie soll das hochgeladene Dunkelspektrum angewendet werden?"
            )
            ext_dark_mode = _dark_options_de[_choice_de]

        # Y-Achse: Standard auf "sample" gesetzt
        y_choice = "sample"
        show_legend = st.checkbox("Legende anzeigen", value=False)
        x_min = float(min(df_i["nm"].min() for df_i in name_to_df.values()))
        x_max = float(max(df_i["nm"].max() for df_i in name_to_df.values()))
        x_range = (x_min, x_max)  # Automatisch voller Bereich
        
        # Polynom-Anpassung
        st.markdown("**Polynom-Anpassung (Least Squares)**")
        fit_polynomial = st.checkbox("Polynom-Kurve anpassen", value=False,
                                    help="Passt eine Polynom-Kurve (Grad 10) mit Least Squares an das Spektrum an.")
        
        poly_degree = 10  # Automatisch auf Grad 10 gesetzt
        
        # Planck-Kurve Anpassung
        st.markdown("**Planck-Kurve Anpassung**")
        fit_planck = st.checkbox("Planck-Kurve anpassen", value=False,
                                help="Passt die Planck'sche Strahlungsformel B(λ,T) an das Spektrum an.")
        
        fit_scaling = True
        if fit_planck:
            fit_scaling = st.checkbox("Skalierungsfaktor A anpassen", value=True,
                                     help="Wenn aktiviert, wird zusätzlich zur Temperatur T auch ein Skalierungsfaktor A angepasst.")

    def build_external_dark_interpolator(dark_df: Optional[pd.DataFrame]):
        if dark_df is None or dark_df.empty:
            return None
        nm_vals = dark_df["nm"].values
        if "dark" in dark_df.columns and np.nanmax(np.abs(dark_df["dark"].values)) > 0:
            dvals = dark_df["dark"].values
        else:
            dvals = dark_df["sample"].values if "sample" in dark_df.columns else None
        if dvals is None:
            return None
        mask = np.isfinite(nm_vals) & np.isfinite(dvals)
        nm_vals = nm_vals[mask]
        dvals = dvals[mask]
        if nm_vals.size < 2:
            return None
        def interp_fn(nm_query: np.ndarray) -> np.ndarray:
            return np.interp(nm_query, nm_vals, dvals, left=dvals[0], right=dvals[-1])
        return interp_fn

    external_dark_fn = build_external_dark_interpolator(ext_dark_df)

    with left:
        fig_pl = make_subplots(rows=1, cols=1)
        fig_pl.update_layout(
            template="simple_white",
            hovermode="x unified",
            margin=dict(l=40, r=10, t=30, b=40),
            height=650,
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)),
        )

        def _short_name(name: str, max_len: int = 28) -> str:
            base = name.rsplit("/", 1)[-1]
            if base.lower().endswith(".txt"):
                base = base[:-4]
            return (base[:max_len] + "…") if len(base) > max_len else base

        planck_axis_added = False

        for fname, df_i in name_to_df.items():
            nm_vals = df_i["nm"].values
            mask = (nm_vals >= x_range[0]) & (nm_vals <= x_range[1])
            y_vals, lbl = compute_y(
                df_i,
                y_choice,
                external_dark_interp=external_dark_fn,
                external_dark_scale=float(ext_dark_scale) if ext_dark_df is not None else 1.0,
                external_dark_mode=ext_dark_mode if ext_dark_df is not None else "Off",
            )
            nm_plot = nm_vals[mask]
            y_plot_raw = y_vals[mask]
            y_plot = y_plot_raw
            
            # WICHTIG: Spektrum-Trace wird IMMER hinzugefügt (auch wenn Planck-Fit aktiviert ist)
            # Dieses Trace wird NIE entfernt oder überschrieben
            # Beide Kurven (Spektrum + Planck) werden im selben Koordinatensystem angezeigt
            fig_pl.add_trace(go.Scatter(
                x=nm_plot,
                y=y_plot,
                mode="lines",
                line=dict(color="black", width=1),
                name=f"Spektrum {_short_name(fname)}",
                hovertemplate="nm=%{x:.1f}<br>counts=%{y:.0f}<extra></extra>",
                showlegend=show_legend,
            ))
            
            # Polynom-Anpassung (wird zuerst durchgeführt, falls aktiviert)
            poly_fit_plot = None
            if fit_polynomial and y_plot_raw.size > 0:
                try:
                    # Führe Polynom-Anpassung durch
                    coeffs, r_squared_poly, poly_fit_values = polynomial_fit(nm_plot, y_plot_raw, degree=poly_degree)
                    
                    # Extrahiere nur die Werte für den geplotteten Bereich
                    poly_fit_plot = poly_fit_values
                    
                    # Stelle sicher, dass die Werte gültig sind
                    valid_mask = np.isfinite(poly_fit_plot)
                    if np.any(valid_mask):
                        # Füge die Polynom-Kurve als zusätzliches Trace hinzu
                        fig_pl.add_trace(go.Scatter(
                            x=nm_plot,
                            y=poly_fit_plot,
                            mode="lines",
                            line=dict(color="blue", width=1.5, dash="dot"),
                            name=f"Polynom (Grad {poly_degree}) {_short_name(fname)}",
                            hovertemplate="nm=%{x:.1f}<br>Polynom=%{y:.0f}<extra></extra>",
                            showlegend=show_legend,
                        ))
                        
                        # Erstelle Polynom-Gleichung als String
                        poly_str_parts = []
                        for i, coeff in enumerate(coeffs):
                            power = len(coeffs) - 1 - i
                            if abs(coeff) > 1e-10:  # Ignoriere sehr kleine Koeffizienten
                                if power == 0:
                                    poly_str_parts.append(f"{coeff:.2e}")
                                elif power == 1:
                                    poly_str_parts.append(f"{coeff:.2e}·x")
                                else:
                                    poly_str_parts.append(f"{coeff:.2e}·x^{power}")
                        
                        poly_eq = " + ".join(poly_str_parts) if poly_str_parts else "0"
                    else:
                        pass  # Ungültige Werte werden ignoriert
                except Exception:
                    pass  # Fehler werden stillschweigend ignoriert
            
            # Planck-Kurve Anpassung
            # Wenn Polynom-Anpassung aktiviert ist, verwende die Polynom-Werte statt der Originaldaten
            if fit_planck and y_plot_raw.size > 0:
                try:
                    # Entscheide, welche Daten für den Planck-Fit verwendet werden sollen
                    if fit_polynomial and poly_fit_plot is not None and np.any(np.isfinite(poly_fit_plot)):
                        # Verwende die Polynom-Anpassung als Basis für den Planck-Fit
                        data_for_planck = poly_fit_plot
                        data_source = "Polynom-Anpassung"
                    else:
                        # Verwende die Originaldaten
                        data_for_planck = y_plot_raw
                        data_source = "Originaldaten"
                    
                    # Führe Planck-Fit durch
                    temperature, scaling_factor, r_squared, fitted_curve_full = fit_planck_curve(
                        nm_plot, data_for_planck, fit_scaling=fit_scaling
                    )
                    
                    if temperature > 0:
                        # Extrahiere nur die Werte für den geplotteten Bereich
                        # fitted_curve_full hat die gleiche Länge wie nm_plot
                        fitted_curve_plot = fitted_curve_full
                        
                        # Stelle sicher, dass die Werte gültig sind
                        valid_mask = np.isfinite(fitted_curve_plot) & (fitted_curve_plot >= 0)
                        if np.any(valid_mask):
                            # Füge die angepasste Planck-Kurve als zusätzliches Trace hinzu
                            if not planck_axis_added:
                                fig_pl.update_layout(
                                    yaxis2=dict(
                                        title="",
                                        overlaying="y",
                                        side="right",
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False,
                                        ticks="",
                                    )
                                )
                                planck_axis_added = True

                            fig_pl.add_trace(go.Scatter(
                                x=nm_plot,
                                y=fitted_curve_plot,
                                mode="lines",
                                line=dict(color="red", width=1.5, dash="dash"),
                                name=f"Planck {_short_name(fname)}",
                                hovertemplate="nm=%{x:.1f}<br>Planck=%{y:.0f}<extra></extra>",
                                showlegend=show_legend,
                                yaxis="y2" if planck_axis_added else "y",
                            ))
                        else:
                            pass  # Ungültige Werte werden ignoriert
                    else:
                        pass  # Fehlgeschlagener Fit wird ignoriert
                        
                except Exception:
                    pass  # Fehler werden stillschweigend ignoriert

        fig_pl.update_xaxes(title_text="Wellenlänge / nm", showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
        fig_pl.update_yaxes(title_text="Zählwerte", showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
        fig_pl.update_layout(hoverlabel=dict(font_size=10))
        st.plotly_chart(fig_pl, use_container_width=True, config={"displayModeBar": False})

        # PNG-Export
        try:
            png_bytes = pio.to_image(fig_pl, format="png", scale=2)
        except Exception:
            png_bytes = b""
        st.download_button(
            label="PNG herunterladen",
            data=png_bytes,
            file_name="spectrum.png",
            mime="image/png"
        )

    # Daten lesen (nicht blockierend)
    def parse_line(line: str) -> Optional[Tuple[float, float]]:
        # Erwartete Formate (beispielhaft): "nm;counts" oder "nm,counts" (Dezimalkomma wird unterstützt)
        line = line.strip()
        if not line:
            return None
        sep = ";" if ";" in line else ","
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) < 2:
            return None
        def to_float(s: str) -> float:
            return float(s.replace(",", "."))
        try:
            nm = to_float(parts[0])
            counts = to_float(parts[1])
            return nm, counts
        except Exception:
            return None

    if st.session_state.get("is_running", False):
        if mock_mode:
            # Erzeuge einfache Testkurve (Glocke) über 380..780 nm
            nm = np.linspace(380, 780, 401)
            center = 550.0
            width = 60.0
            counts = 1000.0 * np.exp(-0.5 * ((nm - center) / width) ** 2) + 50.0 * np.random.rand(nm.size)
            st.session_state["live_df"] = pd.DataFrame({"nm": nm, "counts": counts})
        else:
            try:
                ser = st.session_state.get("serial_conn")
                if ser is not None and ser.readable():
                    raw = ser.read(4096)  # bytes
                    text = raw.decode("utf-8", errors="ignore")
                    lines = text.splitlines()
                    nm_vals: List[float] = []
                    counts_vals: List[float] = []
                    for ln in lines:
                        parsed = parse_line(ln)
                        if parsed is None:
                            continue
                        nm_i, c_i = parsed
                        nm_vals.append(nm_i)
                        counts_vals.append(c_i)
                    if nm_vals:
                        df_new = pd.DataFrame({"nm": nm_vals, "counts": counts_vals}).dropna()
                        df_new = df_new.sort_values("nm")
                        st.session_state["live_df"] = df_new
            except Exception as e:
                st.warning(f"Lesefehler: {e}")

    # Plot anzeigen (mit optionalem kontinuierlichem Update ohne Seiten-Reload)
    plot_placeholder = st.empty()

    def draw_plot(df_plot: pd.DataFrame) -> None:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(template="simple_white", height=600)
        fig.add_trace(go.Scatter(x=df_plot["nm"], y=df_plot["counts"], mode="lines", line=dict(color="black", width=1)))
        fig.update_xaxes(title_text="Wavelength / nm")
        fig.update_yaxes(title_text="Counts")
        plot_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    df_live = st.session_state.get("live_df")
    if df_live is None or df_live.empty:
        return

    if st.session_state.get("is_running", False) and auto_update and continuous_mode:
        import time
        start_t = time.time()
        while time.time() - start_t < float(max_duration_s):
            # Daten ggf. neu lesen
            if mock_mode:
                nm = np.linspace(380, 780, 401)
                center = 550.0
                width = 60.0
                counts = 1000.0 * np.exp(-0.5 * ((nm - center) / width) ** 2) + 50.0 * np.random.rand(nm.size)
                st.session_state["live_df"] = pd.DataFrame({"nm": nm, "counts": counts})
            else:
                try:
                    ser = st.session_state.get("serial_conn")
                    if ser is not None and ser.readable():
                        raw = ser.read(4096)
                        text = raw.decode("utf-8", errors="ignore")
                        lines = text.splitlines()
                        nm_vals: List[float] = []
                        counts_vals: List[float] = []
                        for ln in lines:
                            parsed = parse_line(ln)
                            if parsed is None:
                                continue
                            nm_i, c_i = parsed
                            nm_vals.append(nm_i)
                            counts_vals.append(c_i)
                        if nm_vals:
                            df_new = pd.DataFrame({"nm": nm_vals, "counts": counts_vals}).dropna()
                            df_new = df_new.sort_values("nm")
                            st.session_state["live_df"] = df_new
                except Exception as e:
                    st.warning(f"Lesefehler: {e}")
                    break
            # Zeichnen
            draw_plot(st.session_state.get("live_df"))
            time.sleep(max(0.0, st.session_state.get("auto_interval_ms", 1000)) / 1000.0)
        # Nach Ablauf einmal normal zeichnen
        draw_plot(st.session_state.get("live_df"))
    else:
        # Einmalige Darstellung
        draw_plot(df_live)

        # Bei Auto-Update per Rerun neu laden (Seite wird neu gerendert)
        if st.session_state.get("is_running", False) and auto_update and not continuous_mode:
            import time
            time.sleep(max(0, st.session_state.get("auto_interval_ms", 1000)) / 1000.0)
            st.rerun()

    st.caption("Hinweis: Das Datenformat des Geräts muss zu 'nm;counts' (oder 'nm,counts') passen. Dezimalkomma wird unterstützt.")


if __name__ == "__main__":
    main()


