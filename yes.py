import flet as ft
import asyncio
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from flet.matplotlib_chart import MatplotlibChart
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
# Matplotlib font configuration to avoid missing glyphs in SVG (e.g., Arial)
import matplotlib as mpl
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "DejaVu Serif"]
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
import os
import colorsys
import inspect
from typing import Optional, Tuple, Dict, Any, List
# --- PDF reportlab imports ---
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, CondPageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import tempfile
import shutil

try:
    _MPL_SUPPORTS_INTERACTIVE = "interactive" in inspect.signature(MatplotlibChart.__init__).parameters
except Exception:
    _MPL_SUPPORTS_INTERACTIVE = False


def _create_mpl_chart(fig, **kwargs):
    if _MPL_SUPPORTS_INTERACTIVE:
        kwargs.setdefault("interactive", True)
    return MatplotlibChart(fig, **kwargs)

def _pdf_available_width(doc, fallback=440.0):
    try:
        width = getattr(doc, "width", None)
        if width is None or float(width) <= 0:
            page_width = getattr(doc, "pagesize", (A4[0], A4[1]))[0]
            left = float(getattr(doc, "leftMargin", 0.0))
            right = float(getattr(doc, "rightMargin", 0.0))
            width = float(page_width) - left - right
    except Exception:
        width = fallback
    try:
        width = float(width)
    except Exception:
        width = fallback
    width = min(width, fallback)
    return max(width - 12.0, 0.0)

def _pdf_image(path: str, max_width: float, max_height: float):
    try:
        iw, ih = ImageReader(path).getSize()
        iw = float(iw)
        ih = float(ih)
        scale = min(max_width / max(iw, 1.0), max_height / max(ih, 1.0))
        if not (scale > 0 and scale < float('inf')):
            scale = 1.0
        width = max(1.0, iw * min(scale, 1.0))
        height = max(1.0, ih * min(scale, 1.0))
    except Exception:
        width = max(1.0, float(max_width))
        height = max(1.0, float(max_height))
    return Image(path, width=width, height=height)

APP_VERSION = "v1.0.0"

# =========================
#   Utilidades de rodamientos (frecuencias teóricas)
# =========================
def bearing_freqs_from_geometry(
    rpm: Optional[float],
    n_elements: Optional[int],
    d_mm: Optional[float],
    D_mm: Optional[float],
    theta_deg: Optional[float] = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Calcula FTF/BPFO/BPFI/BSF a partir de la geometría del rodamiento y RPM.
    Parámetros:
      - rpm: velocidad del eje [rev/min]
      - n_elements: número de elementos rodantes
      - d_mm: diámetro del elemento [mm]
      - D_mm: diámetro de paso (pitch) [mm]
      - theta_deg: ángulo de contacto [grados]

    Devuelve dict con claves: ftf, bpfo, bpfi, bsf (en Hz). None si faltan datos.
    Fórmulas estándar (frecuencia del eje f1 = rpm/60):
      FTF  = 0.5 * f1 * (1 - (d/D) * cosθ)
      BPFO = 0.5 * n * f1 * (1 - (d/D) * cosθ)
      BPFI = 0.5 * n * f1 * (1 + (d/D) * cosθ)
      BSF  = (D/d) * 0.5 * f1 * (1 - ((d/D) * cosθ)**2)
    """
    try:
        if rpm is None or n_elements is None or d_mm is None or D_mm is None:
            return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}
        f1 = float(rpm) / 60.0
        if f1 <= 0 or n_elements <= 0 or d_mm <= 0 or D_mm <= 0:
            return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}
        ratio = float(d_mm) / float(D_mm)
        # Físicamente 0 < d/D < 1; limitar para evitar valores no realistas
        if not np.isfinite(ratio):
            return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}
        ratio = float(min(0.999, max(1e-9, ratio)))
        th = float(theta_deg or 0.0)
        cth = float(np.cos(np.deg2rad(th)))
        ftf = 0.5 * f1 * (1.0 - ratio * cth)
        bpfo = 0.5 * float(n_elements) * f1 * (1.0 - ratio * cth)
        bpfi = 0.5 * float(n_elements) * f1 * (1.0 + ratio * cth)
        # Evitar división por cero en BSF si d_mm ~ 0
        bsf = (float(D_mm) / float(d_mm)) * 0.5 * f1 * (1.0 - ( (ratio * cth) ** 2 )) if d_mm > 0 else None
        return {"ftf": ftf, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf}
    except Exception:
        return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}

# =========================
#   Filtros anti-alias y decimación
# =========================
def _kaiser_beta(atten_db: float) -> float:
    A = float(max(0.0, atten_db))
    if A > 50.0:
        return 0.1102 * (A - 8.7)
    if A >= 21.0:
        return 0.5842 * (A - 21.0) ** 0.4 + 0.07886 * (A - 21.0)
    return 0.0

def design_kaiser_lowpass(fs_hz: float, f_pass_hz: float, f_stop_hz: float, atten_db: float = 80.0) -> np.ndarray:
    """
    Diseña un FIR pasa‑bajas de fase lineal (ventana Kaiser) para anti‑aliasing.
    - fs_hz: frecuencia de muestreo actual
    - f_pass_hz: borde de banda de paso (se mantiene plano)
    - f_stop_hz: inicio de banda de rechazo (>= Nyquist de la señal decimada deseada)
    - atten_db: atenuación en banda de rechazo objetivo (dB)

    Devuelve coeficientes (taps) normalizados a ganancia DC = 1.
    """
    fs = float(fs_hz)
    f_pass = float(f_pass_hz)
    f_stop = float(f_stop_hz)
    if not (fs > 0 and 0 < f_pass < f_stop < fs * 0.5):
        raise ValueError("Parámetros inválidos para diseño del FIR (revisa fs, f_pass, f_stop).")

    # Parámetros Kaiser
    beta = _kaiser_beta(atten_db)
    delta_f = max(1e-9, f_stop - f_pass)
    # Ancho de transición en rad/muestra (0..pi)
    d_omega = 2.0 * np.pi * (delta_f / fs)
    # Longitud aproximada (Oppenheim/Schafer). A mayor atenuación o menor transición -> más taps
    numtaps = int(np.ceil((max(atten_db, 0.0) - 8.0) / (2.285 * d_omega))) + 1
    numtaps = max(11, numtaps)
    if numtaps % 2 == 0:
        numtaps += 1  # tipo I, fase lineal y retardo entero

    # Corte en mitad de la banda de transición
    f_c = 0.5 * (f_pass + f_stop)
    f_c_n = f_c / fs  # ciclos por muestra
    n = np.arange(numtaps)
    m = (numtaps - 1) / 2.0
    # Kernel ideal (sinc) y ventana Kaiser
    h = 2.0 * f_c_n * np.sinc(2.0 * f_c_n * (n - m))
    w = np.kaiser(numtaps, beta)
    h *= w
    # Normalizar ganancia DC a 1
    h /= np.sum(h)
    return h.astype(float)

def anti_alias_and_decimate(time_s: np.ndarray,
                            x: np.ndarray,
                            f_max_hz: float,
                            margin: float = 2.8,
                            atten_db: float = 80.0) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Filtra con FIR anti‑alias y decima para evitar problemas por muestrear de más.
    - f_max_hz: frecuencia máxima de análisis que se desea conservar
    - margin: factor >= 2.5–3 sobre f_max para fs_out (regla práctica)
    - atten_db: atenuación objetivo del anti‑alias (dB)

    Devuelve: (t_dec, x_dec, fs_out, info)
    """
    t = np.asarray(time_s).astype(float).ravel()
    y = np.asarray(x).astype(float).ravel()
    if t.size < 3 or y.size != t.size:
        raise ValueError("Se requieren series t y x del mismo tamaño (>=3).")
    dt = float(np.median(np.diff(t)))
    if not (dt > 0):
        raise ValueError("No se pudo estimar dt > 0 para decimación.")
    fs_in = 1.0 / dt

    if not (f_max_hz and f_max_hz > 0):
        # Nada que hacer si no hay banda objetivo
        return t, y, fs_in, {"M": 1, "fs_in": fs_in, "fs_out": fs_in, "note": "sin decimación"}

    # Factor entero de decimación buscando fs_out >= margin * f_max
    M = int(np.floor(fs_in / (margin * f_max_hz)))
    if M < 2:
        return t, y, fs_in, {"M": 1, "fs_in": fs_in, "fs_out": fs_in, "note": "fs ya adecuada"}

    fs_out = fs_in / M
    nyq_out = 0.5 * fs_out
    # Banda del FIR en fs_in: paso hasta f_max, rechazo a 0.9*Nyquist de fs_out (colchón)
    f_pass = min(f_max_hz, 0.9 * nyq_out)
    f_stop = 0.95 * nyq_out
    h = design_kaiser_lowpass(fs_in, f_pass, f_stop, atten_db=atten_db)
    gd = (len(h) - 1) // 2

    # Convolución y recorte de transitorios en los extremos
    y_f = np.convolve(y, h, mode="full")
    # Centrar (compensar retardo de grupo) y quitar bordes con transitorio
    y_lin = y_f[gd:gd + y.size]
    start = gd
    stop = y_lin.size - gd
    if stop <= start:
        # Si la señal es demasiado corta comparada con el FIR, no recortamos extremos
        start, stop = 0, y_lin.size
    yy = y_lin[start:stop]
    tt = t[start:stop]

    # Decimación: tomar una de cada M muestras
    t_dec = tt[::M]
    x_dec = yy[::M]
    fs_out = fs_in / M

    info = {
        "M": int(M),
        "numtaps": int(len(h)),
        "fs_in": float(fs_in),
        "fs_out": float(fs_out),
        "f_pass_hz": float(f_pass),
        "f_stop_hz": float(f_stop),
        "atten_db": float(atten_db),
    }
    return t_dec, x_dec, fs_out, info

# =========================
#   Analizador independiente
# =========================
def analyze_vibration(
    time_s: np.ndarray,
    acc_ms2: np.ndarray,
    rpm: Optional[float] = None,
    line_freq_hz: Optional[float] = None,
    bpfo_hz: Optional[float] = None,
    bpfi_hz: Optional[float] = None,
    bsf_hz: Optional[float] = None,
    ftf_hz: Optional[float] = None,
    gear_teeth: Optional[int] = None,
    segment: Optional[Tuple[float, float]] = None,
    pre_decimate_to_fmax_hz: Optional[float] = None,
    pre_decimate_margin: float = 2.8,
    pre_decimate_atten_db: float = 80.0,
    env_bp_lo_hz: Optional[float] = None,
    env_bp_hi_hz: Optional[float] = None,
    tol_frac: float = 0.02,
    min_bins: int = 2,
    min_snr_db: float = 6.0,
    top_k_peaks: int = 5,
) -> Dict[str, Any]:
    def _to_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).astype(float).ravel()
        return x
    def _clean_pair(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = np.isfinite(t) & np.isfinite(y)
        t2, y2 = t[m], y[m]
        if len(t2) > 1 and not np.all(np.diff(t2) >= 0):
            idx = np.argsort(t2)
            t2, y2 = t2[idx], y2[idx]
        return t2, y2
    def _segment(t: np.ndarray, y: np.ndarray, seg: Optional[Tuple[float,float]]):
        if seg is None or len(t) < 2:
            return t, y
        t0, t1 = seg
        if t0 > t1:
            t0, t1 = t1, t0
        m = (t >= t0) & (t <= t1)
        tt, yy = t[m], y[m]
        return (tt, yy) if len(tt) >= 2 else (t, y)
    def _fs_from_time(t: np.ndarray) -> Tuple[float, float]:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        fs = 1.0 / dt if dt > 0 else 0.0
        return fs, dt
    def _acc_fft_to_vel_mm_s(y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Calcula espectro de aceleración y velocidad (mm/s) usando rFFT con ventana Hann
        para visualización, y además obtiene RMS de velocidad en el tiempo integrando en 
        frecuencia (sin ventana) para mayor fidelidad.
        Devuelve: (f_hz, acc_spec_ms2, vel_spec_mm_s, rms_vel_time_mm_s)
        """
        N = len(y)
        if N < 2 or dt <= 0:
            return np.array([]), np.array([]), np.array([]), 0.0

        # Espectro para visualización (ventana Hann + corrección de ganancia)
        w = np.hanning(N)
        cg = w.mean() if w.mean() != 0 else 1.0
        Yw = np.fft.rfft(y * w)
        xf = np.fft.rfftfreq(N, dt)

        mag_acc = np.abs(Yw) / (N * cg)
        if N % 2 == 0 and mag_acc.size >= 2:
            mag_acc[1:-1] *= 2.0
        elif N % 2 == 1 and mag_acc.size >= 2:
            mag_acc[1:] *= 2.0

        mag_vel = np.zeros_like(mag_acc)
        pos = xf > 0
        mag_vel[pos] = mag_acc[pos] / (2.0 * np.pi * xf[pos])  # m/s
        mag_vel_mm = mag_vel * 1000.0

        # RMS de velocidad temporal via integración en frecuencia (sin ventana)
        Y = np.fft.rfft(y)
        V = np.zeros_like(Y, dtype=complex)
        pos_idx = xf > 0
        V[pos_idx] = Y[pos_idx] / (1j * 2.0 * np.pi * xf[pos_idx])
        V[0] = 0.0
        v_t = np.fft.irfft(V, n=N)
        rms_vel_time_mm = 1000.0 * float(np.sqrt(np.mean(v_t**2)))

        return xf, mag_acc, mag_vel_mm, rms_vel_time_mm
    def _analytic_signal(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        N = len(y)
        if N < 2:
            return y.astype(complex)
        Y = np.fft.fft(y)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = 1
            h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        Z = np.fft.ifft(Y * h)
        return Z
    def _envelope_spectrum(y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        N = len(y)
        if N < 2 or dt <= 0:
            return np.array([]), np.array([])
        z = _analytic_signal(y)
        env = np.abs(z)
        env = env - float(np.mean(env))
        Ef = np.fft.fft(env)
        xf = np.fft.fftfreq(N, dt)[: N // 2]
        mag = 2.0 / N * np.abs(Ef[: N // 2])
        return xf, mag
    def _bandpass_fft(y: np.ndarray, dt: float, f_lo: float, f_hi: float) -> np.ndarray:
        try:
            if dt <= 0:
                return y
            fs = 1.0 / dt
            if not (f_lo and f_hi) or not (0.0 < f_lo < f_hi < 0.5 * fs):
                return y
            N = len(y)
            Y = np.fft.fft(y)
            f = np.fft.fftfreq(N, dt)
            mask = (np.abs(f) >= float(f_lo)) & (np.abs(f) <= float(f_hi))
            Yf = np.where(mask, Y, 0.0)
            yf = np.fft.ifft(Yf).real
            return yf.astype(float)
        except Exception:
            return y
    def _amp_near(xf: np.ndarray, spec: np.ndarray, f: Optional[float], df: float) -> float:
        if xf is None or spec is None or len(xf) == 0 or len(spec) == 0:
            return 0.0
        if f is None or not np.isfinite(f) or f <= 0:
            return 0.0
        bw = max(tol_frac * f, min_bins * df)
        idx = (xf >= (f - bw)) & (xf <= (f + bw))
        return float(np.max(spec[idx])) if np.any(idx) else 0.0
    def _find_top_peaks(xf: np.ndarray, y: np.ndarray, k: int, min_freq: float = 0.5, snr_db: float = 6.0) -> List[Dict[str, float]]:
        if len(xf) == 0 or len(y) == 0:
            return []
        mask = xf >= min_freq
        xv = xf[mask]
        yv = y[mask]
        if len(yv) == 0:
            return []
        ref = float(np.median(yv) + 1e-12)
        snr = 20.0 * np.log10(np.maximum(yv, 1e-12) / ref)
        cand = np.where(snr >= snr_db)[0]
        if len(cand) == 0:
            idx = np.argsort(yv)[-k:][::-1]
        else:
            idx = cand[np.argsort(yv[cand])[-k:]][::-1]
        peaks = []
        for i in idx[:k]:
            peaks.append({"f_hz": float(xv[i]), "amp": float(yv[i]), "snr_db": float(snr[i])})
        return peaks
    def _severity_iso_mm_s(rms_mm_s: float) -> Tuple[str, str]:
        if rms_mm_s <= 2.8:
            return "Buena (Aceptable)", "#2ecc71"
        elif rms_mm_s <= 4.5:
            return "Satisfactoria (Vigilancia)", "#f1c40f"
        elif rms_mm_s <= 7.1:
            return "Insatisfactoria (Crítica)", "#e67e22"
        else:
            return "Inaceptable (Riesgo de daño)", "#e74c3c"
    def _get_1x(dom_freq_guess: Optional[float], rpm_opt: Optional[float]) -> float:
        try:
            if rpm_opt and np.isfinite(rpm_opt) and rpm_opt > 0:
                return float(rpm_opt) / 60.0
        except Exception:
            pass
        try:
            if dom_freq_guess and np.isfinite(dom_freq_guess) and dom_freq_guess > 0:
                return float(dom_freq_guess)
        except Exception:
            pass
        return 0.0
    t = _to_1d(time_s)
    a = _to_1d(acc_ms2)
    t, a = _clean_pair(t, a)
    if len(t) < 2:
        raise ValueError("Datos insuficientes.")
    t, a = _segment(t, a, segment)
    predec_info = None
    if pre_decimate_to_fmax_hz is not None:
        try:
            t, a, fs_try, info = anti_alias_and_decimate(
                t, a, f_max_hz=float(pre_decimate_to_fmax_hz), margin=float(pre_decimate_margin), atten_db=float(pre_decimate_atten_db)
            )
            predec_info = info
        except Exception:
            predec_info = {"error": "falló pre-decimación"}
            # continuar con datos originales
            pass
    fs, dt = _fs_from_time(t)

    # Preprocesado: quitar DC/tendencia para evitar pico LF en FFT/integración
    # Esto mejora la visualización y la estimación de dominante.
    try:
        if len(t) >= 2:
            x0 = t - float(t[0])
            p = np.polyfit(x0, a, 1)
            trend = p[0] * x0 + p[1]
        else:
            trend = np.full_like(a, float(np.mean(a)))
    except Exception:
        trend = np.full_like(a, float(np.mean(a)))
    a_proc = a - trend
    df = fs / len(a) if fs > 0 and len(a) > 0 else 0.0
    xf, mag_acc, mag_vel_mm, rms_vel_time_mm = _acc_fft_to_vel_mm_s(a_proc, dt)
    # RMS de aceleración sin DC/tendencia
    rms_time_acc = float(np.sqrt(np.mean(a_proc**2))) if len(a_proc) else 0.0
    peak_acc = float(np.max(np.abs(a))) if len(a) else 0.0
    pp_acc = float(np.ptp(a)) if len(a) else 0.0
    if len(mag_vel_mm) > 0:
        # Dominante ignorando muy baja frecuencia para evitar sesgos visuales
        dom_min_hz = 0.5
        mask_dom = xf >= dom_min_hz
        if np.any(mask_dom):
            rel_idx = int(np.argmax(mag_vel_mm[mask_dom]))
            idx_dom = np.where(mask_dom)[0][rel_idx]
        else:
            idx_dom = int(np.argmax(mag_vel_mm))
        dom_freq = float(xf[idx_dom])
        dom_amp = float(mag_vel_mm[idx_dom])
    else:
        dom_freq, dom_amp = 0.0, 0.0
    # Severidad basada en RMS de velocidad temporal (mm/s)
    rms_vel_spec_mm = rms_vel_time_mm
    f1 = _get_1x(dom_freq, rpm)
    r2x = _amp_near(xf, mag_vel_mm, 2.0 * f1 if f1 > 0 else 0.0, df) / (dom_amp + 1e-12)
    r3x = _amp_near(xf, mag_vel_mm, 3.0 * f1 if f1 > 0 else 0.0, df) / (dom_amp + 1e-12)
    if len(xf) > 0:
        e_total = float(np.sum(mag_vel_mm**2)) + 1e-12
        e_low = float(np.sum((mag_vel_mm[(xf >= 0.0) & (xf < 30.0)]**2))) if np.any((xf >= 0) & (xf < 30)) else 0.0
        e_mid = float(np.sum((mag_vel_mm[(xf >= 30.0) & (xf < 120.0)]**2))) if np.any((xf >= 30) & (xf < 120)) else 0.0
        e_high = float(np.sum((mag_vel_mm[(xf >= 120.0)]**2))) if np.any(xf >= 120) else 0.0
    else:
        e_total = 1e-12; e_low = e_mid = e_high = 0.0
    peaks_fft = _find_top_peaks(xf, mag_vel_mm, k=top_k_peaks, min_freq=0.5, snr_db=min_snr_db)
    # Envolvente: opcionalmente aplicar band-pass previo
    a_env_src = a_proc
    try:
        _lo = float(env_bp_lo_hz) if env_bp_lo_hz is not None else None
    except Exception:
        _lo = None
    try:
        _hi = float(env_bp_hi_hz) if env_bp_hi_hz is not None else None
    except Exception:
        _hi = None
    if (_lo is not None) and (_hi is not None):
        a_env_src = _bandpass_fft(a_proc, dt, _lo, _hi)
    xf_env, env_spec = _envelope_spectrum(a_env_src, dt)
    peaks_env = _find_top_peaks(xf_env, env_spec, k=top_k_peaks, min_freq=1.0, snr_db=min_snr_db)
    # Resolución de la envolvente para tolerancias correctas
    df_env = 0.0
    try:
        if xf_env is not None and len(xf_env) > 1:
            df_env = float(np.median(np.diff(xf_env)))
    except Exception:
        df_env = 0.0
    sev_label, sev_color = _severity_iso_mm_s(rms_vel_spec_mm)
    findings: List[str] = []
    findings.append(f"Severidad ISO: {sev_label} (RMS={rms_vel_spec_mm:.3f} mm/s)")
    if f1 > 0 and dom_freq > 0:
        if (abs(dom_freq - f1) <= max(tol_frac * f1, min_bins * df)) and (r2x < 0.5) and (r3x < 0.4) and (e_low / e_total > 0.5):
            findings.append("Desbalanceo probable: 1X dominante, 2X/3X bajos, energía en baja frecuencia.")
    if r2x >= 0.6 or r3x >= 0.4:
        findings.append("Desalineación probable: armónicos 2X/3X elevados respecto a 1X.")
    if gear_teeth and gear_teeth > 0 and f1 > 0:
        fmesh = gear_teeth * f1
        a_mesh = _amp_near(xf, mag_vel_mm, fmesh, df)
        if a_mesh > 0.2 * (dom_amp + 1e-12):
            findings.append(f"Engranes: componente en malla ~{fmesh:.1f} Hz.")
    bearing_hits = []
    for name, freq in (("BPFO", bpfo_hz), ("BPFI", bpfi_hz), ("BSF", bsf_hz), ("FTF", ftf_hz)):
        if freq and freq > 0:
            tol_env = df_env if df_env > 0 else (df if df > 0 else (1.0/len(a) if len(a)>0 else 0.1))
            a_env = _amp_near(xf_env, env_spec, freq, tol_env)
            if a_env > 0:
                # Sidebands en envolvente ±k*f1 (k=1..2) si f1 válido
                has_sb = False
                if f1 and f1 > 0:
                    sb_amps = []
                    for k in (1, 2):
                        sb_amps.append(_amp_near(xf_env, env_spec, freq - k * f1, tol_env))
                        sb_amps.append(_amp_near(xf_env, env_spec, freq + k * f1, tol_env))
                    try:
                        sb_vals = [float(s) for s in sb_amps if s is not None]
                        sb_avg = float(np.mean(sb_vals)) if sb_vals else 0.0
                    except Exception:
                        sb_avg = 0.0
                    if sb_avg >= 0.2 * a_env:
                        has_sb = True
                bearing_hits.append(name + (" (SB)" if has_sb else ""))
    if bearing_hits:
        findings.append("Rodamientos: evidencia en envolvente para " + ", ".join(bearing_hits))
    else:
        # Modo automático parcial: sugerir posible defecto de rodamiento si NO hay BPFO/BPFI/BSF/FTF
        # y se observan picos destacados en el espectro de envolvente fuera de armónicos conocidos.
        try:
            if not any([(bpfo_hz and bpfo_hz > 0), (bpfi_hz and bpfi_hz > 0), (bsf_hz and bsf_hz > 0), (ftf_hz and ftf_hz > 0)]):
                # Construir lista de frecuencias conocidas a ignorar (1X..6X, línea y malla si aplica)
                known_env = []
                if f1 and f1 > 0:
                    for k in range(1, 7):
                        known_env.append(k * f1)
                if line_freq_hz and line_freq_hz > 0:
                    known_env.extend([line_freq_hz, 2.0 * line_freq_hz])
                if gear_teeth and gear_teeth > 0 and f1 and f1 > 0:
                    known_env.append(gear_teeth * f1)

                def _near_known_env(f):
                    for fk in known_env:
                        bw = max(tol_frac * max(f, fk), max(2, min_bins) * (df if df > 0 else 0.0))
                        if abs(f - fk) <= (bw if bw > 0 else 1.0):
                            return True
                    return False

                # Elegir picos de la envolvente significativos fuera de las conocidas
                cand = []
                for p in (peaks_env or []):
                    f0 = float(p.get("f_hz", 0.0))
                    a0 = float(p.get("amp", 0.0))
                    if f0 <= 1.0 or a0 <= 0:
                        continue
                    if _near_known_env(f0):
                        continue
                    cand.append((f0, a0))
                # Requiere al menos 2 picos relevantes para sugerir
                if len(cand) >= 2:
                    cand.sort(key=lambda x: x[1], reverse=True)
                    top_fs = ", ".join(f"{f:.1f} Hz" for f, _ in cand[:3])
                    findings.append(f"Rodamientos (modo automático parcial): picos en envolvente ~ {top_fs}.")
        except Exception:
            pass
    if line_freq_hz and line_freq_hz > 0:
        a_line = _amp_near(xf, mag_vel_mm, line_freq_hz, df)
        a_2line = _amp_near(xf, mag_vel_mm, 2.0 * line_freq_hz, df)
        if (a_line > 0.2 * (dom_amp + 1e-12)) or (a_2line > 0.2 * (dom_amp + 1e-12)):
            findings.append(f"Eléctrico: componentes en {line_freq_hz:.0f} Hz y/o {2*line_freq_hz:.0f} Hz.")
    # Resonancias estructurales: picos agudos no armonicos con Q alto
    try:
        if len(peaks_fft) > 0 and len(xf) > 3:
            # Conjunto de frecuencias conocidas a evitar
            known = []
            if f1 and f1 > 0:
                for k in range(1, 9):
                    known.append(k * f1)
            if line_freq_hz and line_freq_hz > 0:
                known.extend([line_freq_hz, 2.0 * line_freq_hz])
            if gear_teeth and gear_teeth > 0 and f1 and f1 > 0:
                known.append(gear_teeth * f1)
            for name, freq in (("BPFO", bpfo_hz), ("BPFI", bpfi_hz), ("BSF", bsf_hz), ("FTF", ftf_hz)):
                if freq and freq > 0:
                    known.append(freq)
            def _near_any(f):
                for fk in known:
                    bw = max(tol_frac * max(f, fk), max(2, min_bins) * (df if df > 0 else 0.0))
                    if abs(f - fk) <= (bw if bw > 0 else 1.0):
                        return True
                return False
            resonances = []
            for p in peaks_fft:
                f0 = float(p.get("f_hz", 0.0))
                a0 = float(p.get("amp", 0.0))
                if f0 <= 0 or a0 <= 0:
                    continue
                if _near_any(f0):
                    continue
                # Estimar Q con ancho a -3 dB (~0.707*A)
                thr = a0 / np.sqrt(2.0)
                idx0 = int(np.argmin(np.abs(xf - f0)))
                # Buscar izquierda
                iL = idx0
                while iL > 0 and mag_vel_mm[iL] > thr:
                    iL -= 1
                # Buscar derecha
                iR = idx0
                max_idx = len(mag_vel_mm) - 1
                while iR < max_idx and mag_vel_mm[iR] > thr:
                    iR += 1
                if iR > iL and (iR - iL) >= 2:
                    fL = float(xf[max(iL, 0)])
                    fR = float(xf[min(iR, max_idx)])
                    bw = max(fR - fL, df if df > 0 else 1e-6)
                    Q = float(f0 / bw) if bw > 0 else 0.0
                    # Umbrales: pico relevante y Q alto
                    if Q >= 8.0 and a0 >= max(0.2 * (dom_amp + 1e-12), 0.3):
                        resonances.append((f0, Q, a0))
            # Reportar hasta 2 resonancias principales
            resonances.sort(key=lambda x: x[2], reverse=True)
            for f0, Q, a0 in resonances[:2]:
                findings.append(f"Resonancia estructural probable: pico agudo ~{f0:.1f} Hz (Q~{Q:.1f}).")
    except Exception:
        pass
    if len(findings) == 1:
        findings.append("Sin anomalías evidentes según reglas actuales.")
    return {
        "segment_used": (float(t[0]), float(t[-1])),
        "fs_hz": fs,
        "dt_s": dt,
        "df_hz": df,
        "n": int(len(a)),
        "pre_decimation": predec_info,
        "time": {
            "t_s": t,
            "acc_ms2": a,
            "rms_acc_ms2": rms_time_acc,
            "peak_acc_ms2": peak_acc,
            "pp_acc_ms2": pp_acc,
        },
        "fft": {
            "f_hz": xf,
            "acc_spec_ms2": mag_acc,
            "vel_spec_mm_s": mag_vel_mm,
            "peaks": peaks_fft,
            "dom_freq_hz": dom_freq,
            "dom_amp_mm_s": dom_amp,
            "r2x": r2x,
            "r3x": r3x,
            "energy": {"low": e_low, "mid": e_mid, "high": e_high, "total": e_total},
        },
        "envelope": {
            "f_hz": xf_env,
            "amp": env_spec,
            "peaks": peaks_env,
        },
        "rpm": rpm,
        "f1_hz": f1,
        "severity": {"label": sev_label, "color": sev_color, "rms_mm_s": rms_vel_spec_mm},
        "diagnosis": findings,
    }



# =========================

#   Botón de Menú Mejorado

# =========================

class MenuButton(ft.Container):

    def __init__(self, icon_name, tooltip, on_click_handler, data=None, is_dark=False):

        self.is_dark = is_dark

        self.icon = ft.Icon(

            name=icon_name,

            color="#e0e0e0" if is_dark else "#2c3e50",

            size=28

        )

        super().__init__(

            width=65,

            height=65,

            border_radius=15,

            tooltip=tooltip,

            content=self.icon,

            ink=True,

            on_hover=self._on_hover,

            on_click=on_click_handler,

            data=data,

            padding=10,

            animate=ft.Animation(200, "easeOut"),

        )

        self.is_active = False



    def _on_hover(self, e: ft.HoverEvent):

        if not self.is_active:

            if e.data == "true":
                accent = getattr(self, "accent", "#3498db")
                self.bgcolor = ft.Colors.with_opacity(0.15, accent)

                self.scale = 1.05

            else:

                self.bgcolor = "transparent"

                self.scale = 1.0

            if self.page:

                self.update()



    def set_active(self, active: bool, safe=False):

        self.is_active = active

        if active:
            self.bgcolor = getattr(self, "accent", "#3498db")
            
            self.icon.color = "white"

            self.scale = 1.05

        else:

            self.bgcolor = "transparent"

            self.icon.color = "#e0e0e0" if self.is_dark else "#2c3e50"

            self.scale = 1.0

        if not safe and self.page:

            self.update()



    def update_theme(self, is_dark: bool):

        self.is_dark = is_dark

        if not self.is_active:

            self.icon.color = "#e0e0e0" if is_dark else "#2c3e50"

            self.bgcolor = "transparent"

        if self.page:

            self.update()





# =========================

#   Aplicación Principal

# =========================

class MainApp:

    def __init__(self, page: ft.Page):

        print("MainApp inicializada")

        self.page = page

        self.page.title = "Sistema de Análisis de Vibraciones Mecánicas"

        self.page.padding = 0

        

        # Configuración de ventana

        self.page.window.width = 1400

        self.page.window.height = 850

        self.page.window.min_width = 1000

        self.page.window.min_height = 700



        # Estado con valores por defecto

        self.clock_24h = self._get_bool_storage("clock_24h", True)



        stored_accent = self.page.client_storage.get("accent")
        self.accent = stored_accent if stored_accent is not None else "#3498db"



        self.is_dark_mode = self._get_bool_storage("is_dark_mode", True)



        self.is_panel_expanded = True  # Add this line after other state variables

        self.is_menu_expanded = True  # Add this line



        self._apply_theme()

        self.last_view = "welcome"

        self.uploaded_files = []

        self.current_df = None

        self.file_data_storage = {}  # Almacenar datos de archivos

        # Estado de análisis 3D (arranque/parada)
        self.waterfall_enabled = False
        self.waterfall_mode = "waterfall"  # "waterfall" o "surface"
        self.waterfall_window_s = 0.5
        self.waterfall_step_s = 0.1
        self._last_waterfall_data = None

        self._fft_zoom_range: Optional[Tuple[float, float]] = None
        self._fft_full_range: Optional[Tuple[float, float]] = None
        self._fft_zoom_syncing = False

        # Estado de análisis/rodamientos
        self.analysis_mode = 'auto'            # 'auto' o 'assist'
        self.selected_bearing_model = ''       # modelo seleccionado para preselección en análisis

        # Base de rodamientos (opcional)
        self.bearing_db_items: List[Dict[str, Any]] = []
        self._load_bearing_db()
        # Favoritos de reportes
        self.report_favorites = self._load_report_favorites()
        self.report_show_favs_only = self._get_bool_storage("report_favs_only", False)
        # Favoritos de rodamientos
        self.bearing_favorites = self._load_bearing_favorites()
        self.bearing_show_favs_only = self._get_bool_storage("bearing_favs_only", False)
        # Favoritos de archivos de datos
        self.data_favorites = self._load_data_favorites()
        self.data_show_favs_only = self._get_bool_storage("data_favs_only", False)



        self.file_picker = ft.FilePicker(on_result=self._handle_file_pick_result)
        self.page.overlay.append(self.file_picker)
        # File picker para CSV de rodamientos
        self.bearing_file_picker = ft.FilePicker(on_result=self._bearing_csv_pick_result)
        self.page.overlay.append(self.bearing_file_picker)



        self.clock_text = ft.Text(self._get_current_time(), size=14, weight="w500")

        self.files_list_view = ft.ListView(expand=1, spacing=8, auto_scroll=False, padding=10)



        self.menu_buttons = {}

        self.menu = self._build_menu()

        self.control_panel = self._build_control_panel()

        self.main_content_area = ft.Container(

            expand=True, 

            padding=25,

            border_radius=20,

            bgcolor=ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black"),

            margin=10,

            alignment=ft.alignment.center

        )

        

        # Layout principal con diseño mejorado

        self.content = ft.Row(

            expand=True,

            spacing=0,

            controls=[

                self.menu, 

                ft.Container(expand=True, content=self.main_content_area, padding=0),

                self.control_panel

            ],

        )

        

        self.main_content_area.content = self._build_welcome_view()

        self.page.run_task(self._start_clock_timer)



    def _get_current_time(self):

        fmt = "%H:%M:%S" if self.clock_24h else "%I:%M:%S %p"

        return time.strftime(fmt)



    def _apply_theme(self):

        self.page.theme_mode = ft.ThemeMode.DARK if self.is_dark_mode else ft.ThemeMode.LIGHT

        self.page.bgcolor = "#1a1a2e" if self.is_dark_mode else "#f5f5f5"

        self.page.update()

    # ===== Rodamientos: DB + asistentes =====
    def _load_bearing_db(self):
        """Intenta cargar 'bearing_db.csv' del directorio actual.
        Formato esperado de columnas: model,brand?,n,d_mm,D_mm,theta_deg
        """
        try:
            path = os.path.join(os.getcwd(), "bearing_db.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                items: List[Dict[str, Any]] = []
                for _, r in df.iterrows():
                    items.append({
                        "model": str(r.get("model", "")).strip(),
                        "brand": (str(r.get("brand", "")).strip() if "brand" in df.columns else None),
                        "n": int(r.get("n", 0)) if pd.notna(r.get("n", None)) else None,
                        "d_mm": float(r.get("d_mm", 0.0)) if pd.notna(r.get("d_mm", None)) else None,
                        "D_mm": float(r.get("D_mm", 0.0)) if pd.notna(r.get("D_mm", None)) else None,
                        "theta_deg": float(r.get("theta_deg", 0.0)) if pd.notna(r.get("theta_deg", None)) else 0.0,
                    })
                # Filtrar modelos no vacíos
                self.bearing_db_items = [it for it in items if it.get("model")] 
            else:
                self.bearing_db_items = []
        except Exception:
            self.bearing_db_items = []

    def _bearing_db_model_options(self) -> List[ft.dropdown.Option]:
        try:
            return [ft.dropdown.Option(it.get("model", "")) for it in (self.bearing_db_items or []) if it.get("model")]
        except Exception:
            return []

    def _on_bearing_model_change(self, e=None):
        try:
            model = getattr(self, "bearing_model_dd", None).value if getattr(self, "bearing_model_dd", None) else None
        except Exception:
            model = None
        if not model:
            return
        try:
            for it in (self.bearing_db_items or []):
                if it.get("model") == model:
                    # Rellenar campos de geometría
                    if getattr(self, "br_n_field", None):
                        self.br_n_field.value = str(it.get("n") or "")
                    if getattr(self, "br_d_mm_field", None):
                        self.br_d_mm_field.value = str(it.get("d_mm") or "")
                    if getattr(self, "br_D_mm_field", None):
                        self.br_D_mm_field.value = str(it.get("D_mm") or "")
                    if getattr(self, "br_theta_deg_field", None):
                        self.br_theta_deg_field.value = str(it.get("theta_deg") or "0")
                    if self.page:
                        try:
                            self.br_n_field.update(); self.br_d_mm_field.update(); self.br_D_mm_field.update(); self.br_theta_deg_field.update()
                        except Exception:
                            pass
                    break
        except Exception:
            pass

    def _on_mode_change(self, e=None):
        mode = None
        try:
            mode = getattr(self, "analysis_mode_dd", None).value if getattr(self, "analysis_mode_dd", None) else None
        except Exception:
            mode = None
        # Persistir estado seleccionado
        try:
            if mode in ("auto", "assist"):
                self.analysis_mode = mode
        except Exception:
            pass
        try:
            if getattr(self, "assisted_box", None) is not None:
                self.assisted_box.visible = (mode == "assist")
                self.assisted_box.update() if self.assisted_box.page else None
        except Exception:
            pass
        # En modo automático, limpiar campos de BPFO/BPFI/BSF/FTF para no condicionar el diagnóstico
        try:
            if mode == "auto":
                for fld_name in ("bpfo_field", "bpfi_field", "bsf_field", "ftf_field"):
                    fld = getattr(self, fld_name, None)
                    if fld:
                        fld.value = ""
                        fld.update() if fld.page else None
        except Exception:
            pass
        # Refrescar análisis
        try:
            self._update_analysis()
        except Exception:
            pass

    def _compute_bearing_freqs_click(self, e=None):
        try:
            rpm_val = float(self.rpm_hint_field.value) if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", "") else None
        except Exception:
            rpm_val = None
        def _tf_float(tf):
            try:
                return float(tf.value) if tf and getattr(tf, "value", "") else None
            except Exception:
                return None
        n_val = None
        try:
            n_val = int(float(self.br_n_field.value)) if getattr(self, "br_n_field", None) and getattr(self, "br_n_field").value not in (None, "") else None
        except Exception:
            n_val = None
        d_val = _tf_float(getattr(self, "br_d_mm_field", None))
        D_val = _tf_float(getattr(self, "br_D_mm_field", None))
        th_val = _tf_float(getattr(self, "br_theta_deg_field", None)) or 0.0
        freqs = bearing_freqs_from_geometry(rpm_val, n_val, d_val, D_val, th_val)
        # Rellenar los campos BPFO/BPFI/BSF/FTF si hay valores
        try:
            if freqs.get("bpfo"):
                self.bpfo_field.value = f"{freqs['bpfo']:.3f}"
            if freqs.get("bpfi"):
                self.bpfi_field.value = f"{freqs['bpfi']:.3f}"
            if freqs.get("bsf"):
                self.bsf_field.value = f"{freqs['bsf']:.3f}"
            if freqs.get("ftf"):
                self.ftf_field.value = f"{freqs['ftf']:.3f}"
            for fld in (self.bpfo_field, self.bpfi_field, self.bsf_field, self.ftf_field):
                try:
                    fld.update()
                except Exception:
                    pass
        except Exception:
            pass
        # Refrescar análisis automáticamente
        try:
            self._update_analysis()
        except Exception:
            pass

    def _save_bearing_db(self):
        try:
            path = os.path.join(os.getcwd(), "bearing_db.csv")
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["model", "brand", "n", "d_mm", "D_mm", "theta_deg"])
                for it in (self.bearing_db_items or []):
                    writer.writerow([
                        it.get("model", ""),
                        it.get("brand", ""),
                        it.get("n", ""),
                        it.get("d_mm", ""),
                        it.get("D_mm", ""),
                        it.get("theta_deg", 0.0),
                    ])
        except Exception:
            pass

    def _refresh_bearing_list_ui(self):
        try:
            if not getattr(self, "bearing_list_view", None):
                return
            self.bearing_list_view.controls.clear()
            q = ""
            try:
                q = str(getattr(self, 'bearing_search', None).value or "").strip().lower()
            except Exception:
                q = ""
            items = list(self.bearing_db_items or [])
            # Filtrar por marca según pestaña seleccionada
            try:
                sel_brand = None
                if getattr(self, 'bearing_tabs', None) and getattr(self.bearing_tabs, 'tabs', None):
                    idx = int(getattr(self.bearing_tabs, 'selected_index', 0) or 0)
                    tabs = self.bearing_tabs.tabs
                    if 0 <= idx < len(tabs):
                        sel_brand = getattr(tabs[idx], 'text', None)
                if sel_brand and sel_brand not in ("Todos", "Todas", "All"):
                    def _get_brand(it):
                        b = it.get('brand') if isinstance(it, dict) else None
                        if b:
                            return str(b)
                        # intenta inferir de 'model' (prefijo alfabético)
                        m = str(it.get('model',''))
                        for cut in (" ", "-", "_"):
                            if cut in m:
                                return m.split(cut)[0]
                        # letras iniciales
                        prefix = ''.join([ch for ch in m if ch.isalpha()])
                        return prefix or "Otros"
                    items = [it for it in items if _get_brand(it) == sel_brand]
            except Exception:
                pass
            if q:
                def _match(it):
                    try:
                        return (q in str(it.get('model','')).lower()) or (q in str(it.get('n','')).lower())
                    except Exception:
                        return False
                items = [it for it in items if _match(it)]
            # Filtrar por favoritos si está activo
            try:
                if getattr(self, 'bearing_show_favs_only', False):
                    favs = getattr(self, 'bearing_favorites', {}) or {}
                    items = [it for it in items if bool(favs.get(str(it.get('model','')), False))]
            except Exception:
                pass
            for _, it in enumerate(items):
                model = str(it.get("model", ""))
                subtitle = f"n={it.get('n','?')}  d={it.get('d_mm','?')}mm  D={it.get('D_mm','?')}mm  θ={it.get('theta_deg',0)}°"
                # Star favorite icon
                is_fav = False
                try:
                    is_fav = bool(getattr(self, 'bearing_favorites', {}).get(model, False))
                except Exception:
                    is_fav = False
                star_icon = ft.Icons.STAR if is_fav else ft.Icons.STAR_BORDER_ROUNDED
                star_color = "#f1c40f" if is_fav else "#bdc3c7"
                tile = ft.ListTile(
                    leading=ft.IconButton(icon=star_icon, icon_color=star_color, tooltip="Favorito", on_click=lambda e, m=model: self._toggle_bearing_favorite(m)),
                    title=ft.Text(model),
                    subtitle=ft.Text(subtitle),
                    on_click=lambda e, m=model: self._select_bearing_by_model(m),
                    trailing=ft.IconButton(icon=ft.Icons.DELETE_FOREVER_ROUNDED, tooltip="Eliminar", on_click=lambda e, m=model: self._bearing_delete_model(m)),
                    dense=True,
                )
                self.bearing_list_view.controls.append(tile)
            if self.bearing_list_view.page:
                self.bearing_list_view.update()
        except Exception:
            pass

    def _bearing_brand_names(self) -> List[str]:
        try:
            brands = []
            for it in (self.bearing_db_items or []):
                b = None
                try:
                    b = it.get('brand')
                except Exception:
                    b = None
                if not b:
                    m = str(it.get('model',''))
                    for cut in (" ", "-", "_"):
                        if cut in m:
                            b = m.split(cut)[0]
                            break
                    if not b:
                        pref = ''.join([ch for ch in m if ch.isalpha()])
                        b = pref or "Otros"
                brands.append(str(b))
            uniq = sorted({b for b in brands if b})
            return ["Todos"] + uniq
        except Exception:
            return ["Todos"]

    def _rebuild_bearing_tabs(self):
        try:
            names = self._bearing_brand_names()
            if not getattr(self, 'bearing_tabs', None):
                self.bearing_tabs = ft.Tabs(tabs=[ft.Tab(text=n) for n in names], selected_index=0, on_change=self._on_bearing_tab_change)
            else:
                self.bearing_tabs.tabs = [ft.Tab(text=n) for n in names]
                self.bearing_tabs.selected_index = min(getattr(self.bearing_tabs, 'selected_index', 0) or 0, len(names)-1)
                if self.bearing_tabs.page:
                    self.bearing_tabs.update()
        except Exception:
            pass

    def _on_bearing_tab_change(self, e=None):
        try:
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _toggle_bearing_favs_filter(self):
        try:
            self.bearing_show_favs_only = bool(getattr(self, 'bearing_favs_only_cb', None).value) if getattr(self, 'bearing_favs_only_cb', None) else False
        except Exception:
            self.bearing_show_favs_only = False
        try:
            self.page.client_storage.set("bearing_favs_only", self.bearing_show_favs_only)
        except Exception:
            pass
        self._refresh_bearing_list_ui()

    # Favoritos de rodamientos
    def _bearing_favorites_path(self) -> str:
        try:
            return os.path.join(os.getcwd(), "bearing_favorites.json")
        except Exception:
            return "bearing_favorites.json"

    def _load_bearing_favorites(self) -> Dict[str, bool]:
        try:
            import json
            path = self._bearing_favorites_path()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_bearing_favorites(self):
        try:
            import json
            path = self._bearing_favorites_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.bearing_favorites or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _toggle_bearing_favorite(self, model: str):
        try:
            m = str(model or "")
            if not m:
                return
            cur = bool((self.bearing_favorites or {}).get(m, False))
            self.bearing_favorites[m] = not cur
            self._save_bearing_favorites()
        except Exception:
            pass
        try:
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _select_bearing_from_list(self, idx: int):
        try:
            if idx < 0 or idx >= len(self.bearing_db_items):
                return
            it = self.bearing_db_items[idx]
            self._bearing_sel_index = idx
            # Rellenar panel detalle del diálogo
            if getattr(self, "br_model_field_dlg", None):
                self.br_model_field_dlg.value = str(it.get("model", ""))
            if getattr(self, "br_n_field_dlg", None):
                self.br_n_field_dlg.value = str(it.get("n", ""))
            if getattr(self, "br_d_mm_field_dlg", None):
                self.br_d_mm_field_dlg.value = str(it.get("d_mm", ""))
            if getattr(self, "br_D_mm_field_dlg", None):
                self.br_D_mm_field_dlg.value = str(it.get("D_mm", ""))
            if getattr(self, "br_theta_deg_field_dlg", None):
                self.br_theta_deg_field_dlg.value = str(it.get("theta_deg", 0))
            try:
                self.br_model_field_dlg.update(); self.br_n_field_dlg.update(); self.br_d_mm_field_dlg.update(); self.br_D_mm_field_dlg.update(); self.br_theta_deg_field_dlg.update()
            except Exception:
                pass
        except Exception:
            pass

    def _select_bearing_by_model(self, model: str):
        try:
            target = str(model or "")
            if not target:
                return
            idx = None
            for i, it in enumerate(self.bearing_db_items or []):
                if str(it.get('model','')) == target:
                    idx = i
                    break
            if idx is None:
                return
            self._select_bearing_from_list(idx)
        except Exception:
            pass

    def _bearing_new_click(self, e=None):
        try:
            self._bearing_sel_index = None
            self.br_model_field_dlg.value = ""
            self.br_n_field_dlg.value = ""
            self.br_d_mm_field_dlg.value = ""
            self.br_D_mm_field_dlg.value = ""
            self.br_theta_deg_field_dlg.value = "0"
            for fld in (self.br_model_field_dlg, self.br_n_field_dlg, self.br_d_mm_field_dlg, self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg):
                try:
                    fld.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _bearing_save_click(self, e=None):
        try:
            model = str(getattr(self, "br_model_field_dlg", None).value or "").strip()
            if not model:
                return
            def _to_float(tf):
                try:
                    return float(tf.value) if tf and getattr(tf, 'value', '') else None
                except Exception:
                    return None
            def _to_int(tf):
                try:
                    return int(float(tf.value)) if tf and getattr(tf, 'value', '') else None
                except Exception:
                    return None
            item = {
                "model": model,
                "n": _to_int(getattr(self, "br_n_field_dlg", None)),
                "d_mm": _to_float(getattr(self, "br_d_mm_field_dlg", None)),
                "D_mm": _to_float(getattr(self, "br_D_mm_field_dlg", None)),
                "theta_deg": _to_float(getattr(self, "br_theta_deg_field_dlg", None)) or 0.0,
            }
            # actualizar si el modelo ya existe, si no agregar
            updated = False
            for i, it in enumerate(self.bearing_db_items or []):
                if str(it.get("model", "")) == model:
                    self.bearing_db_items[i] = item
                    updated = True
                    break
            if not updated:
                self.bearing_db_items.append(item)
            # guardar y refrescar UI
            self._save_bearing_db()
            # refrescar options del dropdown
            try:
                self.bearing_model_dd.options = self._bearing_db_model_options()
                self.bearing_model_dd.update()
            except Exception:
                pass
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _bearing_use_click(self, e=None):
        try:
            model = str(getattr(self, "br_model_field_dlg", None).value or "").strip()
            if model:
                try:
                    # Guardar modelo seleccionado en estado
                    self.selected_bearing_model = model
                    self.bearing_model_dd.value = model
                    self.bearing_model_dd.update()
                except Exception:
                    pass
            # Transferir a los campos principales de geometría
            for src, dst_name in (
                (getattr(self, "br_n_field_dlg", None), "br_n_field"),
                (getattr(self, "br_d_mm_field_dlg", None), "br_d_mm_field"),
                (getattr(self, "br_D_mm_field_dlg", None), "br_D_mm_field"),
                (getattr(self, "br_theta_deg_field_dlg", None), "br_theta_deg_field"),
            ):
                try:
                    getattr(self, dst_name).value = getattr(src, 'value', '')
                    getattr(self, dst_name).update()
                except Exception:
                    pass
            # cerrar diálogo
            if getattr(self, 'bearing_picker_dlg', None):
                self.bearing_picker_dlg.open = False
                if self.page:
                    self.page.update()
        except Exception:
            pass

    def _bearing_use_and_go(self, e=None):
        try:
            self._bearing_use_click()
            # Navegar a análisis
            try:
                if getattr(self, 'analysis_mode_dd', None):
                    self.analysis_mode_dd.value = 'assist'
                    try:
                        self.analysis_mode_dd.update()
                    except Exception:
                        pass
                    self._on_mode_change()
            except Exception:
                pass
            self._select_menu('analysis', force_rebuild=True)
        except Exception:
            try:
                self._select_menu('analysis', force_rebuild=True)
            except Exception:
                pass

    def _bearing_close_click(self, e=None):
        try:
            if getattr(self, 'bearing_picker_dlg', None):
                self.bearing_picker_dlg.open = False
                if self.page:
                    self.page.update()
        except Exception:
            pass

    def _bearing_delete_click(self, e=None):
        try:
            model = str(getattr(self, "br_model_field_dlg", None).value or "").strip()
            if not model:
                return
            before = len(self.bearing_db_items or [])
            self.bearing_db_items = [it for it in (self.bearing_db_items or []) if str(it.get('model','')) != model]
            after = len(self.bearing_db_items)
            if after < before:
                self._save_bearing_db()
                # Limpiar selección
                for fld in (self.br_model_field_dlg, self.br_n_field_dlg, self.br_d_mm_field_dlg, self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg):
                    try:
                        fld.value = "" if fld is not self.br_theta_deg_field_dlg else "0"
                        fld.update()
                    except Exception:
                        pass
                # Refrescar UI
                try:
                    self.bearing_model_dd.options = self._bearing_db_model_options()
                    self.bearing_model_dd.update()
                except Exception:
                    pass
                self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _bearing_delete_model(self, model: str):
        try:
            model = str(model or "").strip()
            if not model:
                return
            self.bearing_db_items = [it for it in (self.bearing_db_items or []) if str(it.get('model','')) != model]
            self._save_bearing_db()
            # refrescar UI y dropdown
            try:
                self.bearing_model_dd.options = self._bearing_db_model_options()
                self.bearing_model_dd.update()
            except Exception:
                pass
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    # ==== Favoritos de reportes ====
    def _favorites_path(self) -> str:
        try:
            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            return os.path.join(reports_dir, "favorites.json")
        except Exception:
            return "favorites.json"

    def _load_report_favorites(self) -> Dict[str, bool]:
        try:
            import json
            path = self._favorites_path()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_report_favorites(self):
        try:
            import json
            path = self._favorites_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.report_favorites or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _toggle_report_favorite(self, path: str):
        try:
            cur = bool((self.report_favorites or {}).get(path, False))
            self.report_favorites[path] = not cur
            self._save_report_favorites()
        except Exception:
            pass
        # Refrescar listado para actualizar icono
        try:
            self._refresh_report_list_scandir()
        except Exception:
            pass

    def _toggle_reports_fav_filter(self):
        try:
            self.report_show_favs_only = bool(getattr(self, 'report_favs_only_cb', None).value)
        except Exception:
            self.report_show_favs_only = False
        try:
            self.page.client_storage.set("report_favs_only", self.report_show_favs_only)
        except Exception:
            pass
        self._refresh_report_list_scandir()

    def _open_bearing_picker(self, e=None):
        try:
            # Crear controles del diálogo si no existen
            if not getattr(self, 'bearing_list_view', None):
                self.bearing_list_view = ft.ListView(expand=True, spacing=4, padding=4, height=400)
            # Panel detalle
            if not getattr(self, 'br_model_field_dlg', None):
                self.br_model_field_dlg = ft.TextField(label="Modelo", width=220)
                self.br_n_field_dlg = ft.TextField(label="# Elementos (n)", width=150)
                self.br_d_mm_field_dlg = ft.TextField(label="d (mm)", width=120)
                self.br_D_mm_field_dlg = ft.TextField(label="D (mm)", width=120)
                self.br_theta_deg_field_dlg = ft.TextField(label="Ángulo (°)", width=120, value="0")
            detail_col = ft.Column([
                ft.Text("Detalle del rodamiento", size=14, weight="bold"),
                self.br_model_field_dlg,
                ft.Row([self.br_n_field_dlg, self.br_d_mm_field_dlg], spacing=10),
                ft.Row([self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg], spacing=10),
                ft.Row([
                    ft.OutlinedButton("Nuevo", icon=ft.Icons.ADD_ROUNDED, on_click=self._bearing_new_click),
                    ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE_ROUNDED, on_click=self._bearing_save_click),
                    ft.OutlinedButton("Eliminar", icon=ft.Icons.DELETE_FOREVER_ROUNDED, on_click=self._bearing_delete_click),
                    ft.ElevatedButton("Usar", icon=ft.Icons.CHECK_CIRCLE_ROUNDED, on_click=self._bearing_use_click),
                ], spacing=10)
            ], spacing=8, width=400)
            list_col = ft.Column([
                ft.Text("Listado de rodamientos", size=14, weight="bold"),
                self.bearing_list_view,
            ], spacing=8, width=350)
            content = ft.Container(
                content=ft.Row([list_col, detail_col], spacing=20),
                padding=10,
                width=800,
            )
            self.bearing_picker_dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text("Rodamientos comunes"),
                content=content,
                actions=[ft.TextButton("Cerrar", on_click=self._bearing_close_click)],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            # Popular listado
            self._refresh_bearing_list_ui()
            # Abrir
            self.page.dialog = self.bearing_picker_dlg
            self.bearing_picker_dlg.open = True
            self.page.update()
        except Exception:
            pass

    # === Importar CSV de rodamientos ===
    def _bearing_open_csv_picker(self, e=None):
        try:
            # Preferir CSV
            try:
                self.bearing_file_picker.pick_files(allow_multiple=False, allowed_extensions=['csv'])
            except Exception:
                self.bearing_file_picker.pick_files(allow_multiple=False)
        except Exception:
            pass

    def _bearing_csv_pick_result(self, e):
        try:
            files = getattr(e, 'files', None)
            if not files:
                return
            path = getattr(files[0], 'path', None)
            if not path or not os.path.exists(path):
                return
            # Leer CSV
            try:
                df = pd.read_csv(path)
            except Exception:
                try:
                    df = pd.read_csv(path, encoding='latin-1')
                except Exception:
                    return
            # Normalizar columnas esperadas
            cols = {c.lower().strip(): c for c in df.columns}
            need = ['model','n','d_mm','D_mm','theta_deg']
            # admitir variantes comunes
            def _getcol(name):
                for k,v in cols.items():
                    if k == name.lower():
                        return v
                return None
            c_model = _getcol('model')
            c_n = _getcol('n')
            c_d = _getcol('d_mm') or _getcol('d')
            c_D = _getcol('D_mm') or _getcol('D')
            c_theta = _getcol('theta_deg') or _getcol('theta') or _getcol('angle')
            c_brand = _getcol('brand')
            if not c_model:
                return
            items = []
            for _, r in df.iterrows():
                try:
                    it = {
                        'model': str(r.get(c_model, '')).strip(),
                        'brand': (str(r.get(c_brand)).strip() if c_brand and pd.notna(r.get(c_brand)) else None),
                        'n': int(r.get(c_n)) if c_n and pd.notna(r.get(c_n)) else None,
                        'd_mm': float(r.get(c_d)) if c_d and pd.notna(r.get(c_d)) else None,
                        'D_mm': float(r.get(c_D)) if c_D and pd.notna(r.get(c_D)) else None,
                        'theta_deg': float(r.get(c_theta)) if c_theta and pd.notna(r.get(c_theta)) else 0.0,
                    }
                except Exception:
                    continue
                if it['model']:
                    items.append(it)
            if not items:
                return
            # Merge por modelo
            by_model = {it['model']: it for it in (self.bearing_db_items or [])}
            for it in items:
                by_model[it['model']] = it
            self.bearing_db_items = list(by_model.values())
            # Guardar y refrescar UI y dropdowns
            self._save_bearing_db()
            try:
                self.bearing_model_dd.options = self._bearing_db_model_options()
                self.bearing_model_dd.update()
            except Exception:
                pass
            # Recrear pestañas por marca
            try:
                self._rebuild_bearing_tabs()
            except Exception:
                pass
            self._refresh_bearing_list_ui()
            # Preseleccionar el primero para mostrar detalle
            try:
                self._select_bearing_by_model(items[0].get('model'))
            except Exception:
                pass
        except Exception:
            pass

    def _bearing_analyze_click(self, e=None):
        try:
            # Copiar al panel asistido principal
            self._bearing_use_click()
            # Calcular frecuencias teóricas
            self._compute_bearing_freqs_click()
            # Ir a Análisis
            self._select_menu('analysis', force_rebuild=True)
        except Exception:
            try:
                self._select_menu('analysis', force_rebuild=True)
            except Exception:
                pass

    # Helpers de configuración/acento
    def _get_bool_storage(self, key: str, default: bool) -> bool:
        try:
            v = self.page.client_storage.get(key)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes", "on")
        except Exception:
            pass
        return default

    # Helpers de lectura segura de campos
    def _fldf(self, fld):
        try:
            return float(fld.value) if fld and getattr(fld, 'value', '') else None
        except Exception:
            return None

    def _tfv(self, tf) -> str:
        try:
            return str(tf.value).strip() if tf and getattr(tf, 'value', '') != '' else ''
        except Exception:
            return ''

    def _accent_ui(self) -> str:
        try:
            a = str(self.accent).strip()
            return a if a else "#3498db"
        except Exception:
            return "#3498db"

    def _accent_hex(self) -> str:
        try:
            a = str(self.accent).strip()
            if a.startswith("#") and (len(a) == 7 or len(a) == 9):
                return a[:7]
        except Exception:
            pass
        return "#3498db"

    def _set_accent(self, hex_color: str):
        try:
            val = str(hex_color or "").strip()
            if not val:
                return
            # Normalizar a #RRGGBB
            if val.startswith("#") and len(val) in (4, 7):
                if len(val) == 4:  # #RGB -> #RRGGBB
                    r, g, b = val[1], val[2], val[3]
                    val = f"#{r}{r}{g}{g}{b}{b}"
            self.accent = val
            self.page.client_storage.set("accent", self.accent)
            # Aplicar cambios
            self._apply_theme()
            self._update_theme_for_all_components()
        except Exception:
            pass

    def _on_accent_swatch_click(self, e):
        try:
            hex_color = getattr(e.control, "data", None)
            if hex_color:
                self._set_accent(hex_color)
        except Exception:
            pass

    def _build_accent_palette(self):
        # Generar paleta: columnas = diferentes H (0..330, paso 30), filas = diferentes V
        cols = 12
        rows = 6
        hues = [i * (360 // cols) for i in range(cols)]
        vals = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

        swatches: List[ft.Control] = []
        for v in vals:
            for h in hues:
                r, g, b = colorsys.hsv_to_rgb(h / 360.0, 1.0, v)
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                hexc = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                swatches.append(
                    ft.Container(
                        width=24,
                        height=24,
                        bgcolor=hexc,
                        border_radius=6,
                        margin=2,
                        data=hexc,
                        on_click=self._on_accent_swatch_click,
                        tooltip=hexc,
                        border=ft.border.all(1, "#00000033"),
                    )
                )

        # Añadir escala de grises
        for i in range(12):
            g = int(255 * (i / 11))
            hexc = f"#{g:02x}{g:02x}{g:02x}"
            swatches.append(
                ft.Container(
                    width=24,
                    height=24,
                    bgcolor=hexc,
                    border_radius=6,
                    margin=2,
                    data=hexc,
                    on_click=self._on_accent_swatch_click,
                    tooltip=hexc,
                    border=ft.border.all(1, "#00000033"),
                )
            )

        # Compatibilidad: algunas versiones no tienen ft.Wrap; componemos filas fijas
        rows: List[ft.Control] = []
        row_size = 12
        for i in range(0, len(swatches), row_size):
            rows.append(ft.Row(controls=swatches[i : i + row_size], spacing=2))

        return ft.Column(
            controls=[
                ft.Text("Elige un color (gradiente)", size=12, color="#7f8c8d"),
                ft.Column(controls=rows, spacing=2),
            ]
        )

    def exportar_pdf(self, e=None):
        prev_style = plt.rcParams.copy()
        try:
            if self.current_df is None or getattr(self.current_df, 'empty', False):
                self._log("No hay datos para exportar")
                return

            time_col = getattr(self.time_dropdown, "value", None)
            fft_signal_col = getattr(self.fft_dropdown, "value", None)
            if not time_col or not fft_signal_col:
                self._log("Selecciona columnas de tiempo y señal antes de exportar.")
                return
            if time_col not in self.current_df.columns or fft_signal_col not in self.current_df.columns:
                self._log("Las columnas seleccionadas no existen en el DataFrame.")
                return

            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name = os.path.splitext(os.path.basename(self.uploaded_files[0]))[0] if getattr(self, "uploaded_files", None) else "sin_nombre"
            pdf_name = f"{timestamp}_{base_name}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_name)

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams["font.family"] = "DejaVu Sans"

            t = self.current_df[time_col].to_numpy()
            signal = self.current_df[fft_signal_col].to_numpy()

            try:
                start_t = float(self.start_time_field.value) if getattr(self.start_time_field, "value", None) else t[0]
                end_t = float(self.end_time_field.value) if getattr(self.end_time_field, "value", None) else t[-1]
            except Exception:
                start_t, end_t = t[0], t[-1]
            mask = (t >= start_t) & (t <= end_t)
            t_seg, sig_seg = t[mask], signal[mask]

            wf_pdf_notes = []
            wf_pdf_notes2 = []

            def compute_fft_dual(y, tv):
                N = len(y)
                if N < 2:
                    return None, None, None
                T = tv[1] - tv[0]
                yf = np.fft.fft(y)
                xf = np.fft.fftfreq(N, T)[:N // 2]
                mag_acc = 2.0 / N * np.abs(yf[0:N // 2])
                mag_vel = np.zeros_like(mag_acc)
                pos = xf > 0
                mag_vel[pos] = mag_acc[pos] / (2 * np.pi * xf[pos])
                mag_vel_mm = mag_vel * 1000.0
                return xf, mag_vel_mm, mag_vel

            def _safe_float(field, fallback):
                try:
                    raw = getattr(field, "value", "") if field else ""
                    return float(raw) if raw not in (None, "") else fallback
                except Exception:
                    return fallback


            xf, mag_vel_mm, mag_vel = compute_fft_dual(sig_seg, t_seg)
            rms_mm = float(np.sqrt(np.mean(mag_vel_mm**2))) if mag_vel_mm is not None else 0.0
            rms_ms = float(np.sqrt(np.mean(mag_vel**2))) if mag_vel is not None else 0.0
            severity_mm = self._classify_severity(rms_mm)

            if xf is not None:
                features_full = self._extract_features(t_seg, sig_seg, xf, mag_vel_mm)
            else:
                features_full = {"dom_freq": 0.0, "crest": 0.0, "rms_time_acc": 0.0, "peak_acc": 0.0, "pp_acc": 0.0,
                                 "e_low": 0.0, "e_mid": 0.0, "e_high": 0.0, "e_total": 1e-12, "r2x": 0.0, "r3x": 0.0}

            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_seg
            self._last_accseg = sig_seg
            findings_pdf = self._diagnose(features_full) if xf is not None else ["Sin espectro válido para diagnóstico."]

            # Unificar cálculo con analizador (RMS de velocidad correcto)
            try:
                rpm_val = None
                if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", ""):
                    rpm_val = float(self.rpm_hint_field.value)
            except Exception:
                rpm_val = None
            try:
                line_val = float(self.line_freq_dd.value) if getattr(self, "line_freq_dd", None) and getattr(self.line_freq_dd, "value", "") else None
            except Exception:
                line_val = None
            try:
                teeth_val = int(self.gear_teeth_field.value) if getattr(self, "gear_teeth_field", None) and getattr(self.gear_teeth_field, "value", "") else None
            except Exception:
                teeth_val = None
            # usando self._fldf para leer campos numéricos opcionales
            # Pre-decimación opcional basada en Máx FFT (Hz)
            try:
                _fmax_pre = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                _fmax_pre = None
            res = analyze_vibration(
                t_seg,
                sig_seg,
                rpm=rpm_val,
                line_freq_hz=line_val,
                bpfo_hz=self._fldf(getattr(self, 'bpfo_field', None)),
                bpfi_hz=self._fldf(getattr(self, 'bpfi_field', None)),
                bsf_hz=self._fldf(getattr(self, 'bsf_field', None)),
                ftf_hz=self._fldf(getattr(self, 'ftf_field', None)),
                gear_teeth=teeth_val,
                pre_decimate_to_fmax_hz=_fmax_pre,
                env_bp_lo_hz=self._fldf(getattr(self, 'env_bp_lo_field', None)),
                env_bp_hi_hz=self._fldf(getattr(self, 'env_bp_hi_field', None)),
            )
            xf = res['fft']['f_hz']
            mag_vel_mm = res['fft']['vel_spec_mm_s']
            rms_mm = res['severity']['rms_mm_s']
            severity_mm = res['severity']['label']
            features_full = self._extract_features(t_seg, sig_seg, xf, mag_vel_mm) if xf is not None else {
                "dom_freq": 0.0, "crest": 0.0, "rms_time_acc": 0.0, "peak_acc": 0.0, "pp_acc": 0.0,
                "e_low": 0.0, "e_mid": 0.0, "e_high": 0.0, "e_total": 1e-12, "r2x": 0.0, "r3x": 0.0
            }
            try:
                features_full["rms_vel_spec"] = float(rms_mm)
            except Exception:
                pass
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_seg
            self._last_accseg = sig_seg
            findings_pdf = res.get('diagnosis', []) if xf is not None else ["Sin espectro valido para diagnostico."]

            # Recalcular con analizador unificado (RMS de velocidad correcto)
            try:
                rpm_val = None
                if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", ""):
                    rpm_val = float(self.rpm_hint_field.value)
            except Exception:
                rpm_val = None
            try:
                line_val = float(self.line_freq_dd.value) if getattr(self, "line_freq_dd", None) and getattr(self.line_freq_dd, "value", "") else None
            except Exception:
                line_val = None
            try:
                teeth_val = int(self.gear_teeth_field.value) if getattr(self, "gear_teeth_field", None) and getattr(self.gear_teeth_field, "value", "") else None
            except Exception:
                teeth_val = None
            # usando self._fldf para leer campos numéricos opcionales
            try:
                _fmax_pre = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                _fmax_pre = None
            res = analyze_vibration(
                t_seg,
                sig_seg,
                rpm=rpm_val,
                line_freq_hz=line_val,
                bpfo_hz=self._fldf(getattr(self, 'bpfo_field', None)),
                bpfi_hz=self._fldf(getattr(self, 'bpfi_field', None)),
                bsf_hz=self._fldf(getattr(self, 'bsf_field', None)),
                ftf_hz=self._fldf(getattr(self, 'ftf_field', None)),
                gear_teeth=teeth_val,
                pre_decimate_to_fmax_hz=_fmax_pre,
                env_bp_lo_hz=self._fldf(getattr(self, 'env_bp_lo_field', None)),
                env_bp_hi_hz=self._fldf(getattr(self, 'env_bp_hi_field', None)),
            )
            xf = res['fft']['f_hz']
            mag_vel_mm = res['fft']['vel_spec_mm_s']
            rms_mm = res['severity']['rms_mm_s']
            severity_mm = res['severity']['label']
            features_full = self._extract_features(t_seg, sig_seg, xf, mag_vel_mm) if xf is not None else {
                "dom_freq": 0.0, "crest": 0.0, "rms_time_acc": 0.0, "peak_acc": 0.0, "pp_acc": 0.0,
                "e_low": 0.0, "e_mid": 0.0, "e_high": 0.0, "e_total": 1e-12, "r2x": 0.0, "r3x": 0.0
            }
            try:
                features_full["rms_vel_spec"] = float(rms_mm)
            except Exception:
                pass
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_seg
            self._last_accseg = sig_seg
            findings_pdf = res.get('diagnosis', [])

            tmp_imgs = []
            def save_plot(fig, *, tight=True):
                path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                bbox = "tight" if tight else None
                fig.savefig(path, dpi=150, bbox_inches=bbox, pad_inches=0.1)
                plt.close(fig)
                tmp_imgs.append(path)
                return path

            fig1, ax1 = plt.subplots(figsize=(8, 3))
            if len(t_seg) > 0:
                ax1.plot(t_seg, sig_seg)
            ax1.set_title(f"Señal {fft_signal_col} ({start_t:.2f}-{end_t:.2f}s)")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel("Aceleración [m/s²]")
            try:
                rms_acc = self._calculate_rms(sig_seg)
                ax1.text(0.02, 0.95, f"RMS acc: {rms_acc:.3e} m/s²", transform=ax1.transAxes, va="top")
            except Exception:
                pass

            # Ajustar etiqueta de eje Y segun unidad seleccionada
            try:
                ax1.set_ylabel(_ylabel)
            except Exception:
                pass

            # Anotar RMS conforme a la unidad seleccionada
            try:
                text_color = "white" if self.is_dark_mode else "black"
                ax1.text(0.02, 0.95, _rms_text, transform=ax1.transAxes, va="top", color=text_color)
            except Exception:
                pass
            img_time = save_plot(fig1)

            top_peaks = []
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            if xf is not None and mag_vel_mm is not None:
                zoom_range = getattr(self, "_fft_zoom_range", None)
                zmin = zmax = None
                if zoom_range and len(zoom_range) == 2 and zoom_range[1] > zoom_range[0]:
                    try:
                        zmin, zmax = float(zoom_range[0]), float(zoom_range[1])
                    except Exception:
                        zmin = zmax = None
                # Aplicar ocultamiento visual de bajas frecuencias en PDF segun configuracion
                try:
                    fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
                except Exception:
                    fc = 0.5
                try:
                    hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
                except Exception:
                    hide_lf = True
                mask_vis = np.ones_like(xf, dtype=bool)
                if hide_lf:
                    mask_vis &= xf >= max(0.0, fc)
                if zmin is not None:
                    mask_vis &= (xf >= zmin) & (xf <= zmax)
                xpdf = xf[mask_vis]
                ypdf = mag_vel_mm[mask_vis]
                if xpdf.size == 0:
                    xpdf = xf
                    ypdf = mag_vel_mm
                ax2.plot(xpdf, ypdf, color="#3498db", linewidth=1.6)
                # Marcar y recolectar Top-N picos para tabla
                try:
                    K = 5
                    min_freq = (max(0.5, fc) if hide_lf else 0.5)
                    mask = xf >= min_freq
                    if zmin is not None:
                        mask &= (xf >= zmin) & (xf <= zmax)
                    xv = xf[mask]
                    yv = mag_vel_mm[mask]
                    if len(yv) > 0:
                        k = min(K, len(yv))
                        idx = np.argpartition(yv, -k)[-k:]
                        idx = idx[np.argsort(yv[idx])[::-1]]
                        peak_f = xv[idx]
                        peak_a = yv[idx]
                        ax2.scatter(peak_f, peak_a, color="#e74c3c", s=20, zorder=5)
                        f1 = self._get_1x_hz(features_full.get("dom_freq", 0.0))
                        peak_points = []
                        peak_labels = []
                        for pf, pa in zip(peak_f, peak_a):
                            try:
                                pf_f = float(pf)
                                pa_f = float(pa)
                            except Exception:
                                continue
                            order = None
                            if f1 and f1 > 0:
                                try:
                                    order = pf_f / float(f1)
                                except Exception:
                                    order = None
                            top_peaks.append((pf_f, pa_f, order))
                            peak_points.append((pf_f, pa_f))
                            peak_labels.append(self._format_peak_label(pf_f, pa_f, order))
                        if peak_points:
                            self._place_annotations(ax2, peak_points, peak_labels, color="#e74c3c")
                except Exception:
                    pass
                # Lineas guia de frecuencias teoricas (modo asistido)
                try:
                    bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                    bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                    bsf  = self._fldf(getattr(self, 'bsf_field', None))
                    ftf  = self._fldf(getattr(self, 'ftf_field', None))
                    marks_raw = [
                        (bpfo, 'BPFO', '#1f77b4'),
                        (bpfi, 'BPFI', '#ff7f0e'),
                        (bsf,  'BSF',  '#2ca02c'),
                        (ftf,  'FTF',  '#9467bd'),
                    ]
                    visible_marks = []
                    for f0, label, col in marks_raw:
                        if not (f0 and f0 > 0):
                            continue
                        try:
                            f0_f = float(f0)
                        except Exception:
                            continue
                        if zmin is not None and (f0_f < zmin or f0_f > zmax):
                            continue
                        ax2.axvline(f0_f, color=col, linestyle='--', alpha=0.8, linewidth=1.2)
                        visible_marks.append((f0_f, label, col))
                    self._draw_frequency_markers(ax2, visible_marks, None if zmin is None else (zmin, zmax))
                except Exception:
                    pass
            ax2.set_title("FFT (Velocidad)")
            ax2.set_xlabel("Frecuencia (Hz)")
            ax2.set_ylabel("Velocidad [mm/s]")
            try:
                ax2_rpm = ax2.twiny()
                xmin, xmax = ax2.get_xlim()
                ax2_rpm.set_xlim(xmin * 60.0, xmax * 60.0)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass
            img_fft = save_plot(fig2)

            # Espectro de Envolvente (gráfica separada para PDF)
            img_env = None
            try:
                xf_env = res.get('envelope', {}).get('f_hz', None)
                env_amp = res.get('envelope', {}).get('amp', None)
                if xf_env is not None and env_amp is not None and len(xf_env) > 0:
                    # Filtros visuales como en FFT
                    try:
                        fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
                    except Exception:
                        fc = 0.5
                    try:
                        hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
                    except Exception:
                        hide_lf = True
                    try:
                        fmax_ui = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
                    except Exception:
                        fmax_ui = None
                    if hide_lf:
                        m_env = xf_env >= max(0.0, fc)
                    else:
                        m_env = np.ones_like(xf_env, dtype=bool)
                    if fmax_ui and fmax_ui > 0:
                        m_env = m_env & (xf_env <= fmax_ui)
                    if zmin is not None:
                        m_env = m_env & (xf_env >= zmin) & (xf_env <= zmax)
                    xenv = xf_env[m_env]
                    yenv = env_amp[m_env]
                    env_fig, env_ax = plt.subplots(figsize=(8, 3))
                    env_ax.plot(xenv, yenv, color="#e67e22", linewidth=1.6)
                    env_ax.set_title("Espectro de Envolvente")
                    env_ax.set_xlabel("Frecuencia (Hz)")
                    env_ax.set_ylabel("Amp [a.u.]")
                    # Líneas guía teóricas (si hay)
                    try:
                        bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                        bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                        bsf  = self._fldf(getattr(self, 'bsf_field', None))
                        ftf  = self._fldf(getattr(self, 'ftf_field', None))
                        marks = [
                            (bpfo, 'BPFO', '#1f77b4'),
                            (bpfi, 'BPFI', '#ff7f0e'),
                            (bsf,  'BSF',  '#2ca02c'),
                            (ftf,  'FTF',  '#9467bd'),
                        ]
                        ymax = env_ax.get_ylim()[1]
                        for f0, label, col in marks:
                            if f0 and f0 > 0:
                                env_ax.axvline(f0, color=col, linestyle='--', alpha=0.85, linewidth=1.2)
                                env_ax.text(f0, ymax*0.92, label, rotation=90, color=col, fontsize=7, va='top', ha='right')
                    except Exception:
                        pass
                    img_env = save_plot(env_fig)
            except Exception:
                img_env = None

            img_wf2 = None
            wf_pdf_notes2 = []
            if bool(getattr(self, "waterfall_enabled_cb", None) and getattr(self.waterfall_enabled_cb, "value", False)):
                window_s_pdf2 = max(_safe_float(getattr(self, "waterfall_window_field", None), self.waterfall_window_s), 0.05)
                step_s_pdf2 = max(_safe_float(getattr(self, "waterfall_step_field", None), self.waterfall_step_s), 0.01)
                fc_pdf2 = _safe_float(getattr(self, "lf_cutoff_field", None), 0.5)
                fmax_pdf2 = _safe_float(getattr(self, "hf_limit_field", None), None)
                hide_lf_pdf2 = bool(getattr(self, "hide_lf_cb", None) and getattr(self.hide_lf_cb, "value", True))
                wf_data_pdf2, wf_error_pdf2 = self._compute_waterfall_data(t_seg, sig_seg, window_s_pdf2, step_s_pdf2, fmax_pdf2, fc_pdf2, hide_lf_pdf2, self._fft_zoom_range)
                if wf_data_pdf2:
                    wf_fig_pdf2 = self._build_waterfall_figure(wf_data_pdf2, getattr(self, "waterfall_mode", "waterfall"))
                    if wf_fig_pdf2:
                        img_wf2 = save_plot(wf_fig_pdf2, tight=False)
                        wf_pdf_notes2 = self._waterfall_start_stop_notes(wf_data_pdf2)
                elif wf_error_pdf2:
                    self._log(f"No se pudo generar cascada para PDF: {wf_error_pdf2}")


            img_wf = None
            wf_pdf_notes = []
            if bool(getattr(self, "waterfall_enabled_cb", None) and getattr(self.waterfall_enabled_cb, "value", False)):
                window_s_pdf = max(_safe_float(getattr(self, "waterfall_window_field", None), self.waterfall_window_s), 0.05)
                step_s_pdf = max(_safe_float(getattr(self, "waterfall_step_field", None), self.waterfall_step_s), 0.01)
                fc_pdf = _safe_float(getattr(self, "lf_cutoff_field", None), 0.5)
                fmax_pdf = _safe_float(getattr(self, "hf_limit_field", None), None)
                hide_lf_pdf = bool(getattr(self, "hide_lf_cb", None) and getattr(self.hide_lf_cb, "value", True))
                wf_data_pdf, wf_error_pdf = self._compute_waterfall_data(t_seg, sig_seg, window_s_pdf, step_s_pdf, fmax_pdf, fc_pdf, hide_lf_pdf, self._fft_zoom_range)
                if wf_data_pdf:
                    wf_fig_pdf = self._build_waterfall_figure(wf_data_pdf, getattr(self, "waterfall_mode", "waterfall"))
                    if wf_fig_pdf:
                        img_wf = save_plot(wf_fig_pdf, tight=False)
                        wf_pdf_notes = self._waterfall_start_stop_notes(wf_data_pdf)
                elif wf_error_pdf:
                    self._log(f"No se pudo generar cascada para PDF: {wf_error_pdf}")



            if wf_pdf_notes:
                for note in wf_pdf_notes:
                    if note not in findings_pdf:
                        findings_pdf.append(note)

            aux_imgs = []
            aux_selected = []
            try:
                aux_selected = [
                    (cb.label, color_dd.value, style_dd.value)
                    for cb, color_dd, style_dd in getattr(self, "aux_controls", [])
                    if getattr(cb, "value", False)
                ]
            except Exception:
                aux_selected = []
            for col, color, style in aux_selected:
                if col in self.current_df.columns:
                    aux_fig, aux_ax = plt.subplots(figsize=(8, 2))
                    aux_ax.plot(self.current_df[time_col], self.current_df[col], color=color, linestyle=style, linewidth=2, label=col)
                    aux_ax.set_title(f"{col} vs Tiempo")
                    aux_ax.legend()
                    aux_ax.set_xlabel("Tiempo (s)")
                    aux_ax.set_ylabel(col)
                    aux_imgs.append(save_plot(aux_fig))


            if img_wf:
                tmp_imgs.append(img_wf)

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            pdf_image_width = _pdf_available_width(doc)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("title", parent=styles['Title'], textColor=colors.HexColor(self._accent_hex()))

            elements = []
            elements.append(Paragraph("Informe de Análisis de Vibraciones", title_style))
            elements.append(Spacer(1, 18))
            elements.append(Paragraph(f"Archivo analizado: {base_name}", styles['Normal']))
            elements.append(Paragraph(f"Periodo analizado: {start_t:.2f}s - {end_t:.2f}s", styles['Normal']))
            elements.append(Paragraph(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Paragraph(f"Aplicacion: V-Analyzer {APP_VERSION}", styles['Normal']))
            elements.append(Spacer(1, 18))

            cover_summary = [
                ["Indicador", "Valor"],
                ["RMS velocidad", f"{rms_mm:.3f} mm/s"],
                ["Clasificacion ISO", severity_mm],
                ["Frecuencia dominante", f"{features_full['dom_freq']:.2f} Hz"],
            ]
            tbl_cover = Table(cover_summary, colWidths=[200, 200])
            tbl_cover.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(tbl_cover)
            elements.append(PageBreak())

            # Nota filtro visual (export): obtiene estado actual de la UI
            try:
                _pdf_fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                _pdf_fc = 0.5
            try:
                _pdf_hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                _pdf_hide_lf = True
            _pdf_fft_filter_note = f"Filtro visual FFT: oculta < {_pdf_fc:.2f} Hz" if _pdf_hide_lf else "Filtro visual FFT: sin ocultar"

            elements.append(Paragraph("Resumen Ejecutivo", styles['Heading1']))
            exec_findings_all = findings_pdf[1:] if len(findings_pdf) > 1 else []
            exec_findings = self._select_main_findings(exec_findings_all)
            if wf_pdf_notes:
                for note in wf_pdf_notes:
                    if note not in exec_findings:
                        exec_findings.append(note)
            if not exec_findings:
                exec_findings = ["Sin anomalias evidentes segun reglas actuales."]
            elements.append(Paragraph(f"Clasificacion ISO: {severity_mm}", styles['Normal']))
            elements.append(Paragraph(f"RMS velocidad: {rms_mm:.3f} mm/s", styles['Normal']))
            elements.append(Paragraph(f"Frecuencia dominante: {features_full['dom_freq']:.2f} Hz", styles['Normal']))
            elements.append(Paragraph(_pdf_fft_filter_note, styles['Normal']))
            elements.append(Spacer(1, 8))
            # Semáforo de severidad (actual + otros atenuados)
            try:
                # Mapear label a índice
                def _sev_index(lbl: str) -> int:
                    s = (lbl or "").lower()
                    if "buena" in s:
                        return 0
                    if "satisfact" in s:
                        return 1
                    if "insatisfact" in s:
                        return 2
                    if "inaceptable" in s:
                        return 3
                    return -1
                cur_idx = _sev_index(severity_mm)
                # Colores base
                base = [
                    ("Buena", "#2ecc71"),
                    ("Satisfactoria", "#f1c40f"),
                    ("Insatisfactoria", "#e67e22"),
                    ("Inaceptable", "#e74c3c"),
                ]
                # Helper para aclarar colores
                def _lighten_hex(hx: str, f: float):
                    try:
                        hx = hx.lstrip('#')
                        r = int(hx[0:2], 16) / 255.0
                        g = int(hx[2:4], 16) / 255.0
                        b = int(hx[4:6], 16) / 255.0
                        r = r + (1.0 - r) * f
                        g = g + (1.0 - g) * f
                        b = b + (1.0 - b) * f
                        from reportlab.lib import colors as _c
                        return _c.Color(r, g, b)
                    except Exception:
                        return colors.lightgrey
                # Construir tabla de 2 filas: bloques de color + etiquetas
                cells = ["", "", "", ""]
                labels = [t for t, _ in base]
                sem_tbl = Table([cells, labels], colWidths=[90, 110, 130, 120])
                ts = []
                for i, (name, hx) in enumerate(base):
                    if i == cur_idx:
                        bg = colors.HexColor(hx)
                    else:
                        bg = _lighten_hex(hx, 0.65)
                    ts.append(("BACKGROUND", (i, 0), (i, 0), bg))
                    ts.append(("BOX", (i, 0), (i, 0), 0.5, colors.black))
                    ts.append(("ALIGN", (i, 0), (i, 0), "CENTER"))
                    ts.append(("ALIGN", (i, 1), (i, 1), "CENTER"))
                ts.append(("GRID", (0, 1), (-1, 1), 0.25, colors.grey))
                sem_tbl.setStyle(TableStyle(ts))
                elements.append(Paragraph("Semáforo de severidad", styles['Heading2']))
                elements.append(sem_tbl)
                elements.append(Spacer(1, 12))
            except Exception:
                pass
            # Omitir bloque duplicado de diagnostico para evitar repeticion
            # elements.append(Paragraph("Diagnostico:", styles['Heading2']))
            for item in []:
                elements.append(Paragraph(f"- {item}", styles['Normal']))

            # Explicacion y recomendaciones (PDF)
            # Explicación y recomendaciones (unificadas con la app)
            exp_lines_pdf2 = self._build_explanations(res, exec_findings)
            if wf_pdf_notes:
                for note in wf_pdf_notes:
                    msg = f"Arranque/parada: {note}"
                    if msg not in exp_lines_pdf2:
                        exp_lines_pdf2.append(msg)
            elements.append(Paragraph("Explicacion y recomendaciones", styles['Heading2']))
            for line in exp_lines_pdf2:
                elements.append(Paragraph(f"- {line}", styles['Normal']))

            

            elements.append(Paragraph("Reporte de Análisis de Vibraciones", title_style))
            elements.append(Paragraph(f"Archivo: {base_name}", styles['Normal']))
            elements.append(Paragraph(f"Periodo: {start_t:.2f}s - {end_t:.2f}s", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Top picos (FFT)
            if top_peaks:
                elements.append(Paragraph("Picos principales (FFT)", styles['Heading2']))
                peaks_data = [["Frecuencia (Hz)", "Amplitud (mm/s)", "Orden (X)"]]
                for pf, pa, order in top_peaks:
                    peaks_data.append([f"{pf:.2f}", f"{pa:.3f}", f"{order:.2f}" if order else "-"])
                tbl_peaks = Table(peaks_data, colWidths=[120, 140, 120])
                tbl_peaks.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]))
                elements.append(tbl_peaks)
                elements.append(Spacer(1, 12))

            data_summary = [
                ["Metrica", "Valor"],
                ["RMS (velocidad)", f"{rms_mm:.3f} mm/s"],
                ["Clasificacion ISO", severity_mm],
                ["Frecuencia dominante", f"{features_full['dom_freq']:.2f} Hz"],
            ]
            table_summary = Table(data_summary, colWidths=[200, 200])
            table_summary.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(table_summary)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Senal temporal", styles['Heading2']))
            elements.append(_pdf_image(img_time, pdf_image_width, 320))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph("Espectro FFT", styles['Heading2']))
            elements.append(_pdf_image(img_fft, pdf_image_width, 320))
            elements.append(Spacer(1, 8))
            if img_wf:
                elements.append(CondPageBreak(360))
                elements.append(Paragraph("Analisis de arranque/parada (3D)", styles['Heading2']))
                elements.append(_pdf_image(img_wf, pdf_image_width, 360))
                elements.append(Spacer(1, 8))
            if img_env:
                elements.append(CondPageBreak(320))
                elements.append(Paragraph("Espectro de envolvente", styles['Heading2']))
                elements.append(_pdf_image(img_env, pdf_image_width, 320))
                elements.append(Spacer(1, 8))

            if aux_imgs:
                elements.append(Paragraph("Variables auxiliares", styles['Heading2']))
                for img in aux_imgs:
                    elements.append(Image(img, width=400, height=120))
                aux_data = [["Variable", "Promedio", "Mínimo", "Máximo"]]
                for col, _, _ in aux_selected:
                    if col in self.current_df.columns:
                        vals = self.current_df[col].dropna().to_numpy()
                        if len(vals) > 0:
                            aux_data.append([col, f"{np.mean(vals):.2f}", f"{np.min(vals):.2f}", f"{np.max(vals):.2f}"])
                if len(aux_data) > 1:
                    aux_table = Table(aux_data, colWidths=[150, 100, 100, 100])
                    aux_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
                    ]))
                    elements.append(aux_table)

            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Diagnóstico", styles['Heading2']))
            elements.append(Paragraph(f"El valor RMS calculado es {rms_mm:.3f} mm/s, lo cual corresponde a: {severity_mm}.", styles['Normal']))
            for item in findings_pdf:
                elements.append(Paragraph(f"- {item}", styles['Normal']))

            # Propiedades del equipo (al final)
            try:
                def _tfv(tf):
                    try:
                        return str(tf.value).strip() if tf and getattr(tf, 'value', '') != '' else ''
                    except Exception:
                        return ''
                props = []
                # Rodamiento (modelo y geometría)
                try:
                    model = getattr(self, 'bearing_model_dd', None).value if getattr(self, 'bearing_model_dd', None) else ''
                except Exception:
                    model = ''
                if model:
                    props.append(["Rodamiento (modelo)", model])
                props.append(["Elementos (n)", _tfv(getattr(self, 'br_n_field', None))])
                props.append(["d (mm)", _tfv(getattr(self, 'br_d_mm_field', None))])
                props.append(["D (mm)", _tfv(getattr(self, 'br_D_mm_field', None))])
                props.append(["Ángulo (°)", _tfv(getattr(self, 'br_theta_deg_field', None))])
                # Máquina
                props.append(["RPM", _tfv(getattr(self, 'rpm_hint_field', None))])
                props.append(["Línea (Hz)", (getattr(self, 'line_freq_dd', None).value if getattr(self, 'line_freq_dd', None) else '')])
                props.append(["Dientes engrane", _tfv(getattr(self, 'gear_teeth_field', None))])
                # Frecuencias teóricas
                props.append(["BPFO (Hz)", _tfv(getattr(self, 'bpfo_field', None))])
                props.append(["BPFI (Hz)", _tfv(getattr(self, 'bpfi_field', None))])
                props.append(["BSF (Hz)", _tfv(getattr(self, 'bsf_field', None))])
                props.append(["FTF (Hz)", _tfv(getattr(self, 'ftf_field', None))])
                # Sensor
                try:
                    sens_type = getattr(self, 'sens_unit_dd', None).value if getattr(self, 'sens_unit_dd', None) else ''
                except Exception:
                    sens_type = ''
                if sens_type:
                    props.append(["Sensor", sens_type])
                props.append(["Sensibilidad", _tfv(getattr(self, 'sensor_sens_field', None))])
                props.append(["Ganancia (V/V)", _tfv(getattr(self, 'gain_field', None))])

                props = [[k, v] for k, v in props if str(v) != '']
                if props:
                    elements.append(Spacer(1, 12))
                    elements.append(Paragraph("Propiedades del equipo", styles['Heading2']))
                    tbl_props = Table([["Propiedad", "Valor"]] + props, colWidths=[200, 200])
                    tbl_props.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ]))
                    elements.append(tbl_props)
            except Exception:
                pass

            doc.build(elements)

            if not hasattr(self, "generated_reports"):
                self.generated_reports = []
            self.generated_reports.append(pdf_path)

            self._log(f"Reporte exportado: {pdf_path}")
            self.page.snack_bar = ft.SnackBar(content=ft.Text(f"✅ Reporte PDF generado: {pdf_name}"), action="OK")
            self.page.snack_bar.open = True
            self.page.update()
            return pdf_path
        except Exception as ex:
            self._log(f"Error exportando PDF: {ex}")
        finally:
            try:
                plt.rcParams.update(prev_style)
            except Exception:
                pass



    def _build_menu(self):

        def mb(icon, tip, key):

            b = MenuButton(icon, tip, self._on_menu_click, key, self.is_dark_mode)
            try:
                b.accent = self.accent
            except Exception:
                pass

            # Asegurar etiqueta de eje Y acorde a unidad seleccionada
            try:
                ax1.set_ylabel(_ylabel)
            except Exception:
                pass
            try:
                ax1.ticklabel_format(style="sci", axis="y", scilimits=(-2, 3))
            except Exception:
                pass

            self.menu_buttons[key] = b

            return b



        return ft.Container(

            width=80,  # Ancho fijo reducido

            expand=0,  # Sin expansión

            border_radius=0,

            bgcolor="#16213e" if self.is_dark_mode else "#ffffff",

            padding=ft.padding.only(

                left=5,

                right=5,

                top=20,

                bottom=20

            ),

            shadow=ft.BoxShadow(

                spread_radius=1,

                blur_radius=10,

                color=ft.Colors.with_opacity(0.3, "black"),

                offset=ft.Offset(2, 0),

            ),

            animate=ft.Animation(300, "easeInOut"),

            content=ft.Column(

                horizontal_alignment=ft.CrossAxisAlignment.CENTER,

                controls=[

                    ft.Container(

                        content=(
                            setattr(self, "menu_logo_icon", ft.Icon(ft.Icons.VIBRATION_ROUNDED, size=40, color=self._accent_ui()))
                            or self.menu_logo_icon
                        ),

                        padding=ft.padding.only(bottom=10),

                        visible=self.is_menu_expanded,

                        animate=ft.Animation(300, "easeInOut"),

                    ),

                    ft.Container(

                        content=(
                            setattr(self, "menu_logo_text", ft.Text(
                                "V-Analyzer",
                                size=12,
                                weight="bold",
                                color=self._accent_ui(),
                            ))
                            or self.menu_logo_text
                        ),

                        visible=self.is_menu_expanded,

                        animate=ft.Animation(300, "easeInOut"),

                    ),

                    ft.Divider(

                        height=20,

                        color=ft.Colors.with_opacity(0.2, "white"),

                        visible=self.is_menu_expanded

                    ),

                    ft.Column(expand=True, controls=[]),

                    *[mb(icon, tip, key) for icon, tip, key in [

                        (ft.Icons.HOME_ROUNDED, "Inicio", "welcome"),

                        (ft.Icons.FOLDER_OPEN_ROUNDED, "Archivos", "files"),

                        (ft.Icons.INSIGHTS_ROUNDED, "Análisis", "analysis"),

                        (ft.Icons.ASSESSMENT_ROUNDED, "Reportes", "reports"),

                        (ft.Icons.SETTINGS_ROUNDED, "Configuración", "settings"),

                    ]],

                    ft.Divider(height=20, color="transparent"),

                    ft.Container(

                        content=ft.Text(

                            APP_VERSION,

                            size=9,

                            color="#7f8c8d",

                            text_align="center",

                        ),

                        visible=self.is_menu_expanded,

                        animate=ft.Animation(300, "easeInOut"),

                    ),

                ],

            ),

        )



    def _toggle_menu(self, e):

        self.is_menu_expanded = not self.is_menu_expanded

        

        # Actualizar icono del botón

        e.control.icon = (

            ft.Icons.CHEVRON_LEFT_ROUNDED 

            if self.is_menu_expanded 

            else ft.Icons.CHEVRON_RIGHT_ROUNDED

        )

        

        # Actualizar el menú

        self.menu.width = 100 if self.is_menu_expanded else 80

        self.menu.padding = ft.padding.only(

            left=15 if self.is_menu_expanded else 10,

            right=15 if self.is_menu_expanded else 10,

            top=20,

            bottom=20

        )

        

        # Actualizar visibilidad de elementos

        for control in self.menu.content.controls:

            if isinstance(control, ft.Container) or isinstance(control, ft.Text) or isinstance(control, ft.Divider):

                if not isinstance(control, MenuButton):

                    control.visible = self.is_menu_expanded

    

        # Actualizar el menú

        self.menu.update()



    def _build_control_panel(self):

        self.tabs = ft.Tabs(

            selected_index=0,

            tabs=[

                ft.Tab(text="Acciones", icon=ft.Icons.TOUCH_APP_ROUNDED),

                ft.Tab(text="Ayuda", icon=ft.Icons.HELP_OUTLINE_ROUNDED),

                ft.Tab(text="Registro", icon=ft.Icons.HISTORY_ROUNDED)

            ],

            expand=1,

            on_change=self._on_tab_change,

            animation_duration=300,

        )

        

        # Crear botón de colapsar panel

        toggle_button = ft.IconButton(

            icon=ft.Icons.CHEVRON_LEFT_ROUNDED,

            icon_size=20,

            tooltip="Colapsar panel",

            on_click=self._toggle_panel,

        )



        panel_header = ft.Row(

            controls=[

                ft.Text("Panel de Control", size=18, weight="bold"),

                toggle_button

            ],

            alignment="space_between"

        )



        # Resto del contenido del panel

        upload_btn = ft.ElevatedButton(

            "Cargar Archivo",

            icon=ft.Icons.UPLOAD_FILE_ROUNDED,

            on_click=self._pick_files,

            style=ft.ButtonStyle(

                bgcolor=self._accent_ui(),

                color="white",

                shape=ft.RoundedRectangleBorder(radius=10),

            ),

            width=200,

            height=45,

        )

        try:
            self.btn_upload = upload_btn
        except Exception:
            pass

        self.quick_actions = ft.Column(

            spacing=15,

            controls=[

                upload_btn,

                ft.Row(

                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,

                    controls=[

                        ft.IconButton(

                            icon=ft.Icons.DARK_MODE_ROUNDED if not self.is_dark_mode else ft.Icons.LIGHT_MODE_ROUNDED,

                            tooltip="Cambiar tema",

                            on_click=self._toggle_theme,

                            icon_size=24,

                        ),

                        ft.IconButton(

                            icon=ft.Icons.ACCESS_TIME_ROUNDED,

                            tooltip="Formato hora",

                            on_click=self._toggle_clock_format,

                            icon_size=24,

                        ),

                    ]

                ),

                (setattr(self, "clock_card", ft.Container(
                    content=self.clock_text,
                    bgcolor=ft.Colors.with_opacity(0.1, self._accent_ui()),
                    padding=10,
                    border_radius=10,
                    alignment=ft.alignment.center,
                )) or self.clock_card),

            ]

        )

        

        self.log_panel = ft.ListView(expand=1, auto_scroll=False, spacing=2)

        self.help_panel = ft.Column(

            scroll="auto",

            spacing=10,

            controls=[

                ft.Container(

                    content=ft.Text("📋 Ayuda contextual", size=16, weight="bold"),

                    padding=ft.padding.only(bottom=10)

                ),

                ft.Text("Información de ayuda aparecerá aquí según la sección actual.", size=13)

            ]

        )

        

        self.tab_content = ft.Container(content=self.quick_actions, expand=True, padding=10)

        

        return ft.Container(

            width=350 if self.is_panel_expanded else 80,

            padding=20 if self.is_panel_expanded else 10,

            border_radius=0,

            bgcolor="#16213e" if self.is_dark_mode else "#ffffff",

            shadow=ft.BoxShadow(

                spread_radius=1,

                blur_radius=10,

                color=ft.Colors.with_opacity(0.3, "black"),

                offset=ft.Offset(-2, 0),

            ),

            animate=ft.Animation(300, "easeInOut"),  # Fixed animation syntax

            content=ft.Column(

                expand=True,

                controls=[

                    panel_header,

                    ft.Divider(color=ft.Colors.with_opacity(0.2, "white")),

                    ft.Container(

                        content=ft.Column(

                            controls=[

                                self.tabs,

                                ft.Divider(color=ft.Colors.with_opacity(0.2, "white")),

                                self.tab_content

                            ]

                        ),

                        visible=self.is_panel_expanded

                    )

                ]

            ),

        )



    def _build_welcome_view(self):

        return ft.Column(

            controls=[

                ft.Container(height=50),

                ft.Icon(ft.Icons.MONITOR_HEART_ROUNDED, size=100, color=self._accent_ui()),

                ft.Container(height=20),

                ft.Text(

                    "Sistema de Diagnóstico Predictivo",

                    size=32,

                    weight="bold",

                    text_align="center"

                ),

                ft.Text(

                    "Análisis de Vibraciones Mecánicas mediante FFT y Machine Learning",

                    size=16,

                    color="#7f8c8d",

                    text_align="center"

                ),

                ft.Container(height=40),

                ft.Row(

                    alignment="center",

                    controls=[

                        ft.ElevatedButton(

                            "Comenzar Análisis",

                            icon=ft.Icons.PLAY_ARROW_ROUNDED,

                            on_click=self._pick_files,

                            style=ft.ButtonStyle(

                                bgcolor=self._accent_ui(),

                                color="white",

                                shape=ft.RoundedRectangleBorder(radius=10),

                                padding=20,

                            ),

                            height=50,

                        ),

                        ft.OutlinedButton(

                            "Ver Documentación",

                            icon=ft.Icons.DESCRIPTION_ROUNDED,

                            on_click=lambda _: self.page.launch_url("https://drive.google.com/file/d/1UqlL1s7jGTq3A38UV2r6AE2eVb915w41/view?usp=sharing")

,

                            style=ft.ButtonStyle(

                                shape=ft.RoundedRectangleBorder(radius=10),

                                padding=20,

                            ),

                            height=50,

                        ),

                    ]

                ),

                ft.Container(height=40),

                ft.Container(

                    content=ft.Column(

                        horizontal_alignment="center",

                        controls=[

                            ft.Text("Características del Sistema", size=18, weight="bold"),

                            ft.Container(height=10),

                            ft.Row(

                                alignment="center",

                                spacing=30,

                                controls=[

                                    self._create_feature_card("FFT", "Transformada Rápida de Fourier", ft.Icons.TRANSFORM_ROUNDED),

                                    self._create_feature_card("ML", "Machine Learning Predictivo", ft.Icons.PSYCHOLOGY_ROUNDED),

                                    self._create_feature_card("ISO", "Normas ISO 20816-3", ft.Icons.VERIFIED_ROUNDED),

                                ]

                            )

                        ]

                    ),

                    padding=20,

                )
            
            
            

            ],

            horizontal_alignment="center",

            alignment="center",

            expand=True,

            

        )



    def _create_feature_card(self, title, subtitle, icon):

        return ft.Container(

            content=ft.Column(

                horizontal_alignment="center",

                spacing=5,

                controls=[

                    ft.Icon(icon, size=40, color=self._accent_ui()),

                    ft.Text(title, size=16, weight="bold"),

                    ft.Text(subtitle, size=12, color="#7f8c8d", text_align="center"),

                ]

            ),

            padding=15,

            border_radius=10,

            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

            width=150,

            height=120,

        )



    def _build_files_view(self):

        self._refresh_files_list()

        return ft.Column(

            controls=[

                ft.Container(

                    content=ft.Column(

                        controls=[

                            ft.Row(

                                alignment="space_between",

                                controls=[

                                    ft.Text("Gestión de Archivos", size=24, weight="bold"),

                                    ft.ElevatedButton(

                                        "Agregar Archivo",

                                        icon=ft.Icons.ADD_ROUNDED,

                                        on_click=self._pick_files,

                                        style=ft.ButtonStyle(

                                            bgcolor=self._accent_ui(),

                                            color="white",

                                            shape=ft.RoundedRectangleBorder(radius=8),

                                        ),

                                    ),

                                ]

                            ),

                            ft.Row([
                                ft.Text(f"Total de archivos: {len(self.uploaded_files)}", size=14, color="#7f8c8d"),
                                setattr(self, 'data_favs_only_cb', ft.Checkbox(label="Mostrar favoritos", value=bool(self.data_show_favs_only), on_change=lambda e: self._toggle_data_favs_filter())) or self.data_favs_only_cb,
                            ], alignment="spaceBetween"),

                            # Buscador de archivos en gestor
                            (setattr(self, 'data_search', ft.TextField(
                                hint_text="Buscar por nombre...",
                                expand=True,
                                on_change=lambda e: self._refresh_files_list(),
                                dense=True,
                            )) or self.data_search),

                        ]

                    ),

                    padding=ft.padding.only(bottom=20)

                ),

                ft.Container(

                    content=self.files_list_view,

                    border=ft.border.all(1, ft.Colors.with_opacity(0.2, "white")),

                    border_radius=10,

                    expand=True,

                    padding=5,

                )

            ],

            expand=True

        )



    def _export_pdf(self, e=None):
        """

        Exporta reporte PDF extendido con diagnóstico automático:

        - Señal principal (tiempo + FFT)

        - Segmento seleccionado

        - Variables auxiliares

        - Tablas y diagnóstico (reglas baseline)

        """
        # Legacy wrapper: delega al exportador unificado
        try:
            return self.exportar_pdf(e)
        except Exception:
            pass

        try:

            if self.current_df is None:

                self._log("No hay datos para exportar")

                return



            reports_dir = os.path.join(os.getcwd(), "reports")

            os.makedirs(reports_dir, exist_ok=True)



            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            base_name = os.path.splitext(os.path.basename(self.uploaded_files[0]))[0]

            pdf_name = f"{timestamp}_{base_name}.pdf"

            pdf_path = os.path.join(reports_dir, pdf_name)



            # --- Estilo claro solo para PDF ---

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams["font.family"] = "DejaVu Sans"



            # Preparar análisis

            time_col = self.time_dropdown.value

            fft_signal_col = self.fft_dropdown.value

            t = self.current_df[time_col].to_numpy()

            signal = self.current_df[fft_signal_col].to_numpy()



            # Periodo seleccionado

            try:

                start_t = float(self.start_time_field.value) if self.start_time_field.value else t[0]

                end_t = float(self.end_time_field.value) if self.end_time_field.value else t[-1]

            except:

                start_t, end_t = t[0], t[-1]

            mask = (t >= start_t) & (t <= end_t)

            t_seg, sig_seg = t[mask], signal[mask]



            # FFT -> velocidad (mm/s y m/s)
            def compute_fft_dual(y, tv):
                N = len(y)
                if N < 2:
                    return None, None, None
                T = tv[1] - tv[0]
                yf = np.fft.fft(y)
                xf = np.fft.fftfreq(N, T)[:N // 2]
                mag_acc = 2.0 / N * np.abs(yf[0:N // 2])
                # m/s
                mag_vel = np.zeros_like(mag_acc)
                pos = xf > 0
                mag_vel[pos] = mag_acc[pos] / (2 * np.pi * xf[pos])
                # mm/s
                mag_vel_mm = mag_vel * 1000.0
                return xf, mag_vel_mm, mag_vel

            xf, mag_vel_mm, mag_vel = compute_fft_dual(sig_seg, t_seg)
            rms_mm = float(np.sqrt(np.mean(mag_vel_mm**2))) if mag_vel_mm is not None else 0.0
            rms_ms = float(np.sqrt(np.mean(mag_vel**2))) if mag_vel is not None else 0.0
            severity_mm = self._classify_severity(rms_mm)
            severity_label_ms, severity_color_ms = self._classify_severity_ms(rms_ms)

            # --- Features + diagnóstico para el PDF (usa mm/s) ---
            features_full = self._extract_features(t_seg, sig_seg, xf, mag_vel_mm)
            # Guardar última FFT/segmento para diagnóstico avanzado (PDF)
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_seg
            self._last_accseg = sig_seg
            findings_pdf = res.get('diagnosis', [])



            # Guardar gráficas como imágenes

            tmp_imgs = []



            def save_plot(fig, *, tight=True):
                path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                bbox = "tight" if tight else None
                fig.savefig(path, dpi=150, bbox_inches=bbox, pad_inches=0.1)
                plt.close(fig)
                tmp_imgs.append(path)
                return path



            # Señal principal
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            if len(t_seg) > 0:
                ax1.plot(t_seg, sig_seg, color="blue")
            ax1.set_title(f"Señal {fft_signal_col} ({start_t:.2f}-{end_t:.2f}s)")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel("Aceleración [m/s²]")
            # Anotar RMS aceleración
            try:
                rms_acc = self._calculate_rms(sig_seg)
                ax1.text(0.02, 0.95, f"RMS acc: {rms_acc:.3e} m/s²", transform=ax1.transAxes, va="top")
            except Exception:
                pass
            img_time = save_plot(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            if xf is not None and mag_vel_mm is not None:
                ax2.plot(xf, mag_vel_mm, color="red")
            ax2.set_title("FFT (Velocidad)")
            ax2.set_xlabel("Frecuencia (Hz)")
            ax2.set_ylabel("Velocidad [mm/s]")
            # Anotación RMS (m/s) y eje superior en RPM
            try:
                text_color = "white" if self.is_dark_mode else "black"
                ax2.text(0.02, 0.95, f"RMS vel: {rms_mm:.3f} mm/s", transform=ax2.transAxes,
                         va="top", color=text_color)
                ax2_rpm = ax2.twiny()
                ax2_rpm.set_xlim(ax2.get_xlim()[0]*60, ax2.get_xlim()[1]*60)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass
            # Eje superior en RPM
            try:
                ax2_rpm = ax2.twiny()
                ax2_rpm.set_xlim(ax2.get_xlim()[0] * 60, ax2.get_xlim()[1] * 60)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass
            img_fft = save_plot(fig2)


            # Variables auxiliares (si las hay marcadas)

            aux_selected = [(cb.label, color_dd.value, style_dd.value)

                            for cb, color_dd, style_dd in self.aux_controls if cb.value]



            aux_imgs = []

            for col, color, style in aux_selected:

                aux_fig, aux_ax = plt.subplots(figsize=(8, 2))

                aux_ax.plot(self.current_df[time_col], self.current_df[col],

                            color=color, linestyle=style, linewidth=2, label=col)

                aux_ax.set_title(f"{col} vs Tiempo")

                aux_ax.legend()

                aux_ax.set_xlabel("Tiempo (s)")

                aux_ax.set_ylabel(col)

                aux_imgs.append(save_plot(aux_fig))



            # Crear PDF


            if img_wf2:
                tmp_imgs.append(img_wf2)

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            pdf_image_width = _pdf_available_width(doc)
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle("title", parent=styles['Title'], textColor=colors.HexColor(self._accent_hex()))

            elements = []
            # Cover page
            elements.append(Paragraph("Informe de Análisis de Vibraciones", title_style))
            elements.append(Spacer(1, 18))
            elements.append(Paragraph(f"Archivo analizado: {base_name}", styles['Normal']))
            elements.append(Paragraph(f"Periodo analizado: {start_t:.2f}s - {end_t:.2f}s", styles['Normal']))
            elements.append(Paragraph(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Paragraph(f"Aplicacion: V-Analyzer {APP_VERSION}", styles['Normal']))
            elements.append(Spacer(1, 18))
            cover_summary = [
                ["Indicador", "Valor"],
                ["RMS velocidad", f"{rms_mm:.3f} mm/s"],
                ["Clasificacion ISO", severity_mm],
                ["Frecuencia dominante", f"{features_full['dom_freq']:.2f} Hz"],
            ]
            tbl_cover = Table(cover_summary, colWidths=[200, 200])
            tbl_cover.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(tbl_cover)
            elements.append(PageBreak())
            # Resumen Ejecutivo
            # Nota filtro visual (export): obtiene estado actual de la UI
            try:
                _pdf_fc2 = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                _pdf_fc2 = 0.5
            try:
                _pdf_hide_lf2 = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                _pdf_hide_lf2 = True
            _pdf_fft_filter_note2 = f"Filtro visual FFT: oculta < {_pdf_fc2:.2f} Hz" if _pdf_hide_lf2 else "Filtro visual FFT: sin ocultar"

            elements.append(Paragraph("Resumen Ejecutivo", styles['Heading1']))
            exec_findings_all2 = findings_pdf[1:] if len(findings_pdf) > 1 else []
            exec_findings = self._select_main_findings(exec_findings_all2)
            if wf_pdf_notes:
                for note in wf_pdf_notes:
                    if note not in exec_findings:
                        exec_findings.append(note)
            if not exec_findings:
                exec_findings = ["Sin anomalias evidentes segun reglas actuales."]
            elements.append(Paragraph(f"Clasificacion ISO: {severity_mm}", styles['Normal']))
            elements.append(Paragraph(f"RMS velocidad: {rms_mm:.3f} mm/s", styles['Normal']))
            elements.append(Paragraph(f"Frecuencia dominante: {features_full['dom_freq']:.2f} Hz", styles['Normal']))
            elements.append(Paragraph(_pdf_fft_filter_note2, styles['Normal']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph("Diagnóstico:", styles['Heading2']))
            for item in exec_findings:
                elements.append(Paragraph(f"- {item}", styles['Normal']))


            elements.append(Paragraph("📊 Reporte de Análisis de Vibraciones", title_style))
            elements.append(Paragraph(f"Archivo: {base_name}", styles['Normal']))
            elements.append(Paragraph(f"Periodo: {start_t:.2f}s – {end_t:.2f}s", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Resumen de métricas espectrales
            # (resumen actualizado)
            data_summary = [
                ["Metrica", "Valor"],
                ["RMS (velocidad)", f"{rms_mm:.3f} mm/s"],
                ["Clasificacion ISO", severity_mm],
                ["Frecuencia dominante", f"{features_full['dom_freq']:.2f} Hz"],
                ["Crest factor (aceleracion)", f"{features_full['crest']:.2f}"]
            ]
            table_summary = Table(data_summary, colWidths=[200, 200])
            elements.append(Paragraph("Metricas detalladas", styles['Heading2']))
            det = [
                ["Metrica", "Valor"],
                ["RMS aceleracion (m/s^2)", f"{features_full['rms_time_acc']:.3e}"],
                ["Pico aceleracion (m/s^2)", f"{features_full['peak_acc']:.3e}"],
                ["Pico a pico (m/s^2)", f"{features_full['pp_acc']:.3e}"],
                ["Crest factor", f"{features_full['crest']:.2f}"],
                ["Energia baja (0-30 Hz)", f"{(100.0*features_full['e_low']/max(features_full['e_total'],1e-12)):.1f}%"],
                ["Energia media (30-120 Hz)", f"{(100.0*features_full['e_mid']/max(features_full['e_total'],1e-12)):.1f}%"],
                ["Energia alta (>=120 Hz)", f"{(100.0*features_full['e_high']/max(features_full['e_total'],1e-12)):.1f}%"],
                ["Relacion 2X", f"{features_full['r2x']:.2f}"],
                ["Relacion 3X", f"{features_full['r3x']:.2f}"],
            ]
            tbl_det = Table(det, colWidths=[220, 180])
            tbl_det.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(tbl_det)
            elements.append(Spacer(1, 12))
            table_summary.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(table_summary)
            elements.append(Spacer(1, 12))



            if wf_pdf_notes:
                for note in wf_pdf_notes:
                    if note not in findings:
                        findings.append(note)
                for note in wf_pdf_notes:
                    expl = f"Arranque/parada: {note}"
                    if expl not in exp_lines:
                        exp_lines.append(expl)
            # Gráficas principales

            elements.append(Paragraph("Senal temporal", styles['Heading2']))

            elements.append(_pdf_image(img_time, pdf_image_width, 320))

            elements.append(Spacer(1, 8))

            elements.append(Paragraph("Espectro FFT", styles['Heading2']))

            elements.append(_pdf_image(img_fft, pdf_image_width, 320))

            elements.append(Spacer(1, 8))

            if img_wf2:
                elements.append(CondPageBreak(360))
                elements.append(Paragraph("Analisis de arranque/parada (3D)", styles['Heading2']))
                elements.append(_pdf_image(img_wf2, pdf_image_width, 360))
                elements.append(Spacer(1, 8))


            # Gracas auxiliares

            # Gráficas auxiliares
            if aux_imgs:
                elements.append(Paragraph("Variables auxiliares", styles['Heading2']))
                for img in aux_imgs:
                    elements.append(Image(img, width=400, height=120))
                # Tabla de métricas auxiliares
                aux_data = [["Variable", "Promedio", "Mínimo", "Máximo"]]
                for col, _, _ in aux_selected:
                    vals = self.current_df[col].dropna().to_numpy()
                    if len(vals) > 0:
                        aux_data.append([col, f"{np.mean(vals):.2f}", f"{np.min(vals):.2f}", f"{np.max(vals):.2f}"])
                if len(aux_data) > 1:
                    aux_table = Table(aux_data, colWidths=[150, 100, 100, 100])
                    aux_table = Table(aux_data, colWidths=[150, 100, 100, 100])
                    aux_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
                    ]))
                    elements.append(aux_table)
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Diagnóstico", styles['Heading2']))
            elements.append(Paragraph(f"El valor RMS calculado es {rms_mm:.3f} mm/s, lo cual corresponde a: {severity_mm}.", styles['Normal']))
            # (line removed: m/s diagnostic)
            for item in findings_pdf:
                elements.append(Paragraph(f"- {item}", styles['Normal']))


            doc.build(elements)



            # Restaurar estilo de la app

            if self.is_dark_mode:

                plt.style.use("dark_background")

            else:

                plt.style.use("seaborn-v0_8-whitegrid")



            if not hasattr(self, "generated_reports"):

                self.generated_reports = []

            self.generated_reports.append(pdf_path)



            self._log(f"Reporte exportado: {pdf_path}")

            self.page.snack_bar = ft.SnackBar(content=ft.Text(f"✅ Reporte PDF generado: {pdf_name}"), action="OK")

            self.page.snack_bar.open = True

            self.page.update()



        except Exception as ex:

            self._log(f"Error exportando PDF: {ex}")



    def _build_analysis_view(self):

        """

        Construye la vista de análisis:

        - Selección de tiempo, FFT y señales

        - Variables auxiliares con checkbox + color + estilo

        - Periodo de análisis

        - Configuración colapsable

        - Gráfica combinada de señales seleccionadas

        """



        if self.current_df is None:

            return ft.Column(

                controls=[

                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=80, color="#e74c3c"),

                    ft.Text("No hay datos cargados para análisis", size=18),

                    ft.ElevatedButton(

                        "Ir a Archivos",

                        icon=ft.Icons.FOLDER_OPEN_ROUNDED,

                        on_click=lambda e: self._select_menu("files"),

                        style=ft.ButtonStyle(bgcolor=self._accent_ui(), color="white")

                    )

                ],

                alignment="center",

                horizontal_alignment="center",

                expand=True

            )



        # --- Detectar columnas ---

        numeric_cols = self.current_df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:

            return ft.Container(

                content=ft.Column(

                    [

                        ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=80, color="#e74c3c"),

                        ft.Text(

                            "El archivo necesita al menos una columna de tiempo y otra de señal numérica",

                            size=16,

                            text_align="center",

                        ),

                        ft.Text(

                            "Verifique el formato del archivo o ajuste la configuración de importación.",

                            size=12,

                            text_align="center",

                            color="#95a5a6",

                        ),

                    ],

                    alignment="center",

                    horizontal_alignment="center",

                    spacing=10,

                ),

                alignment=ft.alignment.center,

                expand=True,

            )



        preferred_time_col = getattr(self, "default_time_col", None)

        initial_time_col = preferred_time_col if preferred_time_col in numeric_cols else None

        if initial_time_col is None:

            initial_time_col = "t_s" if "t_s" in numeric_cols else numeric_cols[0]



        # Dropdown tiempo

        self.time_dropdown = ft.Dropdown(

            label="Tiempo",

            options=[ft.dropdown.Option(col) for col in numeric_cols],

            value=initial_time_col,

            expand=True

        )



        # Dropdown FFT

        available_signals = [col for col in numeric_cols if col != initial_time_col]

        preferred_signal_cols = [

            col for col in getattr(self, "default_signal_cols", []) if col in available_signals

        ]

        initial_fft_col = preferred_signal_cols[0] if preferred_signal_cols else (available_signals[0] if available_signals else None)

        if initial_fft_col is None:

            return ft.Container(

                content=ft.Column(

                    [

                        ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=80, color="#e74c3c"),

                        ft.Text(

                            "No se encontró ninguna señal numérica para analizar",

                            size=16,

                            text_align="center",

                        ),

                        ft.Text(

                            "Seleccione un archivo que incluya columnas de vibración además del tiempo.",

                            size=12,

                            text_align="center",

                            color="#95a5a6",

                        ),

                    ],

                    alignment="center",

                    horizontal_alignment="center",

                    spacing=10,

                ),

                alignment=ft.alignment.center,

                expand=True,

            )



        self.fft_dropdown = ft.Dropdown(

            label="Señal FFT",

            options=[ft.dropdown.Option(col) for col in available_signals],

            value=initial_fft_col,

            expand=True

        )

        # Parámetros de máquina (opcionales) para diagnóstico avanzado
        # Selección de unidad para la señal en tiempo (aceleración vs velocidad)
        self.time_unit_dd = ft.Dropdown(
            label="Señal de tiempo",
            options=[
                ft.dropdown.Option("acc", "Aceleración (m/s^2)"),
                ft.dropdown.Option("acc_g", "Aceleraci3n (g)"),
                ft.dropdown.Option("vel_mm", "Velocidad (mm/s)"),
            ],
            value="vel_mm",
            width=220,
        )

        self.rpm_hint_field = ft.TextField(label="RPM (opc.)", value="", width=120)
        self.line_freq_dd = ft.Dropdown(label="Línea", options=[ft.dropdown.Option("50"), ft.dropdown.Option("60")], value="60", width=110)
        self.gear_teeth_field = ft.TextField(label="Dientes engrane (opc.)", value="", width=160)
        self.bpfo_field = ft.TextField(label="BPFO Hz (opc.)", value="", width=120)
        self.bpfi_field = ft.TextField(label="BPFI Hz (opc.)", value="", width=120)
        self.bsf_field = ft.TextField(label="BSF Hz (opc.)", value="", width=110)
        self.ftf_field = ft.TextField(label="FTF Hz (opc.)", value="", width=110)

        # Modo de análisis: automático parcial vs asistido (rodamientos)
        # Modo de análisis: automático parcial vs asistido (rodamientos)
        # Usa estado previo si existe (self.analysis_mode)
        self.analysis_mode_dd = ft.Dropdown(
            label="Modo de análisis",
            options=[
                ft.dropdown.Option("auto", "Automático parcial"),
                ft.dropdown.Option("assist", "Asistido (rodamientos)"),
            ],
            value=self.analysis_mode if getattr(self, 'analysis_mode', None) in ("auto", "assist") else "auto",
            width=220,
            on_change=self._on_mode_change,
        )
        # Asistido: datos de rodamiento + base opcional
        # Preseleccionar modelo si hay uno en estado
        self.bearing_model_dd = ft.Dropdown(
            label="Modelo rodamiento (opcional)",
            options=self._bearing_db_model_options(),
            width=250,
            on_change=self._on_bearing_model_change,
            value=(self.selected_bearing_model if getattr(self, 'selected_bearing_model', '') else None),
        )
        self.br_n_field = ft.TextField(label="# Elementos (n)", value="", width=150)
        self.br_d_mm_field = ft.TextField(label="d (mm)", value="", width=110)
        self.br_D_mm_field = ft.TextField(label="D (mm)", value="", width=110)
        self.br_theta_deg_field = ft.TextField(label="Ángulo (°)", value="0", width=110)
        self.assisted_box = ft.Container(
            content=ft.Column([
                ft.Text("Asistido (rodamientos)", size=14),
                ft.Row([self.bearing_model_dd, ft.TextButton("Refrescar base", on_click=lambda e: (self._load_bearing_db(), setattr(self.bearing_model_dd, 'options', self._bearing_db_model_options()), self.bearing_model_dd.update()))], spacing=10, wrap=True),
                ft.Row([self.br_n_field, self.br_d_mm_field, self.br_D_mm_field, self.br_theta_deg_field], spacing=10, wrap=True),
                ft.Row([
                    (setattr(self, 'env_bp_lo_field', ft.TextField(label="Env BP lo (Hz)", value="", width=120)) or self.env_bp_lo_field),
                    (setattr(self, 'env_bp_hi_field', ft.TextField(label="Env BP hi (Hz)", value="", width=120)) or self.env_bp_hi_field),
                ], spacing=10, wrap=True),
                ft.Row([ft.ElevatedButton("Calcular frecuencias", icon=ft.Icons.FUNCTIONS, on_click=self._compute_bearing_freqs_click)], alignment="start")
            ], spacing=8),
            visible=False,
        )



        # Señales de tiempo

        self.signal_checkboxes = [

            ft.Checkbox(label=col, value=(col.startswith("a")))

            for col in numeric_cols if col != initial_time_col

        ]



        # 📌 Nueva gráfica principal combinada

        self.multi_chart_container = ft.Container(

            expand=True,

            content=ft.Text("Seleccione señales para graficar..."),

            bgcolor=ft.Colors.with_opacity(0.02, "white" if self.is_dark_mode else "black"),

            border_radius=10,

            padding=15,

            margin=ft.margin.only(top=10)

        )



        # Conectar checkboxes a actualización dinámica

        for cb in self.signal_checkboxes:

            cb.on_change = self._update_multi_chart



        # Variables auxiliares

        aux_cols = [col for col in numeric_cols if col not in [initial_time_col, self.fft_dropdown.value]]

        color_options = [("#3498db", "Azul"), ("#e74c3c", "Rojo"), ("#2ecc71", "Verde"),

                         ("#f39c12", "Naranja"), ("#9b59b6", "Violeta")]

        style_options = [("-", "Sólida"), ("--", "Guiones"), ("-.", "Guion-punto"), (":", "Punteada")]



        self.aux_controls = []

        for col in aux_cols:

            cb = ft.Checkbox(label=col, value=True)

            color_dd = ft.Dropdown(

                options=[ft.dropdown.Option(c, t) for c, t in color_options],

                value=color_options[len(self.aux_controls) % len(color_options)][0],

                width=110

            )

            style_dd = ft.Dropdown(

                options=[ft.dropdown.Option(s, n) for s, n in style_options],

                value="-", width=110

            )

            self.aux_controls.append((cb, color_dd, style_dd))



        # Campos de periodo

        self.start_time_field = ft.TextField(label="Inicio (s)", value="0.0", width=100)

        self.end_time_field = ft.TextField(label="Fin (s)", value="", width=100)

        # Opciones visuales de frecuencias en FFT
        self.hide_lf_cb = ft.Checkbox(label="Ocultar bajas frecuencias", value=True)
        self.lf_cutoff_field = ft.TextField(label="Corte LF (Hz)", value="0.5", width=100)
        self.hf_limit_field = ft.TextField(label="Máx FFT (Hz)", value="", width=120)
        self.fft_zoom_text = ft.Text("Zoom FFT: completo", size=12)
        self.fft_zoom_slider = ft.RangeSlider(
            0.0,
            1.0,
            min=0.0,
            max=1.0,
            divisions=100,
            on_change=self._on_fft_zoom_preview,
            on_change_end=self._on_fft_zoom_commit,
            disabled=True,
            expand=True,
        )
        # Escala en dBV real (re 1 V) y parámetros de calibración
        self.db_scale_cb = ft.Checkbox(label="Ver FFT en dBV (re 1 V)", value=False)
        # Parámetros de calibración para convertir a Voltios
        self.sens_unit_dd = ft.Dropdown(
            label="Tipo de sensor",
            options=[
                ft.dropdown.Option("mV/g", "Acelerómetro (mV/g)"),
                ft.dropdown.Option("V/g", "Acelerómetro (V/g)"),
                ft.dropdown.Option("mV/(mm/s)", "Velocímetro (mV/(mm/s))"),
                ft.dropdown.Option("V/(mm/s)", "Velocímetro (V/(mm/s))"),
            ],
            value="mV/g",
            width=180,
        )
        self.sensor_sens_field = ft.TextField(label="Sensibilidad", value="100", width=120, tooltip="p.ej. 100 mV/g o 10 mV/(mm/s)")
        self.gain_field = ft.TextField(label="Ganancia (V/V)", value="1.0", width=120)
        # Campos de rango Y para dB/dBV
        self.db_ref_field = ft.TextField(label="Ref. dB (genérico)", value="1.0", width=140, tooltip="Solo para dB genérico; dBV usa 1 V por definición.")
        self.db_ymin_field = ft.TextField(label="Y mín (dB)", value="", width=100)
        self.db_ymax_field = ft.TextField(label="Y máx (dB)", value="", width=100)
        # Recalcular al cambiar estas opciones
        self.hide_lf_cb.on_change = self._update_analysis
        self.lf_cutoff_field.on_change = self._update_analysis
        self.db_scale_cb.on_change = self._update_analysis
        self.sens_unit_dd.on_change = self._update_analysis
        self.sensor_sens_field.on_change = self._update_analysis
        self.gain_field.on_change = self._update_analysis
        self.db_ref_field.on_change = self._update_analysis
        self.db_ymin_field.on_change = self._update_analysis
        self.db_ymax_field.on_change = self._update_analysis
        self.hf_limit_field.on_change = self._update_analysis

        # Configuración de cascada/superficie 3D
        def _wf_value(ctrl, fallback):
            try:
                raw = getattr(ctrl, "value", "") if ctrl else ""
                return raw if raw not in (None, "") else fallback
            except Exception:
                return fallback

        self.waterfall_enabled_cb = ft.Checkbox(
            label="Ver análisis 3D",
            value=bool(getattr(self, "waterfall_enabled", False)),
            tooltip="Divide la señal en ventanas y genera un gráfico 3D (cascada o superficie).",
        )
        self.waterfall_enabled_cb.on_change = self._on_waterfall_toggle

        self.waterfall_mode_dd = ft.Dropdown(
            label="Tipo 3D",
            options=[
                ft.dropdown.Option("waterfall", "Cascada"),
                ft.dropdown.Option("surface", "Superficie"),
            ],
            value=self.waterfall_mode if getattr(self, "waterfall_mode", "") in ("waterfall", "surface") else "waterfall",
            width=150,
            on_change=self._on_waterfall_mode_change,
        )

        wf_window_default = _wf_value(getattr(self, "waterfall_window_field", None), f"{self.waterfall_window_s:.3f}")
        wf_step_default = _wf_value(getattr(self, "waterfall_step_field", None), f"{self.waterfall_step_s:.3f}")

        self.waterfall_window_field = ft.TextField(
            label="Ventana (s)",
            value=wf_window_default,
            width=120,
            on_change=self._update_analysis,
        )
        self.waterfall_step_field = ft.TextField(
            label="Paso (s)",
            value=wf_step_default,
            width=120,
            on_change=self._update_analysis,
        )




        # --- Contenedor de configuración ---

        self.config_expanded = True

        data_tab = ft.Column(
            [
                ft.Text("Selecci\u00f3n de datos", size=14, weight="bold"),
                ft.Row([self.time_dropdown, self.fft_dropdown], spacing=10),
                ft.Row([self.time_unit_dd], spacing=10),
                ft.Text("Periodo de an\u00e1lisis", size=14, weight="bold"),
                ft.Row([self.start_time_field, self.end_time_field], spacing=10),
            ],
            spacing=12,
            expand=True,
            scroll="auto",
        )

        signals_tab = ft.Column(
            [
                ft.Text("Se\u00f1ales en tiempo", size=14, weight="bold"),
                ft.Container(content=ft.Row(self.signal_checkboxes, wrap=True, spacing=10), padding=10),
                ft.Text("Variables auxiliares", size=14, weight="bold"),
                ft.Column([ft.Row([cb, color_dd, style_dd], spacing=10) for cb, color_dd, style_dd in self.aux_controls], spacing=8),
            ],
            spacing=12,
            expand=True,
            scroll="auto",
        )

        spectrum_tab = ft.Column(
            [
                ft.Text("Opciones de espectro", size=14, weight="bold"),
                ft.Row([self.hide_lf_cb, self.lf_cutoff_field, self.hf_limit_field], spacing=10, wrap=True),
                ft.Column([self.fft_zoom_text, self.fft_zoom_slider], spacing=4),
                ft.Row([self.db_scale_cb, self.sens_unit_dd, self.sensor_sens_field, self.gain_field], spacing=10, wrap=True),
                ft.Row([self.db_ref_field, self.db_ymin_field, self.db_ymax_field], spacing=10, wrap=True),
            ],
            spacing=12,
            expand=True,
            scroll="auto",
        )

        diagnostics_tab = ft.Column(
            [
                ft.Text("An\u00e1lisis arranque/parada (3D)", size=14, weight="bold"),
                ft.Row([self.waterfall_enabled_cb, self.waterfall_mode_dd, self.waterfall_window_field, self.waterfall_step_field], spacing=10, wrap=True),
                ft.Text("Par\u00e1metros de m\u00e1quina", size=14, weight="bold"),
                ft.Row([self.analysis_mode_dd, self.rpm_hint_field, self.line_freq_dd, self.gear_teeth_field, ft.OutlinedButton("Rodamientos", icon=ft.Icons.LIST_ALT_ROUNDED, on_click=self._goto_bearings_view)], spacing=10, wrap=True),
                self.assisted_box,
                ft.Row([self.bpfo_field, self.bpfi_field, self.bsf_field, self.ftf_field], spacing=10, wrap=True),
            ],
            spacing=12,
            expand=True,
            scroll="auto",
        )

        config_tabs = ft.Tabs(
            animation_duration=220,
            expand=1,
            tabs=[
                ft.Tab(text="Entrada", content=ft.Container(padding=10, content=data_tab)),
                ft.Tab(text="Se\u00f1ales", content=ft.Container(padding=10, content=signals_tab)),
                ft.Tab(text="Espectro", content=ft.Container(padding=10, content=spectrum_tab)),
                ft.Tab(text="Diagn\u00f3stico", content=ft.Container(padding=10, content=diagnostics_tab)),
            ],
        )

        action_row = ft.Row(
            alignment="center",
            spacing=20,
            controls=[
                ft.ElevatedButton("Generar", icon=ft.Icons.ANALYTICS_ROUNDED, on_click=self._update_analysis, style=ft.ButtonStyle(bgcolor=self._accent_ui(), color="white")),
                ft.OutlinedButton("Exportar", icon=ft.Icons.DOWNLOAD_ROUNDED, on_click=self.exportar_pdf),
            ],
        )

        self.config_container = ft.Container(
            content=ft.Column([
                config_tabs,
                action_row,
            ], spacing=16, expand=True),
            visible=self.config_expanded,
        )



        # Botón de colapsar

        toggle_btn = ft.IconButton(

            icon=ft.Icons.ARROW_DROP_DOWN_CIRCLE if self.config_expanded else ft.Icons.ARROW_RIGHT,

            tooltip="Mostrar/Ocultar configuración",

            on_click=self._toggle_config_panel

        )



        # Panel final

        controls_panel = ft.Container(

            padding=20,

            border_radius=15,

            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

            content=ft.Column([

                ft.Row([

                    ft.Text("⚙️ Configuración de Gráficos", weight="bold", size=18),

                    toggle_btn

                ], alignment="spaceBetween"),

                self.config_container

            ], spacing=10)

        )



        # Contenedor de gráficas

        self.chart_container = ft.Container(

            expand=True,

            content=ft.Column([ft.ProgressRing(), ft.Text("Preparando análisis...")],

                              alignment="center", expand=True),

            bgcolor=ft.Colors.with_opacity(0.02, "white" if self.is_dark_mode else "black"),

            border_radius=15,

            padding=20,

            margin=ft.margin.only(top=20)

        )



        # Si venimos desde 'Usar en análisis', activar asistido y aplicar modelo seleccionado
        try:
            if getattr(self, 'analysis_mode', 'auto') == 'assist':
                try:
                    self.assisted_box.visible = True
                    self.assisted_box.update() if self.assisted_box.page else None
                except Exception:
                    pass
                self._on_bearing_model_change()
        except Exception:
            pass

        self._update_waterfall_controls_state()

        self.page.run_task(self._update_analysis_async)



        return ft.Column(

            controls=[

                ft.Text("Análisis FFT y Diagnóstico", size=24, weight="bold"),

                controls_panel,

                self.chart_container         

            ],

            expand=True,

            spacing=10,

            scroll="auto"

        )

    

    async def _update_analysis_async(self, e=None):

        if self.chart_container:

            self.chart_container.content = ft.Column(

                [ft.ProgressRing(), ft.Text("Generando análisis FFT...")],

                horizontal_alignment="center",

                alignment="center",

                expand=True

            )

            if self.chart_container.page:

                self.chart_container.update()

            await asyncio.sleep(0.1)



            try:

                new_chart = self._create_plot()

            except Exception as ex:

                self._log(f"Error al generar análisis: {ex}")

                new_chart = ft.Column(

                    [

                        ft.Icon(ft.Icons.ERROR_OUTLINE_ROUNDED, size=60, color="#e74c3c"),

                        ft.Text("No se pudo generar el análisis.", size=16, weight="bold"),

                        ft.Text(

                            "Revisa los parámetros de entrada e intenta nuevamente.",

                            size=12,

                            color="#bdc3c7",

                            text_align="center",

                        ),

                    ],

                    alignment="center",

                    horizontal_alignment="center",

                    expand=True,

                )

            self.chart_container.content = new_chart

            if self.chart_container.page:

                self.chart_container.update()



    def _update_analysis(self, e=None):
        self.page.run_task(self._update_analysis_async)
        try:
            self._update_multi_chart()
        except Exception:
            pass

    def _goto_bearings_view(self, e=None):
        try:
            self._select_menu("bearings", force_rebuild=True)
        except Exception:
            pass



    def _calculate_rms(self, signal):
        """
        Calcula el valor RMS de una señal en el dominio del tiempo.
        """
        return np.sqrt(np.mean(signal**2))

    def _format_peak_label(self, freq_hz: float, amp_mm_s: float, order: float | None = None, unit: str = "mm/s") -> str:
        label = f"{freq_hz:.2f} Hz | {amp_mm_s:.3f} {unit}"
        try:
            if order is not None and np.isfinite(order):
                label += f" | {float(order):.2f}X"
        except Exception:
            pass
        return label

    def _place_annotations(self, ax, points: List[Tuple[float, float]], labels: List[str], color: str = "#e74c3c", text_color: str | None = None):
        try:
            if not points or not labels:
                return
            text_color = text_color or ("white" if self.is_dark_mode else "black")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x_span = xmax - xmin if xmax > xmin else max(abs(xmax), 1.0)
            y_span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
            y_step = 0.04 * y_span if y_span > 0 else 1.0
            x_offset = 0.02 * x_span if x_span > 0 else 0.5
            placements: List[Tuple[float, float]] = []
            bg_color = "#1b1f24" if self.is_dark_mode else "white"
            for (x, y), label in zip(points, labels):
                try:
                    x = float(x)
                    y = float(y)
                except Exception:
                    continue
                tx = x + x_offset
                align = "left"
                if tx > xmax:
                    tx = x - x_offset
                    align = "right"
                ty = y + y_step
                attempts = 0
                while placements and any(abs(ty - py) < 0.8 * y_step for _, py in placements) and attempts < 20:
                    ty += y_step
                    attempts += 1
                    if ty > ymax:
                        ty = y - y_step
                if ty < ymin:
                    ty = ymin + 0.05 * y_span
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(tx, ty),
                    textcoords="data",
                    ha=align,
                    va="bottom" if ty >= y else "top",
                    fontsize=8,
                    color=text_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=color),
                    zorder=6,
                    clip_on=False,
                )
                placements.append((tx, ty))
        except Exception:
            pass

    def _draw_frequency_markers(self, ax, marks: List[Tuple[float, str, str]], zoom_range: Optional[Tuple[float, float]] = None):
        try:
            if not marks:
                return
            transform = ax.get_xaxis_transform()
            used: List[Tuple[float, float]] = []
            base_y = 0.98
            step = 0.08
            bg_color = "#1b1f24" if self.is_dark_mode else "white"
            for freq, label, color in marks:
                try:
                    freq = float(freq)
                except Exception:
                    continue
                if freq <= 0:
                    continue
                if zoom_range and (freq < zoom_range[0] or freq > zoom_range[1]):
                    continue
                slot = base_y
                attempts = 0
                while used and any(abs(slot - other_y) < 0.05 for _, other_y in used) and attempts < 10:
                    slot -= step
                    attempts += 1
                ax.text(
                    freq,
                    slot,
                    label,
                    rotation=90,
                    color=color,
                    fontsize=8,
                    va="top",
                    ha="center",
                    transform=transform,
                    bbox=dict(boxstyle="round,pad=0.15", fc=bg_color, ec="none", alpha=0.75),
                    clip_on=False,
                )
                used.append((freq, slot))
        except Exception:
            pass

    def _format_fft_zoom_label(self, start: float, end: float, full_range: Tuple[float, float]) -> str:
        min_val, max_val = full_range
        if abs(start - min_val) <= 1e-6 and abs(end - max_val) <= 1e-6:
            return f"Zoom FFT: completo ({min_val:.1f} - {max_val:.1f} Hz)"
        return f"Zoom FFT: {start:.1f} - {end:.1f} Hz"

    def _update_fft_zoom_controls(self, full_range: Optional[Tuple[float, float]], current_range: Optional[Tuple[float, float]]):
        slider = getattr(self, 'fft_zoom_slider', None)
        label = getattr(self, 'fft_zoom_text', None)
        if not slider and not label:
            return
        self._fft_zoom_syncing = True
        try:
            if not full_range or full_range[1] <= full_range[0]:
                if slider:
                    slider.disabled = True
                    slider.start_value = 0.0
                    slider.end_value = 1.0
                    if slider.page:
                        slider.update()
                if label:
                    label.value = 'Zoom FFT: sin datos'
                    if label.page:
                        label.update()
                return
            min_val, max_val = full_range
            if current_range is None:
                start_val, end_val = min_val, max_val
            else:
                start_val = max(min_val, min(current_range[0], max_val))
                end_val = max(start_val + 1e-6, min(current_range[1], max_val))
            if slider:
                slider.min = min_val
                slider.max = max_val
                slider.start_value = start_val
                slider.end_value = end_val
                slider.disabled = False
                if slider.page:
                    slider.update()
            if label:
                label.value = self._format_fft_zoom_label(start_val, end_val, (min_val, max_val))
                if label.page:
                    label.update()
        finally:
            self._fft_zoom_syncing = False

    def _on_fft_zoom_preview(self, e):
        if self._fft_zoom_syncing:
            return
        try:
            slider = e.control
            start = float(slider.start_value)
            end = float(slider.end_value)
        except Exception:
            return
        label = getattr(self, 'fft_zoom_text', None)
        if label and self._fft_full_range:
            label.value = self._format_fft_zoom_label(start, end, self._fft_full_range)
            if label.page:
                label.update()

    def _on_fft_zoom_commit(self, e):
        if self._fft_zoom_syncing:
            return
        slider = e.control
        if not self._fft_full_range:
            return
        try:
            start = float(slider.start_value)
            end = float(slider.end_value)
        except Exception:
            return
        full_start, full_end = self._fft_full_range
        tol = max(1e-6, 0.002 * max(full_end - full_start, 1.0))
        if end <= start + tol:
            end = min(full_end, start + tol)
        if abs(start - full_start) <= tol and abs(end - full_end) <= tol:
            new_range = None
        else:
            new_range = (start, end)
        if new_range == self._fft_zoom_range or (new_range is None and self._fft_zoom_range is None):
            return
        self._fft_zoom_range = new_range
        self._update_fft_zoom_controls(self._fft_full_range, self._fft_zoom_range)
        self._update_analysis()

    def _select_main_findings(self, findings: List[str], max_items: int = 2) -> List[str]:
        """
        Selecciona hallazgos principales para motores eléctricos (prioriza: Eléctrico, Desalineación,
        Desbalanceo, Rodamientos, Engranes, Resonancia). Excluye la línea de severidad ISO.
        """
        if not findings:
            return []
        try:
            items = [f for f in findings if not str(f).startswith("Severidad ISO:")]
        except Exception:
            items = findings[:]
        order = ["Eléctrico", "Elctrico", "El?ctrico", "Desalineaci", "Desbalanceo", "Rodamientos", "Engranes", "Resonancia estructural"]
        selected: List[str] = []
        for key in order:
            for f in items:
                try:
                    if key in f and f not in selected:
                        selected.append(f)
                        if len(selected) >= max_items:
                            return selected
                except Exception:
                    continue
        for f in items:
            if f not in selected:
                selected.append(f)
                if len(selected) >= max_items:
                    break
        return selected

    def _build_explanations(self, res: Dict[str, Any], findings: List[str]) -> List[str]:
        """
        Genera una lista de líneas de "Explicación y recomendaciones" basadas en:
        - Resultado unificado del analizador `res` (rms, severidad, energía por bandas).
        - Hallazgos principales seleccionados (findings) para orientar motivos y revisiones.
        """
        lines: List[str] = []
        try:
            sev = res.get('severity', {})
            rms_mm = float(sev.get('rms_mm_s', 0.0))
            sev_label = str(sev.get('label', 'N/D'))
        except Exception:
            rms_mm = 0.0
            sev_label = 'N/D'

        # Enfoque
        lines.append("Enfoque: motor eléctrico")

        # Severidad
        try:
            lines.append(f"Severidad por RMS de velocidad (ISO): {rms_mm:.3f} mm/s -> {sev_label}.")
        except Exception:
            pass

        # Energía por bandas
        try:
            en = res.get('fft', {}).get('energy', {})
            e_total = float(en.get('total', 1e-12))
            fl = float(en.get('low', 0.0)) / e_total if e_total > 0 else 0.0
            fm = float(en.get('mid', 0.0)) / e_total if e_total > 0 else 0.0
            fh = float(en.get('high', 0.0)) / e_total if e_total > 0 else 0.0
            lines.append(f"Distribución de energía: baja {fl:.0%}, media {fm:.0%}, alta {fh:.0%}.")
        except Exception:
            pass

        # Helper para buscar presencia robusta (variantes de acentos/codificación)
        def _has_any(keys: List[str]) -> bool:
            for f in (findings or []):
                try:
                    s = str(f)
                except Exception:
                    continue
                for k in keys:
                    if k in s:
                        return True
            return False

        # Recomendaciones por patrón
        if _has_any(["Desbalanceo"]):
            lines += [
                "Motivo: 1X dominante con 2X/3X bajos y energía LF.",
                "Revisar: balanceo de rotor/acoplamiento, fijaciones, suciedad/excentricidad.",
            ]
        if _has_any(["Desalineaci", "Desalineación", "Desalineacion"]):
            lines += [
                "Motivo: armónicos 2X/3X elevados respecto a 1X.",
                "Revisar: alineación de ejes, calces y base.",
            ]
        if _has_any(["Engranes", "Engranaje"]):
            lines += [
                "Motivo: componente de malla apreciable.",
                "Revisar: desgaste, juego y lubricación.",
            ]
        if _has_any(["Rodamientos", "Rodamiento"]):
            lines += [
                "Motivo: picos en envolvente en frecuencias del rodamiento.",
                "Revisar: lubricación, holgura y daño en pistas/elementos.",
            ]
        if _has_any(["Eléctrico", "Electrico", "El?ctrico", "El\u0019ctrico"]):
            lines += [
                "Motivo: componentes a frecuencia de línea y/o su 2x.",
                "Revisar: balance de fases, variador, conexiones y carga del motor.",
            ]
        if _has_any(["Resonancia estructural", "Resonancia"]):
            lines += [
                "Motivo: picos agudos con Q alto.",
                "Revisar: rigidez/apoyos, aprietes, prueba modal/FRF.",
            ]

        return lines

    def _acc_to_vel_time_mm(self, acc: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Integra aceleración en el dominio de la frecuencia para obtener velocidad en el tiempo (mm/s).
        - Usa rFFT, divide por j*2*pi*f (f>0), fuerza DC=0 para evitar deriva y hace irFFT.
        """
        acc = np.asarray(acc, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        if acc.size < 2 or t.size < 2:
            return np.asarray([], dtype=float)
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            return np.asarray([], dtype=float)
        N = acc.size
        Af = np.fft.rfft(acc)
        xf = np.fft.rfftfreq(N, dt)
        V = np.zeros_like(Af, dtype=complex)
        pos = xf > 0
        V[pos] = Af[pos] / (1j * 2.0 * np.pi * xf[pos])
        V[0] = 0.0
        v_t = np.fft.irfft(V, n=N)
        return 1000.0 * v_t  # mm/s

    def _waterfall_start_stop_notes(self, wf_data: Dict[str, Any]) -> List[str]:
        notes: List[str] = []
        try:
            amplitude = wf_data.get("amplitude") if isinstance(wf_data, dict) else None
            if amplitude is None:
                return notes
            amp_array = np.asarray(amplitude, dtype=float)
            if amp_array.size == 0:
                return notes
            row_energy = np.nanmean(amp_array, axis=1)
            if row_energy.size < 3:
                return notes
            n = row_energy.size
            seg = max(1, n // 4)
            start_mean = float(np.nanmean(row_energy[:seg]))
            end_mean = float(np.nanmean(row_energy[-seg:]))
            if not (np.isfinite(start_mean) and np.isfinite(end_mean)):
                return notes
            baseline = max(abs(start_mean), 1e-9)
            rel_change = (end_mean - start_mean) / baseline
            if rel_change > 0.25:
                notes.append("La energía aumenta hacia la parada: revisar el proceso de apagado, holguras o resonancias finales.")
            elif rel_change < -0.25:
                notes.append("Energía elevada al arranque con posterior estabilización: revisar amarre inicial, impactos o desbalanceo en arranque.")
            else:
                peak_idx = int(np.nanargmax(row_energy)) if np.all(np.isfinite(row_energy)) else 0
                if peak_idx == 0:
                    notes.append("Picos dominantes al arranque detectados en la cascada 3D.")
                elif peak_idx == row_energy.size - 1:
                    notes.append("Picos dominantes al finalizar la secuencia en la cascada 3D.")
            return notes
        except Exception:
            return notes

    def _classify_severity(self, rms_value):
        """
        Clasifica el nivel de severidad basado en ISO 10816/20816-3
        para motores eléctricos usando velocidad en mm/s.
        """
        if rms_value <= 2.8:
            return "✅ Buena (Aceptable)"
        elif rms_value <= 4.5:
            return "⚠️ Satisfactoria (Zona de vigilancia)"
        elif rms_value <= 7.1:
            return "❌ Insatisfactoria (Crítica)"
        else:
            return "🔥 Inaceptable (Riesgo de daño)"

    def _classify_severity_ms(self, rms_ms):
        """
        Variante en unidades SI (m/s). Devuelve (label, color_hex).
        Umbrales equivalentes a 2.8, 4.5, 7.1 mm/s.
        """
        if rms_ms <= 0.0028:
            return "Buena (Aceptable)", "#2ecc71"
        elif rms_ms <= 0.0045:
            return "Satisfactoria (Zona de vigilancia)", "#f1c40f"
        elif rms_ms <= 0.0071:
            return "Insatisfactoria (Crítica)", "#e67e22"
        else:
            return "Inaceptable (Riesgo de daño)", "#e74c3c"


    def _band_energy(self, xf, spec, f0, f1):

        """Energía (suma de amplitud^2) entre f0–f1 Hz."""

        if xf is None or spec is None or len(xf) == 0:

            return 0.0

        idx = (xf >= f0) & (xf < f1)

        return float(np.sum((spec[idx] ** 2))) if np.any(idx) else 0.0



    def _amp_near(self, xf, spec, f, tol=2.0):
        """Amplitud máx cerca de frecuencia objetivo f ± tol. Devuelve 0.0 si no hay datos o f no es válida."""
        if xf is None or spec is None or len(xf) == 0 or f is None or not np.isfinite(f) or f <= 0:
            return 0.0
        idx = (xf >= (f - tol)) & (xf <= (f + tol))
        return float(np.max(spec[idx])) if np.any(idx) else 0.0


    def _analytic_signal(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        N = len(y)
        if N < 2:
            return y.astype(complex)
        Y = np.fft.fft(y)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = 1
            h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        Z = np.fft.ifft(Y * h)
        return Z

    def _compute_envelope_spectrum(self, y: np.ndarray, T: float):
        N = len(y)
        if N < 2:
            return None, None
        z = self._analytic_signal(y)
        env = np.abs(z)
        env = env - float(np.mean(env))
        Ef = np.fft.fft(env)
        xf = np.fft.fftfreq(N, T)[: N // 2]
        mag = 2.0 / N * np.abs(Ef[: N // 2])
        return xf, mag
        return float(np.max(spec[idx])) if np.any(idx) else 0.0



    def _extract_features(self, t, acc_signal, xf, vel_spec):
        """
        Extrae features de tiempo (aceleración) y frecuencia (velocidad [mm/s]).
        - t: vector de tiempo del segmento
        - acc_signal: aceleración del segmento (m/s^2)
        - xf: frecuencias FFT
        - vel_spec: magnitud espectral de velocidad [mm/s]
        """
        acc = acc_signal.astype(float)

        rms_time_acc = float(np.sqrt(np.mean(acc**2))) if len(acc) else 0.0

        peak = float(np.max(np.abs(acc))) if len(acc) else 0.0

        pp = float(np.ptp(acc)) if len(acc) else 0.0

        mu = float(np.mean(acc)) if len(acc) else 0.0

        std = float(np.std(acc)) + 1e-12

        skew = float(np.mean(((acc - mu) / std) ** 3))

        kurt = float(np.mean(((acc - mu) / std) ** 4) - 3.0)

        crest = float(peak / (rms_time_acc + 1e-12))



        if vel_spec is None or len(vel_spec) == 0:

            dom_freq = 0.0

            dom_amp = 0.0

            total_energy = 1e-12

            e_low = e_mid = e_high = 0.0

            r2x = r3x = 0.0

            rms_vel_spec = 0.0

        else:

            dom_idx = int(np.argmax(vel_spec))

            dom_freq = float(xf[dom_idx]) if len(xf) else 0.0

            dom_amp = float(vel_spec[dom_idx])

            total_energy = float(np.sum(vel_spec**2)) + 1e-12



            # Bandas típicas (ajústalas a tu máquina)

            e_low = self._band_energy(xf, vel_spec, 0.0, 30.0)

            e_mid = self._band_energy(xf, vel_spec, 30.0, 120.0)

            max_f = float(xf.max()) if len(xf) else 500.0

            e_high = self._band_energy(xf, vel_spec, 120.0, max_f)



            r2x = self._amp_near(xf, vel_spec, 2 * dom_freq) / (dom_amp + 1e-12)

            r3x = self._amp_near(xf, vel_spec, 3 * dom_freq) / (dom_amp + 1e-12)

            rms_vel_spec = float(np.sqrt(np.mean(vel_spec**2)))



        return {
            "rms_time_acc": rms_time_acc,
            "peak_acc": peak,
            "pp_acc": pp,
            "crest": crest,
            "skew": skew,
            "kurt": kurt,
            "dom_freq": dom_freq,
            "dom_amp": dom_amp,
            "r2x": r2x,
            "r3x": r3x,
            "e_low": e_low,
            "e_mid": e_mid,
            "e_high": e_high,
            "e_total": total_energy,
            "rms_vel_spec": rms_vel_spec,
        }

    # ---- Helpers avanzados de diagnóstico basado en FFT ----
    def _get_float(self, tf):
        try:
            if tf and getattr(tf, "value", None):
                return float(tf.value)
        except Exception:
            return None
        return None

    def _get_1x_hz(self, dom_freq_guess: float | None = None):
        """Obtiene 1X en Hz desde RPM (si la hay) o una conjetura (dom_freq)."""
        try:
            rpm = self._get_float(getattr(self, "rpm_hint_field", None))
            if rpm and rpm > 0:
                return rpm / 60.0
        except Exception:
            pass
        try:
            if dom_freq_guess and dom_freq_guess > 0:
                return float(dom_freq_guess)
        except Exception:
            pass
        return 0.0

    def _estimate_rpm(self, xf, spec):
        if xf is None or spec is None or len(xf) == 0:
            return None
        try:
            # usar el pico dominante como 1X si está por debajo de 200 Hz
            idx = int(np.argmax(spec))
            f = float(xf[idx])
            if f <= 0:
                return None
            return f * 60.0
        except Exception:
            return None

    def _peak_amp_near(self, xf, spec, f, tol_rel=0.03, tol_abs=1.0):
        if xf is None or spec is None or f is None or f <= 0:
            return 0.0
        bw = max(tol_abs, tol_rel * f)
        idx = (xf >= (f - bw)) & (xf <= (f + bw))
        return float(np.max(spec[idx])) if np.any(idx) else 0.0

    def _sideband_score(self, xf, spec, center, spacing, n=3, tol_rel=0.03):
        if center is None or spacing is None or center <= 0 or spacing <= 0:
            return 0.0
        amps = []
        for k in range(1, n + 1):
            amps.append(self._peak_amp_near(xf, spec, center - k * spacing, tol_rel))
            amps.append(self._peak_amp_near(xf, spec, center + k * spacing, tol_rel))
        return float(np.mean(amps)) if amps else 0.0

    def _detect_faults(self, xf, vel_spec, features):
        """Devuelve hallazgos detallados por subsistema: balanceo, alineación,
        holguras, rodamientos, engranes, eléctrico.
        Usa parámetros opcionales: RPM, BPFO/BPFI/BSF/FTF, dientes, línea.
        """
        findings = []
        if xf is None or vel_spec is None or len(xf) == 0:
            return findings

        # 1X desde RPM o dom_freq (seguro)
        f1 = self._get_1x_hz(features.get("dom_freq", 0.0))

        # Amplitudes 1X..4X
        a1 = self._peak_amp_near(xf, vel_spec, f1)
        a2 = self._peak_amp_near(xf, vel_spec, 2 * f1)
        a3 = self._peak_amp_near(xf, vel_spec, 3 * f1)
        a4 = self._peak_amp_near(xf, vel_spec, 4 * f1)

        # Unbalance
        if f1 > 0 and a1 > 0 and (a2 < 0.5 * a1) and (a3 < 0.4 * a1) and (features.get("e_low", 0) / max(features.get("e_total", 1e-12), 1e-12) > 0.5):
            findings.append(f"Desbalanceo probable (1X dominante, 2X/3X bajos). 1X={f1:.2f} Hz")

        # Misalignment
        if f1 > 0 and (a2 >= 0.6 * a1 or a3 >= 0.4 * a1):
            findings.append("Desalineación probable (armónicos 2X/3X elevados respecto a 1X)")

        # Looseness
        harmonics = [a1, a2, a3, a4]
        if f1 > 0 and sum(1 for a in harmonics if a > 0.3 * max(harmonics)) >= 3:
            findings.append("Holgura mecánica (múltiples armónicos de 1X significativos)")

        # Gear mesh
        gear_teeth = self._get_float(getattr(self, "gear_teeth_field", None))
        if f1 > 0 and gear_teeth and gear_teeth > 0:
            fgm = f1 * gear_teeth
            a_gm = self._peak_amp_near(xf, vel_spec, fgm, tol_rel=0.02, tol_abs=2.0)
            sb = self._sideband_score(xf, vel_spec, fgm, f1, n=3)
            if a_gm > 0 and sb > 0.2 * a_gm:
                findings.append(f"Engranes: frecuencia de malla ~{fgm:.1f} Hz con bandas laterales ±1X")

        # Bearings (si el usuario conoce frecuencias)
        bpfo = self._get_float(getattr(self, "bpfo_field", None))
        bpfi = self._get_float(getattr(self, "bpfi_field", None))
        bsf  = self._get_float(getattr(self, "bsf_field", None))
        ftf  = self._get_float(getattr(self, "ftf_field", None))
        bearing_hits = []
        for name, freq in (("BPFO", bpfo), ("BPFI", bpfi), ("BSF", bsf), ("FTF", ftf)):
            if freq and freq > 0:
                amp = self._peak_amp_near(xf, vel_spec, freq, tol_rel=0.02, tol_abs=2.0)
                if amp > 0.2 * max(vel_spec) if len(vel_spec) else 0:
                    sb = self._sideband_score(xf, vel_spec, freq, f1 if f1 else freq, n=2)
                    bearing_hits.append(f"{name} (~{freq:.1f} Hz){' con bandas laterales' if sb>0 else ''}")
        if bearing_hits:
            findings.append("Rodamientos: patrones en " + ", ".join(bearing_hits))

        # Eléctrico (línea)
        line_opt = getattr(self, "line_freq_dd", None)
        try:
            line_hz = float(line_opt.value) if line_opt and line_opt.value else None
        except Exception:
            line_hz = None
        if line_hz:
            a_line = self._peak_amp_near(xf, vel_spec, line_hz, tol_rel=0.01, tol_abs=1.0)
            a_2line = self._peak_amp_near(xf, vel_spec, 2 * line_hz, tol_rel=0.01, tol_abs=1.0)
            if (a_line > 0.2 * max(vel_spec)) or (a_2line > 0.2 * max(vel_spec)):
                findings.append(f"Eléctrico (suministro): picos en {line_hz:.0f} Hz y/o {2*line_hz:.0f} Hz")

        return findings


    def _diagnose(self, f):
        """
        Reglas simples de diagnóstico (baseline). Devuelve lista de hallazgos.
        Usa velocidad espectral (mm/s) para severidad ISO.
        """
        findings = []

        # Severidad global (tu función ya clasifica con mm/s)
        sev = self._classify_severity(f.get("rms_vel_spec", 0.0))
        findings.append(f"Severidad ISO (velocidad RMS): {sev}")

        # Desbalanceo: pico 1X dominante, armónicos bajos, energía en baja frecuencia
        if f["dom_freq"] > 0 and f["dom_freq"] < 60 and f["r2x"] < 0.5 and (f["e_low"] / f["e_total"]) > 0.5:
            findings.append("⚠️ Posible desbalanceo: pico 1X dominante y baja energía en armónicos (2X/3X).")

        # Desalineación: armónicos 2X/3X fuertes
        if f["r2x"] >= 0.6 or f["r3x"] >= 0.4:
            findings.append("⚠️ Posible desalineación: armónicos altos (2X/3X) significativos.")

        # Rodamientos: energía predominante en alta frecuencia
        if f["dom_freq"] > 200 and (f["e_high"] / f["e_total"]) > 0.5:
            findings.append("❌ Posible falla en rodamientos: energía predominante en alta frecuencia.")

        # Resonancia: RMS muy alto + crest alto
        if f["rms_vel_spec"] >= 7.1 and f["crest"] > 3.0:
            findings.append("🔥 Posible resonancia: RMS muy alto y alto crest factor.")

        # Reglas avanzadas con espectro actual
        try:
            xf = getattr(self, "_last_xf", None)
            spec = getattr(self, "_last_spec", None)
            if xf is not None and spec is not None:
                findings.extend(self._detect_faults(xf, spec, f))
        except Exception:
            pass

        if len(findings) == 1:
            findings.append("Sin anomalías evidentes según reglas actuales.")

        return findings




    def _on_waterfall_toggle(self, e=None):
        self.waterfall_enabled = bool(getattr(self, "waterfall_enabled_cb", None) and getattr(self.waterfall_enabled_cb, "value", False))
        self._update_waterfall_controls_state()
        self._update_analysis()

    def _on_waterfall_mode_change(self, e=None):
        try:
            self.waterfall_mode = self.waterfall_mode_dd.value or "waterfall"
        except Exception:
            self.waterfall_mode = "waterfall"
        if self.waterfall_enabled:
            self._update_analysis()

    def _update_waterfall_controls_state(self):
        enabled = bool(getattr(self, "waterfall_enabled_cb", None) and getattr(self.waterfall_enabled_cb, "value", False))
        self.waterfall_enabled = enabled
        for ctrl in (getattr(self, "waterfall_mode_dd", None), getattr(self, "waterfall_window_field", None), getattr(self, "waterfall_step_field", None)):
            if not ctrl:
                continue
            try:
                ctrl.disabled = not enabled
                if getattr(ctrl, "page", None):
                    ctrl.update()
            except Exception:
                pass
        if getattr(self, "waterfall_enabled_cb", None) and getattr(self.waterfall_enabled_cb, "page", None):
            try:
                self.waterfall_enabled_cb.value = enabled
                self.waterfall_enabled_cb.update()
            except Exception:
                pass

    def _create_plot(self):

        """

        Genera las gráficas y el diagnóstico:

        - Señal principal (tiempo)

        - FFT de velocidad (mm/s)

        - Variables auxiliares

        - Resumen + diagnóstico automático

        Todo en un contenedor con scroll vertical (en la Column interna).

        """

        try:

            time_col = self.time_dropdown.value

            fft_signal_col = self.fft_dropdown.value

            if not time_col or time_col not in self.current_df.columns:
                return ft.Text("Seleccione una columna de tiempo válida", size=14, color="#e74c3c")
            if not fft_signal_col or fft_signal_col not in self.current_df.columns:
                return ft.Text("Seleccione una señal para el análisis FFT", size=14, color="#e74c3c")

            t = self.current_df[time_col].to_numpy()

            signal = self.current_df[fft_signal_col].to_numpy()



            # --- Filtrar periodo ---

            try:

                start_t = float(self.start_time_field.value) if self.start_time_field.value else t[0]

                end_t = float(self.end_time_field.value) if self.end_time_field.value else t[-1]

            except:

                start_t, end_t = t[0], t[-1]



            mask = (t >= start_t) & (t <= end_t)

            t_segment, signal_segment = t[mask], signal[mask]

            if len(signal_segment) < 2:

                return ft.Text("⚠️ Rango inválido", size=14, color="#e74c3c")





            # --- Features + diagnóstico baseline ---

            # Analizar con función robusta
            try:
                rpm_val = None
                if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", ""):
                    rpm_val = float(self.rpm_hint_field.value)
            except Exception:
                rpm_val = None
            try:
                line_val = float(self.line_freq_dd.value) if getattr(self, "line_freq_dd", None) and getattr(self.line_freq_dd, "value", "") else None
            except Exception:
                line_val = None
            try:
                teeth_val = int(self.gear_teeth_field.value) if getattr(self, "gear_teeth_field", None) and getattr(self.gear_teeth_field, "value", "") else None
            except Exception:
                teeth_val = None
            # usar helper unificado para lectura de floats opcionales
            try:
                _fmax_pre = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                _fmax_pre = None
            res = analyze_vibration(
                t_segment,
                signal_segment,
                rpm=rpm_val,
                line_freq_hz=line_val,
                bpfo_hz=self._fldf(getattr(self, 'bpfo_field', None)),
                bpfi_hz=self._fldf(getattr(self, 'bpfi_field', None)),
                bsf_hz=self._fldf(getattr(self, 'bsf_field', None)),
                ftf_hz=self._fldf(getattr(self, 'ftf_field', None)),
                gear_teeth=teeth_val,
                pre_decimate_to_fmax_hz=_fmax_pre,
            )
            # Sustituir espectros por los del analizador
            xf = res['fft']['f_hz']
            mag_vel_mm = res['fft']['vel_spec_mm_s']
            if xf is not None and len(xf) > 0:
                try:
                    arr = np.asarray(xf, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        full_min = float(arr.min())
                        full_max = float(arr.max())
                        if full_max > full_min:
                            self._fft_full_range = (full_min, full_max)
                            current_zoom = self._fft_zoom_range
                            if current_zoom is not None:
                                start_val = max(full_min, min(current_zoom[0], full_max))
                                end_val = max(start_val + 1e-6, min(current_zoom[1], full_max))
                                self._fft_zoom_range = (start_val, end_val)
                            self._update_fft_zoom_controls(self._fft_full_range, self._fft_zoom_range)
                        else:
                            self._fft_full_range = None
                            self._fft_zoom_range = None
                            self._update_fft_zoom_controls(None, None)
                    else:
                        self._fft_full_range = None
                        self._fft_zoom_range = None
                        self._update_fft_zoom_controls(None, None)
                except Exception:
                    self._fft_full_range = None
                    self._fft_zoom_range = None
                    self._update_fft_zoom_controls(None, None)
            else:
                self._fft_full_range = None
                self._fft_zoom_range = None
                self._update_fft_zoom_controls(None, None)
            dom_freq = res['fft']['dom_freq_hz']
            dom_amp = res['fft']['dom_amp_mm_s']
            rms_mm = res['severity']['rms_mm_s']
            severity_label_ms = res['severity']['label']
            severity_color_ms = res['severity']['color']
            findings = res['diagnosis']
            # Explicación y revisiones sugeridas (basado en hallazgos y métricas)
            exp_lines = []
            # Reducir hallazgos a los principales (para explicaciones)
            try:
                _sel = self._select_main_findings(findings)
            except Exception:
                _sel = findings
            findings = _sel
            # Enfoque explícito: motor eléctrico
            exp_lines.append("Enfoque: motor eléctrico")
            try:
                en = res.get('fft', {}).get('energy', {})
                e_total = float(en.get('total', 1e-12))
                frac_low = (float(en.get('low', 0.0)) / e_total) if e_total > 0 else 0.0
                frac_mid = (float(en.get('mid', 0.0)) / e_total) if e_total > 0 else 0.0
                frac_high = (float(en.get('high', 0.0)) / e_total) if e_total > 0 else 0.0
            except Exception:
                frac_low = frac_mid = frac_high = 0.0
            try:
                exp_lines.append(f"Severidad por RMS de velocidad (ISO): {rms_mm:.3f} mm/s → {severity_label_ms}.")
            except Exception:
                pass
            def _has(txt: str) -> bool:
                try:
                    return any((txt in s) for s in (findings or []))
                except Exception:
                    return False
            if _has("Desbalanceo"):
                exp_lines.append("Motivo: 1X dominante, 2X/3X bajos y energía concentrada en baja frecuencia.")
                exp_lines.append("Revisar: balanceo del rotor/acoplamiento, fijaciones y suciedad/excentricidad.")
            if _has("Desalineaci"):
                exp_lines.append("Motivo: armónicos 2X/3X elevados respecto a 1X.")
                exp_lines.append("Revisar: alineación de ejes, calces y planitud de la base.")
            if _has("Engranes"):
                exp_lines.append("Motivo: componente de malla de engranajes apreciable.")
                exp_lines.append("Revisar: desgaste de dientes, juego y lubricación.")
            if _has("Rodamientos"):
                exp_lines.append("Motivo: picos en envolvente en frecuencias características de rodamientos.")
                exp_lines.append("Revisar: lubricación, holgura y posible daño en pistas/elementos.")
            if _has("Elctrico") or _has("Eléctrico") or _has("El?ctrico"):
                exp_lines.append("Motivo: componentes a frecuencia de línea y/o su 2x.")
                exp_lines.append("Revisar: balance de fases, variador, conexiones y carga del motor.")
            if _has("Resonancia estructural"):
                exp_lines.append("Motivo: picos agudos con Q alto fuera de armónicos conocidos.")
                exp_lines.append("Revisar: rigidez/soportes, aprietes y realizar prueba modal/FRF si es posible.")
            if frac_low + frac_mid + frac_high > 0:
                try:
                    exp_lines.append(f"Distribución de energía: baja {frac_low:.0%}, media {frac_mid:.0%}, alta {frac_high:.0%} (guía del tipo de fallo).")
                except Exception:
                    pass
            # Guardar última FFT/segmento para diagnóstico avanzado
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_segment
            self._last_accseg = signal_segment



            # --- Gráficas principales ---

            plt.style.use('dark_background' if self.is_dark_mode else 'seaborn-v0_8-whitegrid')
            plt.rcParams["font.family"] = "DejaVu Sans"

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Preparar senal de tiempo segun unidad seleccionada
            try:
                unit_mode = getattr(self, "time_unit_dd", None).value if getattr(self, "time_unit_dd", None) else "vel_mm"
            except Exception:
                unit_mode = "vel_mm"
            if unit_mode == "vel_mm":
                _y_time = self._acc_to_vel_time_mm(signal_segment, t_segment)
                _ylabel = "Velocidad [mm/s]"
                _rms_text = f"RMS vel: {self._calculate_rms(_y_time):.3f} mm/s" if _y_time.size else "RMS vel: 0.000 mm/s"
            elif unit_mode == "acc_g":
                _y_time = signal_segment / 9.80665
                _ylabel = "Aceleración [g]"
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3f} g"
            else:
                _y_time = signal_segment
                _ylabel = "Aceleración [m/s²]"
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3e} m/s^2"



            ax1.plot(t_segment, _y_time, color="cyan", linewidth=2)

            ax1.set_title("Señal en el tiempo")

            ax1.set_xlabel("Tiempo (s)")

            ax1.set_ylabel("Aceleración [m/s²]")

            # Anotar RMS de Aceleracion (tiempo)
            try:
                text_color = "white" if self.is_dark_mode else "black"
                rms_acc = self._calculate_rms(signal_segment)
                ax1.text(0.02, 0.95, _rms_text, transform=ax1.transAxes,
                         va="top", color=text_color)
            except Exception:
                pass


            # Aplicar filtros visuales de frecuencia (LF y/o límite HF)
            try:
                fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                fc = 0.5
            try:
                fmax_ui = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                fmax_ui = None
            try:
                hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                hide_lf = True
            zoom_range = getattr(self, "_fft_zoom_range", None)
            zmin = zmax = None
            if zoom_range and len(zoom_range) == 2 and zoom_range[1] > zoom_range[0]:
                try:
                    zmin, zmax = float(zoom_range[0]), float(zoom_range[1])
                except Exception:
                    zmin = zmax = None
            if xf is not None and mag_vel_mm is not None:
                mask_vis = np.ones_like(xf, dtype=bool)
                if hide_lf:
                    mask_vis &= xf >= max(0.0, fc)
                if fmax_ui and fmax_ui > 0:
                    mask_vis &= xf <= fmax_ui
                if zmin is not None:
                    mask_vis &= (xf >= zmin) & (xf <= zmax)
                xplot = xf[mask_vis]
                yplot = mag_vel_mm[mask_vis]
                if xplot.size == 0:
                    xplot = xf
                    yplot = mag_vel_mm
            else:
                xplot = xf
                yplot = mag_vel_mm
                mask_vis = None

            # Escala dBV opcional sobre el espectro (re 1 V) usando calibración
            use_dbv = False
            try:
                use_dbv = bool(getattr(self, 'db_scale_cb', None) and getattr(self.db_scale_cb, 'value', False))
            except Exception:
                use_dbv = False

            ax2.plot(xplot, yplot, color=self._accent_ui(), linewidth=2)
            ax2.fill_between(xplot, yplot, alpha=0.3, color=self._accent_ui())
            if use_dbv:
                try:
                    # Espectro de aceleración para dBV si el sensor es de acc
                    acc_spec = res.get('fft', {}).get('acc_spec_ms2', None)
                    if acc_spec is None:
                        acc_spec = np.zeros_like(xf)
                    acc_plot = acc_spec[mask_vis] if (mask_vis is not None) else acc_spec
                    # Leer parámetros de calibración
                    sens_unit = getattr(self.sens_unit_dd, 'value', 'mV/g') if getattr(self, 'sens_unit_dd', None) else 'mV/g'
                    try:
                        sens_val = float(getattr(self.sensor_sens_field, 'value', '100')) if getattr(self, 'sensor_sens_field', None) else 100.0
                    except Exception:
                        sens_val = 100.0
                    try:
                        gain_vv = float(getattr(self, 'gain_field', '1.0').value) if getattr(self, 'gain_field', None) else 1.0
                    except Exception:
                        try:
                            gain_vv = float(getattr(self, 'gain_field', None).value)
                        except Exception:
                            gain_vv = 1.0
                    # Convertir a Voltios según tipo de sensor
                    if sens_unit == 'mV/g':
                        sens_v_per_g = sens_val * 1e-3
                        V_amp = (acc_plot / 9.80665) * sens_v_per_g * gain_vv
                    elif sens_unit == 'V/g':
                        V_amp = (acc_plot / 9.80665) * sens_val * gain_vv
                    elif sens_unit == 'mV/(mm/s)':
                        V_amp = yplot * (sens_val * 1e-3) * gain_vv
                    elif sens_unit == 'V/(mm/s)':
                        V_amp = yplot * sens_val * gain_vv
                    else:
                        V_amp = yplot * 0.0
                    eps = 1e-12
                    yplot_dbv = 20.0 * np.log10(np.maximum(np.asarray(V_amp, dtype=float), eps) / 1.0)
                    ax2_db = ax2.twinx()
                    ax2_db.plot(xplot, yplot_dbv, color="#9b59b6", linewidth=1.6, linestyle="--")
                    ax2_db.set_ylabel("Nivel [dBV]")
                    # Aplicar rango Y si se definió
                    try:
                        ymin = float(self.db_ymin_field.value) if getattr(self, 'db_ymin_field', None) and getattr(self.db_ymin_field, 'value', '') != '' else None
                    except Exception:
                        ymin = None
                    try:
                        ymax = float(self.db_ymax_field.value) if getattr(self, 'db_ymax_field', None) and getattr(self.db_ymax_field, 'value', '') != '' else None
                    except Exception:
                        ymax = None
                    if ymin is not None or ymax is not None:
                        cur = ax2_db.get_ylim()
                        ax2_db.set_ylim(ymin if ymin is not None else cur[0], ymax if ymax is not None else cur[1])
                except Exception:
                    pass

            # Eje espejo en RPM sincronizado con el rango visible
            try:
                ax2_rpm = ax2.twiny()
                xmin, xmax = ax2.get_xlim()
                ax2_rpm.set_xlim(xmin * 60.0, xmax * 60.0)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass

            # Marcar picos principales (Top-N)
            try:
                K = 5
                min_freq = (max(0.5, fc) if hide_lf else 0.5)
                if xf is not None and mag_vel_mm is not None:
                    mask = xf >= min_freq
                    if zmin is not None:
                        mask &= (xf >= zmin) & (xf <= zmax)
                    xv = xf[mask]
                    yv = mag_vel_mm[mask]
                    if len(yv) > 0:
                        k = min(K, len(yv))
                        idx = np.argpartition(yv, -k)[-k:]
                        idx = idx[np.argsort(yv[idx])[::-1]]
                        peak_f = xv[idx]
                        peak_a = yv[idx]
                        ax2.scatter(peak_f, peak_a, color="#e74c3c", s=30, zorder=5)
                        f1 = self._get_1x_hz(dom_freq)
                        peak_points = []
                        peak_labels = []
                        for pf, pa in zip(peak_f, peak_a):
                            try:
                                pf_f = float(pf)
                                pa_f = float(pa)
                            except Exception:
                                continue
                            order = None
                            if f1 and f1 > 0:
                                try:
                                    order = pf_f / float(f1)
                                except Exception:
                                    order = None
                            peak_points.append((pf_f, pa_f))
                            peak_labels.append(self._format_peak_label(pf_f, pa_f, order))
                        if peak_points:
                            self._place_annotations(ax2, peak_points, peak_labels, color="#e74c3c")
            except Exception:
                pass

            # Lineas guia de frecuencias teoricas (modo asistido)
            try:
                bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                bsf  = self._fldf(getattr(self, 'bsf_field', None))
                ftf  = self._fldf(getattr(self, 'ftf_field', None))
                marks_raw = [
                    (bpfo, 'BPFO', '#1f77b4'),
                    (bpfi, 'BPFI', '#ff7f0e'),
                    (bsf,  'BSF',  '#2ca02c'),
                    (ftf,  'FTF',  '#9467bd'),
                ]
                visible_marks = []
                for f0, label, col in marks_raw:
                    if not (f0 and f0 > 0):
                        continue
                    try:
                        f0_f = float(f0)
                    except Exception:
                        continue
                    if zmin is not None and (f0_f < zmin or f0_f > zmax):
                        continue
                    ax2.axvline(f0_f, color=col, linestyle='--', alpha=0.85, linewidth=1.2)
                    visible_marks.append((f0_f, label, col))
                self._draw_frequency_markers(ax2, visible_marks, None if zmin is None else (zmin, zmax))
            except Exception:
                pass

            ax2.set_title("FFT (Velocidad)")

            ax2.set_xlabel("Frecuencia (Hz)")
            ax2.set_ylabel("Velocidad [mm/s]")
            try:
                if zmin is not None:
                    ax2.set_xlim(left=zmin, right=zmax)
                elif fmax_ui and fmax_ui > 0:
                    ax2.set_xlim(left=0.0, right=float(fmax_ui))
            except Exception:
                pass

            chart = _create_mpl_chart(fig, expand=True, isolated=True)
            chart_card = ft.Container(
                content=chart,
                padding=16,
                border_radius=18,
                bgcolor=ft.Colors.with_opacity(0.06, self._accent_ui()),
                expand=True,
            )
            plt.close(fig)

            waterfall_block = None
            wf_notes = []
            self._last_waterfall_data = None
            if bool(getattr(self, "waterfall_enabled_cb", None) and getattr(self.waterfall_enabled_cb, "value", False)):
                def _float_from_field(field, fallback):
                    try:
                        raw = getattr(field, "value", "") if field else ""
                        return float(raw) if raw not in (None, "") else fallback
                    except Exception:
                        return fallback
                window_s = max(_float_from_field(getattr(self, "waterfall_window_field", None), self.waterfall_window_s), 0.05)
                step_s = max(_float_from_field(getattr(self, "waterfall_step_field", None), self.waterfall_step_s), 0.01)
                self.waterfall_window_s = window_s
                self.waterfall_step_s = step_s
                mode = getattr(self, "waterfall_mode", "waterfall")
                wf_data, wf_error = self._compute_waterfall_data(t_segment, signal_segment, window_s, step_s, fmax_ui, fc, hide_lf, self._fft_zoom_range)
                if wf_data:
                    wf_fig = self._build_waterfall_figure(wf_data, mode)
                    if wf_fig:
                        wf_chart = _create_mpl_chart(wf_fig, expand=True, isolated=True)
                        plt.close(wf_fig)
                        info_text = f"Ventana {window_s:.3f} s | Paso {step_s:.3f} s | Ventanas {wf_data['amplitude'].shape[0]}"
                        waterfall_block = ft.Container(
                            content=ft.Column([
                                ft.Text("Análisis de arranque/parada (3D)", size=16, weight="bold"),
                                ft.Text(info_text, size=12, color="#7f8c8d"),
                                wf_chart,
                            ], spacing=8),
                            bgcolor=ft.Colors.with_opacity(0.08, self._accent_ui()),
                            border_radius=18,
                            padding=18,
                        )
                        self._last_waterfall_data = {
                            "data": wf_data,
                            "mode": mode,
                            "window_s": window_s,
                            "step_s": step_s,
                            "fmax": fmax_ui,
                            "fc": fc,
                            "hide_lf": hide_lf,
                        }
                        wf_notes = self._waterfall_start_stop_notes(wf_data)
                else:
                    self._last_waterfall_data = None
                    if wf_error:
                        waterfall_block = ft.Container(
                            content=ft.Column([
                                ft.Text("Análisis de arranque/parada (3D)", size=16, weight="bold"),
                                ft.Text(f"No se pudo generar la cascada: {wf_error}", color="#e74c3c"),
                            ], spacing=6),
                            bgcolor=ft.Colors.with_opacity(0.08, self._accent_ui()),
                            border_radius=18,
                            padding=18,
                        )
            else:
                self._last_waterfall_data = None


            # Gráfica separada de Envolvente con picos
            env_chart = None
            env_card = None
            try:
                xf_env = res.get('envelope', {}).get('f_hz', None)
                env_amp = res.get('envelope', {}).get('amp', None)
                peaks_env = res.get('envelope', {}).get('peaks', [])
                if xf_env is not None and env_amp is not None and len(xf_env) > 0:
                    if hide_lf:
                        m_env = xf_env >= max(0.0, fc)
                    else:
                        m_env = np.ones_like(xf_env, dtype=bool)
                    if fmax_ui and fmax_ui > 0:
                        m_env = m_env & (xf_env <= fmax_ui)
                    xenv = xf_env[m_env]
                    yenv = env_amp[m_env]
                    env_fig, env_ax = plt.subplots(figsize=(14, 3))
                    env_ax.plot(xenv, yenv, color="#e67e22", linewidth=1.6)
                    env_ax.set_title("Espectro de Envolvente")
                    env_ax.set_xlabel("Frecuencia (Hz)")
                    env_ax.set_ylabel("Amp [a.u.]")
                    # Picos anotados
                    try:
                        vis_peaks = []
                        for p in (peaks_env or []):
                            f0 = float(p.get('f_hz', 0.0))
                            a0 = float(p.get('amp', 0.0))
                            if f0 <= 0 or a0 <= 0:
                                continue
                            if hide_lf and f0 < max(0.0, fc):
                                continue
                            if fmax_ui and fmax_ui > 0 and f0 > fmax_ui:
                                continue
                            vis_peaks.append((f0, a0))
                        if vis_peaks:
                            filtered = vis_peaks
                            if zmin is not None:
                                tmp = [(f0, a0) for f0, a0 in vis_peaks if zmin <= f0 <= zmax]
                                if tmp:
                                    filtered = tmp
                            pfx, pfy = zip(*filtered)
                            env_ax.scatter(pfx, pfy, color="#c0392b", s=24, zorder=5)
                            peak_points = [(float(f0), float(a0)) for f0, a0 in filtered]
                            peak_labels = [f"{float(f0):.2f} Hz" for f0, _ in filtered]
                            self._place_annotations(env_ax, peak_points, peak_labels, color="#c0392b", text_color="#c0392b")
                    except Exception:
                        pass
                    # Líneas guía teóricas
                    try:
                        bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                        bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                        bsf  = self._fldf(getattr(self, 'bsf_field', None))
                        ftf  = self._fldf(getattr(self, 'ftf_field', None))
                        marks_raw = [
                            (bpfo, 'BPFO', '#1f77b4'),
                            (bpfi, 'BPFI', '#ff7f0e'),
                            (bsf,  'BSF',  '#2ca02c'),
                            (ftf,  'FTF',  '#9467bd'),
                        ]
                        visible_marks = []
                        for f0, label, col in marks_raw:
                            if not (f0 and f0 > 0):
                                continue
                            try:
                                f0_f = float(f0)
                            except Exception:
                                continue
                            if zmin is not None and (f0_f < zmin or f0_f > zmax):
                                continue
                            visible_marks.append((f0_f, label, col))
                        if not visible_marks:
                            for f0, label, col in marks_raw:
                                if not (f0 and f0 > 0):
                                    continue
                                try:
                                    f0_f = float(f0)
                                except Exception:
                                    continue
                                visible_marks.append((f0_f, label, col))
                        for f0_f, label, col in visible_marks:
                            try:
                                env_ax.axvline(f0_f, color=col, linestyle='--', alpha=0.85, linewidth=1.2)
                            except Exception:
                                pass
                        self._draw_frequency_markers(env_ax, visible_marks, None if zmin is None else (zmin, zmax))
                    except Exception:
                        pass
                    env_chart = _create_mpl_chart(env_fig, expand=True, isolated=True)
                    env_card = ft.Container(
                        content=env_chart,
                        padding=16,
                        border_radius=18,
                        bgcolor=ft.Colors.with_opacity(0.06, self._accent_ui()),
                        expand=True,
                    )
                    plt.close(env_fig)
            except Exception:
                env_chart = None


            # --- Gráficas auxiliares ---

            aux_plots = []

            for cb, color_dd, style_dd in self.aux_controls:

                if cb.value:

                    aux_fig, aux_ax = plt.subplots(figsize=(8, 2))

                    aux_ax.plot(

                        self.current_df[time_col],

                        self.current_df[cb.label],

                        color=color_dd.value,

                        linestyle=style_dd.value,

                        linewidth=2,

                        label=cb.label

                    )

                    aux_ax.set_title(f"{cb.label} vs Tiempo")

                    aux_ax.legend()

                    aux_fig.tight_layout()

                    aux_chart = _create_mpl_chart(aux_fig, expand=True, isolated=True)
                    aux_plots.append(ft.Container(
                        content=aux_chart,
                        padding=12,
                        border_radius=14,
                        bgcolor=ft.Colors.with_opacity(0.04, self._accent_ui()),
                        expand=True,
                    ))
                    plt.close(aux_fig)

            # --- Resumen Ejecutivo (mm/s, formal al inicio) ---
            try:
                sev_label, sev_color = severity_label_ms, severity_color_ms
            except Exception:
                sev_label, sev_color = "N/D", "#7f8c8d"
            exec_findings = findings[1:] if len(findings) > 1 else ["Sin anomalías evidentes según reglas actuales."]
            # Filtrar a hallazgos principales
            try:
                exec_findings_all = list(exec_findings)
            except Exception:
                exec_findings_all = exec_findings if isinstance(exec_findings, list) else []
            exec_findings = self._select_main_findings(exec_findings_all)
            if not exec_findings:
                exec_findings = ["Sin anomalías evidentes según reglas actuales."]
            resumen_exec = ft.Container(
                content=ft.Column([
                    ft.Text("Resumen Ejecutivo", size=18, weight="bold"),
                    ft.Row([
                        ft.Container(width=12, height=12, bgcolor=sev_color, border_radius=6),
                        ft.Text(f"Clasificación ISO: {sev_label}")
                    ]),
                    ft.Text(f"RMS velocidad: {rms_mm:.3f} mm/s"),
                    ft.Text(f"Frecuencia dominante: {dom_freq:.2f} Hz"),
                    ft.Text("Diagnóstico:"),
                    *[ft.Text(f"- {it}") for it in exec_findings],
                ], spacing=6),
                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
                border_radius=10,
                padding=10
            )

            # --- Panel resumen + diagnóstico ---

            resumen = ft.Container(

                content=ft.Column([

                    ft.Text("📊 Resumen del análisis", size=18, weight="bold"),

                    ft.Text(f"Periodo: {start_t:.2f}s – {end_t:.2f}s"),

                    ft.Text(f"Frecuencia dominante: {dom_freq:.2f} Hz"),

                    ft.Text(f"RMS velocidad: {rms_mm:.3f} mm/s"),

                    ft.Text(
                        f"Crest factor (aceleración): "
                        f"{(float(np.max(np.abs(signal_segment))) / (float(self._calculate_rms(signal_segment)) + 1e-12)):.2f}"
                    ),

                    ft.Divider(),

                    ft.Text("🩺 Diagnóstico automático (baseline)", size=16, weight="bold"),

                    *[ft.Text(f"• {it}") for it in findings],

                ], spacing=6),

                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                border_radius=10,

                padding=10

            )



            # Nota de filtro visual FFT para mayor precisión en la interpretación
            try:
                _fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                _fc = 0.5
            try:
                _hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                _hide_lf = True
            _fft_filter_note = f"Filtro visual FFT: oculta < {_fc:.2f} Hz" if _hide_lf else "Filtro visual FFT: sin ocultar"

            # Recalcular explicaciones con helper unificado (evita divergencias)
            try:
                exp_lines = self._build_explanations(res, findings)
            except Exception:
                pass

            # --- Contenedor con scroll (en Column, no en Container) ---

            explanations_container = ft.Container(
                content=ft.Column([
                    ft.Text("Explicación y revisiones sugeridas", size=16, weight="bold"),
                    *[ft.Text(f"- {it}") for it in exp_lines],
                ], spacing=6),
                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
                border_radius=10,
                padding=10,
            )

            content_controls = [
                resumen_exec,
                resumen,
                explanations_container,
                ft.Text(_fft_filter_note),
                chart_card,
            ]
            if waterfall_block:
                content_controls.append(waterfall_block)
            if env_card:
                content_controls.append(env_card)
            content_controls.extend(aux_plots)

            return ft.Container(

                expand=True,

                content=ft.Column(

                    controls=content_controls,

                    spacing=20,

                    scroll="auto",   # scroll vertical aquí (válido en Column)

                    expand=True

                )
            )
        except Exception as e:
            try:
                import traceback
                tb = traceback.format_exc()
                self._log(f"Error en análisis: {e} | {tb}")
            except Exception:
                self._log(f"Error en análisis: {e}")
            return ft.Text(f"Error en análisis: {e}", size=14, color="#e74c3c")

            return ft.Text(f"Error en análisis: {e}", size=14, color="#e74c3c")



    def _compute_waterfall_data(self, t_segment: np.ndarray, acc_segment: np.ndarray, window_s: float, step_s: float, fmax: Optional[float], fc: float, hide_lf: bool, zoom_range: Optional[Tuple[float, float]] = None):
        if t_segment is None or acc_segment is None:
            return None, "Segmento inválido"
        if len(t_segment) < 16 or len(acc_segment) < 16:
            return None, "Segmento demasiado corto"
        try:
            dt = float(np.median(np.diff(t_segment)))
        except Exception:
            return None, "No se puede estimar Δt"
        if not np.isfinite(dt) or dt <= 0:
            return None, "No se puede estimar Δt"
        window_samples = max(int(round(window_s / dt)), 8)
        step_samples = max(int(round(step_s / dt)), 1)
        if window_samples > len(acc_segment):
            window_samples = len(acc_segment)
        if window_samples < 8:
            return None, "Ventana sin suficientes muestras"
        window = np.hanning(window_samples)
        spectra = []
        centers = []
        freq_vector = None
        idx = 0
        while idx + window_samples <= len(acc_segment):
            segment = acc_segment[idx: idx + window_samples]
            if segment.size < window_samples:
                break
            segment = segment - np.mean(segment)
            fft_vals = np.fft.rfft(segment * window)
            freqs = np.fft.rfftfreq(window_samples, dt)
            if freq_vector is None:
                freq_vector = freqs
            mag_acc = np.abs(fft_vals) * (2.0 / np.sum(window))
            mag_vel = np.zeros_like(mag_acc)
            pos = freqs > 0
            mag_vel[pos] = mag_acc[pos] / (2.0 * np.pi * freqs[pos])
            mag_vel_mm = mag_vel * 1000.0
            mag_vel_mm[0] = 0.0
            spectra.append(mag_vel_mm)
            centers.append(t_segment[idx: idx + window_samples].mean())
            idx += step_samples
        if not spectra or freq_vector is None:
            return None, "No se generaron ventanas suficientes"
        amplitude = np.vstack(spectra)
        centers = np.asarray(centers)
        if centers.size:
            centers = centers - centers.min()
        mask = np.ones_like(freq_vector, dtype=bool)
        if hide_lf:
            mask &= freq_vector >= max(0.0, fc)
        if fmax and fmax > 0:
            mask &= freq_vector <= fmax
        if zoom_range and zoom_range[1] > zoom_range[0]:
            mask &= (freq_vector >= zoom_range[0]) & (freq_vector <= zoom_range[1])
        freqs_sel = freq_vector[mask]
        if freqs_sel.size < 3:
            return None, "Rango de frecuencias insuficiente"
        amplitude = amplitude[:, mask]
        return {"freqs": freqs_sel, "times": centers, "amplitude": amplitude}, None

    def _build_waterfall_figure(self, data: Dict[str, np.ndarray], mode: str):
        try:
            freqs = data.get("freqs")
            times = data.get("times")
            amplitude = data.get("amplitude")
            if freqs is None or times is None or amplitude is None:
                return None
            if not len(freqs) or not len(times):
                return None
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111, projection="3d")
            mode_key = str(mode or "waterfall").lower()
            if mode_key == "surface" and amplitude.shape[0] > 1 and amplitude.shape[1] > 1:
                FF, TT = np.meshgrid(freqs, times)
                ax.plot_surface(FF, TT, amplitude, cmap="viridis", linewidth=0, antialiased=False, alpha=0.95)
            else:
                cmap = plt.get_cmap("viridis")
                denom = max(amplitude.shape[0] - 1, 1)
                for idx, row in enumerate(amplitude):
                    color = cmap(idx / denom)
                    ax.plot(freqs, np.full_like(freqs, times[idx]), row, color=color, linewidth=1.2)
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Tiempo relativo (s)")
            ax.set_zlabel("Velocidad (mm/s)")
            ax.set_title("Cascada 3D" if mode_key == "waterfall" else "Superficie 3D")
            ax.view_init(elev=28, azim=-130 if mode_key == "waterfall" else -110)
            return fig
        except Exception:
            return None

    def _build_reports_view(self):
        return self._build_reports_view_impl()

    # ===== Vista de rodamientos =====
    def _build_bearings_view(self):
        # Crear/attach controles si no existen
        if not getattr(self, 'bearing_list_view', None):
            self.bearing_list_view = ft.ListView(expand=True, spacing=4, padding=4)
        if not getattr(self, 'br_model_field_dlg', None):
            self.br_model_field_dlg = ft.TextField(label="Modelo", width=220)
            self.br_n_field_dlg = ft.TextField(label="# Elementos (n)", width=150)
            self.br_d_mm_field_dlg = ft.TextField(label="d (mm)", width=120)
            self.br_D_mm_field_dlg = ft.TextField(label="D (mm)", width=120)
            self.br_theta_deg_field_dlg = ft.TextField(label="Ángulo (°)", width=120, value="0")
        # Refresh list content
        self._refresh_bearing_list_ui()
        # Panel detalle
        detail_col = ft.Column([
            ft.Text("Detalle del rodamiento", size=16, weight="bold"),
            self.br_model_field_dlg,
            ft.Row([self.br_n_field_dlg, self.br_d_mm_field_dlg], spacing=10),
            ft.Row([self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg], spacing=10),
            ft.Row([
                ft.OutlinedButton("Nuevo", icon=ft.Icons.ADD_ROUNDED, on_click=self._bearing_new_click),
                ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE_ROUNDED, on_click=self._bearing_save_click),
                ft.ElevatedButton("Usar en análisis", icon=ft.Icons.CHECK_CIRCLE_ROUNDED, on_click=self._bearing_use_and_go),
                ft.OutlinedButton("Ir a Análisis", icon=ft.Icons.ARROW_FORWARD_ROUNDED, on_click=lambda e: self._select_menu("analysis", force_rebuild=True)),
            ], spacing=10),
        ], spacing=10, expand=True, alignment="start", scroll="auto")

        # Buscador de rodamientos
        if not getattr(self, 'bearing_search', None):
            self.bearing_search = ft.TextField(hint_text="Buscar por modelo o n...", on_change=lambda e: self._refresh_bearing_list_ui(), dense=True)
        # Tabs por marca
        if not getattr(self, 'bearing_tabs', None):
            self.bearing_tabs = ft.Tabs(tabs=[ft.Tab(text=n) for n in self._bearing_brand_names()], selected_index=0, on_change=self._on_bearing_tab_change)
        else:
            self._rebuild_bearing_tabs()
        # Checkbox favoritos sólo
        if not getattr(self, 'bearing_favs_only_cb', None):
            self.bearing_favs_only_cb = ft.Checkbox(label="Mostrar favoritos", value=bool(self.bearing_show_favs_only), on_change=lambda e: self._toggle_bearing_favs_filter())
        list_col = ft.Column([
            ft.Text("Listado de rodamientos", size=16, weight="bold"),
            self.bearing_tabs,
            self.bearing_search,
            self.bearing_favs_only_cb,
            self.bearing_list_view,
        ], spacing=10, expand=True, alignment="start")

        return ft.Column([
            ft.Row([
                ft.Text("Gestor de Rodamientos", size=24, weight="bold"),
                ft.Row([
                    ft.OutlinedButton("Importar CSV", icon=ft.Icons.UPLOAD_FILE_ROUNDED, on_click=self._bearing_open_csv_picker),
                    ft.ElevatedButton("Analizar", icon=ft.Icons.ANALYTICS_ROUNDED, on_click=self._bearing_analyze_click),
                ], spacing=10),
            ], alignment="space_between"),
            ft.Row([
                ft.Container(content=list_col, width=500, height=600, padding=10, bgcolor=ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black"), border_radius=10, alignment=ft.alignment.top_left),
                ft.Container(content=detail_col, height=600, expand=True, padding=10, bgcolor=ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black"), border_radius=10, alignment=ft.alignment.top_left),
            ], spacing=16),
        ], expand=True, scroll="auto")

    # Mantener implementación original de reports separada
    def _build_reports_view_impl(self):

        self.report_search = ft.TextField(

            hint_text="Buscar por nombre...",

            expand=True,

            on_change=lambda e: self._refresh_report_list_scandir()

        )

        self.report_list = ft.ListView(expand=1, spacing=8, padding=10)

        # Filtro de favoritos para reportes
        self.report_favs_only_cb = ft.Checkbox(label="Mostrar favoritos", value=bool(self.report_show_favs_only), on_change=lambda e: self._toggle_reports_fav_filter())



        # Render inicial

        self._refresh_report_list_scandir()



        return ft.Column(

            controls=[

                ft.Text("📑 Reportes Generados", size=24, weight="bold"),

                ft.Row([self.report_search, self.report_favs_only_cb], alignment="spaceBetween"),

                ft.Container(content=self.report_list, expand=True, border_radius=10, padding=10),

            ],

            expand=True

        )



    def _refresh_report_list(self):

        self.report_list.controls.clear()

        if not hasattr(self, "generated_reports") or not self.generated_reports:

            self.report_list.controls.append(ft.Text("Aún no hay reportes generados.", size=14))

        else:

            query = self.report_search.value.lower() if self.report_search.value else ""

            for path in reversed(self.generated_reports):

                name = os.path.basename(path)

                if query in name.lower():

                    self.report_list.controls.append(

                        ft.Container(

                            content=ft.Row(

                                controls=[

                                    ft.Icon(ft.Icons.PICTURE_AS_PDF_ROUNDED, size=30, color="#e74c3c"),

                                    ft.Text(name, expand=True),

                                    ft.IconButton(icon=ft.Icons.FOLDER_OPEN_ROUNDED, on_click=lambda e,p=path: os.startfile(os.path.dirname(p))),

                                    ft.IconButton(icon=ft.Icons.OPEN_IN_NEW_ROUNDED, on_click=lambda e,p=path: os.startfile(p)),

                                ]

                            ),

                            padding=10,

                            border_radius=8,

                            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                        )

                    )

        if self.report_list.page:

            self.report_list.update()

    def _refresh_report_list_scandir(self):
        self.report_list.controls.clear()
        try:
            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            files = []
            for fn in os.listdir(reports_dir):
                if fn.lower().endswith('.pdf'):
                    p = os.path.join(reports_dir, fn)
                    try:
                        mt = os.path.getmtime(p)
                    except Exception:
                        mt = 0.0
                    files.append((p, mt))
            if not files:
                self.report_list.controls.append(ft.Text("Aún no hay reportes generados.", size=14))
            else:
                query = (self.report_search.value.lower() if getattr(self, 'report_search', None) and self.report_search.value else "")
                if query:
                    files = [(p, mt) for (p, mt) in files if query in os.path.basename(p).lower()]
                # Filtrar por favoritos si está activo
                try:
                    if getattr(self, 'report_favs_only_cb', None) and getattr(self.report_favs_only_cb, 'value', False):
                        favs = getattr(self, 'report_favorites', {}) or {}
                        files = [(p, mt) for (p, mt) in files if bool(favs.get(p, False))]
                except Exception:
                    pass
                from datetime import datetime as _dt
                groups = {}
                for p, mt in files:
                    base = os.path.basename(p)
                    try:
                        date_key = _dt.strptime(base[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
                    except Exception:
                        date_key = _dt.fromtimestamp(mt).strftime("%Y-%m-%d")
                    groups.setdefault(date_key, []).append((p, mt))
                for date_key in sorted(groups.keys(), reverse=True):
                    items = sorted(groups[date_key], key=lambda x: x[1], reverse=True)
                    self.report_list.controls.append(ft.Text(date_key, weight="bold"))
                    for p, _mt in items:
                        name = os.path.basename(p)
                        # Estrella de favoritos
                        is_fav = False
                        try:
                            is_fav = bool(getattr(self, 'report_favorites', {}).get(p, False))
                        except Exception:
                            is_fav = False
                        star_icon = ft.Icons.STAR if is_fav else ft.Icons.STAR_BORDER_ROUNDED
                        star_color = "#f1c40f" if is_fav else "#bdc3c7"
                        self.report_list.controls.append(
                            ft.Container(
                                content=ft.Row(
                                    controls=[
                                        ft.IconButton(icon=star_icon, icon_color=star_color, tooltip="Marcar favorito", on_click=lambda e,pp=p: self._toggle_report_favorite(pp)),
                                        ft.Icon(ft.Icons.PICTURE_AS_PDF_ROUNDED, size=30, color="#e74c3c"),
                                        ft.Text(name, expand=True),
                                        ft.IconButton(icon=ft.Icons.FOLDER_OPEN_ROUNDED, on_click=lambda e,pp=p: os.startfile(os.path.dirname(pp))),
                                        ft.IconButton(icon=ft.Icons.OPEN_IN_NEW_ROUNDED, on_click=lambda e,pp=p: os.startfile(pp)),
                                    ]
                                ),
                                padding=10,
                                border_radius=8,
                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
                            )
                        )
        except Exception:
            self.report_list.controls.append(ft.Text("No se pudieron listar los reportes.", size=14))
        if self.report_list.page:
            self.report_list.update()



    def _build_settings_view(self):

        return ft.Column(

            controls=[

                ft.Text("Configuración", size=24, weight="bold"),

                ft.Container(height=20),

                ft.Container(

                    content=ft.Column(

                        spacing=20,

                        controls=[

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.DARK_MODE_ROUNDED, size=24),

                                                ft.Text("Tema oscuro", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Switch(

                                            value=self.is_dark_mode,

                                            on_change=self._toggle_theme_switch,

                                            active_color=self._accent_ui(),

                                        ),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.ACCESS_TIME_ROUNDED, size=24),

                                                ft.Text("Formato 24 horas", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Switch(

                                            value=self.clock_24h,

                                            on_change=self._toggle_clock_format_switch,

                                            active_color=self._accent_ui(),

                                        ),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.PALETTE_ROUNDED, size=24),

                                                ft.Text("Color de acento", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Container(height=10),

                                        # Paleta de gradiente clicable (mejor UX que escribir #RRGGBB)
                                        self._build_accent_palette(),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.STORAGE_ROUNDED, size=24),

                                                ft.Text("Almacenamiento", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Container(height=10),

                                        ft.ElevatedButton(

                                            "Limpiar Datos",

                                            icon=ft.Icons.DELETE_OUTLINE_ROUNDED,

                                            on_click=self._clear_storage,

                                            style=ft.ButtonStyle(

                                                bgcolor="#e74c3c",

                                                color="white",

                                                shape=ft.RoundedRectangleBorder(radius=8),

                                            ),

                                        ),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                        ]

                    ),

                    padding=20,

                )

            ],

            expand=True

        )



    def _change_accent_color(self, e):

        try:
            value = getattr(e.control, "value", None)
            if value is not None:
                self._set_accent(value)
                self._log(f"Color de acento cambiado a: {self.accent}")
        except Exception:
            pass



    def _clear_storage(self, e):

        try:

            self.page.client_storage.clear()

            self._log("Datos de almacenamiento limpiados")

            self.page.snack_bar = ft.SnackBar(

                content=ft.Text("Configuración restablecida correctamente"),

                action="OK",

            )

            self.page.snack_bar.open = True

            self.page.update()

        except Exception as ex:

            self._log(f"Error al limpiar almacenamiento: {ex}")



    def _toggle_theme_switch(self, e):

        self.is_dark_mode = e.control.value

        self.page.client_storage.set("is_dark_mode", self.is_dark_mode)

        self._apply_theme()

        self._update_theme_for_all_components()

        self._log(f"Tema cambiado a: {'oscuro' if self.is_dark_mode else 'claro'}")



    def _toggle_clock_format_switch(self, e):

        self.clock_24h = e.control.value

        self.page.client_storage.set("clock_24h", self.clock_24h)

        self.clock_text.value = self._get_current_time()

        self.clock_text.update()

        self._log(f"Formato de hora cambiado a: {'24h' if self.clock_24h else '12h'}")



    def _update_theme_for_all_components(self):

        # Actualizar botones del menú

        for button in self.menu_buttons.values():
            try:
                button.accent = self.accent
            except Exception:
                pass
            button.update_theme(self.is_dark_mode)
            try:
                if getattr(button, "is_active", False):
                    button.set_active(True, safe=True)
            except Exception:
                pass

        

        # Update background colors

        self.menu.bgcolor = "#16213e" if self.is_dark_mode else "#ffffff"

        self.control_panel.bgcolor = "#16213e" if self.is_dark_mode else "#ffffff"

        self.main_content_area.bgcolor = ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black")

        

        # Update header/menu accent elements if present
        try:
            if hasattr(self, "menu_logo_icon") and self.menu_logo_icon is not None:
                self.menu_logo_icon.color = self._accent_ui()
                self.menu_logo_icon.update()
        except Exception:
            pass
        try:
            if hasattr(self, "menu_logo_text") and self.menu_logo_text is not None:
                self.menu_logo_text.color = self._accent_ui()
                self.menu_logo_text.update()
        except Exception:
            pass

        # Update quick panel accent elements
        try:
            if hasattr(self, "btn_upload") and self.btn_upload is not None:
                self.btn_upload.style = ft.ButtonStyle(bgcolor=self._accent_ui(), color="white")
                self.btn_upload.update()
        except Exception:
            pass
        try:
            if hasattr(self, "clock_card") and self.clock_card is not None:
                self.clock_card.bgcolor = ft.Colors.with_opacity(0.1, self._accent_ui())
                self.clock_card.update()
        except Exception:
            pass

        # Rebuild view if needed

        current_view = self.last_view

        self._select_menu(current_view, force_rebuild=True)



    def _on_menu_click(self, e):

        view_key = e.control.data

        self._select_menu(view_key)



    def _select_menu(self, view_key, force_rebuild=False):

        if view_key == self.last_view and not force_rebuild:

            return



        # Desactivar todos los botones primero

        for key, button in self.menu_buttons.items():

            button.set_active(key == view_key)



        self.last_view = view_key



        # Construir la vista correspondiente

        if view_key == "welcome":

            new_view = self._build_welcome_view()

        elif view_key == "files":

            new_view = self._build_files_view()

        elif view_key == "analysis":

            new_view = self._build_analysis_view()

        elif view_key == "reports":

            new_view = self._build_reports_view()

        elif view_key == "settings":

            new_view = self._build_settings_view()

        elif view_key == "bearings":

            new_view = self._build_bearings_view()

        else:

            new_view = self._build_welcome_view()



        self.main_content_area.content = new_view

        if self.main_content_area.page:  # Only update if added to page

            self.main_content_area.update()



        # Actualizar ayuda contextual después de que la vista principal esté lista

        self._update_contextual_help(view_key)



    def _update_contextual_help(self, view_key):

        help_content = {

            "welcome": "Bienvenido al sistema de análisis de vibraciones. Desde aquí puede comenzar un nuevo análisis o ver la documentación.",

            "files": "Gestione sus archivos de datos. Puede cargar múltiples archivos CSV y seleccionarlos para análisis.",

            "analysis": "Realice análisis FFT y diagnóstico de vibraciones. Configure los parámetros y genere gráficos interactivos.",

            "reports": "Genere reportes detallados de sus análisis (próximamente).",

            "settings": "Configure las preferencias de la aplicación, incluyendo tema colores y formato de hora."

        }

        
        help_content["bearings"] = "Gestione rodamientos: seleccione un modelo, edite geometría y úselo en el análisis."

        self.help_panel.controls = [

            ft.Container(

                content=ft.Text(f"📋 Ayuda - {view_key.capitalize()}", size=16, weight="bold"),

                padding=ft.padding.only(bottom=10)

            ),

            ft.Text(help_content.get(view_key, "Información no disponible"), size=13)

        ]

        if self.help_panel.page:  # Only update if added to page

            self.help_panel.update()



    def _on_tab_change(self, e):

        tab_index = self.tabs.selected_index

        if tab_index == 0:

            self.tab_content.content = self.quick_actions

        elif tab_index == 1:

            self.tab_content.content = self.help_panel

        elif tab_index == 2:

            self.tab_content.content = self.log_panel

        if self.tab_content.page:  # Only update if added to page

            self.tab_content.update()



    def _pick_files(self, e):

        self.file_picker.pick_files(

            allow_multiple=True,

            allowed_extensions=["csv", "txt", "xlsx"],

            file_type=ft.FilePickerFileType.CUSTOM

        )



    def _handle_file_pick_result(self, e: ft.FilePickerResultEvent):

        if e.files:

            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            for file in e.files:
                try:
                    src = file.path
                    # Copiar a carpeta persistente con timestamp para evitar colisiones
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    base = os.path.basename(src)
                    dest = os.path.join(data_dir, f"{ts}_{base}")
                    shutil.copy2(src, dest)
                    if dest not in self.uploaded_files:
                        self.uploaded_files.append(dest)
                    self._log(f"Archivo cargado: {os.path.basename(dest)}")
                except Exception as ex:
                    try:
                        self._log(f"Error guardando archivo: {ex}")
                    except Exception:
                        pass

            

            # Si estamos en la vista de archivos, actualizar la lista

            if self.last_view == "files":
                self._refresh_files_list()

            

            # Si no hay archivo actual seleccionado, usar el primero

            if self.current_df is None and self.uploaded_files:
                self._load_file_data(self.uploaded_files[0])



    def _refresh_files_list(self):

        self.files_list_view.controls.clear()

        try:
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            files = []
            for fn in os.listdir(data_dir):
                if fn.lower().endswith((".csv", ".txt", ".xlsx")):
                    p = os.path.join(data_dir, fn)
                    try:
                        mt = os.path.getmtime(p)
                    except Exception:
                        mt = 0.0
                    files.append((p, mt))
            # Filtro de búsqueda
            try:
                q = (getattr(self, 'data_search', None).value or '').strip().lower() if getattr(self, 'data_search', None) else ''
            except Exception:
                q = ''
            if q:
                files = [(p, mt) for (p, mt) in files if q in os.path.basename(p).lower()]
            from datetime import datetime as _dt
            groups = {}
            for p, mt in files:
                base = os.path.basename(p)
                try:
                    date_key = _dt.strptime(base[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
                except Exception:
                    date_key = _dt.fromtimestamp(mt).strftime("%Y-%m-%d")
                groups.setdefault(date_key, []).append((p, mt))
            for date_key in sorted(groups.keys(), reverse=True):
                items = sorted(groups[date_key], key=lambda x: x[1], reverse=True)
                # Filtrar por favoritos si está activo
                try:
                    fav_only = bool(getattr(self, 'data_favs_only_cb', None) and getattr(self.data_favs_only_cb, 'value', False))
                except Exception:
                    fav_only = False
                filtered = []
                for p, _mt in items:
                    file_name = os.path.basename(p)
                    # Estado de favorito
                    is_fav = False
                    try:
                        is_fav = bool((self.data_favorites or {}).get(p, False))
                    except Exception:
                        is_fav = False
                    if fav_only and not is_fav:
                        continue
                    filtered.append((p, is_fav))
                if not filtered:
                    continue
                # Agregar cabecera de fecha solo si hay elementos
                self.files_list_view.controls.append(ft.Text(date_key, weight="bold"))
                for p, is_fav in filtered:
                    file_name = os.path.basename(p)
                    star_icon = ft.Icons.STAR if is_fav else ft.Icons.STAR_BORDER_ROUNDED
                    star_color = "#f1c40f" if is_fav else "#bdc3c7"
                    file_card = ft.Container(
                        content=ft.Row(
                            controls=[
                                ft.IconButton(icon=star_icon, icon_color=star_color, tooltip="Favorito", on_click=lambda e, path=p: self._toggle_data_favorite(path)),
                                ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=24),
                                ft.Column(
                                    controls=[
                                        ft.Text(file_name, weight="bold", size=14),
                                        ft.Text(p, size=12, color="#7f8c8d"),
                                    ],
                                    expand=True,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE_ROUNDED,
                                    tooltip="Eliminar archivo",
                                    on_click=lambda e, path=p: self._remove_file(path),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.VISIBILITY_ROUNDED,
                                    tooltip="Seleccionar para análisis",
                                    on_click=lambda e, path=p: self._load_file_data(path),
                                ),
                            ],
                            alignment="space_between",
                        ),
                        padding=15,
                        border_radius=10,
                        bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
                        on_hover=lambda e: self._on_file_hover(e),
                    )
                    self.files_list_view.controls.append(file_card)
        except Exception:
            self.files_list_view.controls.append(ft.Text("No se pudieron listar datos persistidos.", size=14))

        if self.files_list_view.page:  # Only update if added to page
            self.files_list_view.update()

            file_name = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]

            

            file_card = ft.Container(

                content=ft.Row(

                    controls=[

                        ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=24),

                        ft.Column(

                            controls=[

                                ft.Text(file_name, weight="bold", size=14),

                                ft.Text(file_path, size=12, color="#7f8c8d"),

                            ],

                            expand=True,

                        ),

                        ft.IconButton(

                            icon=ft.Icons.DELETE_OUTLINE_ROUNDED,

                            tooltip="Eliminar archivo",

                            on_click=lambda e, path=file_path: self._remove_file(path),

                        ),

                        ft.IconButton(

                            icon=ft.Icons.VISIBILITY_ROUNDED,

                            tooltip="Seleccionar para análisis",

                            on_click=lambda e, path=file_path: self._load_file_data(path),

                        ),

                    ],

                    alignment="space_between",

                ),

                padding=15,

                border_radius=10,

                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                on_hover=lambda e: self._on_file_hover(e),

            )

            self.files_list_view.controls.append(file_card)

        

        if self.files_list_view.page:  # Only update if added to page

            self.files_list_view.update()



    def _on_file_hover(self, e):

        if e.data == "true":

            e.control.bgcolor = ft.Colors.with_opacity(0.1, self._accent_ui())

        else:

            e.control.bgcolor = ft.Colors.with_opacity(0.05, self._accent_ui())

        e.control.update()



    def _remove_file(self, file_path):

        if file_path in self.uploaded_files:

            self.uploaded_files.remove(file_path)

            self._log(f"Archivo eliminado: {file_path}")

            

            # Si el archivo eliminado era el actual, limpiar current_df

            if self.current_df is not None and file_path in self.file_data_storage:

                del self.file_data_storage[file_path]

                if not self.uploaded_files:

                    self.current_df = None

                else:

                    self._load_file_data(self.uploaded_files[0])

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass

        self._refresh_files_list()

    # ==== Favoritos de datos ====
    def _data_favorites_path(self) -> str:
        try:
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            return os.path.join(data_dir, "data_favorites.json")
        except Exception:
            return "data_favorites.json"

    def _load_data_favorites(self) -> Dict[str, bool]:
        try:
            import json
            path = self._data_favorites_path()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_data_favorites(self):
        try:
            import json
            path = self._data_favorites_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.data_favorites or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _toggle_data_favorite(self, path: str):
        try:
            cur = bool((self.data_favorites or {}).get(path, False))
            self.data_favorites[path] = not cur
            self._save_data_favorites()
        except Exception:
            pass
        self._refresh_files_list()

    def _toggle_data_favs_filter(self):
        try:
            self.data_show_favs_only = bool(getattr(self, 'data_favs_only_cb', None) and getattr(self.data_favs_only_cb, 'value', False))
        except Exception:
            self.data_show_favs_only = False
        try:
            self.page.client_storage.set("data_favs_only", self.data_show_favs_only)
        except Exception:
            pass
        self._refresh_files_list()



    def _load_file_data(self, file_path):

        try:

            if file_path.endswith('.csv'):

                df = pd.read_csv(file_path)

            elif file_path.endswith('.xlsx'):

                df = pd.read_excel(file_path)

            else:

                df = pd.read_csv(file_path)



            # Detectar columnas numéricas

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not numeric_cols:

                self._log(f"Archivo inválido: {file_path} no tiene columnas numéricas")

                self.page.snack_bar = ft.SnackBar(

                    content=ft.Text("⚠️ El archivo no contiene columnas numéricas para análisis"),

                    bgcolor="#e74c3c",

                )

                self.page.snack_bar.open = True

                self.page.update()

                return



            # Columna de tiempo y señales

            time_candidates = [c for c in numeric_cols if "t" in c.lower()]

            self.default_time_col = time_candidates[0] if time_candidates else numeric_cols[0]

            self.default_signal_cols = [c for c in numeric_cols if c != self.default_time_col]



            self.file_data_storage[file_path] = df

            self.current_df = df

            self._log(f"Datos cargados: {file_path} ({len(df)} filas, {len(df.columns)} columnas)")



            # Cambiar automáticamente a vista de análisis

            self._select_menu("analysis", force_rebuild=True)



        except Exception as e:

            self._log(f"Error al cargar archivo {file_path}: {str(e)}")

            self.page.snack_bar = ft.SnackBar(

                content=ft.Text(f"Error al cargar archivo: {str(e)}"),

                bgcolor="#e74c3c",

            )

            self.page.snack_bar.open = True

            self.page.update()



    def _log(self, message):

        timestamp = datetime.now().strftime("%H:%M:%S" if self.clock_24h else "%I:%M:%S %p")

        log_entry = ft.Text(f"[{timestamp}] {message}", size=12)

        self.log_panel.controls.append(log_entry)

        if self.log_panel.page:  # Only update if added to page

            self.log_panel.update()

        print(f"[LOG] {message}")



    def _toggle_theme(self, e):

        self.is_dark_mode = not self.is_dark_mode

        self.page.client_storage.set("is_dark_mode", self.is_dark_mode)

        self._apply_theme()

        self._update_theme_for_all_components()

        self._log(f"Tema cambiado a: {'oscuro' if self.is_dark_mode else 'claro'}")



    def _toggle_clock_format(self, e):

        self.clock_24h = not self.clock_24h

        self.page.client_storage.set("clock_24h", self.clock_24h)

        self.clock_text.value = self._get_current_time()

        self.clock_text.update()

        self._log(f"Formato de hora cambiado a: {'24h' if self.clock_24h else '12h'}")



    async def _start_clock_timer(self):

        while True:

            self.clock_text.value = self._get_current_time()

            if self.clock_text.page:

                self.clock_text.update()

            await asyncio.sleep(1)



    def _toggle_panel(self, e):

        self.is_panel_expanded = not self.is_panel_expanded

        

        # Actualizar icono del botón

        e.control.icon = (

            ft.Icons.CHEVRON_LEFT_ROUNDED 

            if self.is_panel_expanded 

            else ft.Icons.CHEVRON_RIGHT_ROUNDED

        )

        

        # Actualizar el panel

        self.control_panel.width = 350 if self.is_panel_expanded else 65

        self.control_panel.padding = ft.padding.all(20 if self.is_panel_expanded else 10)

        

        # Actualizar visibilidad del contenido

        panel_header = self.control_panel.content.controls[0]

        panel_header.controls[0].visible = self.is_panel_expanded  # Título

        

        # Actualizar resto del contenido

        content_container = self.control_panel.content.controls[-1]

        content_container.visible = self.is_panel_expanded

        

        # Actualizar el panel

        self.control_panel.update()



    def _toggle_config_panel(self, e):

        self.config_expanded = not self.config_expanded

        self.config_container.visible = self.config_expanded

        e.control.icon = (

            ft.Icons.ARROW_DROP_DOWN_CIRCLE if self.config_expanded else ft.Icons.ARROW_RIGHT

        )

        if self.page:

            self.page.update()



    def _update_multi_chart(self, e=None, normalize=True):

        """

        Genera gráfica combinada de FFTs seleccionadas.

        - normalize=True: escala cada señal entre 0–1 para ver todas.

        """

        try:

            time_col = self.time_dropdown.value

            t = self.current_df[time_col].to_numpy()

            selected_signals = [cb.label for cb in self.signal_checkboxes if cb.value]



            if not selected_signals:

                chart = ft.Text("⚠️ No hay señales seleccionadas")

            else:

                plt.style.use('dark_background' if self.is_dark_mode else 'seaborn-v0_8-whitegrid')

                fig, ax = plt.subplots(figsize=(12, 5))



                # Vista en dBV real opcional (aplica a todas las curvas)
                try:
                    use_dbv = bool(getattr(self, 'db_scale_cb', None) and getattr(self.db_scale_cb, 'value', False))
                except Exception:
                    use_dbv = False
                # Calibración para dBV
                try:
                    sens_unit = getattr(self, 'sens_unit_dd', None).value if getattr(self, 'sens_unit_dd', None) else 'mV/g'
                except Exception:
                    sens_unit = 'mV/g'
                try:
                    sens_val = float(getattr(self, 'sensor_sens_field', None).value) if getattr(self, 'sensor_sens_field', None) else 100.0
                except Exception:
                    sens_val = 100.0
                try:
                    gain_vv = float(getattr(self, 'gain_field', None).value) if getattr(self, 'gain_field', None) else 1.0
                except Exception:
                    gain_vv = 1.0

                # Filtros de frecuencia visuales
                try:
                    fmin_ui = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.0
                except Exception:
                    fmin_ui = 0.0
                try:
                    fmax_ui = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
                except Exception:
                    fmax_ui = None
                zoom_range = self._fft_zoom_range
                zmin = zmax = None
                if zoom_range and len(zoom_range) == 2 and zoom_range[1] > zoom_range[0]:
                    try:
                        zmin, zmax = float(zoom_range[0]), float(zoom_range[1])
                    except Exception:
                        zmin = zmax = None

                for sig in selected_signals:

                    y = self.current_df[sig].to_numpy()

                    N = len(y)

                    if N < 2:

                        continue

                    T = t[1] - t[0]

                    yf = np.fft.fft(y)

                    xf = np.fft.fftfreq(N, T)[:N // 2]

                    mag_acc = 2.0 / N * np.abs(yf[0:N // 2])

                    mag_vel_mm = np.zeros_like(mag_acc)

                    mag_vel_mm[xf > 0] = (mag_acc[xf > 0] / (2 * np.pi * xf[xf > 0])) * 1000



                    # Aplicar máscara de frecuencia
                    mask = xf >= max(0.0, fmin_ui)
                    if fmax_ui and fmax_ui > 0:
                        mask = mask & (xf <= fmax_ui)
                    if zmin is not None:
                        mask = mask & (xf >= zmin) & (xf <= zmax)

                    if use_dbv:
                        if sens_unit == 'mV/g':
                            sens_v_per_g = sens_val * 1e-3
                            V_amp = (mag_acc / 9.80665) * sens_v_per_g * gain_vv
                        elif sens_unit == 'V/g':
                            V_amp = (mag_acc / 9.80665) * sens_val * gain_vv
                        elif sens_unit == 'mV/(mm/s)':
                            V_amp = mag_vel_mm * (sens_val * 1e-3) * gain_vv
                        elif sens_unit == 'V/(mm/s)':
                            V_amp = mag_vel_mm * sens_val * gain_vv
                        else:
                            V_amp = mag_vel_mm * 0.0
                        eps = 1e-12
                        yplot = 20.0 * np.log10(np.maximum(np.asarray(V_amp, dtype=float), eps) / 1.0)
                        ax.plot(xf[mask], yplot[mask], linewidth=2, label=sig)
                    else:
                        if normalize and mag_vel_mm.max() > 0:
                            mag_vel_mm = mag_vel_mm / mag_vel_mm.max()
                        ax.plot(xf[mask], mag_vel_mm[mask], linewidth=2, label=sig)



                ax.set_title("FFT combinada de señales")

                ax.set_xlabel("Frecuencia (Hz)")

                if use_dbv:
                    ax.set_ylabel("Nivel [dBV]")
                    # Rango Y en dBV si se definió
                    try:
                        ymin = float(self.db_ymin_field.value) if getattr(self, 'db_ymin_field', None) and getattr(self.db_ymin_field, 'value', '') != '' else None
                    except Exception:
                        ymin = None
                    try:
                        ymax = float(self.db_ymax_field.value) if getattr(self, 'db_ymax_field', None) and getattr(self.db_ymax_field, 'value', '') != '' else None
                    except Exception:
                        ymax = None
                    if ymin is not None or ymax is not None:
                        cur = ax.get_ylim()
                        ax.set_ylim(ymin if ymin is not None else cur[0], ymax if ymax is not None else cur[1])
                else:
                    ax.set_ylabel("Velocidad [mm/s]" if not normalize else "Amplitud normalizada")

                try:
                    if zmin is not None:
                        ax.set_xlim(left=zmin, right=zmax)
                    elif fmax_ui and fmax_ui > 0:
                        ax.set_xlim(left=0.0, right=float(fmax_ui))
                    if zmin is None and fmin_ui and fmin_ui > 0:
                        cur = ax.get_xlim()
                        ax.set_xlim(left=float(fmin_ui), right=cur[1])
                except Exception:
                    pass
                ax.legend(ncol=2, fontsize=8)



                chart = _create_mpl_chart(fig, expand=True, isolated=True)

                plt.close(fig)

            self.multi_chart_container.content = chart

            if self.multi_chart_container.page:

                self.multi_chart_container.update()

        except Exception as ex:

            self._log(f"Error en gráfica combinada: {ex}")

            self.multi_chart_container.content = ft.Text(f"Error en gráfica combinada: {ex}")

            if self.multi_chart_container.page:

                self.multi_chart_container.update()



# =========================

#   Apartado: Diagnóstico

# =========================

import flet as ft



def diagnostico_view(page: ft.Page, on_generate=None):

    opciones = [

        "Vibraciones generales", "FFT señal completa", "FFT por ventana", "Valores RMS",

        "Valor pico", "Valor pico-pico", "Espectro de frecuencias", "Velocidad crítica",

        "Resonancia", "Distorsión de carcasa", "Armónicos", "IPS (pulgadas/seg)",

        "Comparación de espectros", "Tendencias históricas", "Temperatura relacionada",

        "Desbalanceo", "Desalineación", "Rodamientos", "Excentricidad",

        "Falla catastrófica", "Engranes", "Holguras", "Fuerzas axiales",

        "Modos propios", "Filtros banda", "Cepstrum", "Envelope",

        "Top-N picos", "Order tracking", "Overspeed",

    ]



    chips = [ft.FilterChip(label=op, selected=False) for op in opciones]



    def set_all(val: bool):

        for c in chips:

            c.selected = val

            c.update()



    def generar(_):

        seleccionadas = [c.label for c in chips if c.selected]

        if on_generate:

            on_generate(seleccionadas)

        else:

            page.snack_bar = ft.SnackBar(ft.Text(f"Elegidas: {len(seleccionadas)}"))

            page.snack_bar.open = True

            page.update()



    # --- Dividir en columnas ---

    per_col = 8

    def chunk(lst, n):

        for i in range(0, len(lst), n):

            yield lst[i:i+n]



    columnas = []

    for grupo in chunk(chips, per_col):

        columnas.append(

            ft.Container(

                width=260,

                padding=10,

                content=ft.Column(

                    controls=grupo,

                    scroll="auto",

                )

            )

        )



    # --- Scroll horizontal de columnas ---

    opciones_scroller = ft.Container(

        height=320,

        content=ft.Row(

            controls=columnas,

            spacing=12,

            scroll="auto",

            vertical_alignment=ft.CrossAxisAlignment.START

        )

    )



    acciones = ft.Row(

        controls=[

            ft.TextButton("Seleccionar todo", on_click=lambda e: set_all(True)),

            ft.TextButton("Limpiar", on_click=lambda e: set_all(False)),

            ft.ElevatedButton(

                "Generar gráficas",

                icon=ft.icons.ANALYTICS,

                on_click=generar,

            ),

        ],

        wrap=True,

        alignment=ft.MainAxisAlignment.END,

    )



    return ft.Column(

        controls=[

            ft.Text("Opciones de diagnóstico", size=20, weight="bold"),

            opciones_scroller,

            acciones,

        ],

        expand=True,

        spacing=12,

    )



# =========================

#   Punto de entrada

# =========================

def main(page: ft.Page):

    print("App iniciada")

    app = MainApp(page)

    page.add(app.content)



if __name__ == "__main__":

    ft.app(target=main)



