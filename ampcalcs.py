# SIGMA ELETRÔNICA & TECNOLOGIA
# ampcalcs.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import math
import re
import base64
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ----------------------------
# Constants
# ----------------------------
VT = 0.026  # 26mV (25°C)
E24 = np.array([
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
])

# ----------------------------
# Helper Functions
# ----------------------------
def e24_round(value: float) -> float:
    if value <= 0:
        return value
    decade = 10 ** math.floor(math.log10(value))
    mant = value / decade
    idx = int(np.argmin(np.abs(E24 - mant)))
    return float(E24[idx] * decade)

def safe_float(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

# ----------------------------
# Data Classes
# ----------------------------
@dataclass
class TransistorModel:
    name: str
    kind: str              # "NPN" or "PNP"
    role: str              # "VAS" | "DRIVER" | "OUTPUT" | "INPUT" | "PREDRIVER"
    beta_typ: float
    beta_min: float
    vbe_typ: float
    vceo_V: float = 0.0
    ic_max_A: float = 0.0
    pd_W: float = 0.0
    ft_MHz: float = 0.0
    notes: str = ""
    spice_text: str = ""   # optional: .MODEL or .SUBCKT text blob

@dataclass
class AmpInputs:
    # Supply / target
    vcc: float = 35.0            # magnitude ±Vcc
    r_load: float = 4.0
    p_out: float = 100.0

    # Topology choices
    ef_level: str = "EF2"        # EF1/EF2/EF3
    vas_type: str = "Simple"     # Simple | Enhanced
    n_output_pairs: int = 1

    # Output stage
    re_out: float = 0.22
    vre_idle_mV: float = 22.0

    # Currents
    itail_mA: float = 1.0
    ivas_mA: float = 10.0

    # Compensation
    sr_target_Vus: float = 30.0

    # Practical swing/headroom knobs
    headroom_V: float = 5.0      # per rail estimate
    extra_headroom_enhanced_V: float = 0.8

    # Stability suggestions knobs
    zobel_R: float = 10.0
    zobel_C_nF: float = 100.0
    out_L_uH: float = 1.0
    out_R_damp: float = 8.2
    base_stop_driver_ohm: float = 4.7
    base_stop_output_ohm: float = 2.2

    # Differential degeneration suggestion
    re_dif_ohm: float = 150.0

    # Selected devices
    input_device: str = "MPSA92"
    predriver_device_npn: str = "BD139"
    predriver_device_pnp: str = "BD140"
    vas_device_npn: str = "MJE340"
    vas_device_pnp: str = "MJE350"
    driver_device_npn: str = "BD139"
    driver_device_pnp: str = "BD140"
    output_device_npn: str = "2SC5200"
    output_device_pnp: str = "2SA1943"

@dataclass
class AmpResults:
    # Output fundamentals
    vrms: float
    vpk_required: float
    irms: float
    ipk_total: float
    ipk_per_device: float

    # Swing estimates
    vpk_est_available: float
    pmax_est_W: float

    # Re and bias-related
    vre_pk_per_device: float
    iq_A: float

    # Drive currents (peak)
    ib_out_pk_per_device: float
    ib_drv_pk_total: float
    ib_vas_pk_total: float

    # CCS mirror Rset
    rset_tail: float
    rset_vas: float
    rset_tail_e24: float
    rset_vas_e24: float

    # VAS compensation
    cdom_pF: float
    sr_est_Vus: float

    # Bias estimate based on EF level
    vbias_est: float
    vbe_mult_ratio: float
    rbot_suggest: float
    rtop_suggest: float
    rtop_e24: float
    n_vbe_junctions: int

    # Input stage indicators
    gm_each_S: float
    ic_each_A: float

    # Operating points (approx)
    operating_points: List[Dict[str, float]]

    # Thermal + alerts
    pdiss_max_transistor_W: float
    pdiss_out_avg_W: float
    pdiss_drv_avg_W: float
    pdiss_pre_avg_W: float
    pdiss_vas_avg_W: float
    alerts: List[str]

    # Stability suggestions
    stability_text: str

# ----------------------------
# Transistor Database
# ----------------------------
class TransistorDB:
    def __init__(self):
        self.models: Dict[str, TransistorModel] = {}
        self.load_defaults()
    
    def load_defaults(self):
        """Carrega os transistores padrão"""
        defaults = [
            # OUTPUT (2SC5200/2SA1943)
            TransistorModel("2SC5200", "NPN", "OUTPUT", beta_typ=80, beta_min=30, vbe_typ=0.65, vceo_V=230, ic_max_A=15, pd_W=150, ft_MHz=30,
                           notes="Preset típico para 2SC5200."),
            TransistorModel("2SA1943", "PNP", "OUTPUT", beta_typ=80, beta_min=30, vbe_typ=0.65, vceo_V=230, ic_max_A=15, pd_W=150, ft_MHz=30,
                           notes="Preset típico para 2SA1943."),
            
            # DRIVERS (BD139/BD140)
            TransistorModel("BD139", "NPN", "DRIVER", beta_typ=100, beta_min=50, vbe_typ=0.65, vceo_V=80, ic_max_A=1.5, pd_W=12.5, ft_MHz=100,
                           notes="Driver comum."),
            TransistorModel("BD140", "PNP", "DRIVER", beta_typ=100, beta_min=50, vbe_typ=0.65, vceo_V=80, ic_max_A=1.5, pd_W=12.5, ft_MHz=100,
                           notes="Driver comum."),
            
            # VAS (MJE340/MJE350)
            TransistorModel("MJE340", "NPN", "VAS", beta_typ=80, beta_min=40, vbe_typ=0.65, vceo_V=300, ic_max_A=0.5, pd_W=20, ft_MHz=10,
                           notes="VAS comum."),
            TransistorModel("MJE350", "PNP", "VAS", beta_typ=80, beta_min=40, vbe_typ=0.65, vceo_V=300, ic_max_A=0.5, pd_W=20, ft_MHz=10,
                           notes="VAS comum."),
            
            # Small-signal
            TransistorModel("BC546B", "NPN", "VAS", beta_typ=200, beta_min=100, vbe_typ=0.65, vceo_V=65, ic_max_A=0.1, pd_W=0.5, ft_MHz=150,
                           notes="Para VAS leve/espelhos."),
            TransistorModel("BC556B", "PNP", "VAS", beta_typ=200, beta_min=100, vbe_typ=0.65, vceo_V=65, ic_max_A=0.1, pd_W=0.5, ft_MHz=150,
                           notes="Para VAS leve/espelhos."),
            
            # INPUT
            TransistorModel("MPSA92", "PNP", "INPUT", beta_typ=150, beta_min=80, vbe_typ=0.65, vceo_V=300, ic_max_A=0.05, pd_W=0.625, ft_MHz=50,
                           notes="PNP alta tensão para par diferencial."),
            TransistorModel("2N5401", "PNP", "INPUT", beta_typ=120, beta_min=60, vbe_typ=0.65, vceo_V=150, ic_max_A=0.1, pd_W=0.625, ft_MHz=50,
                           notes="PNP alta tensão."),
            TransistorModel("2N5551", "NPN", "INPUT", beta_typ=120, beta_min=60, vbe_typ=0.65, vceo_V=160, ic_max_A=0.1, pd_W=0.625, ft_MHz=100,
                           notes="NPN alta tensão."),
            
            # PREDRIVER
            TransistorModel("MPSA42", "NPN", "PREDRIVER", beta_typ=120, beta_min=60, vbe_typ=0.65, vceo_V=300, ic_max_A=0.05, pd_W=0.625, ft_MHz=50,
                           notes="Predriver NPN."),
            
            # DRIVERS alternativos
            TransistorModel("2SC4793", "NPN", "DRIVER", beta_typ=100, beta_min=50, vbe_typ=0.65, vceo_V=230, ic_max_A=2.0, pd_W=20, ft_MHz=100,
                           notes="Driver robusto."),
            TransistorModel("2SA1837", "PNP", "DRIVER", beta_typ=100, beta_min=50, vbe_typ=0.65, vceo_V=230, ic_max_A=2.0, pd_W=20, ft_MHz=100,
                           notes="Driver robusto."),
        ]
        
        for preset in defaults:
            self.models[preset.name] = preset
    
    def list_by_role(self, role: str, kind: str = None) -> List[TransistorModel]:
        """Lista transistores por papel (role)"""
        filtered = [m for m in self.models.values() if m.role == role]
        if kind:
            filtered = [m for m in filtered if m.kind == kind]
        return sorted(filtered, key=lambda x: x.name.lower())
    
    def get(self, name: str) -> Optional[TransistorModel]:
        """Obtém um transistor pelo nome"""
        return self.models.get(name)
    
    def add(self, model: TransistorModel):
        """Adiciona um novo transistor"""
        self.models[model.name] = model
    
    def delete(self, name: str):
        """Remove um transistor"""
        if name in self.models:
            del self.models[name]

# ----------------------------
# Engineering Calculations
# ----------------------------
def junction_count_for_ef(ef_level: str) -> int:
    """Conta junções Vbe para bias"""
    return {"EF1": 2, "EF2": 4, "EF3": 6}.get(ef_level, 4)

def compute_amp(inp: AmpInputs, db: TransistorDB) -> AmpResults:
    """Executa todos os cálculos do amplificador"""
    alerts: List[str] = []
    
    vcc = float(inp.vcc)
    r = float(inp.r_load)
    p = float(inp.p_out)
    
    # Output required
    vrms = math.sqrt(p * r)
    vpk_required = vrms * math.sqrt(2)
    irms = vrms / r
    ipk_total = vpk_required / r
    ipk_per_device = ipk_total / max(1, int(inp.n_output_pairs))
    
    # Headroom and available peak
    headroom = float(inp.headroom_V)
    if inp.vas_type == "Enhanced":
        headroom = max(1.0, headroom - float(inp.extra_headroom_enhanced_V))
    
    vpk_est_available = max(0.0, vcc - headroom)
    pmax_est_W = (vpk_est_available ** 2) / (2.0 * r) if vpk_est_available > 0 else 0.0
    
    if pmax_est_W + 1e-6 < p:
        alerts.append(
            f"Headroom: com ±{vcc:.1f}V e headroom {headroom:.1f}V, "
            f"Vpk disponível≈{vpk_est_available:.1f}V → Pmax≈{pmax_est_W:.0f}W (< {p:.0f}W)."
        )
    
    # Devices
    out_npn = db.get(inp.output_device_npn)
    out_pnp = db.get(inp.output_device_pnp)
    drv_npn = db.get(inp.driver_device_npn)
    drv_pnp = db.get(inp.driver_device_pnp)
    pre_npn = db.get(inp.predriver_device_npn)
    pre_pnp = db.get(inp.predriver_device_pnp)
    vas_npn = db.get(inp.vas_device_npn)
    vas_pnp = db.get(inp.vas_device_pnp)
    in_dev = db.get(inp.input_device)
    
    def pick_beta_min(m: Optional[TransistorModel], fallback: float) -> float:
        if not m:
            return fallback
        return max(5.0, float(m.beta_min))
    
    def pick_vbe(m: Optional[TransistorModel], fallback: float) -> float:
        if not m:
            return fallback
        return float(m.vbe_typ) if m.vbe_typ > 0 else fallback
    
    beta_out_min = min(pick_beta_min(out_npn, 30), pick_beta_min(out_pnp, 30))
    beta_drv_min = min(pick_beta_min(drv_npn, 50), pick_beta_min(drv_pnp, 50))
    beta_pre_min = min(pick_beta_min(pre_npn, 80), pick_beta_min(pre_pnp, 80))
    vbe_typ = np.mean([pick_vbe(out_npn, 0.65), pick_vbe(out_pnp, 0.65),
                      pick_vbe(drv_npn, 0.65), pick_vbe(drv_pnp, 0.65),
                      pick_vbe(pre_npn, 0.65), pick_vbe(pre_pnp, 0.65),
                      pick_vbe(vas_npn, 0.65), pick_vbe(vas_pnp, 0.65),
                      pick_vbe(in_dev, 0.65)])
    
    # Current capability checks
    if out_npn and out_npn.ic_max_A > 0 and ipk_per_device > out_npn.ic_max_A * 0.9:
        alerts.append(f"Corrente pico≈{ipk_per_device:.2f}A pode exceder Ic_max do {out_npn.name}.")
    if out_pnp and out_pnp.ic_max_A > 0 and ipk_per_device > out_pnp.ic_max_A * 0.9:
        alerts.append(f"Corrente pico≈{ipk_per_device:.2f}A pode exceder Ic_max do {out_pnp.name}.")
    
    # Re and idle current
    re_out = float(inp.re_out)
    vre_pk_per_device = ipk_per_device * re_out
    iq_A = (inp.vre_idle_mV / 1000.0) / re_out
    
    # Drive currents (peak)
    ib_out_pk_per_device = ipk_per_device / beta_out_min
    ib_out_pk_total_half = ib_out_pk_per_device * max(1, int(inp.n_output_pairs))
    
    ef = inp.ef_level
    if ef == "EF1":
        ib_vas_pk_total = ib_out_pk_total_half
        ib_drv_pk_total = 0.0
    elif ef == "EF2":
        ic_driver_pk = ib_out_pk_total_half
        ib_driver_base_pk = ic_driver_pk / beta_drv_min
        ib_vas_pk_total = ib_driver_base_pk
        ib_drv_pk_total = ib_driver_base_pk
    else:  # EF3
        ic_driver_pk = ib_out_pk_total_half
        ib_driver_base_pk = ic_driver_pk / beta_drv_min
        ic_predriver_pk = ib_driver_base_pk
        ib_predriver_base_pk = ic_predriver_pk / beta_pre_min
        ib_vas_pk_total = ib_predriver_base_pk
        ib_drv_pk_total = ib_driver_base_pk
    
    # VAS drive factor
    vas_drive_factor = 1.35 if inp.vas_type == "Enhanced" else 1.60
    ib_vas_pk_total *= vas_drive_factor
    
    # CCS mirror Rset
    itail_A = inp.itail_mA / 1000.0
    ivas_A = inp.ivas_mA / 1000.0
    
    if inp.vas_type == "Enhanced" and inp.ivas_mA < 6.0:
        alerts.append("VAS Enhanced com Ivas < 6mA pode limitar drive.")
    if inp.vas_type == "Simple" and inp.ivas_mA < 4.0:
        alerts.append("VAS simples com Ivas < 4mA pode limitar SR/drive.")
    
    rset_tail = (vcc - vbe_typ) / max(1e-9, itail_A)
    rset_vas = (vcc - vbe_typ) / max(1e-9, ivas_A)
    
    # VAS compensation
    cdom_F = ivas_A / (inp.sr_target_Vus * 1e6)
    cdom_pF = cdom_F * 1e12
    sr_est_Vus = ivas_A / cdom_F / 1e6 if cdom_F > 0 else 0.0
    
    # Bias junction count + Vbias estimate
    n_vbe = junction_count_for_ef(inp.ef_level)
    vbias_est = n_vbe * vbe_typ + 2.0 * iq_A * re_out
    vbe_mult_ratio = (vbias_est / max(1e-6, vbe_typ)) - 1.0
    rbot = 1000.0
    rtop = max(0.0, vbe_mult_ratio * rbot)
    
    # Input stage
    ic_each = itail_A / 2.0
    gm_each = ic_each / VT
    
    # Power / thermal calculations
    def pavg_halfcycle(Ipk: float, Vcc_eff: float, Vpk: float) -> float:
        return max(0.0, Ipk * (Vcc_eff / math.pi - Vpk / 4.0))
    
    # Output transistor (per device)
    pdiss_out_signal_W = pavg_halfcycle(ipk_per_device, vcc, vpk_required)
    pdiss_out_idle_W = vcc * iq_A
    pdiss_out_avg_W = pdiss_out_signal_W + pdiss_out_idle_W
    
    # Driver / predriver dissipation
    vbe_out = float(np.mean([pick_vbe(out_npn, 0.65), pick_vbe(out_pnp, 0.65)]))
    vbe_drv = float(np.mean([pick_vbe(drv_npn, 0.65), pick_vbe(drv_pnp, 0.65)]))
    vbe_pre = float(np.mean([pick_vbe(pre_npn, 0.65), pick_vbe(pre_pnp, 0.65)]))
    
    pdiss_drv_avg_W = 0.0
    if ef in ("EF2", "EF3"):
        Vcc_eff_drv = max(0.0, vcc - (vbe_out + vbe_drv))
        ic_driver_pk = ib_out_pk_total_half
        pdiss_drv_avg_W = pavg_halfcycle(ic_driver_pk, Vcc_eff_drv, vpk_required)
    
    pdiss_pre_avg_W = 0.0
    if ef == "EF3":
        Vcc_eff_pre = max(0.0, vcc - (vbe_out + vbe_drv + vbe_pre))
        ic_predriver_pk = ib_drv_pk_total / beta_drv_min if beta_drv_min > 0 else 0.0
        pdiss_pre_avg_W = pavg_halfcycle(ic_predriver_pk, Vcc_eff_pre, vpk_required)
    
    # VAS transistor
    vce_vas_avg = max(2.0, vcc - headroom - (vpk_required / 2.0))
    pdiss_vas_avg_W = ivas_A * vce_vas_avg
    
    # Operating points
    vce_idle_hi = max(2.0, vcc - 2.0)
    op = []
    op.append({"stage": "OUTPUT", "ic_idle_A": iq_A, "vce_idle_V": vcc, "ic_peak_A": ipk_per_device, "vce_peak_V": max(0.2, vcc - vpk_required)})
    
    if ef in ("EF2", "EF3"):
        op.append({"stage": "DRIVER", "ic_idle_A": 0.0, "vce_idle_V": max(0.2, vcc - (vbe_out + vbe_drv)), 
                  "ic_peak_A": ib_out_pk_total_half, "vce_peak_V": max(0.2, vcc - vpk_required - (vbe_out + vbe_drv))})
    
    if ef == "EF3":
        op.append({"stage": "PREDRIVER", "ic_idle_A": 0.0, "vce_idle_V": max(0.2, vcc - (vbe_out + vbe_drv + vbe_pre)), 
                  "ic_peak_A": ib_drv_pk_total / beta_drv_min if beta_drv_min > 0 else 0.0, 
                  "vce_peak_V": max(0.2, vcc - vpk_required - (vbe_out + vbe_drv + vbe_pre))})
    
    op.append({"stage": f"VAS ({inp.vas_type})", "ic_idle_A": ivas_A, "vce_idle_V": vcc/2.0, 
              "ic_peak_A": ivas_A + ib_vas_pk_total, "vce_peak_V": max(0.2, vcc - headroom - vpk_required/2.0)})
    op.append({"stage": "DIFF (each)", "ic_idle_A": ic_each, "vce_idle_V": vce_idle_hi, 
              "ic_peak_A": max(ic_each, itail_A), "vce_peak_V": max(0.2, vce_idle_hi - 5.0)})
    
    # Thermal approx
    pdiss_max_transistor = (vcc * vcc) / (math.pi ** 2 * r) / max(1, int(inp.n_output_pairs))
    
    if inp.n_output_pairs == 1 and inp.r_load <= 4.0 and inp.p_out >= 80:
        alerts.append("1 par em 4Ω com potência alta é crítico termicamente. 2 pares recomendado.")
    
    # Stability recommendations
    stability_text = (
        f"- Zobel: {inp.zobel_R:.1f}Ω // {inp.zobel_C_nF:.0f}nF (saída)\n"
        f"- Indutor série: {inp.out_L_uH:.2f}µH com resistor {inp.out_R_damp:.1f}Ω\n"
        f"- Base stopper driver: {inp.base_stop_driver_ohm:.1f}Ω\n"
        f"- Base stopper saída: {inp.base_stop_output_ohm:.1f}Ω\n"
        f"- Re saída: {inp.re_out:.2f}Ω por transistor\n"
        "Valide em simulação/oscópio."
    )
    
    if cdom_pF < 100:
        alerts.append("Cdom < 100pF pode causar instabilidade.")
    if cdom_pF > 680:
        alerts.append("Cdom muito alto reduz banda.")
    
    alerts.append("Reserve ≥1V no espelho do tail e ≥3–5V na CCS do VAS.")
    
    return AmpResults(
        vrms=vrms,
        vpk_required=vpk_required,
        irms=irms,
        ipk_total=ipk_total,
        ipk_per_device=ipk_per_device,
        vpk_est_available=vpk_est_available,
        pmax_est_W=pmax_est_W,
        vre_pk_per_device=vre_pk_per_device,
        iq_A=iq_A,
        ib_out_pk_per_device=ib_out_pk_per_device,
        ib_drv_pk_total=ib_drv_pk_total,
        ib_vas_pk_total=ib_vas_pk_total,
        rset_tail=rset_tail,
        rset_vas=rset_vas,
        rset_tail_e24=e24_round(rset_tail),
        rset_vas_e24=e24_round(rset_vas),
        cdom_pF=cdom_pF,
        sr_est_Vus=sr_est_Vus,
        vbias_est=vbias_est,
        vbe_mult_ratio=vbe_mult_ratio,
        rbot_suggest=rbot,
        rtop_suggest=rtop,
        rtop_e24=e24_round(rtop),
        n_vbe_junctions=n_vbe,
        gm_each_S=gm_each,
        ic_each_A=ic_each,
        operating_points=op,
        pdiss_max_transistor_W=pdiss_max_transistor,
        pdiss_out_avg_W=pdiss_out_avg_W,
        pdiss_drv_avg_W=pdiss_drv_avg_W,
        pdiss_pre_avg_W=pdiss_pre_avg_W,
        pdiss_vas_avg_W=pdiss_vas_avg_W,
        alerts=alerts,
        stability_text=stability_text
    )

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(
        page_title="Sigma Eletrônica - Calculadora AB EFx",
        page_icon="🔊",
        layout="wide"
    )
    
    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = TransistorDB()
    if 'inputs' not in st.session_state:
        st.session_state.inputs = AmpInputs()
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Header
    st.title("🔊 Sigma Eletrônica - Calculadora AB EFx")
    st.markdown("""
    **Calculadora para amplificadores classe AB de 3 estágios**  
    *Topologia: EF1/EF2/EF3 com VAS simples ou enhanced*
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Projeto", "📈 Gráficos", "🗃️ Banco de Transistores", "💾 Exportar"])
    
    # Tab 1: Project inputs and results
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Parâmetros do Projeto")
            
            with st.expander("Alimentação e Carga", expanded=True):
                st.session_state.inputs.vcc = st.number_input(
                    "±Vcc (V)", min_value=10.0, max_value=100.0, value=35.0, step=1.0
                )
                st.session_state.inputs.r_load = st.number_input(
                    "Carga (Ω)", min_value=2.0, max_value=16.0, value=4.0, step=0.1
                )
                st.session_state.inputs.p_out = st.number_input(
                    "Potência alvo (W)", min_value=10.0, max_value=500.0, value=100.0, step=5.0
                )
            
            with st.expander("Topologia", expanded=True):
                col1a, col1b = st.columns(2)
                with col1a:
                    st.session_state.inputs.ef_level = st.selectbox(
                        "Topologia", ["EF1", "EF2", "EF3"], index=1
                    )
                    st.session_state.inputs.vas_type = st.selectbox(
                        "VAS", ["Simple", "Enhanced"], index=0
                    )
                with col1b:
                    st.session_state.inputs.n_output_pairs = st.number_input(
                        "Nº pares saída", min_value=1, max_value=8, value=1, step=1
                    )
            
            with st.expander("Estágio de Saída", expanded=True):
                col2a, col2b = st.columns(2)
                with col2a:
                    st.session_state.inputs.re_out = st.number_input(
                        "Re saída (Ω)", min_value=0.1, max_value=1.0, value=0.22, step=0.01
                    )
                with col2b:
                    st.session_state.inputs.vre_idle_mV = st.number_input(
                        "Vre repouso (mV)", min_value=5.0, max_value=50.0, value=22.0, step=1.0
                    )
            
            with st.expander("Correntes", expanded=True):
                col3a, col3b = st.columns(2)
                with col3a:
                    st.session_state.inputs.itail_mA = st.number_input(
                        "Itail (mA)", min_value=0.5, max_value=10.0, value=1.0, step=0.1
                    )
                with col3b:
                    st.session_state.inputs.ivas_mA = st.number_input(
                        "Ivas (mA)", min_value=2.0, max_value=50.0, value=10.0, step=1.0
                    )
            
            with st.expander("Headroom e SR", expanded=True):
                col4a, col4b = st.columns(2)
                with col4a:
                    st.session_state.inputs.headroom_V = st.number_input(
                        "Headroom base (V)", min_value=1.0, max_value=15.0, value=5.0, step=0.5
                    )
                    st.session_state.inputs.extra_headroom_enhanced_V = st.number_input(
                        "Ganho Enhanced (V)", min_value=0.0, max_value=5.0, value=0.8, step=0.1
                    )
                with col4b:
                    st.session_state.inputs.sr_target_Vus = st.number_input(
                        "SR alvo (V/µs)", min_value=5.0, max_value=100.0, value=30.0, step=5.0
                    )
                    st.session_state.inputs.re_dif_ohm = st.number_input(
                        "Re dif (Ω)", min_value=0.0, max_value=1000.0, value=150.0, step=10.0
                    )
        
        with col2:
            st.header("Seleção de Transistores")
            
            db = st.session_state.db
            
            # Get available transistors by role
            input_devices = db.list_by_role("INPUT")
            vas_npn_devices = db.list_by_role("VAS", "NPN")
            vas_pnp_devices = db.list_by_role("VAS", "PNP")
            driver_npn_devices = db.list_by_role("DRIVER", "NPN")
            driver_pnp_devices = db.list_by_role("DRIVER", "PNP")
            output_npn_devices = db.list_by_role("OUTPUT", "NPN")
            output_pnp_devices = db.list_by_role("OUTPUT", "PNP")
            predriver_npn_devices = db.list_by_role("PREDRIVER", "NPN")
            predriver_pnp_devices = db.list_by_role("PREDRIVER", "PNP")
            
            # Create selectboxes
            st.session_state.inputs.input_device = st.selectbox(
                "Entrada (par dif)",
                options=[d.name for d in input_devices],
                index=next((i for i, d in enumerate(input_devices) if d.name == "MPSA92"), 0)
            )
            
            if st.session_state.inputs.ef_level == "EF3":
                col_pred = st.columns(2)
                with col_pred[0]:
                    st.session_state.inputs.predriver_device_npn = st.selectbox(
                        "Predriver NPN",
                        options=[d.name for d in predriver_npn_devices],
                        index=next((i for i, d in enumerate(predriver_npn_devices) if d.name == "BD139"), 0)
                    )
                with col_pred[1]:
                    st.session_state.inputs.predriver_device_pnp = st.selectbox(
                        "Predriver PNP",
                        options=[d.name for d in predriver_pnp_devices],
                        index=next((i for i, d in enumerate(predriver_pnp_devices) if d.name == "BD140"), 0)
                    )
            
            col_vas = st.columns(2)
            with col_vas[0]:
                st.session_state.inputs.vas_device_npn = st.selectbox(
                    "VAS NPN",
                    options=[d.name for d in vas_npn_devices],
                    index=next((i for i, d in enumerate(vas_npn_devices) if d.name == "MJE340"), 0)
                )
            with col_vas[1]:
                st.session_state.inputs.vas_device_pnp = st.selectbox(
                    "VAS PNP",
                    options=[d.name for d in vas_pnp_devices],
                    index=next((i for i, d in enumerate(vas_pnp_devices) if d.name == "MJE350"), 0)
                )
            
            if st.session_state.inputs.ef_level in ["EF2", "EF3"]:
                col_drv = st.columns(2)
                with col_drv[0]:
                    st.session_state.inputs.driver_device_npn = st.selectbox(
                        "Driver NPN",
                        options=[d.name for d in driver_npn_devices],
                        index=next((i for i, d in enumerate(driver_npn_devices) if d.name == "BD139"), 0)
                    )
                with col_drv[1]:
                    st.session_state.inputs.driver_device_pnp = st.selectbox(
                        "Driver PNP",
                        options=[d.name for d in driver_pnp_devices],
                        index=next((i for i, d in enumerate(driver_pnp_devices) if d.name == "BD140"), 0)
                    )
            
            col_out = st.columns(2)
            with col_out[0]:
                st.session_state.inputs.output_device_npn = st.selectbox(
                    "Saída NPN",
                    options=[d.name for d in output_npn_devices],
                    index=next((i for i, d in enumerate(output_npn_devices) if d.name == "2SC5200"), 0)
                )
            with col_out[1]:
                st.session_state.inputs.output_device_pnp = st.selectbox(
                    "Saída PNP",
                    options=[d.name for d in output_pnp_devices],
                    index=next((i for i, d in enumerate(output_pnp_devices) if d.name == "2SA1943"), 0)
                )
            
            # Calculate button
            if st.button("🔄 Calcular Projeto", type="primary", use_container_width=True):
                st.session_state.results = compute_amp(st.session_state.inputs, st.session_state.db)
                st.rerun()
        
        # Display results
        if st.session_state.results:
            st.header("📋 Resultados do Cálculo")
            r = st.session_state.results
            inp = st.session_state.inputs
            
            # Create metrics
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            with col_metrics1:
                st.metric("Vrms requerido", f"{r.vrms:.2f} V")
                st.metric("Vpk requerido", f"{r.vpk_required:.2f} V")
                st.metric("Irms", f"{r.irms:.2f} A")
            
            with col_metrics2:
                st.metric("Ipk total", f"{r.ipk_total:.2f} A")
                st.metric("Ipk por dispositivo", f"{r.ipk_per_device:.2f} A")
                st.metric("Vpk disponível", f"{r.vpk_est_available:.2f} V")
            
            with col_metrics3:
                st.metric("Pmax estimada", f"{r.pmax_est_W:.1f} W")
                st.metric("Iq por transistor", f"{r.iq_A*1000:.1f} mA")
                st.metric("SR estimado", f"{r.sr_est_Vus:.1f} V/µs")
            
            # Detailed results in expanders
            with st.expander("🔧 Parâmetros de Bias e Drive", expanded=True):
                col_bias1, col_bias2, col_bias3 = st.columns(3)
                with col_bias1:
                    st.write("**Bias Vbe Multiplier**")
                    st.write(f"Vbias estimado: {r.vbias_est:.3f} V")
                    st.write(f"Junções Vbe: {r.n_vbe_junctions}")
                    st.write(f"Razão Rtop/Rbot: {r.vbe_mult_ratio:.3f}")
                
                with col_bias2:
                    st.write("**Resistores CCS**")
                    st.write(f"Rset_tail: {r.rset_tail/1000:.2f} kΩ (E24: {r.rset_tail_e24/1000:.2f} kΩ)")
                    st.write(f"Rset_vas: {r.rset_vas/1000:.2f} kΩ (E24: {r.rset_vas_e24/1000:.2f} kΩ)")
                    st.write(f"Cdom: {r.cdom_pF:.1f} pF")
                
                with col_bias3:
                    st.write("**Correntes de Drive (pico)**")
                    st.write(f"Ib_out: {r.ib_out_pk_per_device*1000:.1f} mA/trans")
                    st.write(f"Ib_driver: {r.ib_drv_pk_total*1000:.2f} mA")
                    st.write(f"Ib_VAS: {r.ib_vas_pk_total*1000:.2f} mA")
            
            with st.expander("🔥 Dissipação Térmica", expanded=False):
                col_therm1, col_therm2 = st.columns(2)
                with col_therm1:
                    st.write("**Potência Média por Transistor**")
                    st.write(f"Saída: {r.pdiss_out_avg_W:.2f} W")
                    if inp.ef_level in ("EF2", "EF3"):
                        st.write(f"Driver: {r.pdiss_drv_avg_W:.2f} W")
                    if inp.ef_level == "EF3":
                        st.write(f"Predriver: {r.pdiss_pre_avg_W:.2f} W")
                    st.write(f"VAS: {r.pdiss_vas_avg_W:.2f} W")
                
                with col_therm2:
                    st.write("**Limites de Potência**")
                    out_npn = st.session_state.db.get(inp.output_device_npn)
                    out_pnp = st.session_state.db.get(inp.output_device_pnp)
                    if out_npn and out_npn.pd_W > 0:
                        st.write(f"Pd_max {out_npn.name}: {out_npn.pd_W} W")
                    if out_pnp and out_pnp.pd_W > 0:
                        st.write(f"Pd_max {out_pnp.name}: {out_pnp.pd_W} W")
                    st.write(f"Referência classe B: {r.pdiss_max_transistor_W:.1f} W")
            
            with st.expander("⚠️ Alertas e Verificações", expanded=False):
                for alert in r.alerts:
                    st.warning(alert)
            
            with st.expander("🎛️ Recomendações de Estabilidade", expanded=False):
                st.text(r.stability_text)
    
    # Tab 2: Graphs
    with tab2:
        if st.session_state.results:
            r = st.session_state.results
            inp = st.session_state.inputs
            
            # Create tabs for different graphs
            graph_tabs = st.tabs(["📈 Potência vs Vpk", "🔥 Térmico", "⚡ VAS/SR", "📊 Pontos de Operação", "🎚️ Bias"])
            
            with graph_tabs[0]:
                # Power plot: Pout vs Vpk
                vpk = np.linspace(0, max(inp.vcc, r.vpk_required) * 1.05, 300)
                pout = (vpk**2) / (2*inp.r_load)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vpk, y=pout, mode='lines', name='Pout (W)', line=dict(color='blue', width=3)))
                
                # Add vertical lines
                fig.add_vline(x=r.vpk_est_available, line=dict(color='green', dash='dash', width=2), 
                             annotation_text="Vpk disponível", annotation_position="top right")
                fig.add_vline(x=r.vpk_required, line=dict(color='red', dash='dash', width=2),
                             annotation_text="Vpk requerido", annotation_position="top left")
                
                # Add horizontal line for target power
                fig.add_hline(y=inp.p_out, line=dict(color='orange', dash='dash', width=2),
                             annotation_text="Potência alvo")
                
                fig.update_layout(
                    title="Potência de Saída vs Tensão de Pico",
                    xaxis_title="Vpk na carga (V)",
                    yaxis_title="Potência (W)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with graph_tabs[1]:
                # Thermal plot
                vpk_sweep = np.linspace(0.0, max(r.vpk_required, r.vpk_est_available, inp.vcc) * 1.02, 200)
                ipk_total = vpk_sweep / inp.r_load
                ipk_per_out = ipk_total / max(1, int(inp.n_output_pairs))
                
                def pavg_half(Ipk, Vcc_eff, Vpk):
                    return np.maximum(0.0, Ipk * (Vcc_eff/np.pi - Vpk/4.0))
                
                # Output dissipation
                p_out = pavg_half(ipk_per_out, inp.vcc, vpk_sweep) + inp.vcc * r.iq_A
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vpk_sweep, y=p_out, mode='lines', name='Saída', 
                                        line=dict(color='blue', width=3)))
                
                # Add markers for operating point
                fig.add_trace(go.Scatter(x=[r.vpk_required], y=[r.pdiss_out_avg_W], 
                                        mode='markers', name='Ponto de Operação',
                                        marker=dict(size=12, color='red')))
                
                # Add vertical lines
                fig.add_vline(x=r.vpk_required, line=dict(color='red', dash='dash', width=2),
                             annotation_text="Vpk requerido")
                fig.add_vline(x=r.vpk_est_available, line=dict(color='green', dash='dash', width=2),
                             annotation_text="Vpk disponível")
                
                # Add Pd limits if available
                out_npn = st.session_state.db.get(inp.output_device_npn)
                if out_npn and out_npn.pd_W > 0:
                    fig.add_hline(y=out_npn.pd_W, line=dict(color='gray', dash='dot', width=2),
                                 annotation_text=f"Pd_max {out_npn.name}")
                    fig.add_hline(y=0.6*out_npn.pd_W, line=dict(color='lightgray', dash='dot', width=1),
                                 annotation_text="60% Pd_max")
                
                fig.update_layout(
                    title="Dissipação Térmica - Transistor de Saída",
                    xaxis_title="Vpk na carga (V)",
                    yaxis_title="Potência média dissipada (W)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with graph_tabs[2]:
                # VAS plot: SR vs Cdom
                cdom_pf = np.linspace(50, 1200, 350)
                sr = (inp.ivas_mA/1000.0) / (cdom_pf*1e-12) / 1e6
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cdom_pf, y=sr, mode='lines', name='SR (V/µs)', 
                                        line=dict(color='purple', width=3)))
                
                # Add target lines
                fig.add_hline(y=inp.sr_target_Vus, line=dict(color='orange', dash='dash', width=2),
                             annotation_text="SR alvo")
                fig.add_vline(x=r.cdom_pF, line=dict(color='red', dash='dash', width=2),
                             annotation_text=f"Cdom calculado: {r.cdom_pF:.1f} pF")
                
                fig.update_layout(
                    title="Slew Rate vs Cdom",
                    xaxis_title="Cdom (pF)",
                    yaxis_title="Slew Rate (V/µs)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with graph_tabs[3]:
                # Operating points plot
                ops = r.operating_points
                
                fig = go.Figure()
                
                colors = px.colors.qualitative.Set1
                for i, op in enumerate(ops):
                    # Add idle point
                    fig.add_trace(go.Scatter(
                        x=[op["vce_idle_V"]], 
                        y=[op["ic_idle_A"]],
                        mode='markers',
                        name=f"{op['stage']} (Idle)",
                        marker=dict(size=12, color=colors[i % len(colors)], symbol='circle'),
                        showlegend=True
                    ))
                    
                    # Add peak point
                    fig.add_trace(go.Scatter(
                        x=[op["vce_peak_V"]], 
                        y=[op["ic_peak_A"]],
                        mode='markers',
                        name=f"{op['stage']} (Pico)",
                        marker=dict(size=12, color=colors[i % len(colors)], symbol='triangle-up'),
                        showlegend=True
                    ))
                    
                    # Add line between points
                    fig.add_trace(go.Scatter(
                        x=[op["vce_idle_V"], op["vce_peak_V"]],
                        y=[op["ic_idle_A"], op["ic_peak_A"]],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Pontos de Operação: Ic × Vce",
                    xaxis_title="Vce (V)",
                    yaxis_title="Ic (A)",
                    yaxis_type="log",
                    hovermode='closest',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with graph_tabs[4]:
                # Bias plot: Iq vs Vre
                vre_mV = np.linspace(5, 70, 320)
                iq_mA = (vre_mV/1000.0) / inp.re_out * 1000.0
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vre_mV, y=iq_mA, mode='lines', name='Iq (mA)',
                                        line=dict(color='green', width=3)))
                
                # Add operating point
                fig.add_trace(go.Scatter(x=[inp.vre_idle_mV], y=[r.iq_A*1000], 
                                        mode='markers', name='Ponto de Operação',
                                        marker=dict(size=12, color='red')))
                
                fig.add_vline(x=inp.vre_idle_mV, line=dict(color='orange', dash='dash', width=2),
                             annotation_text=f"Vre alvo: {inp.vre_idle_mV} mV")
                
                fig.update_layout(
                    title="Corrente de Repouso vs Vre",
                    xaxis_title="Vre (mV) em Re",
                    yaxis_title="Iq (mA) por transistor",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Calcule um projeto primeiro na aba 'Projeto' para ver os gráficos.")
    
    # Tab 3: Transistor Database
    with tab3:
        st.header("Banco de Transistores")
        
        db = st.session_state.db
        
        # List existing transistors
        st.subheader("Transistores Cadastrados")
        
        # Create a DataFrame for display
        transistor_data = []
        for name, model in db.models.items():
            transistor_data.append({
                "Nome": model.name,
                "Tipo": model.kind,
                "Papel": model.role,
                "β típico": model.beta_typ,
                "β mínimo": model.beta_min,
                "Vbe (V)": model.vbe_typ,
                "Vceo (V)": model.vceo_V,
                "Ic max (A)": model.ic_max_A,
                "Pd (W)": model.pd_W,
                "Ft (MHz)": model.ft_MHz
            })
        
        if transistor_data:
            df = pd.DataFrame(transistor_data)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("Nenhum transistor cadastrado.")
        
        # Add new transistor
        st.subheader("Adicionar Novo Transistor")
        
        with st.form("add_transistor"):
            col_new1, col_new2 = st.columns(2)
            
            with col_new1:
                new_name = st.text_input("Nome")
                new_kind = st.selectbox("Tipo", ["NPN", "PNP"])
                new_role = st.selectbox("Papel", ["INPUT", "VAS", "PREDRIVER", "DRIVER", "OUTPUT"])
                new_beta_typ = st.number_input("β típico", min_value=10, max_value=1000, value=100)
                new_beta_min = st.number_input("β mínimo", min_value=5, max_value=500, value=50)
            
            with col_new2:
                new_vbe = st.number_input("Vbe típico (V)", min_value=0.5, max_value=1.0, value=0.65, step=0.01)
                new_vceo = st.number_input("Vceo (V)", min_value=0.0, max_value=1000.0, value=0.0)
                new_icmax = st.number_input("Ic max (A)", min_value=0.0, max_value=100.0, value=0.0)
                new_pd = st.number_input("Pd (W)", min_value=0.0, max_value=500.0, value=0.0)
                new_ft = st.number_input("Ft (MHz)", min_value=0.0, max_value=1000.0, value=0.0)
            
            new_notes = st.text_area("Notas")
            
            if st.form_submit_button("➕ Adicionar Transistor"):
                if new_name:
                    new_model = TransistorModel(
                        name=new_name,
                        kind=new_kind,
                        role=new_role,
                        beta_typ=new_beta_typ,
                        beta_min=new_beta_min,
                        vbe_typ=new_vbe,
                        vceo_V=new_vceo,
                        ic_max_A=new_icmax,
                        pd_W=new_pd,
                        ft_MHz=new_ft,
                        notes=new_notes
                    )
                    db.add(new_model)
                    st.success(f"Transistor {new_name} adicionado!")
                    st.rerun()
                else:
                    st.error("Digite um nome para o transistor.")
        
        # Delete transistor
        st.subheader("Remover Transistor")
        if db.models:
            transistor_names = list(db.models.keys())
            to_delete = st.selectbox("Selecione o transistor para remover", transistor_names)
            
            if st.button("🗑️ Remover Transistor Selecionado", type="secondary"):
                db.delete(to_delete)
                st.success(f"Transistor {to_delete} removido!")
                st.rerun()
    
    # Tab 4: Export
    with tab4:
        st.header("Exportar Dados")
        
        if st.session_state.results:
            inp = st.session_state.inputs
            r = st.session_state.results
            
            # Create export data
            export_data = {
                "projeto": {
                    "fonte": f"±{inp.vcc}V",
                    "carga": f"{inp.r_load}Ω",
                    "potencia_alvo": f"{inp.p_out}W",
                    "topologia": inp.ef_level,
                    "vas": inp.vas_type,
                    "pares_saida": inp.n_output_pairs
                },
                "resultados": {
                    "vrms": r.vrms,
                    "vpk_requerido": r.vpk_required,
                    "vpk_disponivel": r.vpk_est_available,
                    "ipk_total": r.ipk_total,
                    "ipk_por_dispositivo": r.ipk_per_device,
                    "iq_por_transistor": r.iq_A,
                    "pmax_estimada": r.pmax_est_W,
                    "sr_estimado": r.sr_est_Vus,
                    "cdom": r.cdom_pF,
                    "vbias_estimado": r.vbias_est,
                    "dissipacao_saida": r.pdiss_out_avg_W,
                    "dissipacao_vas": r.pdiss_vas_avg_W
                },
                "dispositivos": {
                    "entrada": inp.input_device,
                    "vas_npn": inp.vas_device_npn,
                    "vas_pnp": inp.vas_device_pnp,
                    "driver_npn": inp.driver_device_npn,
                    "driver_pnp": inp.driver_device_pnp,
                    "saida_npn": inp.output_device_npn,
                    "saida_pnp": inp.output_device_pnp
                }
            }
            
            # Export as JSON
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.download_button(
                    label="📥 Baixar como JSON",
                    data=json_str,
                    file_name=f"projeto_ab_{inp.ef_level}_{inp.p_out}W.json",
                    mime="application/json"
                )
            
            with col_exp2:
                # Display JSON
                st.code(json_str, language="json")
            
            # Export summary as text
            st.subheader("Resumo para Relatório")
            summary = f"""PROJETO AMPLIFICADOR CLASSE AB {inp.ef_level}
===============================
Fonte: ±{inp.vcc}V | Carga: {inp.r_load}Ω | Potência: {inp.p_out}W
Topologia: {inp.ef_level} | VAS: {inp.vas_type}

RESULTADOS PRINCIPAIS:
- Vrms requerido: {r.vrms:.2f}V
- Vpk requerido: {r.vpk_required:.2f}V
- Vpk disponível: {r.vpk_est_available:.2f}V
- Ipk total: {r.ipk_total:.2f}A
- Ipk por dispositivo: {r.ipk_per_device:.2f}A
- Iq por transistor: {r.iq_A*1000:.1f}mA
- Pmax estimada: {r.pmax_est_W:.1f}W
- SR estimado: {r.sr_est_Vus:.1f}V/µs
- Cdom: {r.cdom_pF:.1f}pF
- Vbias: {r.vbias_est:.3f}V

DISSIPAÇÃO TÉRMICA:
- Saída: {r.pdiss_out_avg_W:.2f}W por transistor
- VAS: {r.pdiss_vas_avg_W:.2f}W

DISPOSITIVOS:
- Entrada: {inp.input_device}
- VAS: {inp.vas_device_npn} / {inp.vas_device_pnp}
- Driver: {inp.driver_device_npn} / {inp.driver_device_pnp}
- Saída: {inp.output_device_npn} / {inp.output_device_pnp}
"""
            
            st.download_button(
                label="📄 Baixar Resumo (TXT)",
                data=summary,
                file_name=f"resumo_projeto_ab.txt",
                mime="text/plain"
            )
            
            st.text_area("Resumo do Projeto", summary, height=300)
        else:
            st.info("Calcule um projeto primeiro para exportar os dados.")

if __name__ == "__main__":
    main()
