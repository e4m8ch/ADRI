#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto: Analytics para Redes Inteligentes - Ficheiro Único
Tema: Degradação Térmica de Transformadores com Penetração de EVs
Versão: 2.0 (com suporte a CSV local e dados sintéticos melhorados)
"""

# ============================================================================
# SECÇÃO 1: CONFIGURAÇÃO DE VISUALIZAÇÃO
# ============================================================================
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import requests
import os
import warnings
warnings.filterwarnings('ignore')

print(f"[INFO] Backend ativo: {matplotlib.get_backend()}")

# ============================================================================
# SECÇÃO 2: PARÂMETROS E CONFIGURAÇÕES
# ============================================================================
@dataclass
class TransformerSpecs:
    """Especificações do transformador de distribuição"""
    nominal_power: float = 400.0  # kVA
    no_load_loss: float = 0.6     # kW (ferro)
    load_loss: float = 4.0        # kW (cobre)
    delta_theta_or: float = 55.0  # K (subida temp óleo nominal)
    delta_theta_hr: float = 20.0  # K (subida hot-spot sobre óleo)
    tau_oil: float = 150.0        # min (constante de tempo óleo)
    tau_wind: float = 7.0         # min (constante de tempo enrolamento)
    ambient_temp: float = 25.0    # °C base
    
    @property
    def loss_ratio(self) -> float:
        return self.load_loss / self.no_load_loss

@dataclass
class SimulationConfig:
    """Configuração da simulação"""
    ev_penetration: int = 60      # Número de EVs
    time_resolution: int = 15     # minutos
    simulation_days: int = 1
    peak_limit_kw: float = 320.0  # Limite para estratégia gerida

# ============================================================================
# SECÇÃO 3: CARREGADOR DE DADOS FLEXÍVEL (API / CSV / SINTÉTICO)
# ============================================================================
class ACNDataLoader:
    """Carrega dados de 3 fontes: API online, CSV local, ou geração sintética"""
    
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path
        self.base_url = "https://ev.caltech.edu/api/v1/sessions"
        
    def load_data(self, n_sessions: int = 100) -> pd.DataFrame:
        """Tenta carregar na ordem: CSV -> API -> Sintético"""
        
        # 1. Tentar CSV local primeiro
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"[INFO] A carregar dados locais de: {self.csv_path}")
            return self._load_csv_local(n_sessions)
        
        # 2. Tentar API online
        try:
            print("[INFO] A tentar API do ACN-Data...")
            return self._fetch_api(n_sessions)
        except Exception as e:
            print(f"[WARNING] API indisponível: {str(e)[:80]}...")
            
        # 3. Fallback para sintético avançado
        print("[INFO] A gerar dados sintéticos realistas...")
        return self._generate_advanced_synthetic(n_sessions)
    
    def _load_csv_local(self, n_sessions: int) -> pd.DataFrame:
        """Carrega e processa ficheiro CSV do ACN-Data"""
        try:
            # O ACN-Data tem formato específico; ajusta conforme o teu ficheiro
            df = pd.read_csv(self.csv_path)
            
            # Mapear colunas comuns do ACN-Data
            col_mappings = {
                'connectionTime': 'start_time',
                'disconnectTime': 'end_time',
                'kWhDelivered': 'energy_kwh',
                'sessionDuration': 'duration_h'
            }
            
            for old, new in col_mappings.items():
                if old in df.columns:
                    df[new] = df[old]
            
            # Processar timestamps
            if 'start_time' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
            
            # Calcular duração em horas se não existir
            if 'duration_h' not in df.columns and 'end_time' in df.columns:
                df['duration_h'] = (pd.to_datetime(df['end_time']) - 
                                   pd.to_datetime(df['start_time'])).dt.total_seconds() / 3600
            
            # Calcular potência média
            df['power_kw'] = df['energy_kwh'] / df['duration_h'].clip(lower=0.5)
            
            print(f"[SUCCESS] Carregados {len(df)} registos do CSV")
            return df.head(n_sessions)[['start_time', 'energy_kwh', 'duration_h', 'power_kw']]
            
        except Exception as e:
            print(f"[ERROR] Falha no CSV: {e}. A usar sintético...")
            return self._generate_advanced_synthetic(n_sessions)
    
    def _fetch_api(self, limit: int = 100) -> pd.DataFrame:
        """Busca dados da API (pode falhar em redes restritas)"""
        params = {
            'where': '{"site": "caltech"}',
            'max_results': limit
        }
        response = requests.get(self.base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()['_items']
        df = pd.DataFrame(data)
        
        # Processar
        df['start_time'] = pd.to_datetime(df['connectionTime'])
        df['energy_kwh'] = df['kWhDelivered'].fillna(0)
        df['duration_h'] = pd.to_timedelta(df['sessionDuration']).dt.total_seconds() / 3600
        df['power_kw'] = (df['energy_kwh'] / df['duration_h'].clip(lower=0.5)).fillna(6.6)
        
        print(f"[SUCCESS] Carregados {len(df)} registos da API")
        return df[['start_time', 'energy_kwh', 'duration_h', 'power_kw']].head(limit)
    
    def _generate_advanced_synthetic(self, n_sessions: int) -> pd.DataFrame:
        """
        Gera dados sintéticos com padrões realistas:
        - Pico manhã (8-10h): commute
        - Pico tarde (17-20h): regresso
        - Mix de velocidades: 50% slow (3.7kW), 40% regular (7.2kW), 10% fast (11kW)
        """
        np.random.seed(42)
        
        # Perfil de chegadas bimodal (manhã e tarde)
        def arrival_pdf(hour):
            """Distribuição de chegadas: 2 picos normais"""
            morning = 0.4 * np.exp(-((hour - 8.5)**2) / 2)
            evening = 0.6 * np.exp(-((hour - 18.5)**2) / 4)
            return morning + evening + 0.05  # baseline
        
        # Gerar horas de chegada
        hours = np.linspace(0, 24, 1000)
        probs = np.array([arrival_pdf(h) for h in hours])
        probs /= probs.sum()
        
        arrival_hours = np.random.choice(hours, size=n_sessions, p=probs)
        arrival_minutes = (arrival_hours * 60).astype(int)
        
        # Converter para timestamps (hoje)
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_times = [base_date + timedelta(minutes=int(m)) for m in arrival_minutes]
        
        # Durações realistas (log-normal: a maioria 2-6h, alguns curtos, alguns longos)
        durations = np.random.lognormal(mean=1.2, sigma=0.5, size=n_sessions)
        durations = np.clip(durations, 0.5, 12.0)  # entre 30min e 12h
        
        # Energia solicitada (correlacionada com duração mas com variabilidade)
        # kWh = kW * h, mas com eficiência e comportamento realista
        base_power = np.random.choice([3.7, 7.2, 11.0], size=n_sessions, p=[0.5, 0.4, 0.1])
        energies = base_power * durations * np.random.uniform(0.6, 1.0, n_sessions)
        energies = np.clip(energies, 2, 100)
        
        # Calcular potência real (pode ser menor que o carregador se desliga antes)
        power_kw = energies / durations
        
        df = pd.DataFrame({
            'start_time': start_times,
            'energy_kwh': energies,
            'duration_h': durations,
            'power_kw': power_kw
        })
        
        print(f"[INFO] Gerados {n_sessions} perfis sintéticos realistas:")
        print(f"       - Chegadas: Pico 08:30 e 18:30")
        print(f"       - Potências: {(power_kw <= 4).sum()} slow, "
              f"{((power_kw > 4) & (power_kw <= 9)).sum()} regular, "
              f"{(power_kw > 9).sum()} fast")
        
        return df

# ============================================================================
# SECÇÃO 4: MODELO TÉRMICO IEEE C57.91 (LINEARIZADO)
# ============================================================================
class TransformerThermalModel:
    """Modelo térmico com FAA linearizado (piecewise)"""
    
    def __init__(self, specs: TransformerSpecs):
        self.specs = specs
        self.time_step = 15
        self.theta_oil = specs.ambient_temp
        self.theta_hs = specs.ambient_temp
        
        # Segmentos lineares para FAA: [T_min, T_max, slope, intercept]
        self.faa_segments = [
            (60, 95, 0.0010, -0.050),      # Baixa carga
            (95, 110, 0.0080, -0.760),     # Normal
            (110, 150, 0.045, -4.150)      # Sobrecarga
        ]
    
    def update(self, load_pu: float, dt_min: int) -> Tuple[float, float]:
        """Atualiza estado térmico (modelo dinâmico)"""
        # Fator de carga não-linear
        k = max(0.1, load_pu)  # evitar divisão por zero
        
        # 1. Temperatura do óleo (modelo exponencial simplificado)
        delta_theta_or = self.specs.delta_theta_or
        loss_ratio = self.specs.loss_ratio
        
        # Subida de temperatura em regime permanente
        delta_theta_oil_ss = delta_theta_or * ((k**2 * loss_ratio + 1) / (loss_ratio + 1)) ** 0.8
        
        # Dinâmica de 1ª ordem (Euler)
        tau_oil = self.specs.tau_oil
        d_theta_oil = (delta_theta_oil_ss - (self.theta_oil - self.specs.ambient_temp)) * (dt_min / tau_oil)
        self.theta_oil += d_theta_oil
        
        # 2. Hot-spot sobre o óleo
        delta_theta_hs_ss = self.specs.delta_theta_hr * (k ** 1.6)
        tau_wind = self.specs.tau_wind
        d_theta_hs = (delta_theta_hs_ss - (self.theta_hs - self.theta_oil)) * (dt_min / tau_wind)
        self.theta_hs += d_theta_hs
        
        return self.theta_oil, self.theta_hs
    
    def calculate_faa(self, theta_hs: Optional[float] = None) -> float:
        """FAA linearizado por troços"""
        t = theta_hs if theta_hs is not None else self.theta_hs
        
        for t_min, t_max, slope, intercept in self.faa_segments:
            if t_min <= t <= t_max:
                return max(0.01, slope * t + intercept)
        
        return 0.01 if t < 60 else 8.0

# ============================================================================
# SECÇÃO 5: AGREGADOR E GESTÃO DE CARGA
# ============================================================================
class LoadAggregator:
    """Agrega sessões EV em perfil de potência temporal"""
    
    def __init__(self, sessions: pd.DataFrame, config: SimulationConfig):
        self.sessions = sessions
        self.config = config
        self.dt = config.time_resolution
        
        # Criar vetor temporal (24h)
        self.time_index = pd.date_range(
            start=datetime.now().replace(hour=0, minute=0),
            periods=int(24 * 60 / self.dt),
            freq=f'{self.dt}min'
        )
    
    def uncoordinated(self) -> np.ndarray:
        """Plug-and-charge: inicia imediatamente na chegada"""
        profile = np.zeros(len(self.time_index))
        
        for _, sess in self.sessions.iterrows():
            start_idx = self._time_to_index(sess['start_time'])
            duration_steps = int(sess['duration_h'] * 60 / self.dt)
            end_idx = min(start_idx + duration_steps, len(profile))
            
            # Potência constante durante a sessão (simplificação)
            power = min(sess['power_kw'], 11.0)  # limitar a 11kW realista
            profile[start_idx:end_idx] += power
            
        return profile
    
    def managed_valley_filling(self, peak_limit: float) -> np.ndarray:
        """
        Estratégia de gestão simples:
        1. Limita potência instantânea ao peak_limit
        2. Move excesso para períodos de menor carga (vales)
        """
        base = self.uncoordinated()
        managed = base.copy()
        
        # Iterativo: mover carga dos picos para os vales
        max_iterations = 10
        for _ in range(max_iterations):
            peaks = managed > peak_limit
            if not np.any(peaks):
                break
            
            # Energia a mover
            excess = np.sum(managed[peaks] - peak_limit) * (self.dt / 60)
            
            # Procurar vales (abaixo de 60% do limite)
            valleys = managed < (peak_limit * 0.6)
            if not np.any(valleys):
                break
            
            # Capacidade disponível nos vales
            valley_cap = np.sum((peak_limit * 0.6 - managed[valleys])) * (self.dt / 60)
            
            if valley_cap <= 0:
                break
            
            # Redistribuir proporcionalmente
            move_ratio = min(1.0, excess / valley_cap)
            
            # Cortar picos
            managed[peaks] = peak_limit
            
            # Adicionar aos vales
            add_per_step = move_ratio * (peak_limit * 0.6 - managed[valleys])
            managed[valleys] += add_per_step
        
        return managed
    
    def _time_to_index(self, timestamp) -> int:
        """Converte timestamp para índice do vetor"""
        if isinstance(timestamp, pd.Timestamp):
            minutes = timestamp.hour * 60 + timestamp.minute
        else:
            minutes = timestamp.hour * 60 + timestamp.minute
        return int(minutes / self.dt) % len(self.time_index)

# ============================================================================
# SECÇÃO 6: MOTOR DE SIMULAÇÃO
# ============================================================================
class SimulationEngine:
    """Executa simulação térmica passo a passo"""
    
    def __init__(self, specs: TransformerSpecs, config: SimulationConfig):
        self.specs = specs
        self.config = config
        
    def run(self, load_kw: np.ndarray) -> dict:
        """Simula evolução térmica para um perfil de carga"""
        model = TransformerThermalModel(self.specs)
        n_steps = len(load_kw)
        dt_h = self.config.time_resolution / 60
        
        # Arrays de resultados
        oil_temp = np.zeros(n_steps)
        hs_temp = np.zeros(n_steps)
        faa = np.zeros(n_steps)
        load_pu = load_kw / self.specs.nominal_power
        
        # Simulação temporal
        for i in range(n_steps):
            t_oil, t_hs = model.update(load_pu[i], self.config.time_resolution)
            oil_temp[i] = t_oil
            hs_temp[i] = t_hs
            faa[i] = model.calculate_faa(t_hs)
        
        # Métricas acumuladas
        lol = np.sum(faa) * dt_h  # Loss of Life em horas equivalentes
        
        return {
            'oil_temp': oil_temp,
            'hs_temp': hs_temp,
            'faa': faa,
            'load_pu': load_pu,
            'lol_hours': lol,
            'lol_days': lol / 24,
            'max_hs': np.max(hs_temp),
            'avg_faa': np.mean(faa),
            'time': self._generate_time_vector()
        }
    
    def _generate_time_vector(self):
        return pd.date_range(
            start=datetime.now().replace(hour=0, minute=0),
            periods=int(24 * 60 / self.config.time_resolution),
            freq=f'{self.config.time_resolution}min'
        )

# ============================================================================
# SECÇÃO 7: VISUALIZAÇÃO AVANÇADA (CORRIGIDA)
# ============================================================================
class Visualizer:
    """Gera gráficos comparativos em múltiplas janelas"""
    
    @staticmethod
    def full_analysis(uncoord: dict, managed: dict, config: SimulationConfig, specs: TransformerSpecs):
        """Janela principal com 4 subplots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f'Análise: {config.ev_penetration} EVs em {specs.nominal_power:.0f}kVA '
                    f'(Limite: {config.peak_limit_kw:.0f}kW)', 
                    fontsize=13, fontweight='bold')
        
        t = uncoord['time']
        
        # 1. Perfil de Carga
        ax = axes[0, 0]
        ax.plot(t, uncoord['load_pu']*100, 'r-', linewidth=2, label='Não Coordenado', alpha=0.8)
        ax.plot(t, managed['load_pu']*100, 'g-', linewidth=2, label='Gerido', alpha=0.8)
        ax.axhline(100, color='black', linestyle='--', alpha=0.5, label='100% Nominal')
        ax.set_ylabel('Carga (%)')
        ax.set_title('Perfil de Carga do Transformador')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Temperaturas
        ax = axes[0, 1]
        ax.plot(t, uncoord['oil_temp'], 'b--', label='Óleo (Unc.)', alpha=0.7)
        ax.plot(t, managed['oil_temp'], 'b-', label='Óleo (Ger.)', alpha=0.9)
        ax.plot(t, uncoord['hs_temp'], 'r-', linewidth=2, label='Hot-Spot (Unc.)')
        ax.plot(t, managed['hs_temp'], 'g-', linewidth=2, label='Hot-Spot (Ger.)')
        ax.axhline(98, color='orange', linestyle=':', alpha=0.8, label='98°C (Ref.)')
        ax.set_ylabel('Temperatura (°C)')
        ax.set_title('Evolução Térmica')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. FAA (log scale)
        ax = axes[1, 0]
        ax.semilogy(t, uncoord['faa'], 'r-', linewidth=2, label='Não Coordenado')
        ax.semilogy(t, managed['faa'], 'g-', linewidth=2, label='Gerido')
        ax.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='FAA = 1.0')
        ax.set_ylabel('FAA (log scale)')
        ax.set_title('Aging Acceleration Factor')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # 4. Loss of Life Acumulado
        ax = axes[1, 1]
        dt_h = config.time_resolution / 60
        lol_unc = np.cumsum(uncoord['faa']) * dt_h
        lol_man = np.cumsum(managed['faa']) * dt_h
        
        ax.plot(t, lol_unc, 'r-', linewidth=2, label=f'Uncoord: {lol_unc[-1]:.1f}h')
        ax.plot(t, lol_man, 'g-', linewidth=2, label=f'Managed: {lol_man[-1]:.1f}h')
        ax.fill_between(t, lol_man, lol_unc, alpha=0.3, color='green', 
                       label=f' Poupança: {lol_unc[-1]-lol_man[-1]:.1f}h')
        ax.set_ylabel('Loss of Life (horas equiv.)')
        ax.set_title('Envelhecimento Acumulado')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Formatação
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Janela 2: Métricas finais
        Visualizer._metrics_summary(uncoord, managed, config)
    
    @staticmethod
    def _metrics_summary(u, m, config):
        """Janela secundária com barras comparativas"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        metrics = ['Temp. Máx\n(°C)', 'Temp. Média\n(°C)', 'FAA\nMédio', 'Loss of Life\n(horas/dia)']
        u_vals = [u['max_hs'], np.mean(u['hs_temp']), u['avg_faa'], u['lol_hours']]
        m_vals = [m['max_hs'], np.mean(m['hs_temp']), m['avg_faa'], m['lol_hours']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, u_vals, width, label='Não Coordenado', color='#d62728', alpha=0.8)
        bars2 = ax.bar(x + width/2, m_vals, width, label='Gerido (Valley Filling)', color='#2ca02c', alpha=0.8)
        
        ax.set_ylabel('Valor')
        ax.set_title('Métricas de Stress do Transformador - Resumo', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # Adicionar texto de redução percentual
        reducao = (1 - m['lol_hours']/u['lol_hours']) * 100
        ax.text(0.95, 0.95, f'Redução no envelhecimento:\n{reducao:.1f}%',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# SECÇÃO 8: MAIN
# ============================================================================
def main():
    print("="*65)
    print("  DEGRADAÇÃO TÉRMICA DE TRANSFORMADORES COM EVs - ANÁLISE COMPLETA")
    print("="*65)
    
    # CONFIGURAÇÃO
    SPECS = TransformerSpecs(
        nominal_power=400.0,    # 400 kVA
        ambient_temp=25.0
    )
    
    CONFIG = SimulationConfig(
        ev_penetration=80,      # 80 EVs (20% penetração alta)
        time_resolution=15,     # 15 min
        peak_limit_kw=350.0     # Limite de gestão
    )
    
    # CARREGAR DADOS (tenta CSV primeiro se existir)
    loader = ACNDataLoader(csv_path="acn_data.csv")  # <- Altera para o teu ficheiro
    sessions = loader.load_data(n_sessions=CONFIG.ev_penetration)
    
    # AGREGAR
    aggregator = LoadAggregator(sessions, CONFIG)
    
    print(f"\n[INFO] A calcular perfis de carga...")
    load_unc = aggregator.uncoordinated()
    load_man = aggregator.managed_valley_filling(CONFIG.peak_limit_kw)
    
    print(f"[DATA] Pico não coordenado: {np.max(load_unc):.1f} kW ({np.max(load_unc)/SPECS.nominal_power*100:.1f}%)")
    print(f"[DATA] Pico gerido: {np.max(load_man):.1f} kW ({np.max(load_man)/SPECS.nominal_power*100:.1f}%)")
    print(f"[DATA] Energia total: {np.sum(load_unc)*CONFIG.time_resolution/60:.1f} kWh/dia")
    
    # SIMULAR
    print(f"\n[INFO] A simular modelo térmico (Δt={CONFIG.time_resolution}min)...")
    engine = SimulationEngine(SPECS, CONFIG)
    
    res_unc = engine.run(load_unc)
    res_man = engine.run(load_man)
    
    # RESULTADOS
    print("\n" + "="*65)
    print(" RESULTADOS COMPARATIVOS")
    print("="*65)
    print(f"{'Métrica':<35} {'Não Coord.':>12} {'Gerido':>12} {'Variação':>10}")
    print("-"*65)
    
    var = (res_man['lol_hours']/res_unc['lol_hours'] - 1) * 100
    print(f"{'Loss of Life (h/dia)':<35} {res_unc['lol_hours']:>12.3f} {res_man['lol_hours']:>12.3f} {var:>9.1f}%")
    print(f"{'Temp. Hot-Spot Máx (°C)':<35} {res_unc['max_hs']:>12.1f} {res_man['max_hs']:>12.1f}")
    print(f"{'Temp. Hot-Spot Média (°C)':<35} {np.mean(res_unc['hs_temp']):>12.1f} {np.mean(res_man['hs_temp']):>12.1f}")
    print(f"{'FAA Máximo':<35} {np.max(res_unc['faa']):>12.2f} {np.max(res_man['faa']):>12.2f}")
    
    # VISUALIZAR
    print(f"\n[INFO] A abrir janelas gráficas...")
    Visualizer.full_analysis(res_unc, res_man, CONFIG, SPECS)
    
    print("[SUCCESS] Análise concluída!")

if __name__ == "__main__":
    main()