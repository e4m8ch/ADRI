import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def prepare_ev_data(csv_path, output_path='SYNTHETIC_EV_DATA.csv'):
    """
    Converte o CSV sintético num formato utilizável para simulação térmica
    """
    # 1. Carregar dados
    print(f"A carregar {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Limpar dados inválidos
    df = df.dropna()
    df = df[df['chargingDuration'] > 0]  # Remover durações nulas/negativas
    df = df[df['kWhDelivered'] > 0]      # Remover energias nulas
    
    # 3. Calcular potência média (kW)
    df['power_kw'] = df['kWhDelivered'] / df['chargingDuration']
    
    # Filtrar potências absurdas (ex: > 22kW não é residencial)
    df = df[df['power_kw'] <= 22]
    df = df[df['power_kw'] > 0.5]  # Mínimo 500W
    
    # 4. Converter tempo decimal para estrutura de datetime
    # Ex: 15.3329 -> hora=15, minuto=19, segundo=58
    df['start_hour'] = df['connectionTime_decimal'].astype(int)
    df['start_minute'] = ((df['connectionTime_decimal'] % 1) * 60).astype(int)
    
    # 5. Criar timestamp de referência (usando dayIndicator como dia do ano)
    base_date = datetime(2024, 1, 1)
    df['start_time'] = df.apply(lambda row: 
        base_date + 
        timedelta(days=int(row['dayIndicator'])-1) +
        timedelta(hours=int(row['start_hour'])) +
        timedelta(minutes=int(row['start_minute'])), axis=1
    )
    
    # 6. Calcular hora de fim
    df['end_time'] = df['start_time'] + pd.to_timedelta(df['chargingDuration'], unit='h')
    
    # 7. Selecionar apenas colunas necessárias e renomear para formato standard
    df_clean = pd.DataFrame({
        'session_id': range(len(df)),
        'start_time': df['start_time'],
        'end_time': df['end_time'],
        'energy_kwh': df['kWhDelivered'],
        'duration_h': df['chargingDuration'],
        'power_kw': df['power_kw'],
        'day': df['dayIndicator']
    })
    
    # 8. Guardar CSV processado
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✅ Dados processados guardados em: {output_path}")
    print(f"📊 Total de sessões: {len(df_clean)}")
    print(f"⚡ Potência média: {df_clean['power_kw'].mean():.2f} kW")
    print(f"🔋 Energia total: {df_clean['energy_kwh'].sum():.1f} kWh")
    print(f"📅 Período: Dia {df_clean['day'].min()} a Dia {df_clean['day'].max()}")
    
    return df_clean

def create_load_profile(df, day_number=1, resolution_min=15):
    """
    Cria um perfil de carga agregado para um dia específico (resolução em minutos)
    """
    # Filtrar para o dia escolhido
    day_data = df[df['day'] == day_number].copy()
    
    if len(day_data) == 0:
        print(f"Dia {day_number} não encontrado!")
        return None
    
    print(f"\n📅 A processar Dia {day_number} ({len(day_data)} sessões)...")
    
    # Criar vetor temporal (24h)
    start_day = day_data['start_time'].min().normalize()
    time_slots = pd.date_range(start=start_day, periods=int(24*60/resolution_min), 
                               freq=f'{resolution_min}min')
    
    # Inicializar perfil de carga
    load_profile = np.zeros(len(time_slots))
    
    # Para cada sessão, adicionar potência aos slots correspondentes
    for _, session in day_data.iterrows():
        # Encontrar índice de início
        start_idx = int((session['start_time'] - start_day).total_seconds() / 60 / resolution_min)
        duration_slots = int(session['duration_h'] * 60 / resolution_min)
        
        # Garantir limites
        start_idx = max(0, min(start_idx, len(time_slots)-1))
        end_idx = min(start_idx + duration_slots, len(time_slots))
        
        if start_idx < len(time_slots):
            load_profile[start_idx:end_idx] += session['power_kw']
    
    # Criar DataFrame do perfil
    profile_df = pd.DataFrame({
        'timestamp': time_slots,
        'load_kw': load_profile,
        'day': day_number
    })
    
    print(f"⚡ Pico de carga: {load_profile.max():.1f} kW")
    print(f"⚡ Carga média: {load_profile.mean():.1f} kW")
    
    return profile_df

if __name__ == "__main__":
    # 1. Preparar dados
    df = prepare_ev_data('SYNTHETIC_EV_DATA.csv', 'ev_sessions_ready.csv')
    
    # 2. Criar perfil de carga para um dia exemplo (dia 1)
    profile = create_load_profile(df, day_number=1, resolution_min=15)
    
    # 3. Visualizar
    plt.figure(figsize=(12, 5))
    plt.plot(profile['timestamp'], profile['load_kw'], linewidth=2, color='red')
    plt.title('Perfil de Carga Agregado - Dia 1 (15 minutos de resolução)', fontsize=14)
    plt.xlabel('Hora do dia')
    plt.ylabel('Potência (kW)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('load_profile_day1.png', dpi=150)
    plt.show()
    
    # 4. Guardar perfil
    profile.to_csv('load_profile_day1.csv', index=False)
    print("\n✅ Perfil de carga guardado em: load_profile_day1.csv")