#### PROJECT 1 
#### 
#### Impact of EVs on SS transformer and estimation of its loss of life


#### Explicar o workflow do python para este mini projeto
#### Mostrar a imagem das diferentes partes IRL, da SS e assim


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##import as pode ser para a bilbioteca a dar plot


# --- 1. Read data ---
df = pd.read_csv("load_profile_day1.csv")

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

# --- 2. Rename column ---
df = df.rename(columns={'load_kw': 'ev_load_kw'})

# --- 3. Create time-based features ---
df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0

# --- 4. Generate synthetic base load ---
# Daily pattern: low at night, peak during day
# Using a sinusoidal shape + small noise

# Parameters (you can tweak these)
base_min = 20   # minimum base load (kW)
base_max = 60   # peak base load (kW)

# Normalize sinusoid to [0,1]
daily_pattern = 0.5 * (1 + np.sin((df['hour'] - 6) / 24 * 2 * np.pi))

# Scale to desired load range
base_load = base_min + (base_max - base_min) * daily_pattern

# Add some noise
noise = np.random.normal(0, 2, size=len(df))

df['base_load_kw'] = base_load + noise

# --- 5. (Optional but useful) total load ---
df['total_load_kw'] = df['ev_load_kw'] + df['base_load_kw']

# --- 6. Clean up (optional) ---
df['base_load_kw'] = df['base_load_kw'].clip(lower=0)

# --- Done ---
print(df.head())


#Fazer rede neuronal para o laboratorio 4 de ADRI











plt.plot()
plt.title("")

plt.show()


















