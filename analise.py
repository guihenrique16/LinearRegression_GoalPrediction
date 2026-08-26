import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, poisson

# --- CONFIGURAÇÃO VISUAL ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'

print("Iniciando a análise...\n")

# ==========================================
# ETAPA A: CARREGAR E LIMPAR DADOS
# ==========================================
arquivos_csv = [
    'Temporada22-23.csv',
    'Temporada23-24.csv',
    'Temporada24-25.csv',
    'Temporada25-26.csv'
]

lista_dfs = []
for file in arquivos_csv:
    try:
        df_temp = pd.read_csv(file)
        lista_dfs.append(df_temp)
    except FileNotFoundError:
        print(f"Erro: Arquivo {file} não encontrado na pasta.")

# Junta todos os CSVs em um só DataFrame
df = pd.concat(lista_dfs, ignore_index=True)

# Define as colunas que vamos usar para prever (X) e o que queremos prever (y)
colunas_preditoras = ['Min', 'Sh', 'SoT', 'PKatt']
coluna_alvo = 'Gls'

# Remove linhas vazias (jogos que ele não jogou, por exemplo)
df = df.dropna(subset=[coluna_alvo] + colunas_preditoras).copy()

X = df[colunas_preditoras]
y = df[coluna_alvo]

# ==========================================
# ETAPA B: DISTRIBUIÇÃO NORMAL (Z-SCORE)
# ==========================================
# Transforma as variáveis para a mesma escala (Média 0, Desvio Padrão 1)
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)

# ==========================================
# ETAPA C: NÚMEROS PSEUDOALEATÓRIOS
# ==========================================
# Divide os dados (80% treino, 20% teste). O random_state=42 garante o mesmo sorteio sempre.
X_train, X_test, y_train, y_test = train_test_split(
    X_padronizado, y, test_size=0.20, random_state=42
)

# ==========================================
# ETAPA D: TREINAR A INTELIGÊNCIA ARTIFICIAL
# ==========================================
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Calcula os erros do modelo (Resíduos)
previsoes_treino = modelo.predict(X_train)
residuos = y_train - previsoes_treino

# ==========================================
# ETAPA E: PROBABILIDADE (SIMULANDO UM JOGO)
# ==========================================
# Vamos simular: Haaland joga 90 min, 5 chutes totais, 3 no gol, bate 1 pênalti
cenario_teste = scaler.transform([[90, 5, 3, 1]])

# O modelo prevê a EXPECTATIVA DE GOLS (xG) para esse jogo
xg_previsto = max(0.05, modelo.predict(cenario_teste)[0])

print("="*50)
print(f"CENÁRIO SIMULADO: 90 Min, 5 Chutes, 3 no Gol, 1 Pênalti")
print(f"Expectativa de Gols gerada pela IA (xG): {xg_previsto:.2f} gols")
print("-" * 50)

# Aplicamos a Distribuição de Poisson para calcular as % de cada cenário
for k in range(5):
    prob_k = poisson.pmf(k, xg_previsto) * 100
    print(f"Probabilidade de marcar exatamente {k} gol(s): {prob_k:.1f}%")

prob_2_ou_mais = (1 - poisson.cdf(1, xg_previsto)) * 100
print(f"Probabilidade de marcar 2 OU MAIS gols: {prob_2_ou_mais:.1f}%")
print("="*50)

# ==========================================
# ETAPA F: GERAR GRÁFICOS PARA OS SLIDES
# ==========================================
print("\nGerando gráficos...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Gols vs Poisson
sns.histplot(df[coluna_alvo], kde=False, discrete=True, color='#1f77b4', ax=axes[0, 0], stat="density")
x_pois = np.arange(0, df[coluna_alvo].max() + 1)
axes[0, 0].plot(x_pois, poisson.pmf(x_pois, df[coluna_alvo].mean()), 'ro--', label='Poisson Teórica', linewidth=2)
axes[0, 0].set_title('1. Frequência Real de Gols vs Distribuição de Poisson')
axes[0, 0].legend()

# 2. Distribuição Normal das Variáveis
sns.kdeplot(X_padronizado[:, 1], ax=axes[0, 1], label='Chutes (Sh)', color='green', fill=True, alpha=0.3)
sns.kdeplot(X_padronizado[:, 2], ax=axes[0, 1], label='Chutes no Gol (SoT)', color='orange', fill=True, alpha=0.3)
axes[0, 1].set_title('2. Padronização (Z-Score) - Distribuição Normal')
axes[0, 1].legend()

# 3. Distribuição Normal dos Erros (Resíduos)
sns.histplot(residuos, kde=True, color='purple', ax=axes[1, 0], stat="density")
axes[1, 0].set_title('3. Erros da IA (Resíduos) seguem a Curva Normal')

# 4. Probabilidades (Cenário Simulado)
k_gols = np.arange(0, 5)
probs = [poisson.pmf(k, xg_previsto) * 100 for k in k_gols]
bars = axes[1, 1].bar(k_gols, probs, color='#2ca02c', edgecolor='black', alpha=0.85)
axes[1, 1].set_title(f'4. Probabilidade de Gols no Jogo Simulado (xG={xg_previsto:.2f})')
for bar in bars:
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout()

# Salva a imagem na mesma pasta
plt.savefig('graficos_trabalho.png', dpi=300)
print("\nSucesso! Gráficos salvos como 'graficos_trabalho.png'.")