# Лабораторная работа №2, вариант 18 — пункты 1.1–1.5
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR / 'graphs'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def save_and_show(filename):
    path = GRAPHS_DIR / filename
    plt.gcf().savefig(path, dpi=300, bbox_inches='tight')
    print(f'Сохранён график: {path}')
    plt.show()
    plt.close()


# 1.1. Загрузка варианта 18 и корреляции
file_path = BASE_DIR / 'Lab2_Dataset_PCA.xlsx'
df = pd.read_excel(file_path, header=1, usecols='CY:DD')
df.columns = ['a', 'b', 'c', 'd', 'e', 'y']
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['a', 'b', 'c', 'd', 'e', 'y']).reset_index(drop=True)
X = df[['a', 'b', 'c', 'd', 'e']]
y_true = df['y'].to_numpy()
feature_names = list(X.columns)
print('Число объектов:', len(X))
print('\nПервые 5 строк данных:\n', X.head())
corr_matrix = X.corr()
print('\nМатрица корреляций:\n', corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label='Корреляция')
ax.set_xticks(range(5), feature_names)
ax.set_yticks(range(5), feature_names)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')
ax.set_title('Матрица корреляций — вариант 18')
fig.tight_layout()
save_and_show('1_1_correlation.png')

# 1.2. Матрица разброса
axes = scatter_matrix(X, figsize=(12, 12), diagonal='hist', alpha=0.6,
                      color='royalblue', marker='o', hist_kwds={'bins': 20})
fig = axes[0, 0].figure
fig.suptitle('Матрица разброса признаков — вариант 18', fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_and_show('1_2_scatter_matrix.png')

# 1.3. PCA
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
variance_ratio = pca.explained_variance_ratio_
print('\nPCA — объяснённая дисперсия:')
for i, v in enumerate(variance_ratio, 1):
    print(f'PC{i}: {v * 100:.2f}%')
print('Накопленная дисперсия PCA:', np.round(np.cumsum(variance_ratio) * 100, 2))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(1, 6), variance_ratio * 100, alpha=0.7, label='Дисперсия компоненты')
ax.plot(range(1, 6), np.cumsum(variance_ratio) * 100, 'o-', color='red',
        label='Накопленная дисперсия')
ax.set(xlabel='Номер компоненты', ylabel='Объяснённая дисперсия (%)',
       title='PCA — значимость компонент, вариант 18')
ax.set_xticks(range(1, 6))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_and_show('1_3_PCA.png')

# 1.3.2. SVD
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
svd_ratio = S**2 / np.sum(S**2)
print('\nSVD — объяснённая дисперсия:')
for i, v in enumerate(svd_ratio, 1):
    print(f'SV{i}: {v * 100:.2f}%')
print('Накопленная дисперсия SVD:', np.round(np.cumsum(svd_ratio) * 100, 2))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(1, 6), svd_ratio * 100, alpha=0.7, color='mediumseagreen',
       label='Дисперсия компоненты')
ax.plot(range(1, 6), np.cumsum(svd_ratio) * 100, 'o-', color='red',
        label='Накопленная дисперсия')
ax.set(xlabel='Номер компоненты', ylabel='Объяснённая дисперсия (%)',
       title='SVD — значимость компонент, вариант 18')
ax.set_xticks(range(1, 6))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_and_show('1_3_SVD.png')

# 1.3.3. ICA: ошибка восстановления для разного числа компонент
ica_errors = []
for n in range(1, 6):
    model = FastICA(n_components=n, random_state=42, max_iter=3000,
                    tol=0.001, whiten='unit-variance')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        transformed = model.fit_transform(X_scaled)
    reconstructed = model.inverse_transform(transformed)
    error = mean_squared_error(X_scaled, reconstructed)
    ica_errors.append(error)
    print(f'ICA: {n} компонент(ы), ошибка восстановления = {error:.6f}')
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print(f'  ВНИМАНИЕ: ICA с {n} компонентами не сошёлся; оценка предварительная.')

ica = FastICA(n_components=5, random_state=42, max_iter=3000,
              tol=0.001, whiten='unit-variance')
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', ConvergenceWarning)
    X_ica = ica.fit_transform(X_scaled)
if any(issubclass(w.category, ConvergenceWarning) for w in caught):
    print('ВНИМАНИЕ: итоговая модель ICA не сошлась.')
print('ICA — размерность результата:', X_ica.shape)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, 6), ica_errors, 'o-', color='purple', linewidth=2)
ax.set(xlabel='Количество компонент ICA', ylabel='Среднеквадратичная ошибка (MSE)',
       title='ICA — ошибка восстановления, вариант 18')
ax.set_xticks(range(1, 6))
ax.grid(alpha=0.3)
fig.tight_layout()
save_and_show('1_3_ICA.png')

# 1.4. Сравнение методов
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].bar(range(1, 6), variance_ratio * 100, color='steelblue')
axes[0].set(title='PCA', xlabel='Компонента', ylabel='Объяснённая дисперсия (%)')
axes[1].bar(range(1, 6), svd_ratio * 100, color='mediumseagreen')
axes[1].set(title='SVD', xlabel='Компонента', ylabel='Объяснённая дисперсия (%)')
axes[2].plot(range(1, 6), ica_errors, 'o-', color='purple', linewidth=2)
axes[2].set(title='ICA', xlabel='Количество компонент', ylabel='Ошибка восстановления (MSE)')
for ax in axes:
    ax.set_xticks(range(1, 6))
    ax.grid(alpha=0.3)
fig.suptitle('Сравнение PCA, SVD и ICA — вариант 18', fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save_and_show('1_4_comparison.png')

# 1.5. Двумерные проекции и направления признаков
class_codes, class_names = pd.factorize(y_true)


def plot_biplot(ax, scores, loadings, title):
    scales = np.maximum(np.max(np.abs(scores[:, :2]), axis=0), 1e-10)
    points = scores[:, :2] / scales
    scatter = ax.scatter(points[:, 0], points[:, 1], c=class_codes,
                         cmap='viridis', alpha=0.7, s=30,
                         edgecolors='black', linewidths=0.3)
    vectors = loadings[:, :2].copy()
    max_length = max(np.linalg.norm(vectors, axis=1).max(), 1e-10)
    vectors = vectors / max_length * 0.8
    for i, name in enumerate(feature_names):
        vx, vy = vectors[i]
        ax.arrow(0, 0, vx, vy, color='red', alpha=0.8,
                 head_width=0.035, length_includes_head=True)
        ax.text(vx * 1.15, vy * 1.15, name, color='darkred',
                fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set(title=title, xlabel='Компонента 1', ylabel='Компонента 2',
           xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    return scatter

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
scatter = plot_biplot(axes[0], X_pca[:, :2], pca.components_[:2, :].T, 'PCA Biplot')
plot_biplot(axes[1], U[:, :2] * S[:2], Vt[:2, :].T, 'SVD Biplot')
plot_biplot(axes[2], X_ica[:, :2], ica.mixing_[:, :2], 'ICA Biplot')
handles, _ = scatter.legend_elements()
fig.suptitle('Двумерные проекции PCA, SVD и ICA — вариант 18', fontsize=16, y=0.98)
fig.legend(handles, [str(name) for name in class_names], title='Эталонные классы',
           loc='upper center', bbox_to_anchor=(0.5, 0.90),
           ncol=min(len(class_names), 6), fontsize=10)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.75, wspace=0.24)
save_and_show('1_5_biplot.png')

print(f'\nГотово! Все 7 графиков сохранены в: {GRAPHS_DIR}')
