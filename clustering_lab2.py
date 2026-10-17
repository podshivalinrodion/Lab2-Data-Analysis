# -*- coding: utf-8 -*-
"""Лабораторная работа №2, вариант 18: 12 методов кластеризации.
Адаптировано по примеру clustering(1).py.
Эталонные метки y НЕ участвуют в обучении и настройке моделей.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, estimate_bandwidth,
    SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS,
    Birch, BisectingKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, silhouette_samples,
)
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'graphs'
OUT_DIR.mkdir(exist_ok=True)
DATA_FILE = BASE_DIR / 'Lab2_Dataset_Clustering.xlsx'
if not DATA_FILE.exists():
    DATA_FILE = BASE_DIR / 'Lab2_Dataset_Clustering(1).xlsx'
if not DATA_FILE.exists():
    raise FileNotFoundError('Положите Lab2_Dataset_Clustering.xlsx рядом с этим скриптом.')


def save_and_show(name):
    path = OUT_DIR / name
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print('Сохранено:', path)
    plt.show()
    plt.close()


def draw_clusters(ax, coords, labels, title, legend=False):
    labels = np.asarray(labels)
    clusters = sorted(v for v in np.unique(labels) if v != -1)
    cmap = colormaps['tab20']
    for i, label in enumerate(clusters):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22,
                   color=cmap(i % 20), alpha=.85, label=f'Кластер {label}',
                   linewidths=.2, edgecolors='black')
    noise = labels == -1
    if noise.any():
        ax.scatter(coords[noise, 0], coords[noise, 1], s=48, c='black',
                   marker='x', linewidths=1.3, label=f'Шум ({noise.sum()})')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('a (стандартиз.)')
    ax.set_ylabel('b (стандартиз.)')
    ax.grid(alpha=.2)
    if legend:
        ax.legend(fontsize=8, loc='best')


def stats(labels, truth):
    labels = np.asarray(labels)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    valid = labels != -1
    sil = np.nan
    # Силуэт рассчитываем без шума, только если выделены >=2 кластера.
    if n_clusters >= 2 and valid.sum() > n_clusters:
        try:
            sil = silhouette_score(X_scaled[valid], labels[valid])
        except ValueError:
            pass
    return {
        'Кластеров': n_clusters,
        'Выбросов': n_noise,
        'Silhouette': sil,
        'ARI': adjusted_rand_score(truth, labels),
        'NMI': normalized_mutual_info_score(truth, labels),
    }


# 2.1. Данные варианта 18: столбцы AZ:BB = a, b, y.
df = pd.read_excel(DATA_FILE, header=1, usecols='AZ:BB')
df.columns = ['a', 'b', 'y']
df = df.dropna(subset=['a', 'b', 'y']).copy()
for column in ['a', 'b', 'y']:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df = df.dropna(subset=['a', 'b', 'y'])
X = df[['a', 'b']].to_numpy()
y_true = df['y'].astype(int).to_numpy()
X_scaled = StandardScaler().fit_transform(X)
N_CLUSTERS = 3  # Фиксированное k; метки y не используются при обучении.
print('Файл:', DATA_FILE.name)
print('Объектов:', len(X), '| Признаков:', X.shape[1])
print('Эталонные классы (только проверка):', dict(zip(*np.unique(y_true, return_counts=True))))
print('Число кластеров для методов с фиксированным k:', N_CLUSTERS)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], s=25, alpha=.8, color='steelblue')
axes[0].set_title('Исходные данные (без меток)')
axes[0].set_xlabel('a (стандартиз.)'); axes[0].set_ylabel('b (стандартиз.)')
axes[0].grid(alpha=.2)
draw_clusters(axes[1], X_scaled, y_true, 'Эталонные классы (для проверки)', legend=True)
fig.suptitle('Вариант 18 — исходные данные')
fig.tight_layout()
save_and_show('2_1_original_and_reference.png')

# Метод локтя из исходного clustering(1).py
max_k = min(10, len(X_scaled) - 1)
ks = list(range(1, max_k + 1))
inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in ks]
plt.figure(figsize=(7, 4))
plt.plot(ks, inertias, marker='o')
plt.title('Метод локтя для K-Means')
plt.xlabel('Количество кластеров k'); plt.ylabel('Инерция (SSE)')
plt.xticks(ks); plt.grid(alpha=.3); plt.tight_layout()
save_and_show('2_2_elbow.png')

# 2.2. Двенадцать алгоритмов
results = {}
errors = {}

def run(name, fit_function):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            labels = np.asarray(fit_function(), dtype=int)
        if len(labels) != len(X_scaled):
            raise ValueError('Неверное число меток')
        results[name] = labels
        print(f'{name}: кластеров {len(set(labels) - {-1})}, выбросов {(labels == -1).sum()}')
        for warning in caught:
            print(f'  Предупреждение: {warning.message}')
    except Exception as exc:
        errors[name] = str(exc)
        print(f'{name}: ОШИБКА — {exc}')

run('K-Means', lambda: KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(X_scaled))
run('AffinityPropagation', lambda: AffinityPropagation(damping=.8, max_iter=500, random_state=42).fit_predict(X_scaled))
bandwidth = estimate_bandwidth(X_scaled, quantile=.25, n_samples=min(226, len(X_scaled)))
if bandwidth <= 0:
    bandwidth = None
run('MeanShift', lambda: MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X_scaled))
run('SpectralClustering', lambda: SpectralClustering(n_clusters=N_CLUSTERS, affinity='nearest_neighbors',
    n_neighbors=min(10, len(X_scaled)-1), assign_labels='kmeans', random_state=42).fit_predict(X_scaled))
run('Ward', lambda: AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward').fit_predict(X_scaled))
run('Agglomerative', lambda: AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='complete').fit_predict(X_scaled))

# Подбор eps/min_samples без использования эталонных меток.
def tune_dbscan():
    candidates = []
    for eps in np.linspace(.12, .85, 25):
        for min_samples in (3, 5, 8, 12):
            labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X_scaled)
            n = len(set(labels) - {-1})
            valid = labels != -1
            if 2 <= n <= 12 and valid.sum() > n and valid.mean() >= .55:
                score = silhouette_score(X_scaled[valid], labels[valid])
                # Ограничиваем долю шума, чтобы не выигрывала модель с очень малым покрытием.
                objective = score - .35 * (1 - valid.mean())
                candidates.append((objective, eps, min_samples, labels))
    if candidates:
        _, eps, min_samples, labels = max(candidates, key=lambda v: v[0])
        print(f'Подбор DBSCAN: eps={eps:.3f}, min_samples={min_samples}')
        return labels
    print('DBSCAN: подходящие параметры не найдены; используются eps=0.3, min_samples=5')
    return DBSCAN(eps=.3, min_samples=5).fit_predict(X_scaled)

run('DBSCAN', tune_dbscan)
run('HDBSCAN', lambda: HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X_scaled) if HDBSCAN is not None
    else (_ for _ in ()).throw(ImportError('Требуется scikit-learn с поддержкой HDBSCAN')))
run('OPTICS', lambda: OPTICS(min_samples=5, xi=.05, min_cluster_size=.05).fit_predict(X_scaled))
run('GaussianMixture', lambda: GaussianMixture(n_components=N_CLUSTERS, n_init=5, random_state=42).fit_predict(X_scaled))
run('Birch', lambda: Birch(n_clusters=N_CLUSTERS, threshold=.35).fit_predict(X_scaled))
run('BisectingKMeans', lambda: BisectingKMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(X_scaled))

names = ['K-Means', 'AffinityPropagation', 'MeanShift', 'SpectralClustering',
         'Ward', 'Agglomerative', 'DBSCAN', 'HDBSCAN', 'OPTICS',
         'GaussianMixture', 'Birch', 'BisectingKMeans']
fig, axes = plt.subplots(3, 4, figsize=(18, 13), sharex=True, sharey=True)
for ax, name in zip(axes.flat, names):
    if name in results:
        draw_clusters(ax, X_scaled, results[name], name)
        s = stats(results[name], y_true)
        ax.set_title(f'{name}\nk={s["Кластеров"]}, шум={s["Выбросов"]}, ARI={s["ARI"]:.2f}', fontsize=10)
    else:
        ax.text(.5, .5, f'Не выполнен:\n{errors.get(name, "ошибка")}',
                ha='center', va='center', transform=ax.transAxes, fontsize=9, wrap=True)
        ax.set_title(name)
fig.suptitle('Вариант 18 — сравнение 12 алгоритмов кластеризации', fontsize=16)
fig.tight_layout(rect=(0, 0, 1, .97))
save_and_show('2_2_all_12_algorithms.png')

# 2.3–2.4. Выбросы и детальные графики плотностных алгоритмов
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True, sharey=True)
for ax, name in zip(axes, ('DBSCAN', 'HDBSCAN', 'OPTICS')):
    if name in results:
        draw_clusters(ax, X_scaled, results[name], f'{name} — шум и кластеры', legend=True)
    else:
        ax.text(.5, .5, errors.get(name, 'Недоступен'), ha='center', va='center', transform=ax.transAxes)
fig.tight_layout()
save_and_show('2_4_density_outliers.png')

# Иерархическая дендрограмма (по мотивам примера преподавателя)
plt.figure(figsize=(12, 5))
dendrogram(linkage(X_scaled, method='ward'), truncate_mode='lastp', p=20,
           show_leaf_counts=True, leaf_rotation=45)
plt.title('Дендрограмма Ward (последние 20 объединений)')
plt.ylabel('Евклидово расстояние')
plt.tight_layout()
save_and_show('2_3_ward_dendrogram.png')

# Силуэтный график K-Means из исходного clustering(1).py
if 'K-Means' in results:
    labels = results['K-Means']
    vals = silhouette_samples(X_scaled, labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 0
    for i, cluster in enumerate(sorted(np.unique(labels))):
        v = np.sort(vals[labels == cluster])
        y_upper = y_lower + len(v)
        ax.barh(np.arange(y_lower, y_upper), v, height=1, color=colormaps['tab10'](i))
        ax.text(-.08, (y_lower + y_upper)/2, str(cluster))
        y_lower = y_upper
    ax.axvline(vals.mean(), color='red', ls='--', label=f'Среднее = {vals.mean():.3f}')
    ax.set_xlabel('Коэффициент силуэта')
    ax.set_ylabel('Объекты по кластерам')
    ax.set_title('Силуэтный анализ K-Means')
    ax.legend(); fig.tight_layout()
    save_and_show('2_3_kmeans_silhouette.png')

# 2.5. Числовое сравнение с эталоном (y использован только здесь)
rows = []
for name in names:
    if name in results:
        rows.append({'Алгоритм': name, **stats(results[name], y_true)})
summary = pd.DataFrame(rows).sort_values('ARI', ascending=False)
print('\nСРАВНЕНИЕ С ЭТАЛОННЫМИ КЛАССАМИ:')
print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
summary.to_csv(OUT_DIR / '2_5_clustering_metrics.csv', index=False, encoding='utf-8-sig')
if not summary.empty:
    print('\nНаибольший ARI:', summary.iloc[0]['Алгоритм'])
if errors:
    print('\nАлгоритмы, которые не удалось выполнить:', errors)
print('\nГотово. Графики и таблица сохранены в:', OUT_DIR)
