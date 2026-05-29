# -*- coding: utf-8 -*-
"""
  1. TF-IDF por pueblo  — palabras más características (no solo frecuentes)
  2. Heatmap TF-IDF     — palabras discriminativas entre pueblos
  3. Diversidad léxica  — vocabulario único y Type-Token Ratio (TTR)
  4. WordClouds         — por pueblo (top 9 con más reseñas)
  5. Embeddings         — perfiles semánticos con SentenceTransformer
  6. Similitud coseno   — heatmap y pares más similares
  7. Clustering K-Means — silhouette + elbow, cluster óptimo
  8. UMAP               — visualización 2D del espacio turístico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter, defaultdict
from itertools import combinations
import re
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import umap

plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'DejaVu Sans'
})

STOPWORDS_ES = set([
    'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'con', 'que', 'se', 'del',
    'un', 'una', 'por', 'es', 'su', 'al', 'lo', 'muy', 'para', 'como', 'más',
    'pero', 'fue', 'todo', 'nos', 'me', 'mi', 'si', 'este', 'esta', 'también',
    'hay', 'le', 'ya', 'vez', 'bien', 'así', 'lugar', 'sin', 'ser', 'tienen',
    'tiene', 'cuando', 'donde', 'son', 'está', 'estaba', 'les', 'sus', 'cada',
    'te', 'han', 'he', 'ha', 'era', 'no', 'ni', 'sobre', 'entre', 'otro', 'otra',
    'unos', 'unas', 'tan', 'mas', 'porque', 'desde', 'hasta', 'fueron', 'hace',
    'mucho', 'poco', 'solo', 'pueden', 'puede', 'aqui', 'ahi', 'estos', 'estas',
    'eso', 'esa', 'esos', 'esas', 'uno', 'dos', 'tres', 'nuestro', 'nuestra',
    'mis', 'tu', 'realmente', 'siempre', 'aunque', 'luego', 'antes', 'después',
    'durante', 'cuanto', 'tanto', 'algo', 'alguien', 'nada', 'nadie', 'todos',
    'todas', 'siendo', 'haber', 'tener', 'hacer', 'decir', 'ver', 'ir',
    'volver', 'llegar', 'salir', 'quedar', 'tuvimos', 'fuimos', 'vimos',
    'habia', 'habian', 'estaban', 'estamos', 'tenemos', 'dijo', 'dijeron',
    'siguiente', 'primera', 'primer'
])


def tokenize(texts, min_len=4):
    all_text = ' '.join(texts).lower()
    all_text = re.sub(r'[^a-záéíóúüñ ]', ' ', all_text)
    return [w for w in all_text.split() if w not in STOPWORDS_ES and len(w) >= min_len]


def tokenize_doc(text, min_len=3):
    text = re.sub(r'[^a-záéíóúüñ ]', ' ', text.lower())
    return ' '.join([w for w in text.split() if w not in STOPWORDS_ES and len(w) >= min_len])


# TF-IDF por pueblo 

def build_tfidf(df: pd.DataFrame):
    """Construye la matriz TF-IDF (1 documento = 1 pueblo)."""
    corpus_by_town = df.groupby('Town')['full_text'].apply(
        lambda texts: tokenize_doc(' '.join(texts.tolist()))
    )
    towns_list = corpus_by_town.index.tolist()

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_tfidf      = tfidf.fit_transform(corpus_by_town.values)
    feature_names = np.array(tfidf.get_feature_names_out())
    tfidf_df     = pd.DataFrame(X_tfidf.toarray(), index=towns_list, columns=feature_names)

    print(f'Matriz TF-IDF: {X_tfidf.shape}  ({X_tfidf.shape[0]} pueblos × {X_tfidf.shape[1]} términos)')
    return tfidf_df, towns_list, X_tfidf, feature_names


def plot_tfidf_heatmap(df: pd.DataFrame, tfidf_df: pd.DataFrame,
                       X_tfidf, feature_names: np.ndarray) -> None:
    top_towns_idx    = df['Town'].value_counts().head(20).index.tolist()
    variance         = np.array(X_tfidf.toarray()).var(axis=0)
    top_feat_idx     = np.argsort(variance)[::-1][:30]
    top_features     = feature_names[top_feat_idx]
    heatmap_data     = tfidf_df.loc[top_towns_idx, top_features]

    fig, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, linewidths=0.3,
                linecolor='white', cbar_kws={'label': 'TF-IDF Score', 'shrink': 0.6})
    ax.set_title('Heatmap TF-IDF — Palabras más discriminativas por pueblo\n'
                 '(color más intenso = más característica de ese pueblo)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.show()


def plot_tfidf_barplots(df: pd.DataFrame, tfidf_df: pd.DataFrame) -> None:
    towns_plot = df['Town'].value_counts().head(16).index.tolist()
    n_cols = 4
    n_rows = (len(towns_plot) + n_cols - 1) // n_cols
    COLORS_GRAD = plt.cm.Blues(np.linspace(0.4, 0.9, 8))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.2))
    fig.suptitle('Top 8 palabras características por pueblo (TF-IDF)',
                 fontsize=14, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for i, town in enumerate(towns_plot):
        top8     = tfidf_df.loc[town].nlargest(8)
        n_rev    = int(df[df['Town'] == town].shape[0])
        region   = df[df['Town'] == town]['Region'].iloc[0]
        axes[i].barh(top8.index[::-1], top8.values[::-1], color=COLORS_GRAD, edgecolor='white')
        axes[i].set_title(f'{town}\n{region}  ·  {n_rev:,} reseñas', fontsize=9, fontweight='bold')
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].set_xlabel('TF-IDF Score', fontsize=7)

    for j in range(len(towns_plot), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# lexico diverso

def compute_lexical_diversity(df: pd.DataFrame, towns_list: list) -> pd.DataFrame:
    lex_stats = []
    for town in towns_list:
        texts    = df[df['Town'] == town]['Review'].tolist()
        words    = tokenize(texts)
        total_w  = len(words)
        unique_w = len(set(words))
        ttr      = unique_w / max(total_w, 1)
        lex_stats.append({'Town': town, 'total_words': total_w,
                          'unique_words': unique_w, 'TTR': ttr,
                          'n_reviews': len(texts)})
    return pd.DataFrame(lex_stats).sort_values('unique_words', ascending=False)


def plot_lexical_diversity(lex_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Diversidad léxica por pueblo', fontsize=13, fontweight='bold')

    top_lex = lex_df.head(20)
    axes[0].barh(top_lex['Town'][::-1], top_lex['unique_words'][::-1],
                 color=plt.cm.viridis(np.linspace(0.3, 0.9, 20)), edgecolor='white')
    axes[0].set_title('Vocabulario único (top 20)', fontweight='bold')
    axes[0].set_xlabel('Palabras únicas')

    lex_big = lex_df[lex_df['n_reviews'] >= 100].sort_values('TTR', ascending=False)
    axes[1].barh(lex_big['Town'][::-1], lex_big['TTR'][::-1],
                 color=plt.cm.plasma(np.linspace(0.3, 0.9, len(lex_big))), edgecolor='white')
    axes[1].set_title('Type-Token Ratio (≥100 reseñas)', fontweight='bold')
    axes[1].set_xlabel('TTR (mayor = más diverso)')

    sc = axes[2].scatter(lex_df['total_words'], lex_df['TTR'],
                         c=lex_df['n_reviews'], cmap='Blues',
                         s=80, alpha=0.8, edgecolors='grey', linewidth=0.4)
    plt.colorbar(sc, ax=axes[2], label='N reseñas')
    axes[2].set_title('Palabras totales vs TTR', fontweight='bold')
    axes[2].set_xlabel('Total palabras')
    axes[2].set_ylabel('TTR')

    plt.tight_layout()
    plt.show()


# wordclous por pueblo

def plot_wordclouds_by_town(df: pd.DataFrame) -> None:
    towns_wc = df['Town'].value_counts().head(9).index.tolist()
    cmaps_wc = ['viridis', 'Blues', 'Oranges', 'Greens', 'Purples',
                 'YlOrRd', 'cool', 'summer', 'copper']

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle('WordCloud por pueblo — vocabulario más frecuente',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, town in enumerate(towns_wc):
        tokens = tokenize(df[df['Town'] == town]['Review'].tolist())
        n      = len(df[df['Town'] == town])
        region = df[df['Town'] == town]['Region'].iloc[0]
        pol_m  = df[df['Town'] == town]['Polarity'].mean()
        wc = WordCloud(width=600, height=300, background_color='white',
                       colormap=cmaps_wc[i], max_words=60, prefer_horizontal=0.8
                       ).generate(' '.join(tokens))
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'{town}\n{region}  ·  {n:,} reseñas  ·  {pol_m:.2f}',
                          fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.show()


# embeddings y similitud coseno

def build_town_embeddings(df: pd.DataFrame, towns_list: list,
                           model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
                           max_per_town: int = 150, batch_size: int = 64):
    
    """
    -------------------------------------------------------------------------------
    Genera perfiles de embedding por pueblo promediando los embeddings
    de sus reseñas. Requiere sentence-transformers instalado.
    -------------------------------------------------------------------------------
    """

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f'Modelo: {model_name}  ({model.get_sentence_embedding_dimension()} dims)')

    town_profiles = {}
    t0 = time.time()
    for i, town in enumerate(towns_list):
        texts_sample = df[df['Town'] == town]['full_text'].dropna().tolist()[:max_per_town]
        embeds = model.encode(texts_sample, batch_size=batch_size,
                              show_progress_bar=False, normalize_embeddings=True)
        town_profiles[town] = embeds.mean(axis=0)
        if (i + 1) % 5 == 0 or (i + 1) == len(towns_list):
            print(f'  {i+1:>3}/{len(towns_list)}  [{time.time()-t0:.0f}s]  {town}')

    X = np.vstack([town_profiles[t] for t in towns_list])
    print(f'\nPerfiles listos: {X.shape}')
    return town_profiles, X


def plot_similarity_heatmap(X: np.ndarray, towns_list: list) -> None:
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, np.nan)

    dist_matrix = 1 - np.clip(sim_matrix.copy(), -1, 1)
    np.fill_diagonal(dist_matrix, 0)
    linkage_mat  = linkage(squareform(dist_matrix), method='ward')
    dendro       = dendrogram(linkage_mat, labels=towns_list, no_plot=True)
    order        = dendro['leaves']
    towns_ordered = [towns_list[i] for i in order]
    sim_ordered   = sim_matrix[np.ix_(order, order)]
    np.fill_diagonal(sim_ordered, 1.0)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(sim_ordered, ax=ax, xticklabels=towns_ordered, yticklabels=towns_ordered,
                cmap='RdYlGn', vmin=0.5, vmax=1.0, annot=True, fmt='.2f',
                annot_kws={'size': 6.5}, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Similitud coseno', 'shrink': 0.6})
    ax.set_title('Heatmap de similitud semántica entre Pueblos Mágicos',
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    plt.tight_layout()
    plt.show()


# clustering con k-means

def find_optimal_clusters(X: np.ndarray, k_max: int = 11) -> int:
    sil_scores = {}
    inertias   = {}
    k_range    = range(2, min(k_max, len(X)))

    for k in k_range:
        km             = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels         = km.fit_predict(X)
        sil_scores[k]  = silhouette_score(X, labels, metric='cosine')
        inertias[k]    = km.inertia_

    best_k = max(sil_scores, key=sil_scores.get)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Búsqueda del número óptimo de clusters', fontsize=13, fontweight='bold')

    axes[0].plot(list(sil_scores.keys()), list(sil_scores.values()),
                 'o-', color='#1E88E5', lw=2, markersize=7)
    axes[0].axvline(best_k, color='red', linestyle='--', alpha=0.8,
                    label=f'Mejor k={best_k}  (sil={sil_scores[best_k]:.4f})')
    axes[0].set_title('Silhouette Score (coseno)', fontweight='bold')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].legend()

    axes[1].plot(list(inertias.keys()), list(inertias.values()),
                 's-', color='#43A047', lw=2, markersize=7)
    axes[1].axvline(best_k, color='red', linestyle='--', alpha=0.8, label=f'k={best_k}')
    axes[1].set_title('Inercia (método del codo)', fontweight='bold')
    axes[1].set_xlabel('k')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    print(f'\nk óptimo = {best_k}  (Silhouette coseno = {sil_scores[best_k]:.4f})')
    return best_k


# umap 2d

def plot_umap(X: np.ndarray, towns_list: list, cluster_labels=None) -> None:
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42,
                        n_neighbors=min(10, len(towns_list) - 1))
    coords  = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=cluster_labels if cluster_labels is not None else 'steelblue',
                         cmap='tab10', s=120, alpha=0.85, edgecolors='white', linewidth=0.8)
    for i, town in enumerate(towns_list):
        ax.annotate(town, coords[i], fontsize=7.5, ha='left',
                    xytext=(4, 2), textcoords='offset points')
    if cluster_labels is not None:
        plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_title('Mapa UMAP — Espacio semántico de los Pueblos Mágicos\n'
                 '(pueblos cercanos = reseñas más similares)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from data_loading import load_dataset, clean_dataset

    df_raw     = load_dataset()
    df         = clean_dataset(df_raw)
    towns_list = sorted(df['Town'].unique().tolist())

    # TF-IDF
    tfidf_df, towns_list_tfidf, X_tfidf, feature_names = build_tfidf(df)
    plot_tfidf_heatmap(df, tfidf_df, X_tfidf, feature_names)
    plot_tfidf_barplots(df, tfidf_df)

    # Léxico
    lex_df = compute_lexical_diversity(df, towns_list)
    plot_lexical_diversity(lex_df)

    # WordClouds
    plot_wordclouds_by_town(df)

    # Embeddings (requiere GPU recomendado)
    town_profiles, X = build_town_embeddings(df, towns_list)
    plot_similarity_heatmap(X, towns_list)

    # Clustering
    best_k = find_optimal_clusters(X)
    km     = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    plot_umap(X, towns_list, cluster_labels=labels)
