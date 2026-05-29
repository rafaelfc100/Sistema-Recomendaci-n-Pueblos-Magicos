# -*- coding: utf-8 -*-
"""
  - Distribución de Polarity (global, por tipo, boxplot)
  - Proporción y reseñas por tipo de lugar y región
  - Análisis de longitud de reseñas (palabras, caracteres, oraciones)
  - WordClouds y frecuencias de palabras (positivo vs negativo)
  - Estadísticas por pueblo y región
  - Heatmap de quejas dominantes por pueblo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

PALETTE_POL  = ['#d32f2f', '#e57373', '#ffd54f', '#81c784', '#2e7d32']
PALETTE_TYPE = {'Restaurant': '#1565C0', 'Attractive': '#E65100', 'Hotel': '#4A148C'}

plt.rcParams.update({
    'figure.dpi': 130,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
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
    'todas', 'cual', 'siendo', 'haber', 'tener', 'hacer', 'decir', 'ver', 'ir',
    'volver', 'llegar', 'salir', 'quedar', 'tuvimos', 'fuimos', 'vimos',
    'habia', 'habian', 'estaban', 'estamos', 'tenemos', 'dijo', 'dijeron'
])



def plot_polarity_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribución de Polarity', fontsize=15, fontweight='bold', y=1.01)
    labels_pol = {1: '1\n(muy neg)', 2: '2\n(neg)', 3: '3\n(neutro)',
                  4: '4\n(pos)', 5: '5\n(muy pos)'}

    # Global
    counts = df['Polarity'].value_counts().sort_index()
    bars = axes[0].bar([labels_pol[i] for i in counts.index], counts.values,
                       color=PALETTE_POL, edgecolor='white', linewidth=1.2, width=0.6)
    axes[0].set_title('Global', fontweight='bold')
    axes[0].set_ylabel('Número de reseñas')
    axes[0].set_xlabel('Polarity')
    for bar, v in zip(bars, counts.values):
        pct = v / len(df) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{v:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8.5)
    axes[0].set_ylim(0, counts.max() * 1.2)

    # Barras apiladas por Tipo
    type_pol     = df.groupby(['Type', 'Polarity']).size().unstack(fill_value=0)
    type_pol_pct = type_pol.div(type_pol.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(type_pol_pct))
    for i, pol in enumerate([1, 2, 3, 4, 5]):
        if pol in type_pol_pct.columns:
            vals = type_pol_pct[pol].values
            axes[1].bar(type_pol_pct.index, vals, bottom=bottom,
                        color=PALETTE_POL[i], label=str(pol), width=0.5)
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 3:
                    axes[1].text(j, b + v / 2, f'{v:.1f}%',
                                 ha='center', va='center', fontsize=8,
                                 color='white', fontweight='bold')
            bottom += vals
    axes[1].set_title('Distribución (%) por Tipo', fontweight='bold')
    axes[1].set_ylabel('%')
    axes[1].legend(title='Polarity', bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[1].set_ylim(0, 110)

    # Boxplot por Tipo
    order = df['Type'].value_counts().index.tolist()
    colors_box = [list(PALETTE_TYPE.values())[i % len(PALETTE_TYPE)] for i in range(len(order))]
    bp = axes[2].boxplot([df[df['Type'] == t]['Polarity'].values for t in order],
                          labels=order, patch_artist=True, widths=0.4,
                          medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[2].set_title('Boxplot Polarity por Tipo', fontweight='bold')
    axes[2].set_ylabel('Polarity')
    axes[2].set_ylim(0.5, 5.8)

    plt.tight_layout()
    plt.show()

    print('\nPolarity por Tipo:')
    print(df.groupby('Type')['Polarity']
          .agg(['count', 'mean', 'median', 'std'])
          .rename(columns={'count': 'N', 'mean': 'Media', 'median': 'Mediana', 'std': 'Std'})
          .round(3).to_string())


# analisis por tipo y regiones

def plot_type_region(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tipos de Lugar', fontsize=14, fontweight='bold')

    type_counts = df['Type'].value_counts()
    colors_pie  = [PALETTE_TYPE.get(t, '#999') for t in type_counts.index]
    wedges, texts, autotexts = axes[0].pie(
        type_counts.values, labels=type_counts.index,
        autopct='%1.1f%%', colors=colors_pie, startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops={'fontsize': 11})
    for at in autotexts:
        at.set_fontweight('bold')
    axes[0].set_title(f'Proporción (Total: {len(df):,})', fontweight='bold')

    reg_type = df.groupby(['Region', 'Type']).size().unstack(fill_value=0)
    reg_type = reg_type.loc[reg_type.sum(axis=1).sort_values(ascending=True).index]
    reg_type.plot(kind='barh', ax=axes[1], stacked=True,
                  color=[PALETTE_TYPE.get(c, '#999') for c in reg_type.columns])
    axes[1].set_title('Reseñas por Región y Tipo', fontweight='bold')
    axes[1].set_xlabel('Número de reseñas')
    axes[1].set_ylabel('')
    axes[1].legend(title='Tipo', bbox_to_anchor=(1.02, 1))
    for i, (_, row) in enumerate(reg_type.iterrows()):
        total = row.sum()
        axes[1].text(total + 5, i, str(int(total)), va='center', fontsize=8)

    plt.tight_layout()
    plt.show()


# longitud de reseñas

def plot_review_lengths(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Análisis de Longitud de Reseñas', fontsize=14, fontweight='bold')

    axes[0, 0].hist(df['n_words'].clip(0, 400), bins=60, color='#5C6BC0',
                    edgecolor='white', alpha=0.85)
    axes[0, 0].axvline(df['n_words'].median(), color='red', lw=2, linestyle='--',
                       label=f'Mediana: {df["n_words"].median():.0f}')
    axes[0, 0].axvline(df['n_words'].mean(), color='orange', lw=2, linestyle=':',
                       label=f'Media: {df["n_words"].mean():.1f}')
    axes[0, 0].set_title('Distribución de palabras por reseña', fontweight='bold')
    axes[0, 0].set_xlabel('Palabras')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(df['n_chars'].clip(0, 2500), bins=60, color='#26A69A',
                    edgecolor='white', alpha=0.85)
    axes[0, 1].axvline(df['n_chars'].median(), color='red', lw=2, linestyle='--',
                       label=f'Mediana: {df["n_chars"].median():.0f}')
    axes[0, 1].set_title('Distribución de caracteres por reseña', fontweight='bold')
    axes[0, 1].set_xlabel('Caracteres')
    axes[0, 1].legend(fontsize=8)

    wpp = df.groupby('Polarity')['n_words'].mean()
    axes[0, 2].bar([str(i) for i in wpp.index], wpp.values,
                   color=PALETTE_POL, edgecolor='white')
    axes[0, 2].set_title('Palabras promedio por Polarity', fontweight='bold')
    axes[0, 2].set_xlabel('Polarity')
    axes[0, 2].set_ylabel('Palabras promedio')
    for i, v in enumerate(wpp.values):
        axes[0, 2].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)

    data_bp = [df[df['Polarity'] == p]['n_words'].clip(0, 500).values for p in [1, 2, 3, 4, 5]]
    bp = axes[1, 0].boxplot(data_bp, labels=['1', '2', '3', '4', '5'],
                             patch_artist=True, medianprops=dict(color='white', lw=2))
    for patch, color in zip(bp['boxes'], PALETTE_POL):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1, 0].set_title('Distribución de palabras por Polarity', fontweight='bold')
    axes[1, 0].set_xlabel('Polarity')
    axes[1, 0].set_ylabel('Palabras')

    types_ord = df['Type'].value_counts().index.tolist()
    data_bt   = [df[df['Type'] == t]['n_words'].clip(0, 500).values for t in types_ord]
    cols_bt   = [PALETTE_TYPE.get(t, '#999') for t in types_ord]
    bp2 = axes[1, 1].boxplot(data_bt, labels=types_ord, patch_artist=True,
                              medianprops=dict(color='white', lw=2))
    for patch, color in zip(bp2['boxes'], cols_bt):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1, 1].set_title('Palabras por Tipo de lugar', fontweight='bold')
    axes[1, 1].set_xlabel('Tipo')
    axes[1, 1].set_ylabel('Palabras')

    jitter = np.random.uniform(-0.25, 0.25, len(df))
    axes[1, 2].scatter(df['Polarity'] + jitter, df['n_words'].clip(0, 600),
                       alpha=0.08, s=5, color='#5C6BC0')
    axes[1, 2].set_title('Palabras vs Polarity (con jitter)', fontweight='bold')
    axes[1, 2].set_xlabel('Polarity')
    axes[1, 2].set_ylabel('Palabras')
    axes[1, 2].set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.show()

    print('\nEstadísticas de longitud:')
    print(df[['n_chars', 'n_words', 'n_sents']].describe().round(1).to_string())


# wordclouds + y -

def tokenize(texts, min_len=4):
    all_text = ' '.join(texts).lower()
    all_text = re.sub(r'[^a-záéíóúüñ ]', ' ', all_text)
    return [w for w in all_text.split() if w not in STOPWORDS_ES and len(w) >= min_len]


def plot_wordclouds_polarity(df: pd.DataFrame) -> None:
    pos_tokens = tokenize(df[df['Polarity'] == 5]['Review'].tolist())
    neg_tokens = tokenize(df[df['Polarity'] <= 2]['Review'].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('WordCloud: Reseñas muy positivas (⭐5) vs negativas (⭐1-2)',
                 fontsize=13, fontweight='bold')

    for ax, tokens, title, cmap in zip(
        axes,
        [pos_tokens, neg_tokens],
        ['Polarity 5 — Muy Positivas', 'Polarity 1-2 — Negativas'],
        ['Greens', 'Reds']
    ):
        wc = WordCloud(width=700, height=380, background_color='white',
                       colormap=cmap, max_words=80, prefer_horizontal=0.8
                       ).generate(' '.join(tokens))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


# analisis por pueblo

def compute_town_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('Town').agg(
        Region      =('Region', 'first'),
        N_resenas   =('Polarity', 'count'),
        Pol_media   =('Polarity', 'mean'),
        Pol_mediana =('Polarity', 'median'),
        Pol_std     =('Polarity', 'std'),
        Pct_5       =('Polarity', lambda x: (x == 5).mean() * 100),
        Pct_1_2     =('Polarity', lambda x: (x <= 2).mean() * 100),
        Tipos       =('Type', lambda x: ', '.join(sorted(x.unique())))
    ).round(2)


def plot_town_analysis(df: pd.DataFrame, town_stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Análisis por Pueblo Mágico', fontsize=15, fontweight='bold')

    top20_n   = town_stats.nlargest(20, 'N_resenas')
    colors_n  = plt.cm.Blues(np.linspace(0.3, 0.9, 20))
    axes[0, 0].barh(top20_n.index[::-1], top20_n['N_resenas'][::-1],
                    color=colors_n, edgecolor='white')
    axes[0, 0].set_title('Top 20 pueblos por número de reseñas', fontweight='bold')
    axes[0, 0].set_xlabel('Reseñas')
    for i, (name, row) in enumerate(top20_n[::-1].iterrows()):
        axes[0, 0].text(row['N_resenas'] + 0.3, i, str(int(row['N_resenas'])),
                        va='center', fontsize=8)

    top20_pol = town_stats[town_stats['N_resenas'] >= 10].nlargest(20, 'Pol_media')
    colors_p  = plt.cm.RdYlGn(np.linspace(0.4, 1.0, 20))
    axes[0, 1].barh(top20_pol.index[::-1], top20_pol['Pol_media'][::-1],
                    color=colors_p, edgecolor='white')
    axes[0, 1].set_title('Top 20 por Polarity media (mín. 10 reseñas)', fontweight='bold')
    axes[0, 1].set_xlabel('Polarity media')
    axes[0, 1].set_xlim(3, 5.3)

    bot15    = town_stats[town_stats['N_resenas'] >= 10].nsmallest(15, 'Pol_media')
    colors_b = plt.cm.RdYlGn(np.linspace(0.0, 0.4, 15))
    axes[1, 0].barh(bot15.index[::-1], bot15['Pol_media'][::-1],
                    color=colors_b, edgecolor='white')
    axes[1, 0].set_title('Bottom 15 — menor Polarity media', fontweight='bold')
    axes[1, 0].set_xlabel('Polarity media')
    axes[1, 0].set_xlim(2, 5.3)

    valid   = town_stats[town_stats['N_resenas'] >= 5].copy()
    scatter = axes[1, 1].scatter(
        valid['N_resenas'], valid['Pol_media'],
        c=valid['Pol_media'], cmap='RdYlGn', vmin=1, vmax=5,
        s=valid['Pol_std'].fillna(0.5) * 200 + 20,
        alpha=0.75, edgecolors='grey', linewidth=0.4)
    plt.colorbar(scatter, ax=axes[1, 1], label='Polarity media')
    axes[1, 1].set_title('Volumen de reseñas vs Calidad\n(tamaño = std de Polarity)',
                         fontweight='bold')
    axes[1, 1].set_xlabel('Número de reseñas')
    axes[1, 1].set_ylabel('Polarity media')
    axes[1, 1].axhline(4, color='orange', linestyle='--', alpha=0.6, label='Pol=4')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# heatmap de quejas

COMPLAINT_CATS = {
    'Servicio lento/malo': ['tardaron', 'tardanza', 'esperar', 'lento', 'espera', 'tardó'],
    'Precio caro':         ['caro', 'precio', 'costoso', 'cobrar', 'sobreprecio'],
    'Comida/calidad':      ['frío', 'fría', 'malo', 'mala', 'calidad', 'crudo', 'insípido'],
    'Limpieza/higiene':    ['sucio', 'sucia', 'basura', 'olor', 'mosca', 'cucaracha'],
    'Atención/trato':      ['grosero', 'grosera', 'maleducado', 'descortés', 'arrogante'],
    'Instalaciones':       ['viejo', 'vieja', 'deteriorado', 'roto', 'pequeño', 'estrecho'],
    'Ruido/ambiente':      ['ruido', 'ruidoso', 'música', 'bocinas', 'molesto'],
    'Engaño/estafa':       ['estafa', 'engaño', 'cobró', 'cobran', 'mentira'],
    'Espera/colas':        ['cola', 'fila', 'tiempo', 'horas', 'minutos', 'reservación'],
}


def classify_complaint(text: str) -> str:
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower)
              for cat, kws in COMPLAINT_CATS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'Otro'


def plot_complaints_heatmap(df: pd.DataFrame) -> None:
    neg_df = df[df['Polarity'] <= 2].copy()
    neg_df['complaint_cat'] = neg_df['Review'].apply(classify_complaint)

    complaint_by_town = neg_df.groupby(['Town', 'complaint_cat']).size().unstack(fill_value=0)
    complaint_pct     = complaint_by_town.div(complaint_by_town.sum(axis=1), axis=0) * 100

    towns_enough         = neg_df.groupby('Town').size()
    towns_enough         = towns_enough[towns_enough >= 10].index.tolist()
    complaint_pct_filtered = complaint_pct.loc[
        [t for t in towns_enough if t in complaint_pct.index]
    ]

    fig, ax = plt.subplots(figsize=(16, max(7, len(complaint_pct_filtered) * 0.45)))
    sns.heatmap(complaint_pct_filtered, cmap='YlOrRd', ax=ax, annot=True, fmt='.0f',
                linewidths=0.4, linecolor='white',
                cbar_kws={'label': '% de reseñas negativas'})
    ax.set_title('Tipo de queja dominante por pueblo (% de reseñas negativas)',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=35)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from data_loading import load_dataset, clean_dataset

    df_raw    = load_dataset()
    df        = clean_dataset(df_raw)
    town_stats = compute_town_stats(df)

    plot_polarity_distribution(df)
    plot_type_region(df)
    plot_review_lengths(df)
    plot_wordclouds_polarity(df)
    plot_town_analysis(df, town_stats)
    plot_complaints_heatmap(df)
