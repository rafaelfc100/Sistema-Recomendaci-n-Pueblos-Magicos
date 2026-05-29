# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import base64, io, json, time, re, warnings, pickle, os
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import faiss
import torch

print(f'GPU disponible: {torch.cuda.is_available()}')

PATH         = 'Rest-Mex_2025_test_with_labels.csv'
MODEL_NAME   = 'paraphrase-multilingual-mpnet-base-v2'
CACHE_DIR    = './cache'
MAX_PER_TOWN = 100
MAX_CORPUS   = 8000
KNN_K        = 20
BATCH_SIZE   = 64
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)
os.makedirs(CACHE_DIR, exist_ok=True)

PALETTE = [
    '#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA',
    '#00ACC1','#F4511E','#6D4C41','#546E7A','#FFB300'
]
polarity_weight = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

# clean
def fix_enc(t):
    if not isinstance(t, str): return ''
    try:    return t.encode('latin-1').decode('utf-8')
    except: return t

def clean_text(t):
    if not isinstance(t, str): return ''
    t = fix_enc(t)
    t = re.sub(r'\.{3}\s*[Mm][áa]s\s*$', '', t)
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', t).strip()

# load datos
for enc in ['utf-8', 'latin-1', 'utf-8-sig']:
    try:
        df = pd.read_csv(PATH, encoding=enc)
        print(f'leido con: {enc}')
        break
    except: pass

df['Title']    = df['Title'].apply(fix_enc)
df['Review']   = df['Review'].apply(clean_text)
df['Town']     = df['Town'].apply(fix_enc)
df['Region']   = df['Region'].apply(fix_enc)
df['Polarity'] = pd.to_numeric(df['Polarity'], errors='coerce')
df = df.dropna(subset=['Polarity','Town','Type','Review'])
df['Polarity'] = df['Polarity'].astype(int)
df = df[df['Polarity'].between(1,5)].reset_index(drop=True)
df['full_text'] = df['Title'] + '. ' + df['Review']

towns_list  = sorted(df['Town'].unique().tolist())
town_to_idx = {t: i for i, t in enumerate(towns_list)}

town_meta = df.groupby('Town').agg(
    Region       = ('Region', 'first'),
    n_reviews    = ('Polarity', 'count'),
    avg_polarity = ('Polarity', 'mean'),
    pct_5        = ('Polarity', lambda x: (x==5).mean()*100),
    pct_neg      = ('Polarity', lambda x: (x<=2).mean()*100),
    tipos        = ('Type', lambda x: ' / '.join(sorted(x.unique())))
).reset_index()

print(f'Dataset: {df.shape}  |  {len(towns_list)} pueblos')

# embeddings
MODEL_CACHE = os.path.join(CACHE_DIR, 'model_cache')
if os.path.exists(MODEL_CACHE):
    print('cargando modelo desde cache...')
    model = SentenceTransformer(MODEL_CACHE)
else:
    print(f'descargando {MODEL_NAME}...')
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_CACHE)
DIM = model.get_sentence_embedding_dimension()
print(f'modelo listo  |  {DIM} dims')

# construir embeddings
EMBED_FILE = os.path.join(CACHE_DIR, 'embeddings_cache.pkl')

def build_embeddings():
    print('calculando embeddings (~50 min primera vez)...')
    t0 = time.time()

    # Perfiles Content-Based: promedio ponderado por polaridad
    cb_profiles = {}
    for i, town in enumerate(towns_list):
        sub = df[df['Town'] == town]
        texts = []
        for pol in [1, 2, 3, 4, 5]:
            tp = sub[sub['Polarity']==pol]['full_text'].dropna().tolist()
            n  = max(1, min(len(tp), MAX_PER_TOWN // 5))
            texts.extend(tp[:n])
        if not texts:
            texts = sub['full_text'].dropna().tolist()[:MAX_PER_TOWN]
        emb = model.encode(texts, batch_size=BATCH_SIZE,
                           normalize_embeddings=True, show_progress_bar=False)
        cb_profiles[town] = emb.mean(axis=0)
        if (i+1) % 10 == 0 or (i+1) == len(towns_list):
            print(f'  CB {i+1}/{len(towns_list)}  [{time.time()-t0:.0f}s]')

    # Corpus colaborativo
    corpus_rows = []
    n_per = MAX_CORPUS // len(towns_list)
    for town in towns_list:
        sub = df[df['Town'] == town]
        for pol in [1, 2, 3, 4, 5]:
            pr = sub[sub['Polarity']==pol][['full_text','Town','Polarity']].dropna()
            n  = max(1, min(len(pr), n_per // 5))
            if len(pr) > 0:
                corpus_rows.append(pr.sample(min(n, len(pr)), random_state=RANDOM_SEED))
    corpus_df   = pd.concat(corpus_rows, ignore_index=True)
    corpus_embs = model.encode(corpus_df['full_text'].tolist(), batch_size=BATCH_SIZE,
                               normalize_embeddings=True, show_progress_bar=True)

    # Set de evaluación
    eval_rows = []
    n_per_t = max(1, 800 // len(towns_list))
    for town in towns_list:
        sub = df[df['Town'] == town]
        for pol in [1, 2, 3, 4, 5]:
            ps = sub[sub['Polarity']==pol]['full_text'].dropna().tolist()
            n  = max(0, min(len(ps), max(1, n_per_t // 5)))
            for text in ps[:n]:
                eval_rows.append({'text': text, 'town': town, 'town_idx': town_to_idx[town]})
    eval_df   = pd.DataFrame(eval_rows)
    eval_embs = model.encode(eval_df['text'].tolist(), batch_size=BATCH_SIZE,
                             normalize_embeddings=True, show_progress_bar=True)

    cache = {'cb_profiles': cb_profiles, 'corpus_df': corpus_df,
             'corpus_embs': corpus_embs, 'eval_df': eval_df, 'eval_embs': eval_embs}
    with open(EMBED_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'cache guardado [{time.time()-t0:.0f}s]')
    return cache

if os.path.exists(EMBED_FILE):
    print('cargando embeddings desde cache...')
    with open(EMBED_FILE, 'rb') as f:
        cache = pickle.load(f)
else:
    cache = build_embeddings()

cb_profiles     = cache['cb_profiles']
corpus_df       = cache['corpus_df']
corpus_embs     = cache['corpus_embs']
eval_df         = cache['eval_df']
eval_embs       = cache['eval_embs']
corpus_towns    = corpus_df['Town'].tolist()
corpus_polarity = corpus_df['Polarity'].tolist()
print(f'  CB: {len(cb_profiles)} pueblos  |  Corpus: {len(corpus_df)}  |  Eval: {len(eval_df)}')

# indices faiss + knn
print('construyendo índices...')
t0 = time.time()

X_cb = np.vstack([cb_profiles[t] for t in towns_list]).astype('float32')
faiss.normalize_L2(X_cb)
faiss_index = faiss.IndexFlatIP(DIM)
faiss_index.add(X_cb)

knn_index = NearestNeighbors(n_neighbors=KNN_K + 1, metric='cosine',
                              algorithm='brute', n_jobs=-1)
knn_index.fit(corpus_embs)

# alpha dinamico
# alpha alto → confiar en CB (pueblo con pocas reseñas o baja diversidad)
# alpha bajo → confiar en CF (pueblo con muchas reseñas y alta diversidad)
def compute_alpha(town):
    sub       = df[df['Town'] == town]
    n         = len(sub)
    diversity = sub['Polarity'].std() if len(sub) > 1 else 0
    n_max     = df['Town'].value_counts().max()
    cf_conf   = 0.6 * (n / n_max) + 0.4 * min(diversity / 1.5, 1.0)
    return float(np.clip(1.0 - cf_conf, 0.2, 0.95))

alphas    = {t: compute_alpha(t) for t in towns_list}
alpha_arr = np.array([alphas[t] for t in towns_list])

print('Alpha dinámico (muestra):')
for town in sorted(towns_list, key=lambda t: alphas[t])[:3]:
    print(f'  {town:<28}  α={alphas[town]:.3f}  → más CF')
for town in sorted(towns_list, key=lambda t: alphas[t])[-3:]:
    print(f'  {town:<28}  α={alphas[town]:.3f}  → más CB')

# clustering y umap
sil_scores = {}
for k in range(2, min(10, len(towns_list))):
    lbl = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(X_cb)
    sil_scores[k] = silhouette_score(X_cb, lbl, metric='cosine')
best_k = max(sil_scores, key=sil_scores.get)

kmeans         = KMeans(n_clusters=best_k, random_state=42, n_init=15)
cluster_labels = kmeans.fit_predict(X_cb)
town_meta['cluster'] = cluster_labels

cluster_color = {
    t: PALETTE[int(town_meta[town_meta['Town']==t]['cluster'].values[0]) % len(PALETTE)]
    for t in towns_list
}

reducer_base = umap.UMAP(n_components=2, n_neighbors=min(8, len(towns_list)-1),
                          min_dist=0.08, metric='cosine', random_state=42)
reducer_base.fit(X_cb)

print(f'índices listos en {time.time()-t0:.1f}s')
print(f'   FAISS: {faiss_index.ntotal}  |  KNN: {len(corpus_embs)}  |  k={best_k}')

# metricas para evaluar
def hit_rate(ranks, k): return float(np.mean([1 if r <= k else 0 for r in ranks]))
def mrr(ranks):         return float(np.mean([1.0 / r for r in ranks]))
def ndcg(ranks, k):     return float(np.mean([(1/np.log2(r+1))/(1/np.log2(2)) if r<=k else 0 for r in ranks]))
def rank_correct(s, ti): return int(np.where(np.argsort(s)[::-1] == ti)[0][0]) + 1

# score hibrido
def score_hybrid(user_vec):
    """score = α(pueblo) × CB_score + (1-α) × CF_score"""
    q = user_vec.reshape(1, -1).astype('float32').copy()
    faiss.normalize_L2(q)
    rs, ri = faiss_index.search(q, len(towns_list))
    s_cb   = np.zeros(len(towns_list))
    for r_i, i_i in enumerate(ri[0]): s_cb[i_i] = rs[0][r_i]

    dists, idxs = knn_index.kneighbors([user_vec], n_neighbors=KNN_K + 1)
    s_cf = np.zeros(len(towns_list))
    for sim, ci in zip(1 - dists[0][1:], idxs[0][1:]):
        ti = town_to_idx[corpus_towns[ci]]
        s_cf[ti] += sim * polarity_weight[corpus_polarity[ci]]
    if s_cf.max() > 0: s_cf /= s_cf.max()

    return alpha_arr * s_cb + (1 - alpha_arr) * s_cf

#  evaluacion Leave-One-Out
EVAL_FILE = os.path.join(CACHE_DIR, 'eval_results.json')

if os.path.exists(EVAL_FILE):
    print('cargando métricas desde cache...')
    with open(EVAL_FILE) as f:
        EVAL_RESULTS = json.load(f)
else:
    print('evaluando (primera vez, ~4 min)...')
    t0 = time.time()
    ranks_h, ranks_cb_list = [], []
    for i in range(len(eval_df)):
        uv = eval_embs[i]
        ti = eval_df.iloc[i]['town_idx']
        sh = score_hybrid(uv)
        q  = uv.reshape(1, -1).astype('float32').copy()
        faiss.normalize_L2(q)
        rs, ri = faiss_index.search(q, len(towns_list))
        sc = np.zeros(len(towns_list))
        for r_i, i_i in enumerate(ri[0]): sc[i_i] = rs[0][r_i]
        ranks_h.append(rank_correct(sh, ti))
        ranks_cb_list.append(rank_correct(sc, ti))

    EVAL_RESULTS = {
        'n_eval': len(eval_df),
        'hybrid': {
            'HR@1': hit_rate(ranks_h, 1), 'HR@3': hit_rate(ranks_h, 3),
            'HR@5': hit_rate(ranks_h, 5), 'HR@10': hit_rate(ranks_h, 10),
            'MRR': mrr(ranks_h), 'NDCG@5': ndcg(ranks_h, 5),
            'rank_mean': float(np.mean(ranks_h))
        },
        'content_based': {
            'HR@1': hit_rate(ranks_cb_list, 1), 'HR@3': hit_rate(ranks_cb_list, 3),
            'HR@5': hit_rate(ranks_cb_list, 5), 'HR@10': hit_rate(ranks_cb_list, 10),
            'MRR': mrr(ranks_cb_list), 'NDCG@5': ndcg(ranks_cb_list, 5),
            'rank_mean': float(np.mean(ranks_cb_list))
        }
    }
    with open(EVAL_FILE, 'w') as f:
        json.dump(EVAL_RESULTS, f, indent=2)
    print(f'evaluación lista [{time.time()-t0:.0f}s]')

h  = EVAL_RESULTS['hybrid']
cb = EVAL_RESULTS['content_based']
print(f'\n  {"Método":<22} {"HR@1":>6} {"HR@3":>6} {"HR@5":>6} {"MRR":>6} {"NDCG@5":>8}')
print('  ' + '─' * 55)
print(f'  {"Híbrido dinámico":<22} {h["HR@1"]:>6.3f} {h["HR@3"]:>6.3f} {h["HR@5"]:>6.3f} {h["MRR"]:>6.3f} {h["NDCG@5"]:>8.3f}')
print(f'  {"Content-Based":<22} {cb["HR@1"]:>6.3f} {cb["HR@3"]:>6.3f} {cb["HR@5"]:>6.3f} {cb["MRR"]:>6.3f} {cb["NDCG@5"]:>8.3f}')

# funcion de recomendacion
def get_recommendations(user_query: str, top_n: int = 5):
    user_vec = model.encode([user_query], normalize_embeddings=True,
                            show_progress_bar=False)[0]
    scores   = score_hybrid(user_vec)
    top_idx  = np.argsort(scores)[::-1][:top_n]

    recommendations = []
    for rank, idx in enumerate(top_idx, 1):
        town = towns_list[idx]
        meta = town_meta[town_meta['Town'] == town].iloc[0]
        recommendations.append({
            'rank'      : rank,
            'pueblo'    : town,
            'region'    : meta['Region'],
            'similitud' : round(float(scores[idx]), 4),
            'polarity'  : round(float(meta['avg_polarity']), 2),
            'n_reviews' : int(meta['n_reviews']),
            'pct_neg'   : round(float(meta['pct_neg']), 1),
            'tipos'     : meta['tipos'],
            'alpha'     : round(alphas[town], 3),
            'cluster'   : int(meta['cluster']),
            'color'     : cluster_color[town]
        })

    # Proyectar usuario en UMAP ya entrenado (transform es rápido)
    uv_f32 = user_vec.reshape(1, -1).astype('float32')
    X_plus = np.vstack([X_cb, uv_f32])
    pts    = reducer_base.transform(X_plus)
    d2, u2 = pts[:-1], pts[-1]

    top_towns = [r['pueblo'] for r in recommendations]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')

    for c in range(best_k):
        idxs  = [j for j, t in enumerate(towns_list)
                 if int(town_meta[town_meta['Town']==t]['cluster'].values[0]) == c]
        color = PALETTE[c % len(PALETTE)]
        ax.scatter(d2[idxs,0], d2[idxs,1],
                   s=[220 if towns_list[j] in top_towns else 90 for j in idxs],
                   color=color, alpha=0.75, zorder=3, edgecolors='white', linewidth=0.6)
        for j in idxs:
            is_top = towns_list[j] in top_towns
            ax.annotate(towns_list[j], (d2[j,0], d2[j,1]),
                        textcoords='offset points', xytext=(5,3),
                        fontsize=7 if not is_top else 9,
                        color='#CCCCCC' if not is_top else 'white',
                        fontweight='bold' if is_top else 'normal')

    for rec in recommendations:
        j    = towns_list.index(rec['pueblo'])
        size = 500 - (rec['rank'] - 1) * 70
        ax.scatter(d2[j,0], d2[j,1], s=size, color='#FFD700', zorder=6,
                   edgecolors='#FF6F00', linewidth=2)
        ax.annotate(f"#{rec['rank']} {rec['pueblo']}", (d2[j,0], d2[j,1]),
                    textcoords='offset points', xytext=(8,6),
                    fontsize=9.5, color='#FFD700', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#1a1a2e',
                              edgecolor='#FFD700', alpha=0.92))

    ax.scatter(*u2, s=450, color='#FF4444', zorder=10, marker='*',
               edgecolors='white', linewidth=1.5)
    ax.annotate('Estás aquí', u2, textcoords='offset points', xytext=(10,10),
                fontsize=11, color='#FF4444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0F1117',
                          edgecolor='#FF4444', alpha=0.95))

    patches = [mpatches.Patch(color=PALETTE[c % len(PALETTE)], label=f'Cluster {c}')
               for c in range(best_k)]
    ax.legend(handles=patches, loc='lower right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#333', labelcolor='white',
              title='Clusters', title_fontsize=8)

    title = f'"{user_query[:55]}..."' if len(user_query) > 55 else f'"{user_query}"'
    ax.set_title(f'Mapa de Similitud Turística\n{title}', fontsize=11, color='white', pad=10)
    ax.set_xlabel('UMAP Dimensión 1', color='#888', fontsize=9)
    ax.set_ylabel('UMAP Dimensión 2', color='#888', fontsize=9)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0F1117')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return recommendations, img_b64


# Prueba
print('\nPrueba rápida:')
r, _ = get_recommendations('playa, cenotes y ruinas mayas', top_n=3)
for rec in r:
    print(f'  #{rec["rank"]} {rec["pueblo"]}  sim={rec["similitud"]}  α={rec["alpha"]}')