# -*- coding: utf-8 -*-
"""
Endpoints:
  GET  /           → app web HTML
  GET  /metrics    → métricas JSON del sistema
  POST /recommend  → { query, top_n } → recomendaciones + mapa UMAP base64
"""

import os, json, pickle, re, io, base64, time, warnings, threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import umap
import faiss

PATH        = 'Rest-Mex_2025_test_with_labels.csv'
CACHE_DIR   = './cache'
PORT        = 5000
KNN_K       = 20
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PALETTE = [
    '#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA',
    '#00ACC1','#F4511E','#6D4C41','#546E7A','#FFB300'
]
polarity_weight = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def fix_enc(t):
    if not isinstance(t, str): return ''
    try:    return t.encode('latin-1').decode('utf-8')
    except: return t

def clean_text(t):
    if not isinstance(t, str): return ''
    t = fix_enc(t)
    t = re.sub(r'\.{3}\s*[Mm][áa]s\s*$', '', t)
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', t).strip()

for enc in ['utf-8', 'latin-1', 'utf-8-sig']:
    try:
        df = pd.read_csv(PATH, encoding=enc); break
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

# modelo y embeddings desde cache
print('cargando modelo y embeddings desde cache...')
MODEL_CACHE = os.path.join(CACHE_DIR, 'model_cache')
model = SentenceTransformer(MODEL_CACHE)
DIM   = model.get_sentence_embedding_dimension()

with open(os.path.join(CACHE_DIR, 'embeddings_cache.pkl'), 'rb') as f:
    cache = pickle.load(f)
cb_profiles     = cache['cb_profiles']
corpus_df       = cache['corpus_df']
corpus_embs     = cache['corpus_embs']
corpus_towns    = corpus_df['Town'].tolist()
corpus_polarity = corpus_df['Polarity'].tolist()

# indices faiss + knn
X_cb = np.vstack([cb_profiles[t] for t in towns_list]).astype('float32')
faiss.normalize_L2(X_cb)
faiss_index = faiss.IndexFlatIP(DIM)
faiss_index.add(X_cb)

knn_index = NearestNeighbors(n_neighbors=KNN_K + 1, metric='cosine',
                              algorithm='brute', n_jobs=-1)
knn_index.fit(corpus_embs)

# alpha dinámico
def compute_alpha(town):
    sub       = df[df['Town'] == town]
    n         = len(sub)
    diversity = sub['Polarity'].std() if len(sub) > 1 else 0
    n_max     = df['Town'].value_counts().max()
    cf_conf   = 0.6 * (n / n_max) + 0.4 * min(diversity / 1.5, 1.0)
    return float(np.clip(1.0 - cf_conf, 0.2, 0.95))

alphas    = {t: compute_alpha(t) for t in towns_list}
alpha_arr = np.array([alphas[t] for t in towns_list])

# clustering y umap
sil_scores = {}
for k in range(2, min(10, len(towns_list))):
    lbl = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(X_cb)
    sil_scores[k] = silhouette_score(X_cb, lbl, metric='cosine')
best_k         = max(sil_scores, key=sil_scores.get)
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

with open(os.path.join(CACHE_DIR, 'eval_results.json')) as f:
    EVAL_RESULTS = json.load(f)

topic_model_v2   = None
topic_dist_df_v2 = None

BT_FILE = os.path.join(CACHE_DIR, 'bertopic_v2_cache.pkl')
if os.path.exists(BT_FILE):
    with open(BT_FILE, 'rb') as f:
        bt = pickle.load(f)
    topic_model_v2 = bt['topic_model']
    bert_df_v2     = bt['bert_df']
    topics_v2      = bt['topics_assigned']
    bert_df_v2['topic'] = topics_v2
    valid_tids = sorted([t for t in set(topics_v2) if t != -1])
    n_valid    = len(valid_tids)
    tid_to_col = {t: i for i, t in enumerate(valid_tids)}
    topic_mat  = np.zeros((len(towns_list), n_valid))
    for j, town in enumerate(towns_list):
        sub = bert_df_v2[bert_df_v2['town'] == town]
        if len(sub) == 0: continue
        for tid in valid_tids:
            topic_mat[j, tid_to_col[tid]] = (sub['topic'] == tid).sum()
        total = topic_mat[j].sum()
        if total > 0: topic_mat[j] /= total
    topic_dist_df_v2 = pd.DataFrame(
        topic_mat, index=towns_list,
        columns=[f'T{t}' for t in valid_tids]
    )
    print(f'BERTopic cargado — {n_valid} tópicos')

# score hibrido
def score_hybrid(user_vec):
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

def get_topic_explanation(town, top_n=3):
    if topic_model_v2 is None or topic_dist_df_v2 is None: return []
    if town not in topic_dist_df_v2.index: return []
    row = topic_dist_df_v2.loc[town]
    top = row.nlargest(top_n)
    result = []
    for col, pct in top.items():
        if pct < 0.03: continue
        tid   = int(col[1:])
        words = topic_model_v2.get_topic(tid)
        label = ' · '.join([w[0] for w in words[:3]]) if words else col
        result.append({'topico': col, 'pct': round(float(pct) * 100, 1), 'label': label})
    return result

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
            'color'     : cluster_color[town],
            'topicos'   : get_topic_explanation(town)
        })

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
                   s=[240 if towns_list[j] in top_towns else 90 for j in idxs],
                   color=color, alpha=0.75, zorder=3, edgecolors='white', linewidth=0.7)
        for j in idxs:
            is_top = towns_list[j] in top_towns
            ax.annotate(towns_list[j], (d2[j,0], d2[j,1]),
                        textcoords='offset points', xytext=(5,3),
                        fontsize=7 if not is_top else 9,
                        color='#BBBBBB' if not is_top else 'white',
                        fontweight='bold' if is_top else 'normal')

    for rec in recommendations:
        j    = towns_list.index(rec['pueblo'])
        size = 500 - (rec['rank'] - 1) * 70
        ax.scatter(d2[j,0], d2[j,1], s=size, color='#FFD700', zorder=6,
                   edgecolors='#FF6F00', linewidth=2)
        ax.annotate(f"#{rec['rank']} {rec['pueblo']}", (d2[j,0], d2[j,1]),
                    textcoords='offset points', xytext=(8,6), fontsize=9.5,
                    color='#FFD700', fontweight='bold',
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
    ax.set_xlabel('Dimensión 1', color='#888', fontsize=9)
    ax.set_ylabel('Dimensión 2', color='#888', fontsize=9)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0F1117')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return recommendations, img_b64


HTML_APP = '''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pueblos Mágicos — Recomendador</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root{--bg:#0a0a0f;--surface:#12121a;--card:#1a1a28;--border:#2a2a40;--accent:#e8b84b;--text:#e8e8f0;--muted:#7878a0;--green:#27ae60;--red:#c0392b;--radius:12px;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;}
  .header{background:linear-gradient(135deg,#0a0a0f,#16162a);border-bottom:1px solid var(--border);padding:24px 40px;display:flex;align-items:center;gap:16px;}
  .header h1{font-family:'Playfair Display',serif;font-size:24px;color:var(--accent);}
  .header p{font-size:12px;color:var(--muted);margin-top:3px;}
  .main{max-width:1300px;margin:0 auto;padding:28px 40px;}
  .grid{display:grid;grid-template-columns:400px 1fr;gap:24px;}
  .panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;}
  .panel-title{font-family:'Playfair Display',serif;font-size:16px;color:var(--accent);margin-bottom:16px;}
  textarea{width:100%;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:14px;padding:12px 14px;resize:vertical;outline:none;transition:border-color .2s;min-height:110px;}
  textarea:focus{border-color:var(--accent);}
  textarea::placeholder{color:var(--muted);}
  .examples{display:flex;flex-direction:column;gap:5px;margin-top:10px;}
  .ebtn{background:var(--card);border:1px solid var(--border);border-radius:7px;color:var(--muted);font-size:12px;padding:7px 11px;cursor:pointer;text-align:left;font-family:'DM Sans',sans-serif;}
  .ebtn:hover{border-color:var(--accent);color:var(--text);}
  .btn-search{width:100%;background:linear-gradient(135deg,var(--accent),#c8941b);border:none;border-radius:8px;color:#0a0a0f;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:500;padding:13px;cursor:pointer;margin-top:16px;}
  .btn-search:disabled{opacity:.5;cursor:not-allowed;}
  .metrics-bar{margin-top:20px;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;}
  .metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px;}
  .metric-item{text-align:center;}
  .metric-val{font-family:'Playfair Display',serif;font-size:19px;color:var(--accent);}
  .metric-lbl{font-size:10px;color:var(--muted);margin-top:2px;}
  .metric-badge{display:inline-block;background:rgba(39,174,96,.15);color:var(--green);border:1px solid rgba(39,174,96,.3);border-radius:4px;font-size:9px;padding:1px 5px;margin-top:2px;}
  .right-panel{display:flex;flex-direction:column;gap:18px;}
  .section-title{font-family:'Playfair Display',serif;font-size:14px;color:var(--accent);margin-bottom:10px;}
  .results-list{display:flex;flex-direction:column;gap:9px;}
  .result-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px;display:grid;grid-template-columns:38px 1fr auto;align-items:center;gap:14px;animation:slideIn .3s ease both;}
  .result-card:nth-child(1){border-left:3px solid #FFD700;}
  .result-card:nth-child(2){border-left:3px solid #C0C0C0;}
  .result-card:nth-child(3){border-left:3px solid #CD7F32;}
  @keyframes slideIn{from{opacity:0;transform:translateX(-10px);}to{opacity:1;transform:translateX(0);}}
  .rank-badge{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:700;flex-shrink:0;background:var(--surface);color:var(--muted);border:1px solid var(--border);}
  .result-name{font-family:'Playfair Display',serif;font-size:15px;color:white;}
  .result-meta{font-size:11px;color:var(--muted);margin-top:3px;display:flex;gap:10px;flex-wrap:wrap;}
  .score-val{font-family:'Playfair Display',serif;font-size:20px;color:var(--accent);}
  .score-label{font-size:10px;color:var(--muted);}
  .sim-bar{width:75px;height:3px;background:var(--border);border-radius:2px;margin-top:4px;overflow:hidden;}
  .sim-fill{height:100%;background:linear-gradient(90deg,var(--accent),#c8941b);border-radius:2px;}
  .rtopics{margin-top:5px;display:flex;flex-wrap:wrap;gap:4px;}
  .ttag{background:rgba(232,184,75,.08);border:1px solid rgba(232,184,75,.2);border-radius:4px;font-size:10px;color:var(--accent);padding:2px 6px;}
  .map-container{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;min-height:280px;display:flex;align-items:center;justify-content:center;}
  .map-container img{width:100%;display:block;}
  .loading{display:none;align-items:center;justify-content:center;gap:10px;padding:36px;color:var(--muted);font-size:13px;}
  .spinner{width:18px;height:18px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .error-msg{background:rgba(192,57,43,.15);border:1px solid rgba(192,57,43,.3);border-radius:8px;color:var(--red);font-size:12px;padding:10px 14px;display:none;margin-top:10px;}
  @media(max-width:900px){.grid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="header">
  <span style="font-size:34px">&#127474;&#127485;</span>
  <div>
    <h1>Pueblos Mágicos &mdash; Recomendador</h1>
    <p>Sistema híbrido CB+CF &middot; embeddings 768d &middot; alpha dinámico &middot; FAISS &middot; BERTopic (explicabilidad)</p>
  </div>
</div>
<div class="main">
<div class="grid">
  <div>
    <div class="panel">
      <div class="panel-title">Describe tu viaje ideal</div>
      <textarea id="query" rows="5" placeholder="Ej: Quiero playa turquesa, cenotes para nadar, ruinas mayas y vida nocturna."></textarea>
      <div class="examples">
        <button class="ebtn" onclick="setQ(this)">Playa, cenotes y ruinas mayas</button>
        <button class="ebtn" onclick="setQ(this)">Cata de vinos y gastronomía regional</button>
        <button class="ebtn" onclick="setQ(this)">Cascadas, senderismo y bosque</button>
        <button class="ebtn" onclick="setQ(this)">Pirámides y zonas arqueológicas</button>
        <button class="ebtn" onclick="setQ(this)">Artesanías y pueblo colonial</button>
      </div>
      <button class="btn-search" id="btnS" onclick="search()">Buscar destino &nbsp;<small style="opacity:.7">(Ctrl+Enter)</small></button>
      <div class="error-msg" id="err"></div>
    </div>
    <div class="metrics-bar">
      <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;">
        Evaluación — Leave-One-Out (n=<span id="nEval">—</span>)
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-val" id="mHR5">—</div>
          <div class="metric-lbl">HR@5</div>
          <div class="metric-badge" id="gHR5"></div>
        </div>
        <div class="metric-item">
          <div class="metric-val" id="mMRR">—</div>
          <div class="metric-lbl">MRR</div>
          <div class="metric-badge" id="gMRR"></div>
        </div>
        <div class="metric-item">
          <div class="metric-val" id="mNDCG">—</div>
          <div class="metric-lbl">NDCG@5</div>
          <div class="metric-badge" id="gNDCG"></div>
        </div>
      </div>
    </div>
  </div>
  <div class="right-panel">
    <div>
      <div class="section-title">Top 5 Recomendaciones</div>
      <div class="loading" id="loadR"><div class="spinner"></div> Calculando...</div>
      <div class="results-list" id="resList">
        <div style="color:var(--muted);font-size:13px;padding:16px 0;">Escribe lo que buscas y presiona el botón.</div>
      </div>
    </div>
    <div>
      <div class="section-title">Mapa de Similitud Turística</div>
      <div class="loading" id="loadM"><div class="spinner"></div> Generando mapa...</div>
      <div class="map-container" id="mapC">
        <div style="text-align:center;color:var(--muted);padding:50px 20px;font-size:13px;">
          El mapa aparecerá aquí.<br>La estrella ★ marca tu posición en el espacio turístico.
        </div>
      </div>
    </div>
  </div>
</div>
</div>
<script>
const EX={
  'Playa, cenotes y ruinas mayas':'Quiero playa de agua turquesa, cenotes para nadar, ruinas mayas y vida nocturna. Me gustan los mariscos frescos.',
  'Cata de vinos y gastronomía regional':'Busco un pueblo tranquilo con bodegas, cata de vinos, cocina regional y ambiente relajado.',
  'Cascadas, senderismo y bosque':'Me gustan las cascadas, senderismo entre bosques, aire frío de sierra y paisajes naturales.',
  'Pirámides y zonas arqueológicas':'Quiero visitar pirámides, zonas arqueológicas y aprender sobre culturas prehispánicas de México.',
  'Artesanías y pueblo colonial':'Busco un pueblo colonial con mercados de artesanías, calles empedradas y buena comida típica.'
};
function setQ(btn){document.getElementById('query').value=EX[btn.textContent]||btn.textContent;}

fetch('/metrics').then(r=>r.json()).then(d=>{
  const h=d.hybrid,cb=d.content_based;
  document.getElementById('nEval').textContent=d.n_eval;
  document.getElementById('mHR5').textContent=h['HR@5'].toFixed(3);
  document.getElementById('mMRR').textContent=h['MRR'].toFixed(3);
  document.getElementById('mNDCG').textContent=h['NDCG@5'].toFixed(3);
  const g=(a,b)=>((a-b)*100).toFixed(1);
  document.getElementById('gHR5').textContent=`+${g(h['HR@5'],cb['HR@5'])}% vs CB`;
  document.getElementById('gMRR').textContent=`+${g(h['MRR'],cb['MRR'])}% vs CB`;
  document.getElementById('gNDCG').textContent=`+${g(h['NDCG@5'],cb['NDCG@5'])}% vs CB`;
}).catch(()=>{});

function renderRecs(recs){
  document.getElementById('resList').innerHTML=recs.map(r=>{
    const pct=Math.max(0,Math.min(100,((r.similitud-0.85)/(1.0-0.85)*100))).toFixed(0);
    const stars='⭐'.repeat(Math.round(r.polarity));
    const topics=(r.topicos||[]).map(t=>`<span class="ttag">${t.label} ${t.pct}%</span>`).join('');
    const alpha=r.alpha>=0.7?`α=${r.alpha} (más contenido)`:r.alpha<=0.4?`α=${r.alpha} (más colaborativo)`:`α=${r.alpha} (equilibrado)`;
    return `<div class="result-card">
      <div class="rank-badge">${r.rank}</div>
      <div>
        <div class="result-name">${r.pueblo}</div>
        <div class="result-meta"><span>${r.region}</span><span>${stars} ${r.polarity}</span><span>${r.n_reviews.toLocaleString()} reseñas</span><span style="color:var(--red)">${r.pct_neg}% neg</span></div>
        ${topics?`<div class="rtopics">${topics}</div>`:''}
        <div style="font-size:9px;color:var(--muted);margin-top:3px;">${alpha}</div>
      </div>
      <div style="text-align:right;flex-shrink:0;">
        <div class="score-val">${r.similitud.toFixed(3)}</div>
        <div class="score-label">similitud</div>
        <div class="sim-bar"><div class="sim-fill" style="width:${pct}%"></div></div>
      </div>
    </div>`;
  }).join('');
}

async function search(){
  const q=document.getElementById('query').value.trim();
  if(!q){showErr('Escribe algo primero.');return;}
  document.getElementById('btnS').disabled=true;
  document.getElementById('loadR').style.display='flex';
  document.getElementById('loadM').style.display='flex';
  document.getElementById('resList').innerHTML='';
  document.getElementById('mapC').innerHTML='';
  document.getElementById('err').style.display='none';
  try{
    const res=await fetch('/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,top_n:5})});
    if(!res.ok) throw new Error();
    const data=await res.json();
    renderRecs(data.recommendations);
    document.getElementById('mapC').innerHTML=`<img src="data:image/png;base64,${data.map_b64}" alt="Mapa">`;
  }catch{showErr('Error de conexión. ¿Está activo el servidor?');}
  finally{
    document.getElementById('btnS').disabled=false;
    document.getElementById('loadR').style.display='none';
    document.getElementById('loadM').style.display='none';
  }
}
function showErr(m){const e=document.getElementById('err');e.textContent=m;e.style.display='block';}
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')search();});
</script>
</body></html>'''

with open('./app.html', 'w', encoding='utf-8') as f:
    f.write(HTML_APP)
print('HTML guardado en ./app.html')

# flask
app = Flask(__name__, static_folder='.')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'app.html')

@app.route('/metrics')
def metrics():
    return jsonify(EVAL_RESULTS)

@app.route('/recommend', methods=['POST'])
def recommend():
    data  = request.get_json()
    query = data.get('query', '').strip()
    top_n = int(data.get('top_n', 5))
    if not query or len(query) < 3:
        return jsonify({'error': 'query inválido'}), 400
    try:
        recs, img_b64 = get_recommendations(query, top_n=top_n)
        return jsonify({'recommendations': recs, 'map_b64': img_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

print('Flask listo')
print(f'   GET  /          → app web')
print(f'   GET  /metrics   → métricas JSON')
print(f'   POST /recommend → recomendaciones + mapa')

# lanzar servidor
# opcion 1: local
# app.run(port=PORT, debug=False)

# opcion 2: ngrok (Colab)
try:
    from pyngrok import ngrok
    # NGROK_TOKEN = 'tu_token_aqui'
    # ngrok.set_auth_token(NGROK_TOKEN)
    threading.Thread(target=lambda: app.run(port=PORT, debug=False, use_reloader=False),
                     daemon=True).start()
    time.sleep(2)
    url = ngrok.connect(PORT)
    h   = EVAL_RESULTS['hybrid']
    print('=' * 55)
    print('SERVIDOR ACTIVO')
    print('=' * 55)
    print(f'   {url}')
    print(f'   HR@5={h["HR@5"]:.3f}  MRR={h["MRR"]:.3f}  NDCG@5={h["NDCG@5"]:.3f}')
    print('=' * 55)
except ImportError:
    print('pyngrok no instalado. Ejecutando en modo local...')
    app.run(host='0.0.0.0', port=PORT, debug=False)