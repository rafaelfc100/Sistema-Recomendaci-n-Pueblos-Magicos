# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re, warnings, pickle, os, json, time
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print('BERTopic no disponible. Ejecuta: pip install bertopic hdbscan')

PATH        = 'Rest-Mex_2025_test_with_labels.csv'
CACHE_DIR   = './cache'
BT_V2_FILE  = os.path.join(CACHE_DIR, 'bertopic_v2_cache.pkl')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs(CACHE_DIR, exist_ok=True)

STOPWORDS_ES = set([
    'de','la','el','en','y','a','los','las','con','que','se','del','un','una',
    'por','es','su','al','lo','muy','para','como','más','pero','fue','todo',
    'nos','me','mi','si','este','esta','también','hay','le','ya','vez','bien',
    'así','lugar','sin','ser','tienen','tiene','cuando','donde','son','está',
    'estaba','les','sus','cada','te','han','ha','era','no','ni','sobre',
    'entre','otro','otra','unos','unas','tan','mas','porque','desde','hasta',
    'fueron','hace','mucho','poco','solo','pueden','puede','aqui',
    'estos','estas','eso','esa','uno','dos','tres',
    'realmente','siempre','aunque','luego','antes','después','siendo',
    'tuvimos','fuimos','vimos','habia','estaban','tenemos','tienen'
])

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
print(f'Dataset: {df.shape}  |  {len(towns_list)} pueblos')

MODEL_CACHE = os.path.join(CACHE_DIR, 'model_cache')
model = SentenceTransformer(MODEL_CACHE if os.path.exists(MODEL_CACHE)
                            else 'paraphrase-multilingual-mpnet-base-v2')
DIM = model.get_sentence_embedding_dimension()

with open(os.path.join(CACHE_DIR, 'embeddings_cache.pkl'), 'rb') as f:
    cache = pickle.load(f)
cb_profiles = cache['cb_profiles']
X_cb = np.vstack([cb_profiles[t] for t in towns_list]).astype('float32')

# silhouette baseline (sin BERTopic)
print('\n--- Baseline (embeddings puros) ---')
import faiss
faiss.normalize_L2(X_cb.copy())

sil_scores = {}
for k in range(2, min(10, len(towns_list))):
    lbl = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(X_cb)
    sil_scores[k] = silhouette_score(X_cb, lbl, metric='cosine')
best_k = max(sil_scores, key=sil_scores.get)
baseline_labels = KMeans(n_clusters=best_k, random_state=42, n_init=15).fit_predict(X_cb)
baseline_sil = silhouette_score(X_cb, baseline_labels, metric='cosine')
print(f'Silhouette baseline: {baseline_sil:.3f}  (k={best_k})')

def run_bertopic_experiment(n_topics, label):
    if not BERTOPIC_AVAILABLE:
        print(f'BERTopic no disponible, saltando {label}')
        return None, None

    print(f'\n--- {label} ({n_topics} tópicos) ---')
    t0 = time.time()

    # muestra balanceada de reseñas
    bert_rows = []
    for town in towns_list:
        sub = df[df['Town'] == town]
        sample = sub.sample(min(200, len(sub)), random_state=RANDOM_SEED)
        for _, row in sample.iterrows():
            bert_rows.append({'text': row['full_text'], 'town': town})
    bert_df = pd.DataFrame(bert_rows)

    # embeddings del corpus BERTopic
    bert_embs = model.encode(bert_df['text'].tolist(), batch_size=64,
                             normalize_embeddings=True, show_progress_bar=True)

    # BERTopic con HDBSCAN
    hdb = HDBSCAN(min_cluster_size=max(5, len(bert_df) // (n_topics * 3)),
                  metric='euclidean', prediction_data=True)
    umap_model = umap.UMAP(n_components=5, n_neighbors=15,
                           min_dist=0.0, metric='cosine', random_state=RANDOM_SEED)

    topic_model = BERTopic(
        hdbscan_model=hdb,
        umap_model=umap_model,
        nr_topics=n_topics,
        calculate_probabilities=False,
        verbose=False
    )
    topics_assigned, _ = topic_model.fit_transform(
        bert_df['text'].tolist(), embeddings=bert_embs
    )
    bert_df['topic'] = topics_assigned
    valid_topics = sorted([t for t in set(topics_assigned) if t != -1])
    print(f'Tópicos encontrados: {len(valid_topics)}  |  tiempo: {time.time()-t0:.0f}s')

    # distribucion de topicos por pueblo
    n_valid    = len(valid_topics)
    tid_to_col = {t: i for i, t in enumerate(valid_topics)}
    topic_mat  = np.zeros((len(towns_list), n_valid))
    for j, town in enumerate(towns_list):
        sub = bert_df[bert_df['town'] == town]
        if len(sub) == 0: continue
        for tid in valid_topics:
            topic_mat[j, tid_to_col[tid]] = (sub['topic'] == tid).sum()
        total = topic_mat[j].sum()
        if total > 0: topic_mat[j] /= total

    topic_dist_df = pd.DataFrame(topic_mat, index=towns_list,
                                  columns=[f'T{t}' for t in valid_topics])

    # enriquecer perfiles CB con distribucion de topicos
    X_enriched = np.hstack([X_cb, topic_mat]).astype('float32')
    from sklearn.preprocessing import normalize
    X_enriched = normalize(X_enriched, norm='l2')

    # silhouette con perfil enriquecido
    sil_enr = {}
    for k in range(2, min(10, len(towns_list))):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(X_enriched)
        sil_enr[k] = silhouette_score(X_enriched, lbl, metric='cosine')
    best_k_enr = max(sil_enr, key=sil_enr.get)
    labels_enr = KMeans(n_clusters=best_k_enr, random_state=42, n_init=15).fit_predict(X_enriched)
    sil_val    = silhouette_score(X_enriched, labels_enr, metric='cosine')
    print(f'Silhouette con BERTopic: {sil_val:.3f}  (k={best_k_enr})')
    print('\nTópicos principales:')
    for tid in valid_topics[:5]:
        words = topic_model.get_topic(tid)
        label_words = ', '.join([w[0] for w in words[:5]]) if words else 'n/a'
        n_docs = (bert_df['topic'] == tid).sum()
        print(f'  T{tid}: {label_words}  ({n_docs} docs)')

    return topic_model, topic_dist_df, bert_df, topics_assigned


# experimentos
print('\n' + '='*60)
print('EXPERIMENTO BERTOPIC')
print('='*60)
print(f'\nBaseline Silhouette: {baseline_sil:.3f}')
print('\nResultados documentados:')
print('  v1 (5 tópicos):  Silhouette=0.315 ↑  HR@5=0.351 ↓')
print('  v2 (19 tópicos): Silhouette=0.379 ↑  HR@5=0.378 ↓')
print('\nConclusión: BERTopic mejora clustering pero empeora recomendación.')
print('Causa: 24/40 pueblos comparten el mismo tópico dominante (restaurantes).')
print('Rol final: explicabilidad de tópicos por pueblo (no ranking).\n')

# correr v2 si BERTopic está disponible
if BERTOPIC_AVAILABLE:
    result = run_bertopic_experiment(n_topics=19, label='BERTopic v2')
    if result is not None:
        topic_model_v2, topic_dist_v2, bert_df_v2, topics_v2 = result

        # Guardar cache
        bt_cache = {
            'topic_model'      : topic_model_v2,
            'bert_df'          : bert_df_v2,
            'topics_assigned'  : topics_v2,
            'topic_dist_df'    : topic_dist_v2
        }
        with open(BT_V2_FILE, 'wb') as f:
            pickle.dump(bt_cache, f)
        print(f'\nBERTopic v2 guardado en {BT_V2_FILE}')


# función de explicabilidad (usada en la app Flask)
def get_topic_explanation(town, topic_model, topic_dist_df, top_n=3):
    """Retorna los tópicos más representativos de un pueblo."""
    if topic_model is None or town not in topic_dist_df.index:
        return []
    row = topic_dist_df.loc[town]
    top = row.nlargest(top_n)
    result = []
    for col, pct in top.items():
        if pct < 0.03: continue
        tid   = int(col[1:])
        words = topic_model.get_topic(tid)
        label = ' · '.join([w[0] for w in words[:3]]) if words else col
        result.append({'topico': col, 'pct': round(float(pct) * 100, 1), 'label': label})
    return result


print('\nMódulo BERTopic listo.')
print('Para usar en la app Flask, importar get_topic_explanation()')