# -*- coding: utf-8 -*-
"""
  1. Detecta automáticamente el encoding correcto
  2. Corrige texto corrupto (latin-1 doble encodificado)
  3. Limpia las reseñas (caracteres de control, artefactos)
  4. Elimina filas con datos críticos faltantes
  5. Añade columnas de features textuales básicas

"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

PATH = '/content/Rest-Mex_2025_test_with_labels.csv'


# limpieza

def fix_enc(text: str) -> str:
    # corrige texto con doble-encoding latin-1 → utf-8 (ej: Ã³ → ó)."""
    if not isinstance(text, str):
        return ''
    try:
        return text.encode('latin-1').decode('utf-8')
    except Exception:
        return text


def clean_review(text: str) -> str:
    """Limpia una reseña individual."""
    if not isinstance(text, str):
        return ''
    text = fix_enc(text)
    text = re.sub(r'\.{3}\s*[Mm][áa]s\s*$', '', text)   # artefacto "...Más"
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)   # caracteres de control
    return text.strip()


def load_dataset(path: str = PATH) -> pd.DataFrame:
    df_raw = None
    for enc in ['utf-8', 'latin-1', 'utf-8-sig', 'cp1252']:
        try:
            df_raw = pd.read_csv(path, encoding=enc)
            print(f'✓ Leído con encoding: {enc}')
            break
        except Exception as e:
            print(f'  ✗ {enc}: {e}')

    if df_raw is None:
        raise RuntimeError(f'No se pudo leer el archivo: {path}')

    print(f'\nShape original: {df_raw.shape}')
    print(f'Columnas      : {list(df_raw.columns)}')
    return df_raw


# limpieza

def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Corregir encoding en columnas de texto
    df['Title']    = df['Title'].apply(fix_enc)
    df['Review']   = df['Review'].apply(clean_review)
    df['Town']     = df['Town'].apply(fix_enc)
    df['Region']   = df['Region'].apply(fix_enc)

    # Asegurar que Polarity sea numérica
    df['Polarity'] = pd.to_numeric(df['Polarity'], errors='coerce')

    # Eliminar filas sin datos críticos
    before = len(df)
    df = df.dropna(subset=['Polarity', 'Town', 'Type', 'Review'])
    df['Polarity'] = df['Polarity'].astype(int)
    df = df[df['Polarity'].between(1, 5)].reset_index(drop=True)
    print(f'\nFilas: {before:,} → {len(df):,}  (eliminadas: {before - len(df):,})')

    # Features textuales básicas
    df['n_chars']   = df['Review'].str.len()
    df['n_words']   = df['Review'].str.split().str.len()
    df['n_sents']   = df['Review'].str.count(r'[.!?]+')
    df['title_len'] = df['Title'].str.len()
    df['full_text'] = df['Title'] + '. ' + df['Review']

    return df

def print_summary(df: pd.DataFrame) -> None:
    sep = '═' * 55
    print(sep)
    print('RESUMEN DEL DATASET')
    print(sep)
    print(f'  Reseñas totales       : {len(df):>8,}')
    print(f'  Pueblos únicos        : {df["Town"].nunique():>8,}')
    print(f'  Regiones únicas       : {df["Region"].nunique():>8,}')
    print(f'  Tipos de lugar        : {list(df["Type"].unique())}')
    print(f'  Rango Polarity        : {df["Polarity"].min()} – {df["Polarity"].max()}')
    print(f'  Polarity media global : {df["Polarity"].mean():>8.3f}')
    print(f'  Polarity mediana      : {df["Polarity"].median():>8.1f}')
    print(sep)
    print('\n  Valores nulos por columna:')
    print(df.isnull().sum().to_frame('nulos').T.to_string())
    print(sep)

if __name__ == '__main__':
    df_raw = load_dataset()
    df     = clean_dataset(df_raw)
    print_summary(df)
    print('\nPrimeras filas:')
    print(df.head(3).to_string())
