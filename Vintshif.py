#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

# Title
st.title("📊 Scorecard Vintage Analysis (M1, M2, ... buckets)")

# Sidebar: Upload & Settings
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

threshold = st.sidebar.number_input(
    "Threshold for 'Bad' (≥ this value → 1)", 
    min_value=0, max_value=1000, 
    value=60, step=1
)

if uploaded_file is None:
    st.info("👈 Please upload a file in the sidebar.")
    st.stop()

# Load data
try:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.write("### 📁 Raw Data Preview")
st.dataframe(df.head(10))

if df.shape[1] < 3:
    st.error("Input file must have at least 3 columns.")
    st.stop()

# Convert Bucket columns to dates
df.columns = [
    pd.to_datetime(col.split(' ', 1)[1], format='%b-%y').strftime('%Y-%m-%d')
    if col.startswith('Bucket ') else col
    for col in df.columns
]

# Sort OpenDate
df = df.sort_values('OpenDate').reset_index(drop=True)
df['OpenDate'] = df['OpenDate'].dt.to_period('M').dt.start_time


# ------------------------------
# AUTO-SHIFT FUNCTION
# ------------------------------
def auto_shift_rows(df):
    df_copy = df.copy()
    df_copy['OpenDate'] = pd.to_datetime(df_copy['OpenDate'])

    month_cols = [col for col in df_copy.columns if col[:4].isdigit()]
    month_dt_cols = [pd.to_datetime(c) for c in month_cols]

    def shift_row(row):
        open_dt = row['OpenDate']
        start_idx = next((i for i, d in enumerate(month_dt_cols) if d >= open_dt), len(month_cols)-1)
        shifted_cols = month_cols[start_idx:] + month_cols[:start_idx]
        return row[shifted_cols].values

    df_copy[month_cols] = df_copy.apply(shift_row, axis=1, result_type='expand')
    return df_copy


df_aligned = auto_shift_rows(df)

# ------------------------------
# RENAME: first 3 columns same, rest = M1, M2, ...
# ------------------------------
cols = df.columns.tolist()
month_cols = cols[3:]
new_col_names = cols[:3] + [f"M{i+1}" for i in range(len(month_cols))]
df_aligned.columns = new_col_names
st.write("### 🔤 Renamed Columns (First 3 preserved)")
st.write(df_aligned.columns.tolist())

# ------------------------------
# BINARIZE
# ------------------------------
cols_to_update = df_aligned.columns[3:]
df_binarized = df_aligned.copy()
df_binarized[cols_to_update] = df_aligned[cols_to_update].where(
    df_aligned[cols_to_update].isna(),
    (df_aligned[cols_to_update] >= threshold).astype(int)
)

# ------------------------------
# PROPAGATE 1s TO THE RIGHT
# ------------------------------
def propagate_ones(row):
    row = row.copy()
    activated = False
    for i in range(len(row)):
        if not pd.isna(row.iloc[i]):
            if row.iloc[i] == 1:
                activated = True
            if activated:
                row.iloc[i] = 1
    return row

df_propagated = df_binarized.copy()
df_propagated[cols_to_update] = df_binarized[cols_to_update].apply(propagate_ones, axis=1)

st.write("### 🔁 After Propagation (1s spread rightwards)")
st.dataframe(df_propagated.head(10))

# ------------------------------
# SUMMARY
# ------------------------------
sum_vals = df_propagated[cols_to_update].sum()
count_vals = df_propagated[cols_to_update].count()

summary_df = pd.DataFrame({'Sum': sum_vals, 'Count': count_vals})
st.write("### 📊 Column-wise Summary")
st.dataframe(summary_df)

# ------------------------------
# PLOT
# ------------------------------
summary_plot = summary_df
#summary_plot = summary_df.sort_values(by='Count', ascending=False)

fig, ax1 = plt.subplots(figsize=(25, 10))

bars = ax1.bar(summary_plot.index, summary_plot['Count'], label='Observations (Count)')
ax1.set_ylabel('Observations (Count)')
ax1.set_xlabel('Vintage Bucket')

ax2 = ax1.twinx()
line = ax2.plot(summary_plot.index, summary_plot['Sum'],color='red', marker='o', linestyle='-', linewidth=2, label='Bad (Sum)')
ax2.set_ylabel('Bad (Sum)')

plt.title(f'Vintage Buckets: Observations vs Bad (Threshold = {threshold})')
fig.tight_layout()
plt.xticks(rotation=45)

st.pyplot(fig)

# ------------------------------
# ZIP DOWNLOAD
# ------------------------------
csv_processed = df_propagated.to_csv(index=False).encode('utf-8')
csv_summary = summary_df.to_csv().encode('utf-8')

img_buf = io.BytesIO()
fig.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
img_buf.seek(0)

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("processed_vintage_data.csv", csv_processed)
    zf.writestr("vintage_summary.csv", csv_summary)
    zf.writestr("vintage_analysis_plot.png", img_buf.getvalue())

zip_buffer.seek(0)

st.download_button(
    label="📦 Download All (ZIP)",
    data=zip_buffer,
    file_name="vintage_analysis_output.zip",
    mime="application/zip"
)


# In[ ]:




