import os,pandas as pd
base=r"C:\Users\chenz\OneDrive\桌面\预训练微调前结果"
raw=os.path.join(base,'guided_vs_no_nn_raw_20260416_212459.csv')
df=pd.read_csv(raw)
print('RAW_MULTI_METRIC_BEGIN')
for metric in ['guided_cost','no_nn_cost','difference','improve_pct_vs_no_nn']:
    if metric in df.columns:
        gp=df.groupby('dataset')[metric].agg(['count','mean','std','min','max']).reset_index()
        gp['spread']=gp['max']-gp['min']
        print(f'metric={metric}')
        for _,r in gp.sort_values('dataset').iterrows():
            stdv=0.0 if pd.isna(r['std']) else float(r['std'])
            print(f"raw2|metric={metric}|dataset={r['dataset']}|n={int(r['count'])}|mean={float(r['mean']):.10g}|std={stdv:.10g}|spread={float(r['spread']):.10g}")
print('RAW_MULTI_METRIC_END')
