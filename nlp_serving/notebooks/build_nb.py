import nbformat as nbf
nb = nbf.v4.new_notebook()
C = lambda s: nbf.v4.new_code_cell(s)
M = lambda s: nbf.v4.new_markdown_cell(s)
nb.cells = [
M("# SMS spam: data prep, classical ML vs a small TensorFlow model\nRun from `nlp_serving/` after `./download_data.sh` and `python -m python.run`. Everything here reads artefacts produced by that pipeline, so the numbers match the README, Go parity tests and the R check."),
C("import json, sys, os\nsys.path.insert(0, '..'); os.chdir('..')\nimport numpy as np, pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfrom python import data, tokenizer\ndf = data.load()\ndf['n_tokens'] = df.text.map(lambda t: len(tokenizer.tokens(t)))\nprint(df.shape); print(df.label.value_counts())"),
M("## Data preparation\nClass balance is about 87/13, so accuracy is misleading. We use stratified splitting, `class_weight='balanced'`, and report precision, recall, F1 and AUC."),
C("df.groupby('label').n_tokens.describe()"),
C("tr, te = data.split(df)\nprint(len(tr), len(te), tr.y.mean().round(4), te.y.mean().round(4))"),
M("## Tokenizer (shared with the Go service)\nASCII-only on purpose, so Python and Go agree byte for byte."),
C("tokenizer.analyze('WINNER!! Claim your FREE prize, call 09061701461 now')"),
M("## Results"),
C("m = json.load(open('results/metrics.json'))\npd.DataFrame({k: v for k, v in m.items() if isinstance(v, dict)}).T[['tp','fp','fn','tn','precision','recall','f1','auc']].round(4)"),
M("The Keras model ranks messages almost as well (AUC) but at threshold 0.5 over-flags ham, because the class weights push its probabilities up. The threshold is a deployment choice, so the next cells look at it."),
C("p = pd.read_csv('results/test_predictions.csv')\nfrom python.metrics import summarise\nrows = []\nfor name, col in [('tfidf_logreg','p_tfidf'),('tensorflow_keras','p_tf')]:\n    for thr in (0.3,0.5,0.7,0.9):\n        s = summarise(p.y.to_numpy(), p[col].to_numpy(), thr)\n        rows.append({'model': name, 'threshold': thr, **{k: round(s[k],3) for k in ('precision','recall','f1')}})\npd.DataFrame(rows)"),
C("from sklearn.metrics import roc_curve\nfig, ax = plt.subplots(figsize=(4.5,4))\nfor name, col in [('TF-IDF + LogReg','p_tfidf'),('Keras','p_tf')]:\n    f, t, _ = roc_curve(p.y, p[col]); ax.plot(f, t, label=name)\nax.plot([0,1],[0,1],'k:'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(); ax.set_title('ROC on held-out 20%')\nplt.show()"),
M("## Error analysis (baseline false negatives)"),
C("te = te.reset_index(drop=True)\nfn = te[(te.y==1) & (p.p_tfidf.values < 0.5)]\nfor t in fn.text.head(5): print('-', t[:140])"),
]
nbf.write(nb, 'notebooks/nlp_exploration.ipynb')
