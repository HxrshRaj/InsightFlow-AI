# nlp_serving: SMS spam classification in Python, TensorFlow, Go and R

A small, end-to-end NLP module for InsightFlow-AI: data prep and modelling in Python (scikit-learn and TensorFlow/Keras), a dependency-free **Go** inference service, an independent **R** check of the metrics, and a **Jupyter** notebook for exploration.

Dataset: SMS Spam Collection (5,574 messages, 13.4% spam), fetched by `download_data.sh`. Stratified 80/20 split, seed 42.

## Results (held-out 1,115 messages, threshold 0.5)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| TF-IDF (uni+bigrams) + logistic regression | 0.979 | 0.940 | 0.959 | 0.991 |
| Keras: TextVectorization + Embedding + pooling + Dense | 0.624 | 0.980 | 0.762 | 0.986 |

The classical baseline wins on F1. The Keras model ranks about as well (AUC) but its class-weighted probabilities are shifted, so at 0.5 it over-flags ham; the notebook shows the threshold sweep. I did not tune it further, and the first version of that model collapsed to all-ham because of an input-shape/`mask_zero` mistake (fixed).

## Layout

- `python/` tokenizer, model file format, baseline, Keras model, metrics, `run.py` (pipeline)
- `go/` stdlib-only service: `/health`, `POST /predict {"text": "..."}`; Dockerfile (distroless)
- `r/verify_metrics.R` recomputes confusion matrix, F1, AUC (rank-based) and a bootstrap F1 CI in base R, then asserts they match the Python numbers
- `notebooks/nlp_exploration.ipynb` data prep, metrics, ROC, threshold sweep, error analysis (executed, outputs saved)
- `tests/` pytest; `go/spam_test.go` Go tests

## Endianness

The model is shipped as a binary file (MDL1) that records its byte order via a marker word (`0x01020304`), and the Go loader detects it instead of assuming the host order. `model_le.bin` and `model_be.bin` hold identical content in opposite byte orders; Go tests assert both decode identically and score within 1e-9 of Python on 208 golden cases (observed max difference 2.2e-16). The service is cross-compiled for `s390x` (big-endian, IBM Z) in CI.

**Not verified:** I have not executed the binary on a real big-endian machine or under emulation; the big-endian path is tested by reading a big-endian *file* on a little-endian host, plus a successful s390x build.

## Run

```bash
cd nlp_serving
pip install -r requirements.txt
./download_data.sh
python -m pytest tests -q
python -m python.run                # trains, writes results/, go/testdata/
(cd go && go test ./... && go run ./cmd/serve -model testdata/model_le.bin)
Rscript r/verify_metrics.R
docker build -t spam-serve go/ && docker run -p 8088:8088 spam-serve
curl -XPOST localhost:8088/predict -d '{"text":"WINNER! free prize, call now"}'
```

## Scope and limits

Tokenizer is ASCII-only by design so Go and Python agree exactly; non-English text is not handled. Not run on Kubernetes; the Dockerfile builds a container but I have not built it in this environment (no Docker daemon).
