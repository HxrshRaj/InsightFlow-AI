# Independent recomputation of the classifier metrics in base R (no packages), so the numbers
# produced by Python are cross-checked by a second implementation.
# Usage: Rscript r/verify_metrics.R [results/test_predictions.csv] [results/metrics.json]
args <- commandArgs(trailingOnly = TRUE)
csv  <- if (length(args) >= 1) args[1] else "results/test_predictions.csv"
d <- read.csv(csv, stringsAsFactors = FALSE)
stopifnot(all(c("y", "p_tfidf", "p_tf") %in% names(d)))

auc <- function(y, p) {            # Mann-Whitney U with average ranks for ties
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
confusion <- function(y, p, thr = 0.5) {
  yhat <- as.integer(p >= thr)
  tp <- sum(y == 1 & yhat == 1); fp <- sum(y == 0 & yhat == 1)
  fn <- sum(y == 1 & yhat == 0); tn <- sum(y == 0 & yhat == 0)
  prec <- tp / (tp + fp); rec <- tp / (tp + fn)
  c(tp = tp, fp = fp, fn = fn, tn = tn, precision = prec, recall = rec,
    f1 = 2 * prec * rec / (prec + rec), auc = auc(y, p))
}
boot_ci <- function(y, p, fn, B = 2000, seed = 42) {
  set.seed(seed); n <- length(y)
  v <- replicate(B, { i <- sample.int(n, n, replace = TRUE); fn(y[i], p[i]) })
  quantile(v, c(0.025, 0.975), na.rm = TRUE)
}
f1 <- function(y, p) unname(confusion(y, p)["f1"])

out <- list()
for (m in c("tfidf_logreg", "tensorflow_keras")) {
  col <- if (m == "tfidf_logreg") "p_tfidf" else "p_tf"
  cm <- confusion(d$y, d[[col]])
  ci <- boot_ci(d$y, d[[col]], f1)
  cat(sprintf("%-18s tp=%d fp=%d fn=%d tn=%d  P=%.4f R=%.4f F1=%.4f AUC=%.4f  F1 95%% CI [%.3f, %.3f]\n",
              m, cm["tp"], cm["fp"], cm["fn"], cm["tn"], cm["precision"], cm["recall"], cm["f1"], cm["auc"], ci[1], ci[2]))
  out[[m]] <- cm
}

# Cross-check against the Python numbers when metrics.json is available.
mj <- if (length(args) >= 2) args[2] else "results/metrics.json"
if (file.exists(mj)) {
  txt <- paste(readLines(mj, warn = FALSE), collapse = "")
  grab <- function(model, key) {
    blk <- sub(paste0('.*"', model, '": *\\{([^}]*)\\}.*'), "\\1", txt)
    as.numeric(sub(paste0('.*"', key, '": *([-0-9.eE]+).*'), "\\1", blk))
  }
  bad <- 0
  for (m in names(out)) for (k in c("tp", "fp", "fn", "tn", "f1", "auc")) {
    if (abs(grab(m, k) - out[[m]][[k]]) > 1e-6) { cat("MISMATCH", m, k, grab(m, k), out[[m]][[k]], "\n"); bad <- bad + 1 }
  }
  if (bad > 0) quit(status = 1)
  cat("R matches Python metrics.json (tp/fp/fn/tn/F1/AUC within 1e-6)\n")
}
