# Independent check of the A/B statistics in base R (no packages), so the Python numbers in
# results/ab_results.json are confirmed by a second implementation.
# Usage (from repo root): Rscript fairness_ab/r/verify_ab.R [fairness_ab/results/ab_results.json]
args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else "fairness_ab/results/ab_results.json"
txt <- paste(readLines(path, warn = FALSE), collapse = " ")

num <- function(key, from = txt) {                 # first numeric value after "key":
  m <- regmatches(from, regexec(paste0('"', key, '": *([-0-9.eE+]+)'), from))[[1]]
  as.numeric(m[2])
}
block <- function(key) regmatches(txt, regexec(paste0('"', key, '": *\\{([^}]*(\\{[^}]*\\}[^}]*)*)\\}'), txt))[[1]][1]

bad <- 0
check <- function(name, r_val, py_val, tol) {
  ok <- abs(r_val - py_val) <= tol
  cat(sprintf("%-34s R=%-12.6g Python=%-12.6g %s\n", name, r_val, py_val, if (ok) "OK" else "MISMATCH"))
  if (!ok) bad <<- bad + 1
}

# 1. Two-proportion z-test: 12.0% vs 13.5%. Python's example is 480/4000 vs 540/4000 (ab.py main()).
zb <- block("z_test_example")
p1 <- num("p1", zb); p2 <- num("p2", zb)
n <- 4000; x1 <- round(p1 * n); x2 <- round(p2 * n)
pt <- prop.test(c(x2, x1), c(n, n), correct = FALSE)   # chi-square without continuity correction == z^2
check("z-test p-value", pt$p.value, num("p_value", zb), 1e-6)
check("z-test |z|", sqrt(unname(pt$statistic)), abs(num("z", zb)), 1e-6)

# 2. McNemar on discordant predictions: exact binomial test on 326 vs 596.
mc <- block("mcnemar")
b <- num("A_right_B_wrong", mc); c2 <- num("A_wrong_B_right", mc)
bt <- binom.test(c2, b + c2, p = 0.5)
cat(sprintf("%-34s R(exact)=%.3g  Python=%.3g\n", "McNemar p-value", bt$p.value, num("p_value", mc)))
# Both should be astronomically small; compare on the log10 scale so the check is meaningful.
check("McNemar log10(p), tolerance 0.5", log10(bt$p.value), log10(num("p_value", mc)), 0.5)

# 3. Sample size per arm (alpha 0.05, power 0.8, baseline 10%): R's power.prop.test uses a
# slightly different variance formula than the normal approximation in Python, so a small gap is expected.
s <- block("sample_size_per_arm")
for (d in c(0.01, 0.02)) {
  r_n <- ceiling(power.prop.test(p1 = 0.10, p2 = 0.10 + d, sig.level = 0.05, power = 0.8)$n)
  py_n <- num(if (d == 0.01) "baseline_10pct_detect_\\+1pt" else "baseline_10pct_detect_\\+2pt", s)
  check(sprintf("sample size, detect +%d pt", round(d * 100)), r_n, py_n, 0.01 * py_n)
}

if (bad > 0) { cat(sprintf("\n%d check(s) failed\n", bad)); quit(status = 1) }
cat("\nAll R checks match the Python results.\n")
