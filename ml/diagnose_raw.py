"""The two wrong first diagnoses for the unstable 'raw' run, plus the actual fix (seed 0, validation metrics)."""
import train_pytorch as t

cases = [
    ("raw (baseline)", "raw", {}),
    ("raw, lr 1e-4  [hypothesis: LR too high]", "raw", dict(lr=1e-4)),
    ("raw, 300 epochs, patience 300  [hypothesis: needs more epochs]", "raw", dict(epochs=300, patience=300)),
    ("raw, lr 1e-4, 300 epochs", "raw", dict(lr=1e-4, epochs=300, patience=300)),
    ("fixed (scaler fit on train only)", "fixed", {}),
]
for name, variant, upd in cases:
    old = dict(t.HP)
    t.HP.update(upd)
    r = t.run(variant, 0, verbose=False)
    t.HP.clear(); t.HP.update(old)
    print(f"{name:66s} epochs={r['epochs_run']:3d} train_mse={r['train_mse_last']:.3f} "
          f"val_rmse={r['val']['rmse']:.4f} val_r2={r['val']['r2']:.4f}")
