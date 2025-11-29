# experiment.py
import os, json, time, argparse, pathlib, zipfile, io, urllib.request, pickle, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from metrics import calculate_energy_metrics, print_metrics

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
RANDOM_SEED = 42

# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x):  # (B, L, d)
        L = x.size(1)
        return x + self.pe[:, :L, :]

class EnergyTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_length, dropout=0.1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_length)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, x):                # x: (B, L, input_dim)
        x = self.input_embedding(x)      # (B, L, d)
        x = self.positional_encoding(x)  # (B, L, d)
        x = self.encoder(x)              # (B, L, d)
        x = x.mean(dim=1)                # (B, d)
        x = self.dropout(x)
        return self.out(x)               # (B, output_dim)

# ---------------- Data ----------------
class EnergyDataset(Dataset):
    def __init__(self, df, seq_length, pred_length, feature_cols, target_col):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.feat = feature_cols
        self.tgt = target_col

        df = df.copy()
        for col in self.feat + [self.tgt]:
            m, s = df[col].mean(), df[col].std()
            df[col] = (df[col] - m) / (s + 1e-8)

        self.X, self.Y = [], []
        N = len(df)
        for i in range(N - seq_length - pred_length + 1):
            self.X.append(df.iloc[i:i+seq_length][self.feat].values)
            self.Y.append(df.iloc[i+seq_length:i+seq_length+pred_length][self.tgt].values)

        self.X = np.asarray(self.X, dtype=np.float32)
        self.Y = np.asarray(self.Y, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

def _maybe_download_and_extract_txt(target_txt: str):
    """If txt not exists, download zip and extract the .txt into datasets/"""
    txt_path = pathlib.Path(target_txt)
    if txt_path.exists():
        return
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DATA] {txt_path} not found, downloading UCI zip...")
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # the zip contains 'household_power_consumption.txt'
        for name in zf.namelist():
            if name.endswith(".txt"):
                with zf.open(name) as fsrc, open(txt_path, "wb") as fdst:
                    fdst.write(fsrc.read())
                break
    print(f"[DATA] Saved to {txt_path}")

def load_energy_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess energy data with caching support"""
    _maybe_download_and_extract_txt(data_path)
    df = pd.read_csv(
        data_path,
        sep=";",
        low_memory=False,
        na_values=["?"],
    )

    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H:%M:%S',
            dayfirst=True
        )
        df = df.drop(['Date', 'Time'], axis=1)

    # 数据类型转换
    num_cols = [
        "Global_active_power","Global_reactive_power","Voltage","Global_intensity",
        "Sub_metering_1","Sub_metering_2","Sub_metering_3"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 数据清理
    df = df.dropna().sort_values("datetime").reset_index(drop=True)

    # 时间特征
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # 目标变量
    if "total_energy" not in df.columns:
        df["total_energy"] = df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"] + df["Global_active_power"]
    return df

def create_loaders(data_path, seq_length, pred_length, batch_size, val_split=0.2, device="cuda", cache_dir=None):
    """
    Create train/val DataLoaders. To avoid repeated preprocessing, this function caches the
    preprocessed (X, Y) numpy arrays into a compressed .npz file under `cache_dir`.

    Note: normalization is computed on the training split and applied to both train and val.
    This differs from the previous behavior where each EnergyDataset normalized independently.
    """
    data_path = pathlib.Path(data_path)
    df = load_energy_data(str(data_path))

    feat_cols = [
        "Global_active_power","Global_reactive_power","Voltage","Global_intensity",
        "Sub_metering_1","Sub_metering_2","Sub_metering_3","hour","day_of_week","month"
    ]
    tgt = "total_energy"

    # cache directory: default to the same folder as the data file (datasets/)
    if cache_dir is None:
        cache_dir = data_path.parent
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # create a cache filename dependent on key params
    base_name = data_path.stem
    cache_file = cache_dir / f"{base_name}_s{seq_length}_p{pred_length}_v{int(val_split*100)}.npz"

    if cache_file.exists():
        try:
            npz = np.load(cache_file, allow_pickle=True)
            X_train = npz["X_train"]
            Y_train = npz["Y_train"]
            X_val = npz["X_val"]
            Y_val = npz["Y_val"]
            feat_cols = list(npz["feat_cols"].tolist()) if "feat_cols" in npz else feat_cols
            input_dim = int(npz["input_dim"]) if "input_dim" in npz else len(feat_cols)
            print(f"[DATA] Loaded cached arrays from {cache_file}")

            # build TensorDatasets and DataLoaders
            train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
            val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

            num_workers = min(8, os.cpu_count() or 4)
            kwargs = dict(
                batch_size=batch_size,
                pin_memory=(str(device).startswith("cuda")),
                persistent_workers=True,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            train_loader = DataLoader(train_ds, shuffle=True, num_workers=num_workers, **kwargs)
            val_loader = DataLoader(val_ds, shuffle=False, num_workers=num_workers, **kwargs)
            return train_loader, val_loader, input_dim
        except Exception as e:
            print(f"[DATA] Failed to load cache (will rebuild): {e}")

    # build raw train/val split from dataframe
    n_val = int(len(df) * val_split)
    train_df = df.iloc[:-n_val].copy() if n_val > 0 else df.copy()
    val_df = df.iloc[-n_val:].copy() if n_val > 0 else df.iloc[-1000:].copy()

    # compute normalization stats on training split and apply to both splits
    stats = {}
    for col in feat_cols + [tgt]:
        m = train_df[col].mean()
        s = train_df[col].std()
        stats[col] = (m, s + 1e-8)
        train_df[col] = (train_df[col] - m) / (s + 1e-8)
        val_df[col] = (val_df[col] - m) / (s + 1e-8)

    # helper to slice windows into arrays
    def slice_windows(df_local):
        Xs, Ys = [], []
        N = len(df_local)
        for i in range(N - seq_length - pred_length + 1):
            Xs.append(df_local.iloc[i:i+seq_length][feat_cols].values)
            Ys.append(df_local.iloc[i+seq_length:i+seq_length+pred_length][tgt].values)
        if len(Xs) == 0:
            return np.zeros((0, seq_length, len(feat_cols)), dtype=np.float32), np.zeros((0, pred_length), dtype=np.float32)
        return np.asarray(Xs, dtype=np.float32), np.asarray(Ys, dtype=np.float32)

    X_train, Y_train = slice_windows(train_df)
    X_val, Y_val = slice_windows(val_df)

    # save cache
    try:
        np.savez_compressed(cache_file, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                            feat_cols=np.array(feat_cols, dtype=object), tgt=tgt, input_dim=len(feat_cols))
        print(f"[DATA] Saved preprocessed arrays to {cache_file}")
    except Exception as e:
        print(f"[DATA] Failed to save cache: {e}")

    # build dataloaders from arrays
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    num_workers = min(8, os.cpu_count() or 4)
    kwargs = dict(
        batch_size=batch_size,
        pin_memory=(str(device).startswith("cuda")),
        persistent_workers=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=num_workers, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=num_workers, **kwargs)
    return train_loader, val_loader, len(feat_cols)

# ---------------- Train / Eval ----------------
def train_epoch(model, loader, crit, opt, device, use_amp=False):
    model.train()
    total = 0.0
    scaler = torch.amp.GradScaler() if use_amp else None

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        opt.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                pred = model(xb)
                # align pred and yb shapes for loss computation
                if pred.ndim == 2 and yb.ndim == 2:
                    loss = crit(pred, yb)
                elif pred.ndim == 2 and yb.ndim == 1:
                    if pred.shape[1] == 1:
                        loss = crit(pred.squeeze(-1), yb)
                    else:
                        loss = crit(pred[:, -1], yb)
                elif pred.ndim == 1 and yb.ndim == 2:
                    if yb.shape[1] == 1:
                        loss = crit(pred, yb.squeeze(-1))
                    else:
                        loss = crit(pred, yb[:, -1])
                else:
                    loss = crit(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(xb)
            # align pred and yb shapes for loss computation
            if pred.ndim == 2 and yb.ndim == 2:
                loss = crit(pred, yb)
            elif pred.ndim == 2 and yb.ndim == 1:
                if pred.shape[1] == 1:
                    loss = crit(pred.squeeze(-1), yb)
                else:
                    loss = crit(pred[:, -1], yb)
            elif pred.ndim == 1 and yb.ndim == 2:
                if yb.shape[1] == 1:
                    loss = crit(pred, yb.squeeze(-1))
                else:
                    loss = crit(pred, yb[:, -1])
            else:
                loss = crit(pred, yb)
            loss.backward()
            opt.step()

        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, crit, device, use_amp=False):
    model.eval()
    total = 0.0
    all_preds, all_tgts = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                out = model(xb)
        else:
            out = model(xb)

        # align out and yb shapes for loss
        if out.ndim == 2 and yb.ndim == 2:
            loss = crit(out, yb)
        elif out.ndim == 2 and yb.ndim == 1:
            if out.shape[1] == 1:
                loss = crit(out.squeeze(-1), yb)
            else:
                loss = crit(out[:, -1], yb)
        elif out.ndim == 1 and yb.ndim == 2:
            if yb.shape[1] == 1:
                loss = crit(out, yb.squeeze(-1))
            else:
                loss = crit(out, yb[:, -1])
        else:
            loss = crit(out, yb)

        total += float(loss.item())

        # 分批收集结果，避免内存爆炸 — 保留原始形状以便 calculate_energy_metrics 处理
        all_preds.append(out.detach().cpu().numpy())
        all_tgts.append(yb.detach().cpu().numpy())

    # 合并所有batch的结果
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    tgts  = np.concatenate(all_tgts)  if all_tgts else np.array([])

    metrics = calculate_energy_metrics(tgts, preds)
    return total / max(1, len(loader)), metrics

@torch.no_grad()
def _get_predictions(model, loader, device, use_amp=False, max_samples=1000):
    """获取模型的预测结果，用于可视化"""
    model.eval()
    all_preds, all_tgts = [], []
    sample_count = 0

    for xb, yb in loader:
        if sample_count >= max_samples:
            break
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                out = model(xb)
        else:
            out = model(xb)

        all_preds.append(out.detach().cpu().numpy())
        all_tgts.append(yb.detach().cpu().numpy())
        sample_count += len(xb)

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    tgts = np.concatenate(all_tgts) if all_tgts else np.array([])
    return preds, tgts

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Energy Consumption Forecasting")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--data_path", type=str, default="datasets/household_power_consumption.txt")
    ap.add_argument("--seq_length", type=int, default=48)
    ap.add_argument("--pred_length", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=1)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_interval", type=int, default=1)
    args, _ = ap.parse_known_args()

    # seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    try:
        print("[INFO] Loading data ...")
        # store cache next to the original data file (datasets/), not in the run out_dir
        cache_folder = pathlib.Path(args.data_path).parent
        train_loader, val_loader, input_dim = create_loaders(
            args.data_path, args.seq_length, args.pred_length, args.batch_size, device=str(device), cache_dir=cache_folder
        )
        print(f"[INFO] Data ready. input_dim={input_dim}, "
              f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

        model = EnergyTransformer(
            input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, output_dim=args.pred_length,
            seq_length=args.seq_length, dropout=args.dropout
        ).to(device)
        print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")

        crit = nn.MSELoss()
        opt  = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

        best_loss, best_metrics, patience = float("inf"), None, 0
        best_preds, best_tgts = None, None  # 保存最佳模型的预测结果用于可视化

        print(f"[INFO] Start training (max_epochs={args.max_epochs})")
        for epoch in range(1, args.max_epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, crit, opt, device)

            if epoch % args.val_interval == 0:
                va_loss, va_metrics = validate(model, val_loader, crit, device)

                if va_loss < best_loss:
                    best_loss, best_metrics, patience = va_loss, va_metrics, 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "val_loss": best_loss,
                            "metrics": best_metrics,
                        },
                        out_dir / "best_model.pth",
                    )
                    # 保存最佳模型的预测结果用于可视化
                    best_preds, best_tgts = _get_predictions(model, val_loader, device)
                    print(f"[E{epoch}] improved: val_loss={va_loss:.4f} (saved)")
                else:
                    patience += 1
                    print(f"[E{epoch}] val_loss={va_loss:.4f} (no improve, patience={patience})")

                print_metrics(va_metrics, prefix=f"[E{epoch}] ")

                if patience >= args.patience:
                    print(f"[INFO] Early stopping at epoch {epoch}")
                    break

                sched.step(va_loss)
            else:
                sched.step(tr_loss)

            print(f"[E{epoch}] time={time.time() - t0:.1f}s, train_loss={tr_loss:.4f}")

        print(f"[INFO] Training completed after {args.max_epochs} epochs")

        # ----- Save final artifacts -----
        # metrics.json （便于上层解析）
        primary = float(best_metrics.get("rmse", best_loss if best_loss < 1e9 else np.nan)) if best_metrics else best_loss
        metrics_payload = {"metric": "rmse", "value": primary, "aux": best_metrics or {}}
        (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

        # final_info.json
        # Align format with other tasks: top-level key is task name -> {"means": {...}}
        def _to_serializable(x):
            """Recursively convert numpy scalars/arrays to Python native types for JSON."""
            if isinstance(x, dict):
                return {k: _to_serializable(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_serializable(v) for v in x]
            try:
                # numpy types
                import numpy as _np
                if isinstance(x, (_np.floating,)):
                    return float(x)
                if isinstance(x, (_np.integer,)):
                    return int(x)
                if isinstance(x, _np.ndarray):
                    return _to_serializable(x.tolist())
            except Exception:
                pass
            # fallback for native python
            if isinstance(x, (float, int, str, type(None), bool)):
                return x
            return str(x)

        means = _to_serializable(best_metrics or {})
        import numpy as _np
        best_loss_value = float(best_loss) if (isinstance(best_loss, (float, int)) and _np.isfinite(best_loss)) else None
        print("Best validation loss:", best_loss_value)

        final_info = {
            "AutoForecast": {
                "means": means
            }
        }

        (out_dir / "final_info.json").write_text(json.dumps(final_info, ensure_ascii=False, indent=2))

        # 保存预测结果用于可视化（只保存部分样本以避免文件过大）
        if best_preds is not None and best_tgts is not None:
            # 限制保存的样本数量（最多1000个样本）
            max_plot_samples = 1000
            if len(best_preds) > max_plot_samples:
                indices = np.linspace(0, len(best_preds) - 1, max_plot_samples, dtype=int)
                plot_preds = best_preds[indices]
                plot_tgts = best_tgts[indices]
            else:
                plot_preds = best_preds
                plot_tgts = best_tgts

            plot_data = {
                "predictions": _to_serializable(plot_preds.tolist()),
                "targets": _to_serializable(plot_tgts.tolist()),
                "pred_length": args.pred_length,
                "seq_length": args.seq_length
            }
            (out_dir / "plot_data.json").write_text(json.dumps(plot_data, ensure_ascii=False, indent=2))
            print(f"[INFO] Saved plot data with {len(plot_preds)} samples to {out_dir / 'plot_data.json'}")

        # 统一 RESULT 输出
        print("RESULT: " + json.dumps({"metric": "rmse", "value": primary}, ensure_ascii=False))

    except Exception as e:
        # 可被上层捕获的错误提示
        (out_dir / "error.log").write_text(str(e))
        print("ERROR: " + str(e))
        raise

if __name__ == "__main__":
    main()
