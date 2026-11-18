# C2 Base

PyTorch codebase cho C2 composition robustness theo blueprint ngày 11.11.2026.
Training dùng IR-18 + ArcFace và một trong ba pair objectives trên cùng kernel
\(H\). Single-TRS giải signed relaxation bằng một eigendecomposition và secular
solve. Feasible witness và numerical upper được ghi cùng KKT và positivity diagnostics.

## Kiểm tra trên CPU

Chạy từ project root trong Codex Terminal. Python 3.10 trở lên và PyTorch 2.4
trở lên là interface mục tiêu. Phiên bản đã thực chạy nằm trong report.

```powershell
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
python -m pytest -q -rs
python -m scripts.bootstrap_coverage
```

`tests/test_pipeline.py` chạy từ các ảnh RGB tổng hợp tạm qua dataset manifest,
IR-18 warmup, cả ba arms, checkpoint và locked evaluation. Tests CUDA được
đánh dấu `cuda`. Trên runtime GPU, chạy `python -m pytest -q -rs -m cuda` để
kiểm tra mixed precision của runtime đó.

## Google Colab Pro

| Notebook | Công việc |
|---|---|
| [00_setup_colab](notebooks/00_setup_colab.ipynb) | Mount Drive, setup, GPU detection, regression tests |
| [01_dataset_audit](notebooks/01_dataset_audit.ipynb) | Kiểm tra bản đã giải nén, freeze manifest và capture hashes |
| [02_arcface_warmup](notebooks/02_arcface_warmup.ipynb) | Common warmup theo seed, best dev checkpoint, \(\tau_{\rm tr}\) lock |
| [03_c2_pilot](notebooks/03_c2_pilot.ipynb) | Matched nominal, row, C2, periodic checkpoint và profiler |
| [04_evaluation](notebooks/04_evaluation.ipynb) | Dev threshold lock, calibration tùy chọn, test, frontier và seed summary |

Đặt source trong Drive và cấu hình `REPO`, `WORK`, `DATA_ROOT` nhất quán giữa
các notebook. `WORK` lưu checkpoints và reports qua runtime restart. Giữ source
archive, manifests và outputs ngoài Git. Dữ liệu có thể nằm trên disk runtime
để giảm I/O trên Drive. `AMP_MODE="off"` là mặc định. Lựa chọn `fp16` hoặc
`bf16` trong notebook sẽ chạy numerical gate CUDA cho dtype tương ứng.
Checkpoint ghi model, optimizer, scheduler, scaler, Python/NumPy/PyTorch RNG và
sampler cursor. Resume giữ cùng config và input manifest. Reproducibility
bitwise đã được test trên cùng CPU runtime.

## Chạy bằng CLI

Các lệnh nhận bản DigiFace đã được cấp quyền, giải nén và có layout một
thư mục cho mỗi identity, dưới một hoặc nhiều part directories. Dùng phần
72 captures/ID. Audit kiểm tra số lượng, exact file và decoded-pixel duplicates,
rồi tạo outer identity splits 6.000/1.000/1.000/2.000. Pilot gồm subset 2.000
train IDs × 20 captures. Optional `--archives` nhận JSON mapping part directory
đến `{"archive": "name.zip", "sha256": "<SHA-256>"}`. Dùng key `"."` khi
root trực tiếp chứa identity directories. Loader kiểm tra hash của mỗi ảnh
được đọc lại. [Microsoft DigiFace-1M](https://github.com/microsoft/DigiFace1M) ·
[R-UDA](https://github.com/microsoft/DigiFace1M/blob/main/LICENSE)

```powershell
python -m scripts.audit_dataset --data-root datasets/digiface72 --manifest manifests/digiface72.csv --audit manifests/audit.json

python -m scripts.train warmup --device cuda --data-root datasets/digiface72 --manifest manifests/digiface72.csv --out runs/pilot/seed-17/warmup

foreach ($arm in @('nominal', 'row', 'c2')) {
    python -m scripts.train $arm --device cuda --data-root datasets/digiface72 --manifest manifests/digiface72.csv --warmup-dir runs/pilot/seed-17/warmup --out "runs/pilot/seed-17/$arm"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $arm" }
}

python -m scripts.report_runs profile --run-root runs/pilot/seed-17 --limit 500 --output runs/pilot/seed-17/profile.json

python -m scripts.evaluate lock --device cuda --run-root runs/pilot/seed-17 --data-root datasets/digiface72 --manifest manifests/digiface72.csv --output runs/pilot/seed-17/evaluation
```

Dùng `--resume` trên lệnh training của arm đang dở. Sau khi xem dev report và
khóa protocol, dùng `test` hoặc `cal` với cùng các tham số của lệnh
`scripts.evaluate lock`. Mỗi seed có thư mục warmup riêng và cả ba arms
chia sẻ checkpoint cùng seed. `scripts.report_runs seeds --reports ... --output ...`
gộp các `test.json` theo seed. Full track dùng `--scope full` cho cả warmup
và matched arms.

## Phạm vi evidence

- Pilot dùng sigmoid surrogate và đo hard verification tại thresholds fit
  trên dev. Bracket là finite-population optimization uncertainty. Bootstrap
  là identity-level uncertainty conditional trên threshold/captures đã khóa.
- Giá trị relax ở \(\rho=0.10,B=64\) nằm trong universal exact region trong
  exact arithmetic. Held-out \(B\) lớn hơn có thể cho signed weights ngay tại
  radius đó, nên kiểm tra bracket ở evaluation.
- Trạng thái CPU, numerical limits và kết quả coverage simulation nằm trong
  [implementation report](IMPLEMENTATION_REPORT.md). [Audit](IMPLEMENTATION_AUDIT.md)
  và [protocol](experiments/PROTOCOL.md) ghi phạm vi claim và điều kiện chạy pilot.
