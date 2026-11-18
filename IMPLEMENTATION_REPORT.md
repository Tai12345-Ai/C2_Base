# C2 implementation report

## M0 — PASS

- Initial tree chỉ có blueprint PDF. `git status --short --branch --untracked-files=all` trả exit 1 vì thiếu repository. Đã kiểm tra trước khi ghi.
- Đọc đủ 32 trang bằng PyMuPDF, render các trang phương trình và appendix trọng tâm, kiểm tra PDF embedded files và path links.
- Reference scripts của Appendix F — BLOCKED, các file không được cung cấp. Các kết quả Appendix F giữ nhãn [Reported].
- Đã tạo `IMPLEMENTATION_SPEC.md`, `IMPLEMENTATION_AUDIT.md`, `IMPLEMENTATION_REPORT.md` và `.gitignore`.
- Local runtime Python 3.13.1, PyTorch 2.11.0+cpu, NumPy 2.2.3, SciPy 1.15.2, pytest 9.1.0. CUDA không khả dụng.
- Bước kế tiếp là M1 kernel, oracle, gradient và CPU mathematical tests.

## M1 — PASS

- Đã tạo `src/optimizers/{trust_region,composition}.py`, `src/objectives/{pairwise,losses}.py`, `tests/test_math.py` và `pytest.ini`.
- `python -m pytest -q -s tests/test_math.py` với `OPENBLAS_NUM_THREADS=2` → **28 passed in 10.97s**.
- 200 oracle comparisons với independent basis và generic fractional root (21) → max absolute disagreement `2.243e-14`, max scaled KKT residual `1.015e-14`, max sampled positivity gap `2.217e-01`. 3.200 nonnegative random competitors và SLSQP từ hai starts đều nằm dưới numerical upper với test tolerance.
- Regression cases gồm graph curves (19), hard/near-hard, concave interior, một eigendecomposition, `rho=0`, signed weights, close-to-singular denominator, constant kernel và envelope finite differences.
- Remaining risk là finite arithmetic khi `D_min` rất nhỏ. Oracle kiểm tra direct ratio, KKT và dual gap trước khi trả witness.
- Bước kế tiếp là M2 IR-18/ArcFace và kiểm tra gradient/model AMP.

## M2 — PASS CPU / NOT RUN CUDA

- Đã tạo `src/models/face.py` và `tests/test_models.py`, nối ba arms qua `src/objectives/losses.py`.
- `python -m pytest -q -rs tests/test_models.py` với BLAS/OMP threads bằng 2 → **9 passed, 1 skipped in 7.75s**. CUDA test skip vì local PyTorch CPU.
- Đã kiểm tra Eq. (23) qua finite difference ở embeddings, gradient qua cả hai endpoints và về stem IR-18, ArcFace poles, BatchNorm train/eval, zero-norm, NaN/Inf, CPU bfloat16 encoder với pair/head FP32 và oracle FP64.
- First test run có warning Tensor-to-scalar ở test ArcFace. Đã đổi expected value sang `.item()` trước khi gọi `math.acos`. Sẽ chạy lại cùng test suite tích hợp.
- Bước kế tiếp là M3 manifests, training state và Colab orchestration.

## M3 — PASS hạ tầng local / BLOCKED DigiFace ingestion

- Đã tạo `src/data/{manifest,captures}.py`, `src/utils/io.py`, `src/training/{config,checkpoint,engine}.py`, `scripts/{audit_dataset,train}.py`, `configs/pilot.yaml`, `requirements.txt`, `README.md` và năm notebook trong `notebooks/`.
- Dataset fixtures kiểm tra identity-disjoint splits, dev subpartitions, pilot counts, deterministic sampler, cross-ID pixel duplicates và mismatch khi ảnh đổi bytes. Đã đối chiếu official release và R-UDA. Bản DigiFace đã cấp quyền chưa có tại workspace để audit bytes và alignment.
- `nbformat.validate` và `ast.parse` kiểm tra cả năm notebook. Notebook cells sạch outputs. `python -m compileall -q src scripts` hoàn tất.
- SHA-256 PDF sau kiểm tra vẫn là `a289e4d199562397de4134f56d518ab792b73019ed1ddacbec45cbb064a9438b`. Git status vẫn trả exit 1 cho thư mục chưa có Git metadata. `.gitignore` bảo vệ datasets, manifests, checkpoints và local artifacts.
- Remaining risk là validation trên extracted files và Drive/GPU runtime của Colab. Bước kế tiếp là audit DigiFace, warmup và matched pilot trên Colab.

## M4 — PASS CPU smoke / BLOCKED pilot GPU

- `tests/test_pipeline.py` chạy actual IR-18 trên ảnh RGB synthetic, từ manifest đến warmup, training-threshold lock, nominal/row/C2 và locked dev/cal/test. Mỗi arm có một successful update cùng checkpoint khởi tạo và zero skips. C2 chạy nontrivial TRS và encoder weights thay đổi.
- `tests/test_training.py` chạy C2 liên tục, pause/resume giữa epoch và lặp cùng seed trên model nhỏ. CPU state cuối bằng nhau bitwise cho model, optimizer, scheduler và RNG. Test cũng kiểm tra restore BatchNorm và skip khi oracle hoặc tensors lỗi.
- `scripts/report_runs.py` tổng hợp profiling từ JSONL. Colab Pro pilot 40.000 ảnh, CUDA fp16/bf16, GPU throughput, epoch time và peak VRAM trên GPU — NOT RUN. Cần dữ liệu đã audit và GPU runtime do người dùng khởi chạy.

## M5 — PASS evaluation plumbing / FAIL rare-hub coverage / BLOCKED confirmatory

- Đã tạo `src/evaluation/{data,verification,statistics,stress}.py`, `scripts/{evaluate,report_runs,bootstrap_coverage}.py`, `experiments/PROTOCOL.md` và `tests/test_evaluation.py`.
- Locked evaluation kiểm tra manifest/checkpoint hashes, dev-only threshold fit, hard FMR/FNMR/TAR, finite-graph bounds, ordinary and role shifts, cross-adversary transfer và paired identity bootstrap. Fixture kiểm tra same-ID copies bị loại đúng, cùng pipeline báo cáo across seeds trên test data tổng hợp.
- `python -m scripts.bootstrap_coverage --output .c2-local/bootstrap_coverage.json` → 64 IDs, 300 trials, 300 replicates/trial. Rare hub (p=0.005) cho 77/300 intervals phủ truth và 223/300 samples hụt rare hub. Equal-row case cho 300/300. Scope là known-truth simulation, percentile CI ở rare regime vẫn exploratory.
- Confirmatory runs và các baselines sau pilot — BLOCKED cho tới khi có DigiFace pilot hợp lệ. Novelty, real-face transfer và empirical superiority còn mở.

## Regression fixes

- Bootstrap đầu tiên FAIL khi truyền integer weights. Chuyển raw weights sang FP64 trước khi normalize multiplicities, giữ nguyên original identity keys trong denominator.
- Oracle stress audit ban đầu từ chối 84/1.500 near-constant cases do rounding nominal mean ở mức một ULP làm lệch curvature. Tính centered residual bằng base level tách khỏi mean deviation. 1.500/1.500 cases chạy lại đạt, max scaled KKT `1.551e-15`.

## Final local suite — PASS

- `python -m pytest -q -rs -s --basetemp <thư mục tạm trong .c2-local>` với BLAS/OMP threads bằng 2 → **50 passed, 2 skipped in 43.72s**. Hai skips là CUDA fp16/bf16, thiếu CUDA device trên local runtime.
- Trong 200 fractional cross-checks sau fix centering, max absolute disagreement `2.243e-14`, max scaled KKT `9.779e-15`, max observed positivity gap `2.217e-01`.
- Supplemental CPU check dùng seeded symmetric matrices. \(B=64,\rho=0.10\) trả nonnegative witness. Với \(B=2000\), cả ba radii \(0.10,0.25,0.50\) trả feasible brackets, max scaled KKT `3.687e-14` trên graph tổng hợp.
- Bước kế tiếp là chạy `00_setup_colab.ipynb` rồi `01_dataset_audit.ipynb` với bản DigiFace hợp lệ. Sau đó chạy shared warmup và pilot đã cấu hình, review dev trước khi mở test.
