# C2 implementation specification

[Reported] Nguồn đặc tả là bản C2 ngày 11.11.2026, 32 trang vật lý, SHA-256
`a289e4d199562397de4134f56d518ab792b73019ed1ddacbec45cbb064a9438b`. Số trang dưới đây tính từ bìa.

## Equation-to-code mapping

| Blueprint | Contract | Module |
|---|---|---|
| (1), Algorithm 3 | IR-18, RGB 112 × 112, unit embeddings 512 chiều, cosine, accept nếu score ≥ threshold | `src/models/face.py`, `src/evaluation/verification.py` |
| (2)–(10), §5.3 | Biểu diễn population, hai đồ thị hard-cosine và phân biệt common shift với one-sided shift | `tests/test_math.py`, `src/evaluation/verification.py` |
| (11)–(16), trang 11 | Uniform base trên actual identities, \(H=H^\top,\ H_{ii}=0\), \(\rho^2<B-1\), \(R_H(q)=q^\top Hq/(1-\|q\|^2)\). \(N^\top 1=0,\ N^\top N=I\), \(q=\mathbf1/B+Nv/\sqrt B\) | `src/optimizers/composition.py` |
| (17)–(20), trang 11–12 | \(C=N^\top HN/B+r_0 I/B\), \(\operatorname{tr}C=0\), signed response \(r_0+G_\rho/d_{\min}\), \(d_{\min}=(B-1-\rho^2)/B\) | `src/optimizers/composition.py`, `tests/test_math.py` |
| (21), (35) | Generic fractional root check đối chiếu trên CPU | `tests/test_math.py` |
| (22) | Radial shrink, feasible witness và numerical upper qua dual | `src/optimizers/composition.py` |
| (23) | Detached signed witness, zero diagonal, ordered sum với \(q_iq_j/(1-\|q\|^2)\), active subgradient tại tie | `src/objectives/losses.py` |
| (24)–(26), trang 14 | Với \(S=2\), genuine sigmoid một cosine mỗi ID, impostor sigmoid từng capture pair rồi average, \(H_{ij}=\tfrac12\ell^-_{ij}+\tfrac14(\ell_i^++\ell_j^+)\) | `src/objectives/pairwise.py` |
| (27), Algorithm 1 | ArcFace và pair term, full encoder training, mặc định \(B=64,S=2,T=0.05,\beta=1,\rho=0.10\) | `src/objectives/losses.py`, `src/training/engine.py` |
| (29)–(31) | Optional finite-family audit cho frozen weights trên calibration IDs | `src/evaluation/statistics.py` |
| (32)–(34), trang 26 | Một eigendecomposition, secular equation, hard case, KKT và Lagrangian dual upper | `src/optimizers/trust_region.py` |
| §11, Appendix D | Identity manifests, common ArcFace warm start, locked training threshold, matched pilot | `src/data/manifest.py`, `scripts/train.py` |
| §12–15, Appendix E | Hard verification, held-out stress, matched contrasts và reporting | `src/evaluation/`, `experiments/PROTOCOL.md` |

## Oracle

[Derived] Đặt \(E=H-r_0(\mathbf1\mathbf1^\top-I)\). Khi chiếu lên
mean-zero subspace, \(N^\top E N/B=C\) và \(N^\top E q_0/\sqrt B=b\).
Residual form giảm cancellation cho near-constant kernels. Với kernel constant hoặc
\(\rho=0\), chọn uniform witness. Ngoài hai trường hợp này, Theorem 5 đưa
witness lên boundary. Quadratic solver scale bài toán về unit radius, solve
secular root theo offset từ spectral endpoint, và dùng top eigenspace trong hard case.

Số học float64 trên CPU. Kết quả cung cấp signed witness, feasible witness
tốt nhất trong uniform và radial shrink, risk đạt ở signed witness, dual-based
numerical upper, numerical gap, relaxation gap, bracket width, denominator,
minimum weight và KKT residual. Khi residual hay đối chiếu direct ratio vượt
tolerance, oracle phát sinh lỗi số học có thể ghi log. Giữ dấu của từng
envelope weight. Dùng cùng \(H\) có gradient trong outer loss.

[Derived] Tại \(B=64\), \(1/\sqrt{63}\approx0.125988\). Radius pilot 0.10
nằm trong vùng universal exactness. Ở radius lớn hơn, một signed witness
có phần tử âm vẫn cho relaxation và feasible bracket theo (22).

## Quyết định triển khai

- [Implementation assumption] Row arm dùng \(r_0+2\rho\|b\|_2/d_0\),
  nghiệm support function của linearization (16) tại \(q_0\). Nominal arm dùng
  cùng \(H,T,\tau_{\rm tr},\beta\) với \(\rho=0\). C2 dùng (27).
- [Implementation assumption] IR-18 dùng layout bốn stages 2–2–2–2, PReLU,
  BatchNorm và embedding dropout 0.4. ArcFace dùng \(s=64,m=0.5\), FP32
  ngoài encoder autocast, với FP64 cho input double trong mathematical tests.
  Chuẩn hóa vector với norm floor và fallback unit vector cố định tại norm 0.
- [Implementation assumption] Trong outer dev 1.000 IDs, hash chia 200 IDs để
  fit \(\tau_{\rm tr}\) tại FMR \(10^{-2}\) từ shared checkpoint, 800 IDs
  để chọn common checkpoint và estimate final thresholds. Hai captures/ID
  của evaluation được khóa theo hash trước khi xem score.
- [Implementation assumption] Cùng một budget successful updates và fixed
  final checkpoint ở matched pilot. Common warmup chọn checkpoint theo dev
  nominal TAR tại target \(10^{-2}\), ưu tiên epoch sớm hơn khi hòa.
- [Implementation assumption] Tính một epoch budget theo số ảnh chia cho
  \(BS\), làm tròn lên. Sampler chọn đều \(B\) IDs không lặp và hai captures
  không lặp trong từng step. Coverage qua một epoch là ngẫu nhiên.
- [Implementation assumption] Ảnh DigiFace chính thức đã crop, loader resize
  RGB về 112 × 112 và scale sang [-1, 1]. Cần audit alignment thực tế trên
  bytes được cấp quyền. Flip ngang là augmentation chung.
