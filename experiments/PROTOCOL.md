# C2 experimental protocol

## Pilot lock

[Implementation assumption] Seed đầu là 17. Screen sau đó dùng các paired
seeds 17, 37, 61. Confirmatory recipe có thể dùng 17, 37, 61, 83, 101 sau khi
khóa đủ hypotheses và budgets. Mỗi seed dùng cùng checkpoint warmup cho cả
ba arms. Khóa hash của dataset manifest, best checkpoint, training threshold
và cả ba final checkpoints.

| Arm | Pair term cộng với ArcFace |
|---|---|
| Nominal | Off-diagonal mean của \(H\), tương đương C2 ở \(\rho=0\) |
| Row-risk | Cùng nominal risk với phần worst linear response \(2\rho\|b\|/d_0\) |
| C2 | Signed-relaxed risk trong (20), dùng gradient (23) |

Recipe bắt đầu với IR-18, \(B=64,S=2,T=0.05,\beta=1,\rho=0.10\), SGD
momentum 0.9, weight decay \(5\times10^{-4}\), cosine LR \(10^{-2}\) đến
\(10^{-5}\). Common warmup 20 epochs, mỗi arm fine-tune 10 epochs, với epoch
budget \(\lceil 40000/128\rceil=313\) updates. Chọn common warmup trên
dev/selection. So sánh final matched checkpoints theo 3.130 successful
updates. `training.jsonl` ghi skips, số attempts, wall time và gradient
health. Một skipped update làm sequence có thể khác, cần nêu rõ khi báo
matched comparison. Profile 500 updates đầu, ghi cùng batch size, AMP
mode và GPU thực tế. `pair_path_fraction_including_oracle` bao gồm kernel,
transfer/synchronization và solve. `peak_vram_allocated_bytes` là allocator
peak của PyTorch.

## Endpoints và leakage controls

1. Cùng score-blind capture selection qua arms. Fit riêng thresholds của từng
   model trên dev/selection với target FMR \(10^{-2},10^{-3}\), accept khi
   \(s\ge\tau\). Ghi FMR thực tế và TAR/FNMR trên untouched test. Ngưỡng
   \(10^{-4}\) chỉ xét sau khi thiết kế tail measurement và uncertainty thích hợp.
2. Hard impostor kernel cho mỗi identity pair là average của bốn capture
   indicators. Stress radius cố định \(\{0.10,0.25,0.50\}\). Báo nominal TAR
   cùng feasible lower và relaxed numerical upper của worst-case hard FMR.
   Với bracket overlap, ordering về optimized FMR còn unresolved. Numerical
   gap và nonnegative-relaxation gap có ý nghĩa riêng.
3. Ordinary shifts dùng bốn score-blind identity hash buckets, mỗi lần tăng
   raw weight một bucket từ 1 lên 2, normalize cùng law ở cả endpoints.
   Báo actual radius theo split. Có separate fixed gallery và independently
   reweighted gallery controls. Transfer mỗi feasible fitted adversary qua
   các graphs cùng original IDs. Các nhóm hash là pilot stress cơ bản,
   chưa đại diện operational subpopulations hoặc demographics.
4. Paired identity bootstrap giữ original identity multiplicities, dùng
   cùng draws cho mọi arm và loại toàn bộ same-ID trials. Intervals
   conditional trên locked thresholds và observed captures. Báo riêng
   mean ± std và paired differences across seeds. Với rare hub hoặc sparse
   errors, xem coverage simulations và ghi nhận uncertainty chưa đủ.
   Bootstrap percentile trong suite là exploratory. Robust brackets
   được báo trên finite graph gốc. Eq. (31) tùy chọn chỉ áp dụng frozen
   family với calibration identities theo giả thiết sampling của theorem.
5. So hard metrics và surrogate-at-training-threshold trên cùng identities.
   Giữ nguyên independent role-shift controls và per-ID error distributions.
   Feature model, thresholds, stress menu, operating points và captures được
   khóa trước khi chạy test. Báo held-out FMR ở threshold đã fit của từng model.

## Bác bỏ và follow-up

- [Hypothesis] C2 cải thiện frontier so với nominal pair và row-risk controls
  tại matched nominal utility. Nếu chỉ hơn warmup ArcFace hoặc gain biến mất
  khi khống chế nominal TAR, chưa support learning contribution ở H2.
- [Hypothesis] Signed interaction có giá trị sau khi kiểm soát row risk và
  canonical variance. Sau pilot, so CIFP, edge-DRO, direct nonnegative
  product-DRO với multi-start, và full-variance/uniformity dưới comparable
  tuning budget. Direct product-DRO là objective/solver equivalence check
  ở exact radius. So sánh larger batch cần cả matched update và wall time.
- [Hypothesis] Robustness có signal với ordinary common-population shifts
  và transfer adversary. Nếu gain chỉ nằm ở own adversary hoặc bracket rộng
  tương đương effect, thu hẹp interpretation. Fixed-gallery gain có thể là
  ordinary recognition gain.
- Trước confirmatory runs, xác định smallest effect of interest, allowed
  nominal TAR drop, multiplicity corrections và interval coverage trong
  relevant design. Nếu test images hoặc low-FMR events quá ít, ghi
  insufficient evidence. FEI là real-face mechanism check có quyền riêng.
  CelebA cần identity annotations và hợp lệ theo agreement. WebFace4M là
  optional scale-up.
