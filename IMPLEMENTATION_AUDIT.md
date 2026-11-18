# C2 implementation audit

## M0 — Provenance và phạm vi

Đã inspect root trước khi ghi. `git status --short --branch --untracked-files=all`
trả exit 1 với thông báo chưa có Git repository. Initial tree chỉ có
`_reference/C2_Research_Blueprint_VI.pdf`. Đã đọc text cả 32 trang, đối chiếu
render cho phần oracle, gradient, kernel và Appendix C–D. PDF chỉ có embedded
item “Content Credentials”, không có code attachment hoặc external link đến
code C2. Appendix F [Reported] liệt kê `composition_core.py`,
`check_design.py` và `make_figure.py` với kết quả CPU trước đây.
Ba file vắng trong workspace. Code mới được triển khai từ equations, numerical
checks của bản PDF được giữ ở trạng thái [Reported].

## Kiểm tra toán trước implementation

- [Derived, Cao] Trace correction đúng vì
  \(\operatorname{tr}(N^\top HN/B)=-q_0^\top Hq_0\) khi \(\operatorname{diag}H=0\).
  Nếu \(C=0\), \(P_0 E P_0=0\) với \(E=H-r_0(\mathbf1\mathbf1^\top-I)\).
  Symmetry suy \(E=\mathbf1 a^\top+a\mathbf1^\top\), rồi zero diagonal suy \(E=0\).
  Với \(B=2\), mọi symmetric zero-diagonal kernel đều constant. Nếu \(H\)
  nonconstant thì \(C\) traceless và có positive eigenvalue. Do đó maximizer
  của quadratic nằm trên sphere và \(D(v)=d_{\min}\). Không phát hiện mâu thuẫn
  trong phần formulation của (11)–(27) dưới các giả thiết này.
- [Derived, Cao] Genuine marginal của ordered different-ID trials là
  \(q_i(1-q_i)/D(q)\). Tổng ordered off-diagonal weights bằng một, kể cả
  signed relaxation. Tại \(\rho=0\), nominal risk là off-diagonal mean.
- [Concern, Cao] Tại repeated eigenvalue một maximizer có thể signed trong khi
  một maximizer khác nonnegative. Bracket của một witness có thể rộng dù
  optimum relaxation trùng optimum nonnegative. Appendix B.1 nêu chính
  trường hợp này. Kiểm tra cả analytic feasible witness và returned bracket.
- [Concern, Cao] Hard case làm classical gradient có thể không duy nhất.
  Kiểm tra centered finite differences ở nghiệm unique, và subgradient
  inequality tại tie. Solver kiểm tra KKT, direct ratio và dual gap trong
  float64. Khi \(d_{\min}\) nhỏ, sai số có thể bị khuếch đại.
- [Concern, Cao] Pointwise bound \(1\{t\ge0\}\le 2\sigma(t/T)\) truyền qua
  nonnegative weights. Với signed weights, dấu âm phá monotonicity của tổng.
  Hard FMR và surrogate risk được đo riêng. Numerical upper có phạm vi
  floating-point, independent proof review vẫn pending.
- [Concern, Cao] Equation (20) dùng uniform base và zero diagonal. Identity
  bootstrap tạo multiplicity trên cùng original identity. Denominator phải
  loại mọi same-identity pairs, kể cả giữa bản sao. Đưa multiset vào
  Single-TRS với uniform slots sẽ đổi protocol. Bootstrap nominal/ordinary
  shifts dùng original identity keys. Robust bracket là kết quả trên
  finite held-out graph đã khóa.

## Scope thực nghiệm

- [Concern, Cao] H2 cần nominal-pair baseline làm control cho additional
  supervision. So cùng initialization, \(H,T,\tau_{\rm tr}\), data,
  preprocessing, optimizer và update budget. Oracle failures hoặc AMP
  overflows có thể làm effective sample sequence lệch. Log và kiểm tra
  trước khi giải thích gain.
- [Concern, Cao] Hard FMR thấp và identity-cluster uncertainty khác pair
  count. Chọn thresholds trên dev, freeze trước test, dùng paired identity
  resampling rồi kiểm tra coverage trong regime rare/near-degenerate trước
  confirmatory inference. Robust empirical graph và Eq. (31) có scope khác nhau.
- [Concern, Trung bình] Fixed random subsets là stress menu cơ bản. Có thể
  đo susceptibility chung, cần real-face và independent score-blind clusters
  hoặc metadata hợp lệ để đánh giá shift có ý nghĩa vận hành. Synthetic gain
  có thể phụ thuộc renderer, identity morphology hoặc capture scheme.
- [Reported] Microsoft công bố 10.000 IDs × 72 captures và R-UDA cho
  non-commercial computational research. R-UDA quy định việc dùng data lẫn
  Results trong commercial offering. Chưa có archive hoặc xác minh bytes trong
  workspace. [Official release](https://github.com/microsoft/DigiFace1M) ·
  [R-UDA](https://github.com/microsoft/DigiFace1M/blob/main/LICENSE)
- [Reported] FEI dùng cho research và giữ giới hạn phân phối của nguồn.
  CelebA công bố identity annotations theo request. Dữ liệu, annotations và
  quyền của người dùng chưa được xác minh tại workspace.
  [FEI](https://fei.edu.br/~cet/facedatabase.html) ·
  [CelebA agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html#agreement)
- [Reported] IR-18 stage layout được đối chiếu với implementation chính thức
  của AdaFace. ArcFace classifier theo angular-margin objective của Deng et al.
  Weights của tất cả arms sẽ được train từ scratch trong protocol này.
  [AdaFace IR-18](https://github.com/mk-minchul/AdaFace/blob/master/net.py) ·
  [ArcFace CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)

## Bootstrap coverage check

[Concern, Cao] Chạy known-truth simulation với 64 IDs, 300 trials, 300
bootstrap replicates mỗi trial và nominal 95%. Rare-hub scenario với
`p=0.005` cho coverage **77/300 (25.7%)**. Có 223 trials hụt rare hub.
Equal-row scenario cho **300/300** trong phép đo này. Những số này giới hạn
scope của CI thực nghiệm, chưa xác nhận coverage trên DigiFace. Giữ
percentile intervals ở trạng thái exploratory và báo insufficient evidence
cho rare tails khi sampling không đủ.

Novelty chưa thể xác nhận. Classical TRS và quadratic fractional root là
prior. Scope của code audit là specialization và protocol trong blueprint.
[Fortin–Wolkowicz](https://optimization-online.org/2002/09/530/)
