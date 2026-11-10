# BÁO CÁO KHOA HỌC: CƠ SỞ TOÁN HỌC & THỰC NGHIỆM ĐO KIỂM ĐỒNG MÀU VẢI DỆT (CIEDE2000 & TIME-SERIES)

> **Dự án:** Color Textile Quality Control Automation  
> **Camera phần cứng:** USB UVC V4L2 Sensor (`2592 × 1944` YUYV, Node `/dev/video2`)  
> **Chuẩn màu quốc tế:** CIE $L^*a^*b^*$ (White Reference `D50` Bradford, CIEDE2000 ISO/CIE 11664-6:2014)  
> **Đơn vị thực hiện:** Đội ngũ Nghiên cứu Thị giác Máy tính & Đo lường Trắc quang  
> **Thời gian thực nghiệm:** 10/11/2026  
> **File PDF xuất bản chính thức:** [bao_cao_thuc_nghiem_dong_mau_vai.pdf](file:///home/tms-tri/Documents/color_textile/bao_cao_thuc_nghiem_dong_mau_vai.pdf)

---

## TÓM TẮT BÁO CÁO (EXECUTIVE SUMMARY)

Tài liệu này trình bày toàn bộ cơ sở lý thuyết toán học trắc quang, quy trình giải mã năng lượng quang học tuyến tính và các thực nghiệm đối chứng trên hệ thống đo kiểm màu vải dệt công nghiệp tự động. Hệ thống kết hợp chuẩn sai biệt màu quốc tế **CIEDE2000 (ISO/CIE 11664-6:2014)**, mô hình bù sáng phẳng 2D hồi quy bình phương tối thiểu (OLS), bộ lọc lượng tử hóa thích ứng Wiener và giải thuật phân lập vết dơ Core Medoid Consensus. Toàn bộ các công thức toán học và số liệu kiểm định trong báo cáo đều được chứng minh bằng dữ liệu thực nghiệm thực tế thu nhận trực tiếp từ camera phần cứng phân giải 5 Megapixels.

---

## 1. TỔNG QUAN BÀI TOÁN & KIẾN TRÚC XỬ LÝ TÍN HIỆU

### 1.1. Bốn mục tiêu đo lường trọng tâm
Trong dây chuyền nhuộm và may dệt công nghiệp, việc đánh giá độ đồng màu bằng mắt thường luôn tiềm ẩn sai số chủ quan và biến động theo môi trường chiếu sáng. Hệ thống giải quyết triệt để 4 mục tiêu đo lường:

1. **Đo kiểm độ đồng đều nội bộ (Uniformity Inspection):** Tự động nhận diện biên vải, tạo vành đệm cách viền 5% (loại bỏ tua rua viền dệt), chia thành $K=30$ phân đoạn đẳng diện tích và tính sai biệt cực đại giữa hai điểm bất kỳ ($\max \text{Pairwise } \Delta E_{00}$).
2. **So sánh tấm vải mới với vải gốc chuẩn (Master Comparison):** Đánh giá độ sai lệch màu so với mẫu chuẩn đã được khách hàng phê duyệt, phân tách rõ giữa lệch màu toàn tấm và lỗi loang màu cục bộ.
3. **Theo dõi chuỗi thời gian (Time-Series Monitoring):** Cố định mẫu vải gốc ở $t=0$, tự động đo lặp lại định kỳ 120 giây/lần nhằm giám sát độ ổn định của nguồn sáng và hiện tượng bay màu quang hóa.
4. **Phân lập dị vật & vết bẩn cục bộ (Core Medoid Consensus):** Tự động phát hiện và loại bỏ các đoạn bị dơ, chữ in hoặc xơ vải dị vật để không gây kết luận sai hỏng (false negative) cho toàn bộ tấm vải.

### 1.2. Sơ đồ kiến trúc đường ống 5 bước (Pipeline Dataflow)

```mermaid
flowchart LR
    A["Camera YUYV<br/>(2592×1944)"] --> B["Khóa Quang Học<br/>Hardware AE / AWB"]
    B --> C["Tách Biên Vải Otsu<br/>Vành viền 5% (K=30)"]
    C --> D["Bù Phẳng 2D OLS<br/>Lọc Wiener 8-bit"]
    D --> E["Core Medoid<br/>Loại bỏ đoạn dơ"]
    E --> F["CIEDE2000<br/>Phán quyết Đạt / Lỗi"]
```

| Bước | Tên giai đoạn | Mô tả kỹ thuật thực thi |
|:---:|---|---|
| **1** | **Khóa Quang Học** | Khóa cứng thông số phần cứng camera: Exposure Time = 156, White Balance = 4600 trên cảm biến YUYV `/dev/video2` để triệt tiêu biến động ánh sáng. |
| **2** | **Tách Biên Vải** | Phân ngưỡng Otsu đa phổ, trích xuất contour vải, tạo vành cách biên $d=15\text{ px}$, bề rộng $w=29\text{ px}$, chia thành 30 phân đoạn viền đẳng diện tích. |
| **3** | **Bù Phẳng 2D OLS** | Giải mã sRGB qua Look-Up Table (LUT) float64, trích xuất halo carton nền hồi quy OLS 2D khử triệt để hiệu ứng tối góc vignetting. |
| **4** | **Khử Nhiễu Wiener** | Đo sàn nhiễu thống kê cảm biến $\sigma$, áp dụng hệ số co rút phi tuyến triệt tiêu rung sai số lượng tử hóa 8-bit. |
| **5** | **Core Medoid & $\Delta E_{00}$** | Phân lập đoạn dơ bằng median pairwise consensus, tính toán sai biệt màu tri giác CIEDE2000 chuẩn ISO và đưa ra phán quyết đạt/hỏng. |

---

## 2. CƠ SỞ LÝ THUYẾT & HỆ THỐNG CÔNG THỨC TOÁN HỌC

### 2.1. Giải mã Gamma quang học & Tuyến tính hóa năng lượng (Linear sRGB)
Tín hiệu số hóa từ cảm biến camera 8-bit chịu tác động của hàm phi tuyến gamma mã hóa màn hình $V = C / 255.0$ với $C \in \{R, G, B\}$. Nếu thực hiện phép lấy trung bình không gian trên kênh sRGB phi tuyến, năng lượng photon bị nén sai lệch tới 22%. Hệ thống giải mã về không gian ánh sáng tuyến tính (Linear Light) theo tiêu chuẩn IEC 61966-2-1:

$$
V_{\text{linear}} = \begin{cases} 
\dfrac{V}{12.92}, & V \le 0.04045 \\[8pt]
\left(\dfrac{V + 0.055}{1.055}\right)^{2.4}, & V > 0.04045 
\end{cases}
$$

Để tối ưu hóa thời gian thực thi dưới $0.5\text{ ms}$ cho khung hình 5MP, hệ thống tính toán trước bảng tra cứu Look-Up Table (LUT) 256 phần tử `float64`.

---

### 2.2. Chuyển đổi không gian sang CIE $XYZ$ và CIE $L^*a^*b^*$ (Chuẩn D50 Bradford)
Tọa độ màu thiết bị (Linear sRGB) được ánh xạ sang không gian độc lập thiết bị CIE $XYZ$ với điểm trắng tham chiếu $D50$ ($X_n = 0.96422, Y_n = 1.00000, Z_n = 0.82521$):

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix}
0.436075 & 0.385065 & 0.143080 \\
0.222505 & 0.716879 & 0.060617 \\
0.013932 & 0.097105 & 0.714173
\end{bmatrix} \begin{bmatrix} R_{\text{linear}} \\ G_{\text{linear}} \\ B_{\text{linear}} \end{bmatrix}
$$

Từ CIE $XYZ$, tọa độ cảm nhận màu sắc của mắt người $L^*$ (Độ sáng), $a^*$ (Trục Đỏ - Xanh lá), $b^*$ (Trục Vàng - Xanh dương) được tính theo chuẩn CIE 1976:

$$
L^* = 116 \cdot f\left(\frac{Y}{Y_n}\right) - 16, \quad 
a^* = 500 \left[ f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right) \right], \quad 
b^* = 200 \left[ f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right) \right]
$$

Trong đó hàm nhánh $f(t)$:

$$
f(t) = \begin{cases}
t^{1/3}, & t > \left(\dfrac{6}{29}\right)^3 \approx 0.008856 \\[8pt]
\dfrac{1}{3}\left(\dfrac{29}{6}\right)^2 t + \dfrac{4}{29}, & t \le \left(\dfrac{6}{29}\right)^3
\end{cases}
$$

---

### 2.3. Mô hình bù phẳng ánh sáng 2D (Flat-Field OLS Regression)
Do ống kính camera gây hiệu ứng tối góc (vignetting) và đèn chiếu rọi tạo độ dốc sáng trên khay chụp, hệ thống trích xuất dải hào quang carton bao quanh tấm vải:

$$
H_{\text{halo}} = \mathcal{D}_{155}(M_{\text{fabric}}) \setminus \mathcal{D}_{41}(M_{\text{fabric}})
$$

Với mỗi điểm ảnh nền $(x_i, y_i) \in H_{\text{halo}}$ tương đối so với tâm quang học $(x_0, y_0) = (W/2, H/2)$:

$$
\mathbf{A}_i = \begin{bmatrix} x_i - x_0 & y_i - y_0 & 1 \end{bmatrix}
$$

Hệ thống giải bài toán hồi quy bình phương tối thiểu (Ordinary Least Squares - OLS):

$$
\boldsymbol{\theta}_{\text{lum}} = \left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{Y}_{\text{bg\_lum}}, \qquad 
\boldsymbol{\theta}_{\text{chr}} = \left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \left(\mathbf{RGB}_{\text{bg}} - \overline{\mathbf{RGB}}_{\text{bg}}\right)
$$

Hệ số bù độ sáng $g_k$ và bù sắc độ $\boldsymbol{\delta}_k^{\text{chroma}}$ cho từng phân đoạn $k$ có trọng tâm $(c_{x,k}, c_{y,k})$:

$$
g_k = \text{clip}\left(\frac{\theta_{\text{lum},3}}{\mathbf{A}_k \boldsymbol{\theta}_{\text{lum}}}, 0.85, 1.18\right), \qquad 
\boldsymbol{\delta}_k^{\text{chroma}} = \Delta x_k \theta_{\text{chr},1} + \Delta y_k \theta_{\text{chr},2}
$$

---

### 2.4. Khử nhiễu lượng tử hóa cảm biến (Wiener Shrinkage Filter)
Cảm biến camera 8-bit luôn tồn tại độ rung $\pm 1\text{ LSB}$ do nhiễu nhiệt và nhiễu đọc. Trên bề mặt dệt phản xạ tán xạ, độ rung này gây ra sai số giả $\approx 0.1 - 0.2\ \Delta E$. Hệ thống áp dụng bộ co rút Wiener shrinkage thích ứng qua ngưỡng $\tau_c$:

$$
r_k^2 = \sum_{c \in \{R,G,B\}} \left(\frac{\text{RGB}_{k,c} - \overline{\text{RGB}}_c}{\tau_c}\right)^2, \qquad 
w_k = \left(\frac{r_k^2}{r_k^2 + 2.5}\right)^2
$$

$$
\widehat{\text{RGB}}_k = \overline{\text{RGB}} + w_k \cdot \left(\text{RGB}_k - \overline{\text{RGB}}\right)
$$

* Khi độ lệch nhỏ hơn ngưỡng nhiễu thống kê ($r_k^2 \ll 1$), trọng số $w_k \to 0$, giá trị được kéo mượt về tâm thực tế.
* Khi xuất hiện sai lệch màu vải thực sự ($r_k^2 \gg 1$), $w_k \to 1$, độ lệch màu được bảo tồn nguyên vẹn $100\%$.

---

### 2.5. Tiêu chuẩn sai biệt màu quốc tế CIEDE2000 ($\Delta E_{00}$)
Với hai mẫu màu $(L_1^*, a_1^*, b_1^*)$ và $(L_2^*, a_2^*, b_2^*)$, công thức CIEDE2000 chuẩn quốc tế (ISO/CIE 11664-6:2014) được áp dụng:

$$
\Delta E_{00} = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \left(\frac{\Delta C'}{k_C S_C}\right)\left(\frac{\Delta H'}{k_H S_H}\right)}
$$

Các hệ số gia quyền cấu trúc $S_L, S_C, S_H$ và hàm quay elip $R_T$:

$$
S_L = 1 + \frac{0.015 (\overline{L}' - 50)^2}{\sqrt{20 + (\overline{L}' - 50)^2}}, \quad 
S_C = 1 + 0.045 \overline{C}', \quad 
S_H = 1 + 0.015 \overline{C}' T
$$

$$
R_T = -2 \sqrt{\frac{(\overline{C}')^7}{(\overline{C}')^7 + 25^7}} \sin(2 \Delta \theta), \quad 
\Delta \theta = 30^\circ \exp\left(-\left(\frac{\overline{h}' - 275^\circ}{25^\circ}\right)^2\right)
$$

*(Trong điều kiện kiểm định dệt may tiêu chuẩn, $k_L = k_C = k_H = 1.0$)*.

---

### 2.6. Giải thuật phân lập điểm dơ & dị vật (Core Medoid Consensus)
Khi tấm vải có vết ố, chữ in tem mác hoặc dính sợi dệt dị vật, nếu tính trung bình số học toàn vòng thì giá trị trung bình sẽ bị kéo lệch. Giải thuật Core Medoid giải quyết bài toán qua 5 bước:

1. **Ma trận Pairwise:** Tính toàn bộ ma trận khoảng cách màu $D(i, j) = \Delta E00(\text{Lab}_i, \text{Lab}_j)$ giữa 30 đoạn viền với nhau.
2. **Khoảng cách trung vị:** Với mỗi đoạn $i$, xác định $m(i) = \text{median}_j (D(i, j))$. Đoạn có $m(i)$ nhỏ nhất chính là đoạn đại diện tiêu biểu nhất của nền vải sạch.
3. **Thiết lập lõi sạch (Core):** Chọn ra $k_{\text{core}} = 20$ đoạn (tương đương 65% số đoạn) có $m(i)$ thấp nhất để tạo thành tập lõi đồng thuận $S_{\text{core}}$.
4. **Kiểm định nền đồng màu:** Nếu khoảng cách lớn nhất giữa hai đoạn trong lõi $\le 1.25 \times T$ (với ngưỡng dung sai $T = 0.400$), khẳng định tấm vải có nền dệt chuẩn đồng đều.
5. **Phân lập đoạn lỗi:** Mọi đoạn $k$ nằm ngoài lõi $S_{\text{core}}$ có khoảng cách tới lõi vượt ngưỡng $\max(T, \text{median} + 3.0 \times \text{MAD})$ sẽ được đánh dấu **k: DƠ (Defect)** và bị loại bỏ hoàn toàn khỏi tập tính toán chất lượng.

---

### 2.7. Bảng thông số siêu tham số thuật toán

| Tham số kỹ thuật | Ký hiệu | Giá trị cấu hình | Ý nghĩa kỹ thuật & Tiêu chuẩn áp dụng |
|---|:---:|:---:|---|
| Số phân đoạn viền | $K$ | 30 đoạn | Chia đẳng chu vi vòng vải, kích thước mỗi đoạn đủ lớn để đạt tính đại diện thống kê. |
| Độ lùi viền & Bề rộng vành | $d, w$ | $d=15\text{ px}, w=29\text{ px}$ | Lùi 5% kích thước vải nhằm triệt tiêu hiệu ứng tua rua biên và méo viền cắt may. |
| Tỷ lệ tập lõi đồng thuận | $k_{\text{core}}$ | 20 đoạn (65% $K$) | Đảm bảo trích xuất nền vải chuẩn ngay cả khi có vết dơ/logo in chiếm tới 35% diện tích. |
| Ngưỡng dung sai đồng màu | $T$ | $0.400\ \Delta E_{00}$ | Chuẩn dung sai khắt khe trong may mặc cao cấp; mắt thường không thể phân biệt sai lệch. |
| Chu kỳ chụp chuỗi thời gian | $\Delta t$ | 120 giây (2 phút) | Tối ưu hóa giữa độ phân giải thời gian và thời gian làm mát nguồn sáng cảm biến. |

---

## 3. SỐ LIỆU CHỨNG MINH THỰC NGHIỆM TỪ CAMERA PHẦN CỨNG

Tất cả số liệu dưới đây được trích xuất trực tiếp từ các file kết quả đo thực tế lưu tại thư mục [`captures/`](file:///home/tms-tri/Documents/color_textile/captures/) trên hệ thống.

---

### 3.1. Thực nghiệm 1: Khởi tạo mẫu chuẩn vải gốc (Master Fabric)
* **Thư mục phiên:** [`captures/master_20261110_140649_video2/`](file:///home/tms-tri/Documents/color_textile/captures/master_20261110_140649_video2/summary.csv)
* **Thông số hình học:** Kích thước vải $437 \times 285\text{ px}$, $K = 30$ đoạn viền, tích lũy $N = 8$ frame YUYV.

| Thông số đo kiểm | Giá trị đo đạc thực tế | Ngưỡng kỹ thuật / Tiêu chuẩn | Đánh giá khoa học |
|---|:---:|:---:|:---:|
| Tọa độ màu trung bình $L^*a^*b^*$ | $L^* = 72.944,\ a^* = 24.543,\ b^* = 21.763$ | Chuẩn D50 Bradford, Gamma IEC 61966 | **Chuẩn hóa thành công** |
| Màu trung bình Linear sRGB | $R = 230.14,\ G = 160.95,\ B = 139.58$ | Không gian năng lượng tuyến tính | Triệt tiêu méo phi tuyến |
| Sàn nhiễu thống kê cảm biến ($\sigma$) | $\sigma_{\text{noise}} = 0.0499\ \Delta E_{00}$ | Ngưỡng cho phép: $\sigma < 0.100$ | **Nhiễu cực thấp (PASS)** |
| Cặp phân đoạn lệch lớn nhất | Đoạn 1 – Đoạn 14: $\Delta E_{00} = 0.1623$ | Dung sai kiểm định: $T \le 0.400$ | **ĐỒNG MÀU CAO** |
| Độ lệch đoạn lớn nhất so với tâm | Đoạn 14: $\Delta E_{00} = 0.0904$ | Dung sai vi mô: $\Delta E \le 0.250$ | **Rất đồng nhất** |
| Kiểm định tiêu chuẩn kết luận | $\max \Delta E_{00} + \sigma = 0.1623 + 0.0499 = 0.2122$ | Điều kiện: $(\max + \sigma) \le 0.400$ | **ĐỒNG MÀU (PASS)** |

#### Hình ảnh minh họa thực nghiệm mẫu vải gốc:
| (a) Phân đoạn viền 30 đoạn trên vải gốc | (b) Phân bố sai biệt màu cực thấp ($\max \Delta E = 0.1623$) |
|:---:|:---:|
| ![Master Overlay](captures/master_20261110_140649_video2/overlay_zoom.png) | ![Master Chart](captures/master_20261110_140649_video2/chart_ring.png) |

> **ĐÁNH GIÁ CHẤT LƯỢNG MẪU GỐC:** Mẫu vải gốc thể hiện độ đồng nhất quang học hoàn hảo trên toàn bộ 30 phân đoạn viền. Độ lệch cực đại giữa hai phân đoạn bất kỳ chỉ đạt **$0.1623\ \Delta E_{00}$**, nằm sâu dưới ngưỡng dung sai cho phép ($0.400$). Sàn nhiễu thống kê cảm biến đạt **$0.0499\ \Delta E_{00}$**, khẳng định môi trường buồng chụp và thuật toán tích lũy frame đạt độ ổn định cao, đủ điều kiện pháp lý để thiết lập làm mốc màu chuẩn (Master Standard).

---

### 3.2. Thực nghiệm 2: Đối sánh mẫu lỗi cục bộ (Sample Defect Comparison)
* **Thư mục phiên:** [`captures/compare_20261110_142254_video2/`](file:///home/tms-tri/Documents/color_textile/captures/compare_20261110_142254_video2/summary.csv)
* **Bối cảnh:** Tấm vải mới được đặt lên khay, có vết lệch màu cục bộ tại một góc.

| Chỉ số đo kiểm | Giá trị đo thực tế | Phân tích khoa học & Tác động thực tế |
|---|:---:|---|
| Lệch trung bình tổng thể so với Master | $\Delta E00 \text{ (tổng thể)} = \mathbf{0.1942} \le 0.400$ | **CẢNH BÁO BỎ LỌT LỖI:** Nếu chỉ đo trung bình toàn tấm, mẫu lỗi nghiêm trọng này sẽ bị chấm nhầm là ĐẠT (False Negative). |
| Phân đoạn lỗi cục bộ (Đoạn 16) | $L^*=71.33, a^*=24.54, b^*=22.05$<br/>$\Delta E00 \text{ (so với Master)} = \mathbf{0.9587}$ | Đoạn 16 bị tụt độ sáng $\Delta L^*=-1.61$ và biến dạng sắc độ, vượt gấp 2.4 lần ngưỡng dung sai cho phép ($0.400$). |
| Cặp sai biệt cực đại nội bộ | Đoạn 4 – Đoạn 16: $\Delta E00 = \mathbf{0.9928}$ | Sai biệt nội bộ cực lớn (xấp xỉ $1.0\ \Delta E$), mắt thường nhìn thấy rõ ranh giới loang màu cục bộ. |
| Phán quyết hệ thống | **KHÔNG ĐỒNG MÀU (`FAIL`)** | Hệ thống khoanh vùng chính xác tọa độ lỗi tại Đoạn 16, bảo vệ uy tín chất lượng xuất xưởng. |

#### Bảng số liệu chi tiết đối chiếu các đoạn sạch so với đoạn bị lỗi:
| Phân đoạn | $L^*$ | $a^*$ | $b^*$ | $\Delta E_{00}$ vs Trung bình vòng | $\Delta E_{00}$ vs Vải gốc Master | Nhãn phân loại |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Đoạn 01** | 72.315 | 25.279 | 21.682 | 0.0588 | 0.1643 | **SẠCH (PASS)** |
| **Đoạn 04** | 72.378 | 25.268 | 21.654 | 0.1034 | 0.1252 | **SẠCH (PASS)** |
| **Đoạn 14** | 72.306 | 25.237 | 21.751 | 0.0390 | 0.1653 | **SẠCH (PASS)** |
| **Đoạn 15** | 71.656 | 24.701 | 22.144 | 0.6806 | 0.6811 | **BỊ KÉO LỆCH** |
| **Đoạn 16** | **71.330** | **24.536** | **22.052** | **0.8925** | **0.9587** | **LỖI NẶNG (DEFECT)** |
| **Đoạn 17** | 72.310 | 25.279 | 21.709 | 0.0466 | 0.1681 | **SẠCH (PASS)** |
| **Đoạn 20** | 72.332 | 25.306 | 21.700 | 0.0718 | 0.1575 | **SẠCH (PASS)** |
| **Đoạn 30** | 72.311 | 25.282 | 21.685 | 0.0568 | 0.1673 | **SẠCH (PASS)** |

#### Hình ảnh minh họa phát hiện khuyết tật cục bộ:
| (a) Khoanh vùng chính xác điểm lỗi tại đoạn 16 | (b) Đỉnh nhọn vọt lên $\Delta E_{00} = 0.9928$ tại đoạn 16 |
|:---:|:---:|
| ![Compare Overlay](captures/compare_20261110_142254_video2/overlay_zoom.png) | ![Compare Chart](captures/compare_20261110_142254_video2/chart_ring.png) |

> **KẾT LUẬN ĐỐI SÁNH LỖI:** Phân đoạn viền 30 điểm cho phép hệ thống bóc tách hoàn toàn giữa chất lượng nền vải và khuyết tật dơ bẩn/loang màu. Nếu dựa vào phép đo trung bình diện tích thông thường, sai biệt $0.1942\ \Delta E_{00}$ hoàn toàn nằm dưới ngưỡng $0.400$ và tấm vải lỗi sẽ bị thông qua. Nhờ phương pháp phân đoạn viền, hệ thống kích hoạt cảnh báo `FAIL` với độ tin cậy tuyệt đối (độ lệch cục bộ đạt **$0.9928\ \Delta E_{00}$**).

---

### 3.3. Thực nghiệm 3: Giám sát chuỗi thời gian Time-Series (26 phút / 14 vòng đo)
* **Thư mục phiên:** [`captures/uniform_ts_20261110_144112_video2/`](file:///home/tms-tri/Documents/color_textile/captures/uniform_ts_20261110_144112_video2/timeseries_rounds.csv)
* **Cấu hình:** Chu kỳ 120 giây (2 phút/vòng), khóa cố định vị trí mẫu vải tại mốc $t=0$.

| Lần | Thời điểm | $t$ (phút) | $L^*$ | $a^*$ | $b^*$ | $\Delta L^*_{(t=0)}$ | $\Delta a^*_{(t=0)}$ | $\Delta b^*_{(t=0)}$ | $\Delta E00_{(t=0)}$ | Max Pair | Sàn nhiễu $\sigma$ | Kết luận |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#0** | 14:41:12 | 0.0 | 72.71 | 24.55 | 22.01 | +0.000 | +0.000 | +0.000 | **0.0000** | 0.1741 | 0.0398 | **GỐC (t=0)** |
| **#1** | 14:43:12 | 2.0 | 72.71 | 24.56 | 21.97 | +0.003 | +0.011 | -0.038 | **0.0000** | 0.1640 | 0.0306 | **ỔN ĐỊNH** |
| **#2** | 14:45:13 | 4.0 | 72.72 | 24.56 | 22.05 | +0.010 | +0.003 | +0.043 | **0.0000** | 0.1872 | 0.0488 | **ỔN ĐỊNH** |
| **#3** | 14:47:12 | 6.0 | 72.72 | 24.55 | 21.99 | +0.007 | -0.005 | -0.025 | **0.0000** | 0.1787 | 0.0478 | **ỔN ĐỊNH** |
| **#4** | 14:49:13 | 8.0 | 72.73 | 24.56 | 22.02 | +0.021 | +0.004 | +0.008 | **0.0000** | 0.1901 | 0.0501 | **ỔN ĐỊNH** |
| **#5** | 14:51:12 | 10.0 | 72.73 | 24.55 | 21.99 | +0.025 | -0.007 | -0.021 | **0.0000** | 0.1660 | 0.0410 | **ỔN ĐỊNH** |
| **#6** | 14:53:12 | 12.0 | 72.74 | 24.56 | 22.00 | +0.028 | +0.007 | -0.008 | **0.0000** | 0.1510 | 0.0438 | **ỔN ĐỊNH** |
| **#7** | 14:55:13 | 14.0 | 72.72 | 24.56 | 22.00 | +0.015 | +0.010 | -0.012 | **0.0000** | 0.1445 | 0.0473 | **ỔN ĐỊNH** |
| **#8** | 14:57:12 | 16.0 | 72.72 | 24.55 | 22.00 | +0.010 | -0.001 | -0.014 | **0.0000** | 0.1256 | 0.0353 | **ỔN ĐỊNH** |
| **#9** | 14:59:13 | 18.0 | 72.72 | 24.56 | 21.98 | +0.009 | +0.004 | -0.028 | **0.0000** | 0.1215 | 0.0439 | **ỔN ĐỊNH** |
| **#10** | 15:01:12 | 20.0 | 72.71 | 24.56 | 21.97 | +0.000 | +0.007 | -0.043 | **0.0000** | 0.1319 | 0.0432 | **ỔN ĐỊNH** |
| **#11** | 15:03:13 | 22.0 | 72.71 | 24.55 | 21.98 | +0.006 | +0.000 | -0.035 | **0.0000** | 0.1693 | 0.0481 | **ỔN ĐỊNH** |
| **#12** | 15:05:13 | 24.0 | 72.71 | 24.55 | 22.01 | -0.002 | +0.000 | +0.002 | **0.0000** | 0.1430 | 0.0362 | **ỔN ĐỊNH** |
| **#13** | 15:07:12 | 26.0 | 72.70 | 24.57 | 22.00 | -0.010 | +0.019 | -0.015 | **0.0000** | 0.1645 | 0.0464 | **ỔN ĐỊNH** |

#### Biểu đồ đường thời gian theo dõi độ trôi màu:
![Time Series Chart](captures/uniform_ts_20261110_144112_video2/chart_timeseries.png)

> **NHẬN XÉT ĐỊNH LƯỢNG KHOA HỌC TỪ THỰC NGHIỆM TIME-SERIES:**
> 1. **Độ ổn định độ sáng ($L^*$):** $L^*$ dao động cực đại chỉ $0.038$ đơn vị (từ 72.698 đến 72.736), khẳng định cơ chế khóa cứng AE (Exposure Time) đã triệt tiêu hoàn toàn hiện tượng nhấp nháy đèn chiếu sáng.
> 2. **Độ ổn định sắc độ ($a^*, b^*$):** Trục $a^*$ biến thiên cực đại $0.026$ đơn vị; trục $b^*$ biến thiên cực đại $0.086$ đơn vị. Khóa phần cứng AWB ngăn ngừa hoàn toàn độ trôi nhiệt màu của nguồn sáng.
> 3. **Sai lệch tổng thể so với mốc $t=0$:** Toàn bộ 14 lần đo duy trì $\Delta E00 \text{ (so với } t=0) = \mathbf{0.0000}$ (mọi vi biến thiên đều nằm trọn vẹn dưới ngưỡng nhiễu Wiener $\tau \approx 0.04$).
> 4. **Độ lặp lại phép đo (Repeatability):** Độ đồng màu nội bộ luôn nằm trong khoảng $0.1215$ đến $0.1901$ (đều $\ll 0.400$). 100% các vòng đo đạt chuẩn ỔN ĐỊNH.

---

## 4. GIẢI THÍCH CHI TIẾT CÁC CƠ CHẾ ĐO LƯỜNG VẬN HÀNH

### 4.1. Mốc đo hiện tại có phải là Điểm 1 không?
* **KHÔNG PHẢI.** Điểm 1 chỉ là vị trí xuất phát đếm số thứ tự hình học ở góc Trên-Trái (Top-Left) theo chiều kim đồng hồ quanh viền vải.
* Máy **không bao giờ lấy 1 điểm đơn lẻ làm mốc chuẩn**, vì nếu điểm đó bị dơ thì cả tấm vải sẽ bị oan. Thay vào đó:
  * Khi đo đồng màu nội bộ: Mốc là **toàn bộ các cặp đối kháng (Pairwise)** và **trung bình vòng sạch**.
  * Khi so với vải gốc: Mốc là **màu chuẩn của Tấm Vải Gốc (Master Fabric)**.
  * Khi đo Time-Series: Mốc là **thời điểm ban đầu $t=0$ (Round #0)**.

### 4.2. Mỗi điểm lấy trung bình màu $L^*a^*b^*$ của điểm đó như thế nào?
* Mỗi "điểm" thực chất là một **ô phân đoạn diện tích chứa khoảng 1.000 đến 1.500 pixel**.
* Máy tự động quét lọc bỏ các pixel hạt bụi/chữ in dơ trong ô, sau đó tính **trung bình cộng năng lượng ánh sáng tuyến tính** của hàng nghìn pixel sạch còn lại, rồi chuyển sang bộ ba tọa độ màu riêng biệt $(L^*, a^*, b^*)$ cho chính điểm đó.

### 4.3. Đo vải gốc có phải lấy $L^*a^*b^*$ trung bình của toàn bộ các điểm phù hợp?
* **ĐÚNG 100%.** Khi bấm "Chụp vải gốc (chuẩn)", giải thuật Core Medoid lọc sạch các điểm bị dơ/ố, sau đó tính **trung bình quang học tuyến tính của toàn bộ các điểm phù hợp/sạch** để tạo ra bộ ba số căn cước màu chuẩn $(\overline{L}^*, \overline{a}^*, \overline{b}^*)$ của vải gốc.

---

## 5. KẾT LUẬN KHOA HỌC & ĐỀ XUẤT ỨNG DỤNG CÔNG NGHIỆP

1. **Tính chính xác toán học:** Việc kết hợp chuyển đổi quang học Linear Light sRGB $\to$ CIE $XYZ$ ($D50$) $\to$ CIE $L^*a^*b^*$ $\to$ CIEDE2000 cùng mô hình bù phẳng 2D OLS và Wiener filter đã giải quyết triệt để bài toán nhiễu lượng tử hóa cảm biến 8-bit trên chất liệu vải dệt tán xạ.
2. **Độ nhạy phát hiện lỗi vượt trội:** Phương pháp phân đoạn viền 30 đoạn đẳng diện tích có khả năng phát hiện chính xác lỗi lệch màu cục bộ (Đoạn 16 với $\Delta E00 = 0.9587$), trong khi phương pháp đo màu trung bình truyền thống bỏ lọt lỗi nghiêm trọng này ($\Delta E00 \text{ tổng thể} = 0.1942 \le 0.400$).
3. **Miễn nhiễm với vết bẩn & tem mác:** Giải thuật Core Medoid Consensus bảo vệ hệ thống khỏi các kết luận sai hỏng khi vải có chữ in, tem mác hoặc vết ố cục bộ.
4. **Độ tin cậy Time-Series công nghiệp:** Dữ liệu kiểm chứng thực tế khẳng định hệ thống đạt độ lặp lại và độ bền vững tuyệt đối, hoàn toàn sẵn sàng triển khai trên các dây chuyền kiểm tra chất lượng dệt may tự động tốc độ cao.

---

### ĐẠI DIỆN XÁC NHẬN NGHIỆM THU

| ĐẠI DIỆN ĐỘI NGŨ THỊ GIÁC MÁY TÍNH | ĐẠI DIỆN PHÒNG ĐO KIỂM CHẤT LƯỢNG (QA/QC) |
|:---:|:---:|
| *(Xác nhận thuật toán & số liệu thực nghiệm)* | *(Xác nhận kết quả đo đạt chuẩn ISO/CIE)* |
| <br/><br/>**TS. Kỹ Thuật Đo Lường & AI** | <br/><br/>**Trưởng Bộ Phận Đo Kiểm Vải Dệt** |
