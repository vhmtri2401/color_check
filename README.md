# Color Textile · Camera Lab

Giao diện desktop để chọn camera USB, đổi định dạng MJPEG/YUYV, độ phân giải, FPS và thuật toán decode; chụp ảnh và so sánh với YUYV gốc.

## Chạy

Máy hiện tại đã có OpenCV, NumPy, PyQt5 và `v4l2-ctl`:

```bash
cd /home/tms-tri/Documents/color_textile
./run.sh
```

Mở camera 2 ở 1080p hoặc xem danh sách thiết bị:

```bash
./run.sh --cam /dev/video2 --size 1920x1080 --fourcc MJPG --fps 30
./run.sh --list
```

`python3 main.py` và `python3 mjpeg_yuyv_lab.py` cũng mở giao diện mới. Entry point giữ tên script cũ; các cờ CLI benchmark/capture của bản cũ không được chuyển sang entry point này.

Trên máy khác, cài các gói Python trong `requirements.txt` và gói hệ thống `v4l-utils` của Linux. Ứng dụng hiện hỗ trợ V4L2 MJPG/YUYV với độ phân giải và khoảng thời gian frame dạng discrete. Không tự đoán mode khi driver không công bố danh sách.

## Thao tác

1. Chọn **Camera → Định dạng truyền → Độ phân giải → FPS** ở bảng trái. Chỉ các lựa chọn được camera công bố mới xuất hiện. FPS tự cập nhật khi đổi định dạng hoặc độ phân giải.
2. Chọn decoder và chế độ xem, rồi bấm **Áp dụng cấu hình**. Decoder không áp dụng cho YUYV gốc. Dòng trên ảnh báo mode camera thực nhận; các ô dưới ảnh báo FPS thực và thời gian xử lý.
3. Bấm **Tìm lại camera** sau khi cắm/rút thiết bị. Camera được nhớ bằng `/dev/v4l/by-id` hoặc `by-path` nếu có, để giữ lựa chọn khi số `/dev/videoN` thay đổi. Nếu camera đang dùng biến mất, luồng dừng để người dùng chọn thiết bị và bắt đầu lại.
4. **Cài đặt…** mở cửa sổ riêng gồm Camera, Chụp & lưu, Hiển thị, và bảng Mode hỗ trợ. Bấm **Lưu và áp dụng** để sử dụng thay đổi; **Hủy** bỏ thay đổi trong cửa sổ.
5. **Lưu ảnh hiện tại** lưu PNG, YUYV thô và JPEG gốc nếu decoder đang lấy MJPEG thô. **Capture & compare** chụp lần lượt MJPEG và YUYV ở cùng độ phân giải, lưu báo cáo và tự trở lại luồng xem trước.

Phím tắt: `Ctrl+R` tìm lại camera, `Ctrl+S` lưu ảnh, `Ctrl+,` mở cài đặt.

Cài đặt được lưu tại `~/.config/color_textile/settings.json`, hoặc theo `XDG_CONFIG_HOME`. Có thể dùng `--settings /đường/dẫn/settings.json` để có cấu hình riêng, và `--no-start` để mở giao diện mà chưa bật camera. Ảnh mặc định lưu trong `captures/` của dự án; thay đổi thư mục trong Cài đặt.

## So sánh ảnh

- **camera**: MJPEG từ camera → decoder → YUYV, so với lần chụp YUYV gốc. Hai lần chụp khác thời điểm, có thể khác cảnh/phơi sáng.
- **roundtrip**: chính ảnh YUYV chuẩn → BGR → JPEG → decoder → YUYV. Dùng để so sánh codec/decoder trên cùng một ảnh.
- **no-jpeg / convert_only**: YUYV → BGR → YUYV, dùng xem sai số của phép chuyển màu.

Cửa sổ so sánh có ảnh chuẩn, ảnh decode, bản đồ `|dY| ×8`, bản đồ SSIM và bảng PSNR Y/U/V/BGR, SSIM Y, MAE/max dY, độ lệch Y/U/V, thời gian decode/convert và dung lượng JPEG. Chọn từng dòng để xem kết quả. Mỗi lần capture có thư mục riêng chứa `metrics.csv`, `compare.png`, JPEG và các tệp `.yuyv`.

Thời gian trong bảng compare chỉ đo một frame. Khi dùng decoder OpenCV tự decode, thời gian giải nén nằm trong cột ĐỌC ở màn hình chính. Giới hạn FPS/kích thước xem trước chỉ ảnh hưởng hiển thị, không giảm độ phân giải ảnh được lưu.

## Đo màu 3 điểm (độ ổn định RGB / LAB)

Bấm **Đo màu 3 điểm…** ở màn hình chính để mở cửa sổ đo. Khi cửa sổ này đang mở, bấm chuột trái lên ảnh xem trước để đặt tối đa 3 điểm (ô ROI, mặc định 31 px). Khi đã đủ 3 điểm, bấm gần một điểm để dời nó; chuột phải để xóa điểm gần nhất.

**Một phiên đo** (mặc định 30 phút, mỗi 2 phút đo một lần, mỗi lần 30 frame liên tiếp, tức 16 lần đo ở phút 0, 2, …, 30):

1. **Chờ ổn định:** lấy trung bình mỗi giây của các ROI; ảnh được coi là ổn định khi trong cửa sổ 5 s mọi giá trị nằm trong ΔE00 ≤ 0,3 so với trung bình cửa sổ. Hết 60 s mà chưa ổn định thì vẫn tiếp tục nhưng có cảnh báo.
2. **Khóa thông số** (tùy chọn, mặc định bật): tắt AE/AWB, giữ exposure, gain, nhiệt độ màu và tắt dynamic framerate qua `v4l2-ctl`. Nếu camera trước đó đang để tự động, phần mềm dò exposure (chia đôi theo L*) và nhiệt độ màu (golden-section theo sắc độ) để ảnh sau khi khóa khớp với ảnh tự động. Có driver báo exposure cũ khi AE bật. Nếu chế độ tay không đạt được độ sáng của AE (AE dùng gain ẩn, như camera USB 1), phần mềm giữ AE tự động, chỉ khóa AWB và cảnh báo. Sau đó chờ ổn định lần nữa.
3. **Các lần đo:** mỗi lần chụp N frame. Thông số camera được đọc lại sau mỗi lần; nếu bị đổi thì bảng các lần đo và `rounds.csv` ghi lại.

**Frame rác (MAD + Mean):** trong mỗi lần đo, với từng kênh R, G, B của ROI, tính z = |c − trung vị| / (1,4826·MAD + ε). Frame có kênh z > 3,5 (và ΔE00 tới trung vị > 0,5; đặt 0 để chỉ dùng MAD) bị loại. Giá trị của lần đo là Mean các frame hợp lệ. MAD chỉ bắt frame bất thường trong một lần chụp; trôi màu dài hạn được theo dõi riêng.

**Biểu đồ:** Time series RGB và LAB (trung bình mỗi giây suốt phiên, chấm là giá trị từng lần đo ± σ, × là frame rác; mặc định vẽ độ lệch so với đầu phiên); Phân phối frame RGB/LAB (box plot các frame của từng lần đo); ΔE00 (trôi màu so với lần 1, ΔE00 từng frame so với trung vị); Bảng các lần đo; Tổng kết (σ trong lần / giữa các lần, độ dốc RGB/phút, trôi ΔE00, ΔE00 giữa các điểm); So sánh Mean / Median / Trimmed mean 10% / MAD + Mean.

**Kết quả** nằm trong `captures/measure_<thời gian>_<camera>/`: `session.json` (tham số, điểm, các bước ổn định, thông số camera trước/sau khi khóa), `frames.csv` (mọi frame), `rounds.csv`, `monitor.csv` (chuỗi 1 s), `summary.csv` và ảnh biểu đồ PNG. Các tệp được ghi sau mỗi lần đo, nên dừng giữa chừng vẫn giữ các lần đã xong.

**Lưu ý:**
- RGB/LAB giả định sRGB D65 từ YUYV BT.601, chưa hiệu chuẩn bằng bảng màu, nên chỉ dùng để so sánh tương đối.
- Camera USB 2 (Realtek 0bda:5858, UVC 1.00) chỉ công bố MJPEG và YUYV; không có RAW Bayer. Nên đo bằng YUYV (không nén JPEG). Ví dụ YUYV 1280×720 @ 5 FPS: 30 frame mất khoảng 6 s mỗi lần đo.
- Tránh để ROI bị cháy sáng; cột Clip % báo tỉ lệ điểm ảnh bị chạm 0/255.
- Không chạy Capture & compare trong khi đo: thao tác này đổi luồng camera nên phiên đo sẽ dừng.

## Kiểm tra đồng màu 1 vòng cách biên 5%

Bấm **Kiểm tra đồng màu…** ở màn hình chính để mở cửa sổ kiểm tra đồng màu 1 vòng sát biên vải (chưa đo điểm giữa):

1. **Phát hiện tấm vải nhẹ (< 30 ms CPU):** hạ mẫu về rộng 648 px, đổi sang CIE $L^*a^*b^*$, lấy viền khung 5% ngoài cùng làm mẫu nền carton (hoặc bấm **Chụp nền trống** để lưu nền tham chiếu), ngưỡng tự động (bounded Otsu + cắt bóng viền tối), nối vùng lớn nhất và **lấp kín mọi lỗ thủng bên trong** (`_fill_internal_holes`) để chữ in/tem ở giữa tấm vải không làm thủng mặt nạ.
2. **Dải vòng đo lùi 5% từ biên vải:** tính cạnh ngắn $S = \min(W_{\text{rect}}, H_{\text{rect}})$ từ `minAreaRect`, tạo dải vòng cách biên ngoài $d = \max(8\text{ px}, 0.05 \times S)$ với bề rộng $w = \max(10\text{ px}, 0.05 \times S)$ bằng `cv2.distanceTransform`, chia đều chu vi thành $K = 12$ đoạn có cung bằng nhau theo chiều kim đồng hồ từ góc trên-trái.
3. **Loại nhiễu chữ in trong từng đoạn:** điểm ảnh lệch $\Delta E_{76} > 8.0$ so với trung vị đoạn bị loại trước khi lấy mẫu; đoạn có $> 20\%$ điểm ảnh bị loại được gắn cờ `CHỮ/BẨN` và loại khỏi kết luận ĐẠT/LỖI.
4. **Đo & kết luận 3 mức ($\Delta E_{76}$ / D50):** chụp $N$ frame YUYV (mặc định 10 frame ở 2592×1944 @ 2 FPS), gộp bằng **MAD + Mean**, so sánh **các đoạn trên vòng với nhau** ($\max_{i<j} \Delta E_{76}(S_i, S_j) \le 0.4$) kèm ngưỡng nhiễu nội tại `noise_floor_de` để kết luận **ĐẠT (`pass`)**, **KHÔNG ĐỒNG MÀU (`fail`)** hoặc **KHÔNG ĐỦ ĐỘ TIN CẬY (`uncertain`)**. Kết quả lưu tại `captures/uniform_<thời gian>_<camera>/`.

## Tổ chức mã

| Tệp | Vai trò |
| --- | --- |
| `main.py` | Entry point, tham số dòng lệnh |
| `color_textile/devices.py` | Quét camera, đọc mode và định danh thiết bị |
| `color_textile/capture.py` | Luồng đọc camera, đổi cấu hình, snapshot, capture/compare |
| `color_textile/decoders.py` | Registry decoder và chuyển đổi YUYV |
| `color_textile/comparison.py` | PSNR/SSIM, xây dựng và xuất báo cáo |
| `color_textile/controls.py` | Bộ chọn camera/mode dùng chung |
| `color_textile/gui.py` | Màn hình chính và cửa sổ so sánh |
| `color_textile/settings.py` | Kiểu dữ liệu, kiểm tra và lưu cài đặt |
| `color_textile/settings_dialog.py` | Cửa sổ cài đặt riêng |
| `color_textile/colorimetry.py` | sRGB ↔ tuyến tính, L*a*b* D65, CIEDE2000 |
| `color_textile/measurement.py` | Lấy mẫu ROI, chờ ổn định, lịch đo, lọc frame rác MAD, tổng hợp, xuất CSV |
| `color_textile/camera_controls.py` | Đọc/khóa/trả thông số UVC (v4l2-ctl), dò exposure/WB khi khóa |
| `color_textile/measure_window.py` | Cửa sổ đo màu 3 điểm, biểu đồ matplotlib và bảng |
| `color_textile/fabric_detect.py` | Phát hiện tấm vải nhẹ trên nền carton ($L^*a^*b^*$ + Otsu + lấp lỗ chữ in) |
| `color_textile/uniformity.py` | Tạo vòng đo cách biên 5%, chia $K$ đoạn đều nhau, lọc chữ in, tính $\Delta E$ vòng và xuất CSV/JSON/ảnh |
| `color_textile/uniformity_window.py` | Cửa sổ kiểm tra đồng màu 1 vòng, biểu đồ phân bố vòng và bảng ma trận $\Delta E$ |

Thêm decoder qua `METHODS` trong `decoders.py`; hàm nhận frame và trả về `bgr`, `yuyv`, `dec`, `cvt`, `jpeg`. `raw=True` yêu cầu byte MJPEG thô, `raw=False` nhận BGR do OpenCV giải nén. Giao diện tự lấy danh sách decoder. Nếu cần tên dễ đọc hơn, bổ sung `METHOD_LABELS` trong `controls.py`.

Thêm trang cài đặt qua `SettingsDialog.add_page(title, widget)`; thêm giá trị lưu vào `AppSettings`, kiểm tra trong `validate()`, và nối phần đọc/ghi điều khiển mới trong cửa sổ. Camera chỉ được truy cập từ một worker; không đọc camera trong callback giao diện. Thống kê chỉ giữ số đo, không giữ một danh sách ảnh lớn trong RAM.

## Ngữ cảnh kế thừa

Nguồn: hội thoại Claude Code `d3fe2976-690d-4e7b-a644-fb903492959d`, tệp `/home/tms-tri/.claude/projects/-home-tms-tri-mjpeg-yuyv-lab/d3fe2976-690d-4e7b-a644-fb903492959d.jsonl`, và mã gốc `/home/tms-tri/mjpeg-yuyv-lab/mjpeg_yuyv_lab.py`.

- **Mục tiêu ban đầu:** chọn camera, thử nghiệm decode MJPEG thành YUYV, chuyển luồng và đo hiệu năng/chất lượng.
- **Quyết định:** dùng MJPEG để giảm lưu lượng USB; giữ YUYV gốc làm tham chiếu; dùng roundtrip cùng frame để đánh giá codec, tách ảnh hưởng đổi cảnh/phơi sáng. Không coi mục tiêu 60 FPS là thông số camera hỗ trợ.
- **Đã có trong bản cũ:** `cv_auto`, `imdecode_cvt`, `imdecode_numpy`, decoder TurboJPEG tùy thư viện; capture/compare, chỉ số và xuất file; nút tìm lại camera. Bản cũ chủ yếu chuyển mode bằng phím tắt trong cửa sổ OpenCV.
- **Còn dang dở trong hội thoại:** tích hợp thêm simplejpeg, TurboJPEG theo các biến thể nghiên cứu và VA-API. Các số benchmark cũ là nội dung lịch sử, không được dùng làm kết quả kiểm thử của bản này.
- **Đã thực hiện lần nâng cấp này:** chuyển sang giao diện PyQt5, lựa chọn mode bằng danh sách, cửa sổ cài đặt riêng, worker camera, lưu cấu hình, định danh camera ổn định và khôi phục khi đổi cấu hình lỗi. Tái sử dụng các phép chuyển màu/so sánh từ mã cũ. Bản gốc và các ảnh capture cũ vẫn ở thư mục nguồn.

## Kiểm tra

Kiểm thử tự động, không dùng camera:

```bash
python3 -m unittest discover -s tests -v
```

Kiểm thử tích hợp trên hai camera USB đang kết nối, chỉ chạy khi camera không được ứng dụng khác sử dụng:

```bash
QT_QPA_PLATFORM=offscreen python3 -m tests.smoke_camera
```

Kiểm thử tích hợp đổi camera, MJPEG/YUYV 1080p, decoder, tìm lại camera, lưu ảnh, áp dụng cài đặt, capture/compare, trở lại luồng và giải phóng thiết bị. Ảnh kiểm tra giao diện ghi ở `/tmp/color-textile-qa`; dữ liệu capture kiểm thử được lưu tạm và dọn sau khi kết thúc.
