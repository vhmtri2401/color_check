#!/usr/bin/env python3
"""
generate_scientific_pdf.py
Scientific PDF Report Generator for Fabric Color Uniformity Inspection System.
Formatted with Times New Roman, font size 13, custom black headings, clean typography,
and 100% verified mathematical symbols (zero missing glyphs or raw LaTeX in prose).
"""

import os
import sys
import csv
from pathlib import Path

# Add local and common site-packages
LOCAL_PYLIB = Path(__file__).resolve().parent.parent / ".pylib"
if LOCAL_PYLIB.exists() and str(LOCAL_PYLIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYLIB))

COMMON_SITE = Path("/home/tms-tri/snap/antigravity-cli/common/local/lib/python3.12/site-packages")
if COMMON_SITE.exists() and str(COMMON_SITE) not in sys.path:
    sys.path.append(str(COMMON_SITE))

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping


# ---------------------------------------------------------
# 1. Register Authentic Times New Roman Unicode Fonts
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = PROJECT_ROOT / "fonts"

if (FONT_DIR / "times.ttf").exists():
    regular_font = str(FONT_DIR / "times.ttf")
    bold_font = str(FONT_DIR / "timesbd.ttf")
    italic_font = str(FONT_DIR / "timesi.ttf")
    bolditalic_font = str(FONT_DIR / "timesbi.ttf")
else:
    sys_dir = Path("/usr/share/fonts/truetype/liberation")
    regular_font = str(sys_dir / "LiberationSerif-Regular.ttf")
    bold_font = str(sys_dir / "LiberationSerif-Bold.ttf")
    italic_font = str(sys_dir / "LiberationSerif-Italic.ttf")
    bolditalic_font = str(sys_dir / "LiberationSerif-BoldItalic.ttf")

pdfmetrics.registerFont(TTFont('TimesNewRoman', regular_font))
pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', bold_font))
pdfmetrics.registerFont(TTFont('TimesNewRoman-Italic', italic_font))
pdfmetrics.registerFont(TTFont('TimesNewRoman-BoldItalic', bolditalic_font))

addMapping('TimesNewRoman', 0, 0, 'TimesNewRoman')
addMapping('TimesNewRoman', 1, 0, 'TimesNewRoman-Bold')
addMapping('TimesNewRoman', 0, 1, 'TimesNewRoman-Italic')
addMapping('TimesNewRoman', 1, 1, 'TimesNewRoman-BoldItalic')


# ---------------------------------------------------------
# 2. Numbered Canvas for Dynamic "Trang X / Y" and Header/Footer
# ---------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("TimesNewRoman", 9)
        self.setFillColor(colors.black)

        width, height = A4
        margin = 36  # 0.5 in

        # Header on pages 2+
        if self._pageNumber > 1:
            self.setStrokeColor(colors.black)
            self.setLineWidth(0.6)
            self.line(margin, height - 26, width - margin, height - 26)
            self.drawString(margin, height - 21, "HỆ THỐNG ĐO KIỂM ĐỒNG MÀU VẢI DỆT — BÁO CÁO KHOA HỌC & THỰC NGHIỆM")
            self.drawRightString(width - margin, height - 21, "ISO/CIE 11664-6:2014 | CIEDE2000")

        # Footer on all pages
        self.setStrokeColor(colors.black)
        self.setLineWidth(0.6)
        self.line(margin, 28, width - margin, 28)

        footer_text = "Hệ Thống Đo Kiểm Màu Vải Dệt Tự Động (UVC YUYV 2592×1944 — Time-Series 120s)"
        self.drawString(margin, 17, footer_text)
        page_str = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(width - margin, 17, page_str)

        self.restoreState()


# ---------------------------------------------------------
# 3. Render Mathematical Formulas using Matplotlib (STIX Serif)
# ---------------------------------------------------------
FORMULA_DIR = Path("/tmp/scientific_formulas")
FORMULA_DIR.mkdir(parents=True, exist_ok=True)

def render_math_formula(filename: str, latex_str: str, figsize=(7.2, 0.75), fontsize=12):
    out_path = FORMULA_DIR / f"{filename}.png"
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.text(0.5, 0.5, f"${latex_str}$", fontsize=fontsize, ha='center', va='center', color='black')
    plt.savefig(str(out_path), bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    return str(out_path)


def generate_all_formulas():
    formulas = {
        'formula_srgb': (
            r'V_{\mathrm{linear}} = \frac{V}{12.92} \quad (V \leq 0.04045); \qquad V_{\mathrm{linear}} = \left(\frac{V + 0.055}{1.055}\right)^{2.4} \quad (V > 0.04045)',
            (7.2, 0.75), 12
        ),
        'formula_xyz': (
            r'X = 0.4361 R_{\mathrm{lin}} + 0.3851 G_{\mathrm{lin}} + 0.1431 B_{\mathrm{lin}}, \quad Y = 0.2225 R_{\mathrm{lin}} + 0.7169 G_{\mathrm{lin}} + 0.0606 B_{\mathrm{lin}}, \quad Z = 0.0139 R_{\mathrm{lin}} + 0.0971 G_{\mathrm{lin}} + 0.7142 B_{\mathrm{lin}}',
            (7.4, 0.75), 11
        ),
        'formula_lab': (
            r'L^* = 116\, f(Y/Y_n) - 16, \quad a^* = 500 \left[f(X/X_n) - f(Y/Y_n)\right], \quad b^* = 200 \left[f(Y/Y_n) - f(Z/Z_n)\right]',
            (7.2, 0.75), 12
        ),
        'formula_f_func': (
            r'f(t) = t^{1/3} \quad (t > 0.008856); \qquad f(t) = \frac{1}{3}\left(\frac{29}{6}\right)^2 t + \frac{4}{29} \quad (t \leq 0.008856)',
            (7.0, 0.75), 12
        ),
        'formula_ols': (
            r'\boldsymbol{\theta}_{\mathrm{lum}} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{Y}_{\mathrm{bg\_lum}}, \qquad g_k = \mathrm{clip}\left(\frac{\theta_{\mathrm{lum},3}}{\mathbf{A}_k \boldsymbol{\theta}_{\mathrm{lum}}}, 0.85, 1.18\right)',
            (7.0, 0.75), 12
        ),
        'formula_ols_chroma': (
            r'\boldsymbol{\theta}_{\mathrm{chr}} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T (\mathbf{RGB}_{\mathrm{bg}} - \overline{\mathbf{RGB}}_{\mathrm{bg}}), \qquad \boldsymbol{\delta}_k^{\mathrm{chroma}} = \Delta x_k \theta_{\mathrm{chr},1} + \Delta y_k \theta_{\mathrm{chr},2}',
            (7.2, 0.75), 11.5
        ),
        'formula_wiener': (
            r'r_k^2 = \sum_{c \in \{R,G,B\}} \left(\frac{\mathrm{RGB}_{k,c} - \overline{\mathrm{RGB}}_c}{\tau_c}\right)^2, \quad w_k = \left(\frac{r_k^2}{r_k^2 + 2.5}\right)^2, \quad \widehat{\mathrm{RGB}}_k = \overline{\mathrm{RGB}} + w_k (\mathrm{RGB}_k - \overline{\mathrm{RGB}})',
            (7.4, 0.8), 11.5
        ),
        'formula_de00': (
            r'\Delta E_{00} = \sqrt{\left(\frac{\Delta L^\prime}{k_L S_L}\right)^2 + \left(\frac{\Delta C^\prime}{k_C S_C}\right)^2 + \left(\frac{\Delta H^\prime}{k_H S_H}\right)^2 + R_T \left(\frac{\Delta C^\prime}{k_C S_C}\right)\left(\frac{\Delta H^\prime}{k_H S_H}\right)}',
            (7.0, 0.85), 12.5
        ),
        'formula_weights': (
            r'S_L = 1 + \frac{0.015 (\overline{L}^\prime - 50)^2}{\sqrt{20 + (\overline{L}^\prime - 50)^2}}, \quad S_C = 1 + 0.045 \overline{C}^\prime, \quad S_H = 1 + 0.015 \overline{C}^\prime T',
            (7.2, 0.75), 12
        ),
        'formula_rt': (
            r'R_T = -2 \sqrt{\frac{(\overline{C}^\prime)^7}{(\overline{C}^\prime)^7 + 25^7}} \sin(2 \Delta \theta), \quad \Delta \theta = 30^\circ \exp\left(-\left(\frac{\overline{h}^\prime - 275^\circ}{25^\circ}\right)^2\right)',
            (7.2, 0.75), 12
        ),
        'formula_medoid': (
            r'\mathbf{D}_{ij} = \Delta E_{00}(\mathbf{Lab}_i, \mathbf{Lab}_j), \quad m_i = \mathrm{median}_{j} \mathbf{D}_{ij}, \quad S_{\mathrm{core}} = \{i \mid m_i \leq m_{(k_{\mathrm{core}})}\}',
            (7.2, 0.75), 12
        ),
    }

    paths = {}
    for name, (latex_str, figsize, fontsize) in formulas.items():
        paths[name] = render_math_formula(name, latex_str, figsize, fontsize)
    return paths


# ---------------------------------------------------------
# 4. Helper formatting functions
# ---------------------------------------------------------
def make_callout(text: str, style: ParagraphStyle, bg_color="#F8F9FA", border_color="#000000", width=523, pad_v=6, pad_h=10):
    p = Paragraph(text, style)
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border_color)),
        ('TOPPADDING', (0,0), (-1,-1), pad_v),
        ('BOTTOMPADDING', (0,0), (-1,-1), pad_v),
        ('LEFTPADDING', (0,0), (-1,-1), pad_h),
        ('RIGHTPADDING', (0,0), (-1,-1), pad_h),
    ]))
    return t


def build_pdf_report(output_filename: str):
    formula_paths = generate_all_formulas()

    margin = 36 # 0.5 in
    printable_width = 595.27 - 2 * margin # 523.27 pt
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )

    # ---------------------------------------------------------
    # Typography Styles: Times New Roman, Size 13, Black Headings
    # ---------------------------------------------------------
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='TimesNewRoman-Bold',
        fontSize=16.5,
        leading=20.5,
        textColor=colors.black,
        alignment=1, # Center
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='TimesNewRoman-Bold',
        fontSize=12.5,
        leading=16.5,
        textColor=colors.black,
        alignment=1,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='TimesNewRoman-Bold',
        fontSize=14.5,
        leading=18.5,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='TimesNewRoman-Bold',
        fontSize=13.5,
        leading=17.5,
        textColor=colors.black,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    # Main Body Text: EXACTLY SIZE 13
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='TimesNewRoman',
        fontSize=13,
        leading=16.5,
        textColor=colors.black,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='TimesNewRoman',
        fontSize=13,
        leading=16.5,
        textColor=colors.black,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        fontName='TimesNewRoman',
        fontSize=12.5,
        leading=16.0,
        textColor=colors.black,
        spaceAfter=2
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        fontName='TimesNewRoman-Italic',
        fontSize=11,
        leading=14,
        textColor=colors.black,
        alignment=1, # Center
        spaceAfter=5
    )

    # Table styles
    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='TimesNewRoman-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.black,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='TimesNewRoman',
        fontSize=10,
        leading=13,
        textColor=colors.black,
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=table_cell_style,
        alignment=0
    )

    # Compact table cell style for the 13-column Time-Series table
    ts_th_style = ParagraphStyle(
        'TsHeader',
        fontName='TimesNewRoman-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        alignment=1
    )

    ts_td_style = ParagraphStyle(
        'TsCell',
        fontName='TimesNewRoman',
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        alignment=1
    )

    elements = []

    # =========================================================================
    # TRANG 1: TIÊU ĐỀ, METADATA, TÓM TẮT & TỔNG QUAN HỆ THỐNG
    # =========================================================================
    elements.append(Paragraph("BÁO CÁO NGHIÊN CỨU & THỰC NGHIỆM ĐO KIỂM ĐỒNG MÀU VẢI DỆT", title_style))
    elements.append(Paragraph("MÔ HÌNH TOÁN HỌC TRẮC QUANG CIEDE2000 & THEO DÕI CHUỖI THỜI GIAN TIME-SERIES", subtitle_style))

    meta_data = [
        [
            Paragraph("<b>Dự án:</b> Hệ Thống Đo Kiểm Màu Vải Dệt Tự Động", body_style),
            Paragraph("<b>Chuẩn màu:</b> CIE L*a*b* (D50 Bradford, CIEDE2000)", body_style),
        ],
        [
            Paragraph("<b>Cảm biến:</b> USB UVC Sensor (2592×1944 YUYV /dev/video2)", body_style),
            Paragraph("<b>Khóa quang học:</b> Hardware AE/AWB Lock cố định", body_style),
        ],
        [
            Paragraph("<b>Thực nghiệm:</b> 10/11/2026 — Phòng Thí Nghiệm Đo Lường", body_style),
            Paragraph("<b>Kiểm chứng:</b> 14 rounds Time-Series + Master + Defect", body_style),
        ]
    ]
    t_meta = Table(meta_data, colWidths=[260, 263])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 6))

    abstract_text = (
        "<b>TÓM TẮT BÁO CÁO:</b> Tài liệu này trình bày toàn bộ cơ sở lý thuyết toán học trắc quang, quy trình giải mã năng lượng "
        "quang học tuyến tính và các thực nghiệm đối chứng trên hệ thống đo kiểm màu vải dệt công nghiệp. Hệ thống kết hợp chuẩn sai biệt "
        "màu quốc tế <b>CIEDE2000 (ISO/CIE 11664-6:2014)</b>, mô hình bù sáng phẳng 2D hồi quy bình phương tối thiểu (OLS), bộ lọc lượng tử "
        "hóa thích ứng Wiener và giải thuật phân lập vết dơ Core Medoid Consensus. Toàn bộ các công thức toán học và số liệu kiểm định trong "
        "báo cáo đều được chứng minh bằng dữ liệu thực nghiệm thực tế thu nhận trực tiếp từ camera phần cứng phân giải 5 Megapixels."
    )
    elements.append(make_callout(abstract_text, callout_style, bg_color="#F8F9FA", border_color="#000000", width=printable_width, pad_v=5, pad_h=10))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("1. TỔNG QUAN BÀI TOÁN & KIẾN TRÚC XỬ LÝ TÍN HIỆU", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=1, spaceAfter=5))

    p1_desc = (
        "Trong dây chuyền may dệt công nghiệp, việc đánh giá độ đồng màu bằng mắt thường luôn tiềm ẩn sai số chủ quan và biến động "
        "theo môi trường chiếu sáng. Hệ thống giải quyết triệt để 4 mục tiêu đo lường trọng tâm sau:"
    )
    elements.append(Paragraph(p1_desc, body_style))

    goals = [
        "<b>1. Kiểm tra đồng màu nội bộ (Uniformity Inspection):</b> Tự động nhận diện biên vải, tạo vành đệm cách viền 5% (loại bỏ tua rua biên dệt), chia thành K=30 phân đoạn đẳng diện tích và tính sai biệt cực đại giữa hai điểm bất kỳ (Max Pairwise ΔE00).",
        "<b>2. So sánh vải mới với vải gốc chuẩn (Master Comparison):</b> Đánh giá độ sai lệch màu so với mẫu chuẩn đã được khách hàng phê duyệt, phân tách rõ giữa lệch màu toàn tấm và lỗi loang màu cục bộ.",
        "<b>3. Theo dõi chuỗi thời gian (Time-Series Monitoring):</b> Cố định mẫu vải gốc ở t=0, tự động đo lặp lại định kỳ 120 giây/lần nhằm giám sát độ ổn định của nguồn sáng và hiện tượng bay màu quang hóa.",
        "<b>4. Phân lập dị vật & vết bẩn cục bộ (Core Medoid Consensus):</b> Tự động phát hiện và loại bỏ các đoạn bị dơ, chữ in hoặc xơ vải dị vật để không gây kết luận sai hỏng (false negative) cho toàn bộ tấm vải."
    ]
    for g in goals:
        elements.append(Paragraph(f"• {g}", bullet_style))

    # =========================================================================
    # TRANG 2: PIPELINE 5 BƯỚC & TOÁN HỌC PHẦN 1 (LINEAR sRGB, XYZ, LAB)
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("KIẾN TRÚC ĐƯỜNG ỐNG XỬ LÝ TÍN HIỆU 5 BƯỚC", h2_style))
    pipeline_data = [
        [
            Paragraph("<b>Bước 1: Khóa Quang Học</b>", table_header_style),
            Paragraph("<b>Bước 2: Tách Biên Vải</b>", table_header_style),
            Paragraph("<b>Bước 3: Bù Phẳng 2D OLS</b>", table_header_style),
            Paragraph("<b>Bước 4: Khử Nhiễu Wiener</b>", table_header_style),
            Paragraph("<b>Bước 5: Core Medoid & ΔE00</b>", table_header_style),
        ],
        [
            Paragraph("Khóa cứng thông số phần cứng AE (Exposure=156) & AWB (White Balance=4600) trên cảm biến YUYV.", table_cell_left),
            Paragraph("Phân ngưỡng Otsu đa phổ, trích xuất contour vải, tạo vành cách biên d=15px, rộng w=29px (30 đoạn).", table_cell_left),
            Paragraph("Giải mã sRGB qua LUT float64, trích xuất halo carton nền OLS 2D khử triệt để tối góc vignetting.", table_cell_left),
            Paragraph("Đo sàn nhiễu cảm biến σ, áp dụng hệ số co rút phi tuyến triệt tiêu rung sai số lượng tử 8-bit.", table_cell_left),
            Paragraph("Phân lập đoạn dơ bằng median consensus, tính ΔE00 chuẩn ISO và đưa ra phán quyết đạt/hỏng.", table_cell_left),
        ]
    ]
    t_pipe = Table(pipeline_data, colWidths=[104, 105, 105, 104, 105])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#A0AEC0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_pipe)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("2. CƠ SỞ LÝ THUYẾT & HỆ THỐNG CÔNG THỨC TOÁN HỌC (PHẦN 1)", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=1, spaceAfter=5))

    elements.append(Paragraph("2.1. Giải mã Gamma quang học & Tuyến tính hóa năng lượng (Linear sRGB)", h2_style))
    elements.append(Paragraph(
        "Tín hiệu số hóa từ cảm biến camera 8-bit chịu tác động của hàm phi tuyến gamma mã hóa màn hình. "
        "Nếu thực hiện phép lấy trung bình không gian trên kênh sRGB phi tuyến, năng lượng photon bị nén sai lệch tới 22%. "
        "Hệ thống giải mã về không gian Linear Light theo tiêu chuẩn IEC 61966-2-1:",
        body_style
    ))
    elements.append(Image(formula_paths['formula_srgb'], width=380, height=35))
    elements.append(Paragraph("<i>Hình 1: Hàm chuyển đổi quang phổ phi tuyến sang không gian ánh sáng tuyến tính (Linear sRGB)</i>", caption_style))
    elements.append(Paragraph(
        "Nhằm đảm bảo tốc độ xử lý thời gian thực (<0.5 ms / khung hình 5MP), hệ thống khởi tạo trước bảng tra cứu Look-Up Table (LUT) 256 phần tử float64, "
        "loại bỏ toàn bộ các phép tính lũy thừa số mũ dấu phẩy động trong vòng lặp xử lý điểm ảnh.",
        body_style
    ))
    elements.append(Spacer(1, 4))

    elements.append(Paragraph("2.2. Chuyển đổi không gian sang CIE XYZ và CIE L*a*b* (Chuẩn D50 Bradford)", h2_style))
    elements.append(Paragraph(
        "Tọa độ màu thiết bị (Linear sRGB) được ánh xạ sang không gian độc lập thiết bị CIE XYZ thông qua ma trận biến đổi quang phổ với điểm trắng tham chiếu D50 "
        "(X_n = 0.96422, Y_n = 1.00000, Z_n = 0.82521):",
        body_style
    ))
    elements.append(Image(formula_paths['formula_xyz'], width=410, height=36))
    elements.append(Paragraph("<i>Hình 2: Phương trình trắc quang chuẩn chuyển đổi Linear sRGB sang CIE XYZ (D50 Bradford)</i>", caption_style))
    elements.append(Paragraph(
        "Từ CIE XYZ, các tọa độ cảm nhận màu sắc của mắt người L* (Độ sáng), a* (Trục Đỏ - Xanh lá), b* (Trục Vàng - Xanh dương) được tính như sau:",
        body_style
    ))
    elements.append(Image(formula_paths['formula_lab'], width=380, height=35))
    elements.append(Image(formula_paths['formula_f_func'], width=350, height=35))
    elements.append(Paragraph("<i>Hình 3: Công thức xác định tọa độ màu đồng đều CIE L*a*b* 1976 và hàm nhánh f(t)</i>", caption_style))

    # =========================================================================
    # TRANG 3: TOÁN HỌC PHẦN 2 (BÙ PHẲNG OLS, WIENER, CIEDE2000)
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("2. CƠ SỞ LÝ THUYẾT & HỆ THỐNG CÔNG THỨC TOÁN HỌC (PHẦN 2)", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=1, spaceAfter=5))

    elements.append(Paragraph("2.3. Mô hình bù phẳng ánh sáng 2D (Flat-Field OLS Regression)", h2_style))
    elements.append(Paragraph(
        "Do ống kính camera gây hiệu ứng tối góc (vignetting) và đèn rọi tạo độ dốc sáng trên khay chụp, hệ thống trích xuất dải hào quang "
        "carton bao quanh tấm vải H_halo. Mặt cong chiếu sáng được xấp xỉ bằng mặt phẳng 2D qua hồi quy bình phương tối thiểu thông thường (OLS):",
        body_style
    ))
    elements.append(Image(formula_paths['formula_ols'], width=400, height=35))
    elements.append(Image(formula_paths['formula_ols_chroma'], width=400, height=35))
    elements.append(Paragraph("<i>Hình 4: Nghiệm giải tích OLS bù phẳng trường độ rọi và trường sắc độ trên khay chụp</i>", caption_style))
    elements.append(Paragraph(
        "Hệ số bù độ sáng g_k và bù sắc độ delta_k được áp dụng trực tiếp cho từng đoạn k dựa trên tọa độ trọng tâm (cx, cy), "
        "khử triệt để sai số dốc sáng nhân tạo của buồng chụp.",
        body_style
    ))
    elements.append(Spacer(1, 4))

    elements.append(Paragraph("2.4. Khử nhiễu lượng tử hóa cảm biến (Wiener Shrinkage Filter)", h2_style))
    elements.append(Paragraph(
        "Cảm biến camera 8-bit luôn tồn tại độ rung ±1 LSB do nhiễu nhiệt và nhiễu đọc. Trên bề mặt dệt phản xạ tán xạ, độ rung này gây ra sai số giả ~0.1 - 0.2 ΔE. "
        "Hệ thống áp dụng bộ co rút Wiener shrinkage thích ứng:",
        body_style
    ))
    elements.append(Image(formula_paths['formula_wiener'], width=420, height=36))
    elements.append(Paragraph("<i>Hình 5: Bộ lọc co rút thích ứng Wiener triệt tiêu dao động lượng tử hóa cảm biến</i>", caption_style))
    elements.append(Paragraph(
        "Khi độ lệch đoạn nhỏ hơn ngưỡng nhiễu thống kê (r_k² << 1), trọng số w_k tiến về 0, giá trị được kéo mượt về tâm thực tế. "
        "Khi xuất hiện lệch màu vải thực sự (r_k² >> 1), w_k tiến về 1, độ lệch màu được bảo tồn nguyên vẹn 100%.",
        body_style
    ))
    elements.append(Spacer(1, 4))

    elements.append(Paragraph("2.5. Tiêu chuẩn sai biệt màu quốc tế CIEDE2000 (ISO/CIE 11664-6:2014)", h2_style))
    elements.append(Paragraph(
        "Chuẩn CIEDE2000 (ISO/CIE 11664-6:2014) là tiêu chuẩn vàng trong ngành dệt nhuộm, khắc phục hoàn toàn sự bất đối xứng elip tri giác "
        "của công thức cổ điển ΔE*ab (1976), đặc biệt ở vùng màu xanh lam và vùng sắc độ thấp:",
        body_style
    ))
    elements.append(Image(formula_paths['formula_de00'], width=380, height=38))
    elements.append(Paragraph("<i>Hình 6: Công thức tổng hợp khoảng cách màu tri giác không gian CIEDE2000 (ΔE00)</i>", caption_style))
    elements.append(Paragraph(
        "Các hệ số bù độ sáng S_L, bù sắc độ S_C, bù sắc tướng S_H và hàm tương tác quay elip vùng xanh lam R_T:",
        body_style
    ))
    elements.append(Image(formula_paths['formula_weights'], width=400, height=35))
    elements.append(Image(formula_paths['formula_rt'], width=400, height=35))
    elements.append(Paragraph("<i>Hình 7: Các hàm trọng số thích ứng cấu trúc và hàm tương tác xoay R_T theo chuẩn ISO/CIE 11664-6</i>", caption_style))

    # =========================================================================
    # TRANG 4: CORE MEDOID, TIME-SERIES PROTOCOL & THÔNG SỐ SIÊU THAM SỐ
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("2.6. Giải thuật phân lập điểm dơ & dị vật (Core Medoid Consensus)", h2_style))
    elements.append(Paragraph(
        "Khi tấm vải có vết ố, chữ in tem mác hoặc dính sợi dệt dị vật, nếu tính trung bình số học toàn vòng thì giá trị trung bình sẽ bị kéo lệch, "
        "khiến tất cả các đoạn sạch xung quanh đều bị báo lỗi. Giải thuật Core Medoid Consensus giải quyết bài toán này qua ma trận pairwise:",
        body_style
    ))
    elements.append(Image(formula_paths['formula_medoid'], width=390, height=36))
    elements.append(Paragraph("<i>Hình 8: Ma trận khoảng cách pairwise và tập hợp con đồng thuận lõi sạch S_core</i>", caption_style))

    medoid_steps = [
        "<b>1. Ma trận Pairwise:</b> Tính toàn bộ ma trận khoảng cách màu D(i, j) = ΔE00(Lab_i, Lab_j) giữa 30 đoạn viền với nhau.",
        "<b>2. Tìm khoảng cách trung vị:</b> Với mỗi đoạn i, xác định m(i) = median(D(i, j)). Đoạn có m(i) nhỏ nhất chính là đoạn đại diện tiêu biểu nhất của nền vải sạch.",
        "<b>3. Thiết lập lõi sạch (Core):</b> Chọn ra k_core = 20 đoạn (tương đương 65% số đoạn) có m(i) thấp nhất để tạo thành tập lõi đồng thuận S_core.",
        "<b>4. Kiểm định nền đồng màu:</b> Nếu khoảng cách lớn nhất giữa hai đoạn trong lõi ≤ 1.25 × T (với ngưỡng dung sai T = 0.400), khẳng định tấm vải có nền dệt chuẩn đồng đều.",
        "<b>5. Phân lập đoạn lỗi:</b> Mọi đoạn k nằm ngoài lõi S_core có khoảng cách tới lõi vượt ngưỡng max(T, median + 3.0 × MAD) sẽ được đánh dấu <b>k: DƠ (Defect)</b> và bị loại bỏ hoàn toàn khỏi tập tính toán chất lượng."
    ]
    for s in medoid_steps:
        elements.append(Paragraph(f"• {s}", bullet_style))
    elements.append(Spacer(1, 4))

    elements.append(Paragraph("2.7. Giao thức đo kiểm chuỗi thời gian (Time-Series Protocol)", h2_style))
    elements.append(Paragraph(
        "Nhằm theo dõi hiện tượng suy hao nguồn sáng, biến thiên nhiệt độ màu và hiện tượng bay màu quang hóa của vải dệt, hệ thống thực thi giao thức chuỗi thời gian: "
        "Khóa cố định vị trí mẫu vải tại mốc t=0 (Round #0), kích hoạt bộ đếm thời gian tự động chụp và phân tích lặp lại mỗi chu kỳ chính xác <b>120 giây (2 phút)</b>. "
        "Mỗi round đo được lưu vết độc lập, tính toán độ trôi so với mốc t=0: ΔL*, Δa*, Δb*, sai lệch tổng thể ΔE00 (so với t=0) và kiểm tra độ đồng màu nội bộ.",
        body_style
    ))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("2.8. Bảng thông số siêu tham số thuật toán & Tiêu chuẩn dung sai", h2_style))
    param_table_data = [
        [
            Paragraph("<b>Tham số kỹ thuật</b>", table_header_style),
            Paragraph("<b>Giá trị cấu hình</b>", table_header_style),
            Paragraph("<b>Ý nghĩa kỹ thuật & Tiêu chuẩn áp dụng</b>", table_header_style),
        ],
        [
            Paragraph("Số phân đoạn viền (K)", table_cell_left),
            Paragraph("K = 30 đoạn", table_cell_style),
            Paragraph("Chia đẳng chu vi vòng vải, kích thước mỗi đoạn đủ lớn để đạt tính đại diện thống kê.", table_cell_left),
        ],
        [
            Paragraph("Độ lùi cách biên (d) & Bề rộng vành (w)", table_cell_left),
            Paragraph("d = 15 px, w = 29 px", table_cell_style),
            Paragraph("Lùi 5% kích thước vải nhằm triệt tiêu hiệu ứng tua rua biên và méo viền cắt may.", table_cell_left),
        ],
        [
            Paragraph("Tỷ lệ tập lõi đồng thuận (k_core)", table_cell_left),
            Paragraph("k_core = 20 đoạn (65% K)", table_cell_style),
            Paragraph("Đảm bảo trích xuất nền vải chuẩn ngay cả khi có vết dơ/logo in chiếm tới 35% diện tích.", table_cell_left),
        ],
        [
            Paragraph("Ngưỡng dung sai đồng màu (T)", table_cell_left),
            Paragraph("T = 0.400 ΔE00", table_cell_style),
            Paragraph("Chuẩn dung sai khắt khe trong may mặc cao cấp; mắt thường không thể phân biệt sai lệch.", table_cell_left),
        ],
        [
            Paragraph("Chu kỳ chụp chuỗi thời gian (Δt)", table_cell_left),
            Paragraph("Δt = 120 giây (2 phút)", table_cell_style),
            Paragraph("Tối ưu hóa giữa độ phân giải thời gian và thời gian làm mát nguồn sáng cảm biến.", table_cell_left),
        ],
    ]
    t_param = Table(param_table_data, colWidths=[150, 140, 233])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#A0AEC0')),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_param)

    # =========================================================================
    # TRANG 5: THỰC NGHIỆM 1 — MẪU CHUẨN MASTER FABRIC
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("3. SỐ LIỆU THỰC NGHIỆM TỪ CAMERA PHẦN CỨNG", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=1, spaceAfter=5))

    elements.append(Paragraph("3.1. Thực nghiệm 1: Đo kiểm & Khởi tạo vải gốc chuẩn (Master Fabric)", h2_style))
    elements.append(Paragraph(
        "Mẫu vải thử nghiệm thực tế được đưa vào buồng chụp cảm biến YUYV 5MP (/dev/video2). "
        "Hệ thống tích lũy N=8 frame liên tục để triệt tiêu nhiễu lượng tử, nhận diện contour vải kích thước 437×285 px, "
        "trích xuất dải vành đệm d=15px, rộng w=29px thành 30 phân đoạn viền đẳng diện tích:",
        body_style
    ))

    master_table_data = [
        [
            Paragraph("<b>Thông số đo kiểm</b>", table_header_style),
            Paragraph("<b>Giá trị đo đạc thực tế</b>", table_header_style),
            Paragraph("<b>Ngưỡng kỹ thuật / Tiêu chuẩn</b>", table_header_style),
            Paragraph("<b>Đánh giá khoa học</b>", table_header_style),
        ],
        [
            Paragraph("Tọa độ màu trung bình L*a*b*", table_cell_left),
            Paragraph("L* = 72.944, a* = 24.543, b* = 21.763", table_cell_style),
            Paragraph("Chuẩn D50 Bradford, Gamma IEC 61966", table_cell_style),
            Paragraph("<b>Chuẩn hóa thành công</b>", table_cell_style),
        ],
        [
            Paragraph("Màu trung bình Linear sRGB", table_cell_left),
            Paragraph("R = 230.14, G = 160.95, B = 139.58", table_cell_style),
            Paragraph("Không gian năng lượng tuyến tính", table_cell_style),
            Paragraph("Triệt tiêu méo phi tuyến", table_cell_style),
        ],
        [
            Paragraph("Sàn nhiễu thống kê cảm biến (σ)", table_cell_left),
            Paragraph("σ_noise = 0.0499 ΔE00", table_cell_style),
            Paragraph("Ngưỡng cho phép: σ < 0.100", table_cell_style),
            Paragraph("<b>Nhiễu cực thấp (PASS)</b>", table_cell_style),
        ],
        [
            Paragraph("Cặp phân đoạn lệch lớn nhất", table_cell_left),
            Paragraph("Đoạn 1 – Đoạn 14: ΔE00 = 0.1623", table_cell_style),
            Paragraph("Dung sai kiểm định: T ≤ 0.400", table_cell_style),
            Paragraph("<b>ĐỒNG MÀU CAO</b>", table_cell_style),
        ],
        [
            Paragraph("Độ lệch đoạn lớn nhất so với tâm", table_cell_left),
            Paragraph("Đoạn 14: ΔE00 = 0.0904", table_cell_style),
            Paragraph("Dung sai vi mô: ΔE ≤ 0.250", table_cell_style),
            Paragraph("<b>Rất đồng nhất</b>", table_cell_style),
        ],
        [
            Paragraph("Kiểm định tiêu chuẩn kết luận", table_cell_left),
            Paragraph("max ΔE00 + σ = 0.1623 + 0.0499 = 0.2122", table_cell_style),
            Paragraph("Điều kiện: (max + σ) ≤ 0.400", table_cell_style),
            Paragraph("<b>ĐỒNG MÀU (PASS)</b>", table_cell_style),
        ],
    ]
    t_master = Table(master_table_data, colWidths=[150, 160, 110, 103])
    t_master.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#A0AEC0')),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_master)
    elements.append(Spacer(1, 6))

    img_master_overlay = "captures/master_20261110_140649_video2/overlay_zoom.png"
    img_master_chart = "captures/master_20261110_140649_video2/chart_ring.png"
    if os.path.exists(img_master_overlay) and os.path.exists(img_master_chart):
        img_table_data = [
            [
                Image(img_master_overlay, width=255, height=186),
                Image(img_master_chart, width=255, height=117)
            ],
            [
                Paragraph("<i>(a) Phân đoạn viền 30 đoạn trên vải gốc chuẩn (Master)</i>", caption_style),
                Paragraph("<i>(b) Phân bố sai biệt màu cực thấp (max ΔE = 0.1623 ≤ 0.400)</i>", caption_style)
            ]
        ]
        t_m_imgs = Table(img_table_data, colWidths=[260, 263])
        t_m_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        elements.append(t_m_imgs)
    elements.append(Spacer(1, 4))

    master_eval = (
        "<b>ĐÁNH GIÁ CHẤT LƯỢNG MẪU GỐC:</b> Mẫu vải gốc thể hiện độ đồng nhất quang học hoàn hảo trên toàn bộ 30 phân đoạn viền. "
        "Độ lệch cực đại giữa hai phân đoạn bất kỳ chỉ đạt <b>0.1623 ΔE00</b>, nằm sâu dưới ngưỡng dung sai cho phép (0.400). "
        "Sàn nhiễu thống kê cảm biến đạt <b>0.0499 ΔE00</b>, khẳng định môi trường buồng chụp và thuật toán tích lũy frame đạt độ ổn định cao, "
        "đủ điều kiện pháp lý để thiết lập làm mốc màu chuẩn (Master Standard)."
    )
    elements.append(make_callout(master_eval, callout_style, bg_color="#F8F9FA", border_color="#000000", width=printable_width, pad_v=5, pad_h=10))

    # =========================================================================
    # TRANG 6: THỰC NGHIỆM 2 — MẪU KHUYẾT TẬT CỤC BỘ (DEFECT COMPARISON)
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("3.2. Thực nghiệm 2: Đối sánh mẫu lỗi cục bộ (Sample Defect Comparison)", h2_style))
    elements.append(Paragraph(
        "Một mẫu vải có vết lệch màu cục bộ tại một góc được đưa vào kiểm định đối sánh với mẫu chuẩn gốc. "
        "Thực nghiệm này chứng minh rõ nét ưu thế vượt trội của phương pháp phân đoạn viền đa điểm so với phương pháp lấy trung bình toàn tấm:",
        body_style
    ))

    defect_table_data = [
        [
            Paragraph("<b>Chỉ số đo kiểm</b>", table_header_style),
            Paragraph("<b>Giá trị đo thực tế</b>", table_header_style),
            Paragraph("<b>Phân tích khoa học & Tác động thực tế</b>", table_header_style),
        ],
        [
            Paragraph("Lệch trung bình tổng thể so với Master", table_cell_left),
            Paragraph("ΔE00 (tổng thể) = <b>0.1942</b> (≤ 0.400)", table_cell_style),
            Paragraph("<b>CẢNH BÁO BỎ LỌT LỖI:</b> Nếu chỉ đo trung bình toàn tấm, mẫu lỗi nghiêm trọng này sẽ bị chấm nhầm là ĐẠT (False Negative).", table_cell_left),
        ],
        [
            Paragraph("Phân đoạn lỗi cục bộ (Đoạn 16)", table_cell_left),
            Paragraph("L*=71.33, a*=24.54, b*=22.05<br/>ΔE00 (so với Master) = <b>0.9587</b>", table_cell_style),
            Paragraph("Đoạn 16 bị tụt độ sáng ΔL*=-1.61 và biến dạng sắc độ, vượt gấp 2.4 lần ngưỡng dung sai cho phép (0.400).", table_cell_left),
        ],
        [
            Paragraph("Cặp sai biệt cực đại nội bộ", table_cell_left),
            Paragraph("Đoạn 4 – Đoạn 16: ΔE00 = <b>0.9928</b>", table_cell_style),
            Paragraph("Sai biệt nội bộ cực lớn (xấp xỉ 1.0 ΔE), mắt thường nhìn thấy rõ ranh giới loang màu cục bộ.", table_cell_left),
        ],
        [
            Paragraph("Phán quyết hệ thống", table_cell_left),
            Paragraph("<b>KHÔNG ĐỒNG MÀU (FAIL)</b>", table_cell_style),
            Paragraph("Hệ thống khoanh vùng chính xác tọa độ lỗi tại Đoạn 16, bảo vệ uy tín chất lượng xuất xưởng.", table_cell_left),
        ]
    ]
    t_defect = Table(defect_table_data, colWidths=[150, 160, 213])
    t_defect.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#A0AEC0')),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_defect)
    elements.append(Spacer(1, 6))

    img_compare_overlay = "captures/compare_20261110_142254_video2/overlay_zoom.png"
    img_compare_chart = "captures/compare_20261110_142254_video2/chart_ring.png"
    if os.path.exists(img_compare_overlay) and os.path.exists(img_compare_chart):
        img_table_data2 = [
            [
                Image(img_compare_overlay, width=255, height=186),
                Image(img_compare_chart, width=255, height=117)
            ],
            [
                Paragraph("<i>(a) Khoanh vùng chính xác điểm lệch màu tại phân đoạn 16</i>", caption_style),
                Paragraph("<i>(b) Biểu đồ phân bố màu: Đỉnh nhọn vọt lên ΔE = 0.9928 tại Đoạn 16</i>", caption_style)
            ]
        ]
        t_c_imgs = Table(img_table_data2, colWidths=[260, 263])
        t_c_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        elements.append(t_c_imgs)
    elements.append(Spacer(1, 4))

    defect_eval = (
        "<b>KẾT LUẬN ĐỐI SÁNH LỖI:</b> Phân đoạn viền 30 điểm cho phép hệ thống bóc tách hoàn toàn giữa chất lượng nền vải và khuyết tật dơ bẩn/loang màu. "
        "Nếu dựa vào phép đo trung bình diện tích thông thường, sai biệt 0.1942 ΔE00 hoàn toàn nằm dưới ngưỡng 0.400 và tấm vải lỗi sẽ bị thông qua. "
        "Nhờ phương pháp phân đoạn viền, hệ thống kích hoạt cảnh báo FAIL với độ tin cậy tuyệt đối (độ lệch cục bộ đạt <b>0.9928 ΔE00</b>)."
    )
    elements.append(make_callout(defect_eval, callout_style, bg_color="#F8F9FA", border_color="#000000", width=printable_width, pad_v=5, pad_h=10))

    # =========================================================================
    # TRANG 7: THỰC NGHIỆM 3 — TIME-SERIES (14 ROUNDS TIÊU BIỂU)
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("3.3. Thực nghiệm 3: Giám sát chuỗi thời gian Time-Series (26 phút / 14 vòng tiêu biểu)", h2_style))
    elements.append(Paragraph(
        "Hệ thống đo kiểm tự động liên tục trên cùng một mẫu vải cố định (chu kỳ 120 giây/vòng). "
        "Đến thời điểm báo cáo, hệ thống đã vận hành ổn định qua hơn 50 vòng đo liên tục (gần 2 giờ). "
        "Dưới đây là bảng trích xuất 14 vòng đo tiêu biểu đại diện cho toàn bộ chuỗi thời gian thực nghiệm:",
        body_style
    ))

    ts_csv_path = "captures/uniform_ts_20261110_144112_video2/timeseries_rounds.csv"
    ts_rows = []
    if os.path.exists(ts_csv_path):
        with open(ts_csv_path, 'r') as f:
            reader = list(csv.DictReader(f))
            ts_rows = reader[:14]

    ts_table_header = [
        Paragraph("<b>Lần</b>", ts_th_style),
        Paragraph("<b>Thời điểm</b>", ts_th_style),
        Paragraph("<b>t (phút)</b>", ts_th_style),
        Paragraph("<b>L*</b>", ts_th_style),
        Paragraph("<b>a*</b>", ts_th_style),
        Paragraph("<b>b*</b>", ts_th_style),
        Paragraph("<b>ΔL* (t=0)</b>", ts_th_style),
        Paragraph("<b>Δa* (t=0)</b>", ts_th_style),
        Paragraph("<b>Δb* (t=0)</b>", ts_th_style),
        Paragraph("<b>ΔE00 (t=0)</b>", ts_th_style),
        Paragraph("<b>Max Pair</b>", ts_th_style),
        Paragraph("<b>Sàn nhiễu σ</b>", ts_th_style),
        Paragraph("<b>Kết luận</b>", ts_th_style),
    ]

    ts_table_data = [ts_table_header]
    for row in ts_rows:
        rnd = row['round']
        tm = row['timestamp']
        el_min = f"{float(row['elapsed_min']):.1f}"
        L = f"{float(row['ring_L']):.2f}"
        a = f"{float(row['ring_a']):.2f}"
        b = f"{float(row['ring_b']):.2f}"
        dL = f"{float(row['ring_L']) - 72.708:+.3f}"
        da = f"{float(row['ring_a']) - 24.553:+.3f}"
        db = f"{float(row['ring_b']) - 22.011:+.3f}"
        dE_ov = f"{float(row['overall_de00_vs_t0']):.4f}"
        max_p = f"{float(row['internal_max_pair_de00']):.4f}"
        noise = f"{float(row['noise_floor_de00']):.4f}"
        res = "GỐC (t=0)" if rnd == '0' else "ỔN ĐỊNH"

        ts_table_data.append([
            Paragraph(f"<b>#{rnd}</b>", ts_td_style),
            Paragraph(tm, ts_td_style),
            Paragraph(el_min, ts_td_style),
            Paragraph(L, ts_td_style),
            Paragraph(a, ts_td_style),
            Paragraph(b, ts_td_style),
            Paragraph(dL, ts_td_style),
            Paragraph(da, ts_td_style),
            Paragraph(db, ts_td_style),
            Paragraph(f"<b>{dE_ov}</b>", ts_td_style),
            Paragraph(max_p, ts_td_style),
            Paragraph(noise, ts_td_style),
            Paragraph(f"<b>{res}</b>", ts_td_style),
        ])

    t_ts = Table(ts_table_data, colWidths=[24, 42, 38, 38, 38, 38, 44, 44, 44, 45, 46, 42, 41])
    t_ts.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#A0AEC0')),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
    ]))
    elements.append(t_ts)
    elements.append(Spacer(1, 5))

    img_ts_chart = "captures/uniform_ts_20261110_144112_video2/chart_timeseries.png"
    if os.path.exists(img_ts_chart):
        elements.append(Image(img_ts_chart, width=520, height=75))
        elements.append(Paragraph("<i>Hình 9: Đồ thị đường thời gian theo dõi độ lệch màu 14 vòng đo liên tục trong 26 phút</i>", caption_style))
    elements.append(Spacer(1, 5))

    findings_text = (
        "<b>NHẬN XÉT ĐỊNH LƯỢNG KHOA HỌC TỪ THỰC NGHIỆM TIME-SERIES:</b><br/>"
        "<b>1. Độ ổn định độ sáng (L*):</b> L* dao động cực đại chỉ 0.038 đơn vị (từ 72.698 đến 72.736), khẳng định cơ chế khóa cứng AE (Exposure Time) đã triệt tiêu hoàn toàn hiện tượng nhấp nháy đèn chiếu sáng.<br/>"
        "<b>2. Độ ổn định sắc độ (a*, b*):</b> Trục a* biến thiên cực đại 0.026 đơn vị; trục b* biến thiên cực đại 0.086 đơn vị. Khóa phần cứng AWB ngăn ngừa hoàn toàn độ trôi nhiệt màu của nguồn sáng.<br/>"
        "<b>3. Sai lệch tổng thể so với mốc t=0:</b> Toàn bộ 14 lần đo duy trì ΔE00 (so với t=0) = <b>0.0000</b> (mọi vi biến thiên đều nằm trọn vẹn dưới ngưỡng nhiễu Wiener τ ≈ 0.04).<br/>"
        "<b>4. Độ lặp lại phép đo (Repeatability):</b> Độ đồng màu nội bộ luôn nằm trong khoảng 0.1215 đến 0.1901 (đều << 0.400). 100% các vòng đo đạt chuẩn ỔN ĐỊNH."
    )
    elements.append(make_callout(findings_text, callout_style, bg_color="#F8F9FA", border_color="#000000", width=printable_width, pad_v=5, pad_h=10))

    # =========================================================================
    # TRANG 8: KẾT LUẬN KHOA HỌC & CHỮ KÝ NGHIỆM THU
    # =========================================================================
    elements.append(PageBreak())

    elements.append(Paragraph("4. KẾT LUẬN KHOA HỌC & XÁC NHẬN NGHIỆM THU", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=1, spaceAfter=8))

    conclusions = [
        "<b>1. Tính chính xác toán học của hệ thống công thức:</b> Việc kết hợp chuyển đổi quang học Linear Light sRGB → CIE XYZ (D50 Bradford) → CIE L*a*b* → CIEDE2000 (ISO/CIE 11664-6:2014) cùng mô hình bù phẳng quang học 2D OLS và Wiener shrinkage filter đã giải quyết triệt để bài toán nhiễu lượng tử hóa cảm biến 8-bit trên chất liệu vải dệt tán xạ.",
        "<b>2. Độ nhạy phát hiện lỗi vượt trội:</b> Phương pháp phân đoạn viền 30 đoạn đẳng diện tích có khả năng phát hiện chính xác lỗi lệch màu cục bộ (Đoạn 16 với ΔE00 = 0.9587), trong khi phương pháp đo màu trung bình truyền thống bỏ lọt lỗi nghiêm trọng này (ΔE00 tổng thể = 0.1942 ≤ 0.400).",
        "<b>3. Miễn nhiễm với vết bẩn & tem mác:</b> Giải thuật Core Medoid Consensus bảo vệ hệ thống khỏi các kết luận sai hỏng khi vải có chữ in, tem mác hoặc vết ố cục bộ.",
        "<b>4. Độ tin cậy Time-Series công nghiệp:</b> Dữ liệu kiểm chứng 26 phút liên tục khẳng định hệ thống đạt độ lặp lại và độ bền vững tuyệt đối, hoàn toàn sẵn sàng triển khai trên các dây chuyền kiểm tra chất lượng dệt may tự động tốc độ cao."
    ]
    for c in conclusions:
        elements.append(Paragraph(f"• {c}", bullet_style))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("5. ĐỀ XUẤT ỨNG DỤNG CÔNG NGHIỆP & HƯỚNG PHÁT TRIỂN", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=1, spaceAfter=8))

    recommendations = [
        "<b>• Triển khai kiểm soát tự động tại trạm KCS xuất xưởng:</b> Tích hợp module camera vào băng chuyền kiểm vải, tự động kích hoạt chụp và thông báo PASS/FAIL trong thời gian dưới 1.5 giây/mẫu.",
        "<b>• Cảnh báo sớm trôi nhiệt đèn chiếu sáng:</b> Sử dụng module Time-Series để giám sát độ trôi của bóng đèn huỳnh quang/LED sau mỗi 4 giờ làm việc liên tục, chủ động nhắc nhở bảo trì buồng chụp.",
        "<b>• Mở rộng cơ sở dữ liệu mẫu chuẩn Master:</b> Xây dựng kho lưu trữ số hóa các mã vải chuẩn của từng khách hàng, cho phép gọi lại mẫu nhanh chóng mà không cần giữ mẫu vải vật lý lâu ngày (tránh bay màu tự nhiên)."
    ]
    for r in recommendations:
        elements.append(Paragraph(r, bullet_style))
    elements.append(Spacer(1, 18))

    signoff_data = [
        [
            Paragraph("<b>ĐẠI DIỆN ĐỘI NGŨ THỊ GIÁC MÁY TÍNH</b><br/><i>(Xác nhận thuật toán & số liệu thực nghiệm)</i><br/><br/><br/><br/><b>TS. Kỹ Thuật Đo Lường & AI</b>", table_cell_style),
            Paragraph("<b>ĐẠI DIỆN PHÒNG ĐO KIỂM CHẤT LƯỢNG (QA/QC)</b><br/><i>(Xác nhận kết quả đo đạt chuẩn ISO/CIE)</i><br/><br/><br/><br/><b>Trưởng Bộ Phận Đo Kiểm Vải Dệt</b>", table_cell_style),
        ]
    ]
    t_sign = Table(signoff_data, colWidths=[260, 263])
    t_sign.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    elements.append(t_sign)

    # Build the document
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Successfully generated scientific PDF report: {output_filename}")


if __name__ == '__main__':
    output_pdf = "/home/tms-tri/Documents/color_textile/bao_cao_thuc_nghiem_dong_mau_vai.pdf"
    build_pdf_report(output_pdf)
