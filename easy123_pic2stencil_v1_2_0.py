
# easy123_pic2stencil_v1_2_0.py
# v1.2.0 — ultra-simple default with Advanced tab; uses ImageReader to avoid BytesIO errors.
import io, math, re
from typing import Tuple, Optional
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader

APP_NAME = "Easy123Pic2Stencil"
APP_VER = "1.2.0"

st.set_page_config(page_title=f"{APP_NAME} v{APP_VER}", page_icon="🎃", layout="centered")
st.title(f"{APP_NAME} v{APP_VER}")
st.caption("Upload → set size → print at 100%. Simple by default.")

# -------- helpers --------
def to_stencil(im: Image.Image, threshold: int = 120, median: int = 3, invert: bool = False) -> Image.Image:
    g = im.convert("L")
    g = ImageOps.autocontrast(g)
    bw = g.point(lambda p: 255 if p > threshold else 0).convert("1")
    if median and median > 1:
        tmp = bw.convert("L").filter(ImageFilter.MedianFilter(median))
        bw = tmp.point(lambda p: 255 if p > 127 else 0).convert("1")
    if invert:
        bw = ImageOps.invert(bw.convert("L")).convert("1")
    return bw

def unit_factor(units: str) -> float:
    if units.lower().startswith("in"): return inch
    if units.lower().startswith("cm"): return cm
    raise ValueError("Units must be 'in' or 'cm'")

def page_size(name: str):
    n = name.lower().strip()
    if n in ("letter","us-letter"): return letter
    if n == "a4": return A4
    if "x" in n:
        w, h = n.split("x", 1)
        return float(w) * inch, float(h) * inch
    raise ValueError("paper must be 'letter', 'a4', or custom like '11x17'")

def parse_tile(tile: str):
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", tile.lower())
    if not m: return 1,1
    return int(m.group(1)), int(m.group(2))

def draw_crop_marks(c: canvas.Canvas, x: float, y: float, w: float, h: float, bleed: float = 12):
    c.setLineWidth(0.5)
    # left
    c.line(x - bleed, y, x, y); c.line(x - bleed, y + h, x, y + h)
    # right
    c.line(x + w, y, x + w + bleed, y); c.line(x + w, y + h, x + w + bleed, y + h)
    # bottom
    c.line(x, y - bleed, x, y); c.line(x + w, y - bleed, x + w, y)
    # top
    c.line(x, y + h, x, y + h + bleed); c.line(x + w, y + h, x + w, y + h + bleed)

def round_surface_size(h_circ: float, v_circ: float, units: str, coverage: float) -> tuple[float,float]:
    uf = unit_factor(units)
    return (h_circ / math.pi) * uf * coverage, (v_circ / math.pi) * uf * coverage

def flat_surface_size(w_in: float, h_in: float, units: str) -> tuple[float,float]:
    uf = unit_factor(units)
    return w_in * uf, h_in * uf

def _draw_img(c, pil_img: Image.Image, x, y, w, h):
    buf = io.BytesIO(); pil_img.save(buf, format="PNG"); buf.seek(0)
    c.drawImage(ImageReader(buf), x, y, width=w, height=h, mask="auto")

def build_pdf(stencil: Image.Image, tgt_w: float, tgt_h: float, *, paper: str, margins_in: float, cols: int, rows: int, overlap_in: float, add_crop: bool) -> bytes:
    page_w, page_h = page_size(paper)
    margin = margins_in * inch
    max_w = page_w - 2*margin
    max_h = page_h - 2*margin
    overlap = max(0.0, overlap_in) * inch
    layout_w = cols * max_w - (cols - 1) * overlap
    layout_h = rows * max_h - (rows - 1) * overlap
    scale = min(layout_w / max(tgt_w,1), layout_h / max(tgt_h,1), 1.0)

    sw, sh = stencil.size
    s_scale = min((tgt_w*scale)/sw, (tgt_h*scale)/sh)
    s_new = (max(1,int(sw*s_scale)), max(1,int(sh*s_scale)))
    s_img = stencil.resize(s_new, Image.NEAREST)

    big_w, big_h = int(tgt_w*scale), int(tgt_h*scale)
    big = Image.new("1", (big_w, big_h), 255)
    off_x = (big_w - s_img.size[0]) // 2
    off_y = (big_h - s_img.size[1]) // 2
    big.paste(s_img, (off_x, off_y))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    for r in range(rows):
        for col in range(cols):
            x0 = int(round(col * (max_w - overlap)))
            y0 = int(round((rows - 1 - r) * (max_h - overlap)))
            x1 = min(x0 + int(round(max_w)), big_w)
            y1 = min(y0 + int(round(max_h)), big_h)
            x0 = max(0, x0); y0 = max(0, y0)
            tile_img = big.crop((x0,y0,x1,y1))

            if add_crop: draw_crop_marks(c, margin, margin, max_w, max_h)
            _draw_img(c, tile_img, margin, margin, tile_img.size[0], tile_img.size[1])

            c.setFont("Helvetica", 9)
            c.drawString(margin, margin*0.6, f"Print at 100% | Paper={paper.upper()} | Margins={margins_in:.2f}in")
            c.showPage()

    c.save(); buf.seek(0)
    return buf.read()

# -------- UI: Simple (default) + Advanced tab --------
uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","bmp","tif","tiff"])
tabs = st.tabs(["🪄 Simple", "⚙️ Advanced"])

with tabs[0]:
    st.subheader("Simple Mode")
    col = st.columns(2)
    with col[0]:
        kind = st.radio("Sizing type", ["Round (pumpkin)", "Flat (wall)"])
        units = st.radio("Units", ["in", "cm"], horizontal=True)
    with col[1]:
        if kind.startswith("Round"):
            st.image("assets/horizontal.png", caption="Horizontal circumference", use_column_width=True)
            st.image("assets/vertical.png", caption="Vertical circumference", use_column_width=True)
        else:
            st.image("assets/flat.png", caption="Flat surface width × height", use_column_width=True)

    if kind.startswith("Round"):
        a = st.number_input("Horizontal circumference", min_value=0.0, value=34.0, step=0.1)
        b = st.number_input("Vertical circumference", min_value=0.0, value=34.0, step=0.1)
        coverage = 0.80
    else:
        a = st.number_input("Target width", min_value=0.0, value=12.0 if units=='in' else 30.0, step=0.1)
        b = st.number_input("Target height", min_value=0.0, value=10.0 if units=='in' else 25.0, step=0.1)
        coverage = None

    # Hidden but opinionated defaults
    paper="letter"; margins=0.25
    if uploaded and st.button("Generate Stencil PDF", type="primary", use_container_width=True):
        try:
            src = Image.open(uploaded).convert("RGBA")
            stencil = to_stencil(src, threshold=120, median=3, invert=False)
            tgt_w, tgt_h = (round_surface_size(a,b,units,coverage) if kind.startswith("Round") else flat_surface_size(a,b,units))
            pdf_bytes = build_pdf(stencil, tgt_w, tgt_h, paper=paper, margins_in=margins, cols=1, rows=1, overlap_in=0.0, add_crop=True)
            st.success("PDF ready.")
            st.download_button("Download PDF", data=pdf_bytes, file_name="easy123pic2stencil_simple_v1_2_0.pdf", mime="application/pdf")
            st.image(stencil.convert("L"), caption="Stencil preview")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[1]:
    st.subheader("Advanced Mode")
    c1, c2 = st.columns(2)
    with c1:
        kind2 = st.radio("Sizing type", ["Round (pumpkin)", "Flat (wall)"])
        units2 = st.radio("Units", ["in","cm"], horizontal=True)
        if kind2.startswith("Round"):
            coverage2 = st.slider("Coverage (fraction of face)", 0.60, 0.95, 0.80, 0.01)
            a2 = st.number_input("Horizontal circumference", min_value=0.0, value=34.0, step=0.1, key="a2")
            b2 = st.number_input("Vertical circumference", min_value=0.0, value=34.0, step=0.1, key="b2")
        else:
            coverage2 = None
            a2 = st.number_input("Target width", min_value=0.0, value=12.0, step=0.1, key="aw2")
            b2 = st.number_input("Target height", min_value=0.0, value=10.0, step=0.1, key="ah2")

        threshold = st.slider("Threshold", 0, 255, 120, 1)
        median = st.number_input("Median filter (odd, 0=off)", min_value=0, value=3, step=1)
        invert = st.checkbox("Invert black/white", value=False)

    with c2:
        paper = st.text_input("Paper (letter, a4, or custom like 11x17)", value="letter")
        margins = st.number_input("Margins (inches)", min_value=0.0, value=0.25, step=0.05)
        tile = st.text_input("Tiling grid (CxR)", value="1x1")
        overlap = st.number_input("Overlap (inches)", min_value=0.0, value=0.25, step=0.05)
        crop = st.checkbox("Add crop marks", value=True)

    if uploaded and st.button("Generate Advanced PDF", use_container_width=True):
        try:
            cols, rows = parse_tile(tile)
            src = Image.open(uploaded).convert("RGBA")
            stencil = to_stencil(src, threshold=int(threshold), median=int(median), invert=bool(invert))
            tgt_w, tgt_h = (round_surface_size(a2,b2,units2,coverage2) if kind2.startswith("Round") else flat_surface_size(a2,b2,units2))
            pdf_bytes = build_pdf(stencil, tgt_w, tgt_h, paper=paper, margins_in=margins, cols=cols, rows=rows, overlap_in=overlap, add_crop=crop)
            st.success("PDF ready.")
            st.download_button("Download PDF", data=pdf_bytes, file_name="easy123pic2stencil_advanced_v1_2_0.pdf", mime="application/pdf")
            st.image(stencil.convert("L"), caption="Stencil preview")
        except Exception as e:
            st.error(f"Error: {e}")
