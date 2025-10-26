# easy123_pic2stencil.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app for "Easy123Pic2Stencil"
# - Baseline stencil pipeline (grayscale → contrast → blur/edge → threshold)
# - Consistent previews & session state (orig + stencil cached once)
# - Floating islands, auto-bridge
# - Draw/Touch-up canvas (no refresh needed; correct scaling; safe apply)
#
# Recommended deps:
#   pip install streamlit==1.39.0 pillow==10.4.0 opencv-python-headless==4.10.0.84
#   pip install streamlit-drawable-canvas==0.9.3
# Run:
#   streamlit run easy123_pic2stencil.py

import io, json, math
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

import streamlit as st
from streamlit_drawable_canvas import st_canvas


# ────────────────────────── PAGE / STYLE ──────────────────────────
st.set_page_config(page_title="Easy123Pic2Stencil", page_icon="🎃", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1260px;}
.small {font-size:.9rem; color:#777}
.refcap {font-size:.8rem; color:#666; margin-top:-.35rem}
.panel {padding:.75rem 1rem; border:1px solid rgba(255,255,255,.08); border-radius:.5rem; background:rgba(255,255,255,.02);}
h3, h4 { margin-top: .4rem; }
hr { margin: .6rem 0 .8rem 0; }
</style>
""", unsafe_allow_html=True)
st.title("🎃 Easy123Pic2Stencil")


# ────────────────────────── PRESET / STATE ──────────────────────────
@dataclass
class Preset:
    stencil_type: str = "Simple (2 colors)"   # or "Multi-layer (3-4 colors)"
    threshold: int = 128
    blur_amount: int = 2
    edge_enhance: bool = True
    invert_colors: bool = False
    contrast: float = 1.5
    bridge_width_px: int = 8
    bridge_color: int = 255
    size_mode: str = "Round (pumpkin)"
    units: str = "in"
    horiz_circ: float = 34.0
    vert_circ: float = 34.0
    flat_w: float = 12.0
    flat_h: float = 10.0


def _init_state():
    if "presets" not in st.session_state:
        st.session_state.presets = {"Default": asdict(Preset())}
    if "current" not in st.session_state:
        st.session_state.current = asdict(Preset())
    # assets:
    # - original_bytes: the last uploaded image (bytes)
    # - original_img: PIL Image (full res)
    # - stencil_gray_full: PIL L (full res)
    if "assets" not in st.session_state:
        st.session_state.assets = {}


_init_state()


def cur_preset() -> Preset:
    return Preset(**st.session_state.current)


def set_preset(p: Preset):
    st.session_state.current = asdict(p)


# ────────────────────────── UTIL / PIPELINE ──────────────────────────
def fit_for_display(img: Image.Image, max_w: int = 560) -> Tuple[Image.Image, float]:
    w, h = img.size
    if w <= max_w:
        return img.copy(), 1.0
    scale = max_w / float(w)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, Image.LANCZOS), scale


def _to_gray(img: Image.Image) -> Image.Image:
    return img.convert("L")


def preprocess_image(img: Image.Image, contrast_val: float) -> Image.Image:
    g = _to_gray(img)
    return ImageEnhance.Contrast(g).enhance(contrast_val)


def create_simple_stencil(gray_img: Image.Image, threshold_val: int, blur_val: int, edge_enhance_val: bool) -> Image.Image:
    img = gray_img
    if blur_val > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_val))
    if edge_enhance_val:
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    bw = img.point(lambda x: 255 if x > threshold_val else 0).convert("L")
    return bw.convert("RGB")


def create_multilayer_stencil(gray_img: Image.Image, blur_val: int) -> Image.Image:
    img = gray_img
    if blur_val > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_val))
    g = np.array(img, dtype=np.uint8)
    layer1 = (g > 192).astype(np.uint8) * 255
    layer2 = ((g > 128) & (g <= 192)).astype(np.uint8) * 170
    layer3 = ((g > 64) & (g <= 128)).astype(np.uint8) * 85
    layer4 = (g <= 64).astype(np.uint8) * 0
    combined = (layer1 + layer2 + layer3 + layer4).astype(np.uint8)
    return Image.fromarray(combined, mode="L").convert("RGB")


def pil_to_cv_gray(img: Image.Image) -> np.ndarray:
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.uint8)


def cv_to_pil_gray(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="L")


# ── Floating Islands / Auto-bridge ──
def detect_floating_islands(stencil_gray: Image.Image):
    arr = pil_to_cv_gray(stencil_gray)
    black = (arr < 128).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(black, connectivity=4)

    h, w = black.shape
    border_mask = np.zeros_like(labels, dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_ids = np.unique(labels[border_mask])

    islands = []
    for comp_id in range(1, num_labels):
        if comp_id not in border_ids:
            comp_mask = (labels == comp_id)
            if int(comp_mask.sum()) > 50:
                islands.append(comp_id)

    overlay = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    for comp_id in islands:
        cnts, _ = cv2.findContours((labels == comp_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)
    return overlay, islands, labels


def auto_bridge(stencil_gray: Image.Image, bridge_width_px: int = 6, bridge_color: int = 255):
    arr = pil_to_cv_gray(stencil_gray)
    h, w = arr.shape
    black = (arr < 128).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(black, connectivity=4)

    border_mask = np.zeros_like(labels, dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_ids = np.unique(labels[border_mask])

    islands = []
    for comp_id in range(1, num_labels):
        if comp_id not in border_ids:
            if (labels == comp_id).sum() > 50:
                islands.append(comp_id)

    out = arr.copy()
    for comp_id in islands:
        ys, xs = np.where(labels == comp_id)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        d = {"up": cy, "down": h-1-cy, "left": cx, "right": w-1-cx}
        dirn = min(d, key=d.get)
        if dirn == "up":
            p0, p1 = (cx, cy), (cx, 0)
        elif dirn == "down":
            p0, p1 = (cx, cy), (cx, h-1)
        elif dirn == "left":
            p0, p1 = (cx, cy), (0, cy)
        else:
            p0, p1 = (cx, cy), (w-1, cy)
        cv2.line(out, p0, p1, color=bridge_color, thickness=max(1, int(bridge_width_px)))
    return cv_to_pil_gray(out)


# ────────────────────────── REF DIAGRAMS ──────────────────────────
def _pumpkin_ref(size=(140, 100)):
    W, H = size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2 + 8
    rx, ry = int(W * 0.34), int(H * 0.36)
    for s in (1.0, 0.9):
        d.ellipse([cx - int(rx * s), cy - int(ry * s), cx + int(rx * s), cy + int(ry * s)], outline=(0, 0, 0), width=2)
    d.rectangle([cx - 10, cy - ry - 16, cx + 10, cy - ry - 6], outline=(0, 0, 0), width=2)
    return img


def ref_h():
    img = _pumpkin_ref(); d = ImageDraw.Draw(img)
    W, H = img.size; cx, cy = W // 2, H // 2 + 8; rx = int(W * 0.34)
    d.line([cx - rx - 12, cy, cx + rx + 12, cy], fill=(0, 0, 0), width=3)
    return img


def ref_v():
    img = _pumpkin_ref(); d = ImageDraw.Draw(img)
    W, H = img.size; cx, cy = W // 2, H // 2 + 8; ry = int(H * 0.36)
    d.line([cx, cy - ry - 12, cx, cy + ry + 12], fill=(0, 0, 0), width=3)
    return img


def ref_flat():
    W, H = 140, 100
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img); cx, cy = W // 2, H // 2
    d.rectangle([cx - 50, cy - 30, cx + 50, cy + 30], outline=(0, 0, 0), width=2)
    return img


# ────────────────────────── COMMON UPLOAD ──────────────────────────
with st.sidebar:
    st.header("Upload")
    up = st.file_uploader(
        "Upload your image",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
        key="upl_main"
    )
    if up is not None:
        # Cache original in session for all tabs
        st.session_state.assets["original_bytes"] = up.read()
        st.session_state.assets["original_img"] = Image.open(io.BytesIO(st.session_state.assets["original_bytes"])).convert("RGB")
        # Invalidate prior stencil so it will be recomputed with current sliders
        st.session_state.assets.pop("stencil_gray_full", None)


# ────────────────────────── TABS ──────────────────────────
tabs = st.tabs(["Stencil", "Sizing", "Floating Islands", "Draw", "Presets"])


# ── Tab 1: Stencil ──
with tabs[0]:
    st.subheader("Stencil")
    p = cur_preset()

    with st.sidebar:
        st.header("Stencil Settings")
        p.stencil_type = st.radio("Stencil Type", ["Simple (2 colors)", "Multi-layer (3-4 colors)"],
                                  index=0 if p.stencil_type.startswith("Simple") else 1, key="stype")
        st.caption("Tip: 'Simple' for carving; 'Multi-layer' for painting/shading.")
        p.threshold = st.slider("Detail Level", 0, 255, p.threshold,
                                help="Lower = more black (more cut), higher = more white.", key="thresh")
        p.blur_amount = st.slider("Blur/Smooth", 0, 10, p.blur_amount, help="Reduce small noise/details.", key="blur")
        p.edge_enhance = st.checkbox("Edge Enhancement", value=p.edge_enhance, key="edgeenh")
        p.invert_colors = st.checkbox("Invert Colors", value=p.invert_colors, key="inv")
        p.contrast = st.slider("Contrast", 0.5, 3.0, p.contrast, 0.1, key="contrast")
    set_preset(p)

    orig = st.session_state.assets.get("original_img")
    if orig is None:
        st.info("👆 Upload an image to generate a stencil.")
    else:
        gproc = preprocess_image(orig, p.contrast)
        if p.stencil_type.startswith("Simple"):
            stencil_rgb_full = create_simple_stencil(gproc, p.threshold, p.blur_amount, p.edge_enhance)
            stencil_gray_full = stencil_rgb_full.convert("L")
        else:
            stencil_rgb_full = create_multilayer_stencil(gproc, p.blur_amount)
            stencil_gray_full = stencil_rgb_full.convert("L")

        if p.invert_colors:
            stencil_rgb_full = ImageOps.invert(stencil_rgb_full.convert("RGB"))
            stencil_gray_full = ImageOps.invert(stencil_gray_full)

        # cache for other tabs
        st.session_state.assets["stencil_gray_full"] = stencil_gray_full

        colL, colR = st.columns(2, gap="large")
        with colL:
            st.markdown("**Original Image**")
            disp, _ = fit_for_display(orig, 560)
            st.image(disp, width=disp.size[0])
            st.caption(f"Original size: {orig.size[0]} × {orig.size[1]} px")
        with colR:
            st.markdown("**Stencil Output**")
            sdisp, _ = fit_for_display(stencil_rgb_full, 560)
            st.image(sdisp, width=sdisp.size[0])

        st.subheader("Download")
        c1, c2 = st.columns(2)

        buf = io.BytesIO()
        stencil_rgb_full.save(buf, format="PNG"); buf.seek(0)
        c1.download_button("📥 Download PNG", data=buf, file_name="stencil.png", mime="image/png", key="dl_png")

        buf2 = io.BytesIO()
        bw = stencil_gray_full.point(lambda x: 255 if x > 127 else 0).convert("1")
        bw.save(buf2, format="PNG"); buf2.seek(0)
        c2.download_button("📥 Download B&W (Print)", data=buf2, file_name="stencil_bw.png", mime="image/png", key="dl_bw")


# ── Tab 2: Sizing ──
with tabs[1]:
    st.subheader("Sizing & References")
    p = cur_preset()

    colA, colB = st.columns(2)
    with colA:
        p.size_mode = st.radio("Sizing mode", ["Round (pumpkin)", "Flat (wall)"],
                               index=0 if p.size_mode.startswith("Round") else 1, key="sizemode")
        p.units = st.radio("Units", ["in", "cm"], index=0 if p.units == "in" else 1, horizontal=True, key="units")
    set_preset(p)

    if p.size_mode.startswith("Round"):
        c1, c2 = st.columns(2)
        with c1:
            st.image(ref_h(), width=140)
            st.caption("Horizontal circumference")
            p.horiz_circ = st.number_input("Horizontal circumference", 0.0, value=float(p.horiz_circ), step=0.1, key="hcirc")
        with c2:
            st.image(ref_v(), width=140)
            st.caption("Vertical circumference")
            p.vert_circ = st.number_input("Vertical circumference", 0.0, value=float(p.vert_circ), step=0.1, key="vcirc")
        set_preset(p)

        def circ_to_d(c): return c / math.pi
        dw, dh = circ_to_d(p.horiz_circ), circ_to_d(p.vert_circ)
        st.markdown(f"Recommended usable face ~ **{0.8*dw:.1f} × {0.8*dh:.1f} {p.units}** (≈80% of diameter).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(ref_flat(), width=140)
            st.caption("Target width")
            p.flat_w = st.number_input("Flat width", 0.0, value=float(p.flat_w), step=0.1, key="flatw")
        with c2:
            st.image(ref_flat(), width=140)
            st.caption("Target height")
            p.flat_h = st.number_input("Flat height", 0.0, value=float(p.flat_h), step=0.1, key="flath")
        set_preset(p)
        st.markdown(f"Flat surface: **{p.flat_w:.1f} × {p.flat_h:.1f} {p.units}**")

    st.markdown('<div class="small">Printer reference: Letter usable ≈ <b>8.0×10.5 in</b> (≈2550×3300 px @300DPI). A4 usable ≈ <b>7.8×11.2 in</b>.</div>', unsafe_allow_html=True)


# ── Tab 3: Floating Islands ──
with tabs[2]:
    st.subheader("Floating Islands")
    sg = st.session_state.assets.get("stencil_gray_full")
    if sg is None:
        st.info("Generate a stencil first (Tab: Stencil).")
    else:
        overlay, islands, _ = detect_floating_islands(sg)
        c1, c2 = st.columns(2)
        with c1:
            d1, _ = fit_for_display(sg, 560)
            st.markdown("**Current Stencil (Gray)**")
            st.image(d1, width=d1.size[0])
        with c2:
            ov = Image.fromarray(overlay)
            d2, _ = fit_for_display(ov, 560)
            st.markdown("**Detected Islands (red outlines)**")
            st.image(d2, width=d2.size[0])

        if islands:
            st.warning(f"⚠️ Found {len(islands)} floating islands. Add bridges or use Auto-Bridge below.")
        else:
            st.success("No floating islands detected.")

        st.markdown("---")
        st.markdown("### Auto-Bridge Islands")
        p = cur_preset()
        p.bridge_width_px = st.slider("Bridge thickness (px)", 1, 20, p.bridge_width_px, key="bwidth")
        bc = st.selectbox("Bridge color", ["White (recommended)", "Black"], index=0 if p.bridge_color == 255 else 1, key="bcolor")
        p.bridge_color = 255 if bc.startswith("White") else 0
        set_preset(p)

        if st.button("🔧 Auto-Bridge All Islands", key="autobridge"):
            bridged = auto_bridge(sg, bridge_width_px=p.bridge_width_px, bridge_color=p.bridge_color)
            st.session_state.assets["stencil_gray_full"] = bridged
            st.success("Bridges added.")
            bdisp, _ = fit_for_display(bridged, 560)
            st.image(bdisp, width=bdisp.size[0], caption="Bridged stencil")

            buf = io.BytesIO()
            out = bridged.point(lambda x: 255 if x > 127 else 0).convert("1")
            out.save(buf, format="PNG"); buf.seek(0)
            st.download_button("📥 Download Bridged B&W", data=buf, file_name="stencil_bridged_bw.png", mime="image/png")


# ── Tab 4: Draw / Touch-up ──
with tabs[3]:
    st.subheader("Draw / Touch-up")

    # Always ensure we can draw: if no stencil yet, try to build one from the current sliders + original
    if st.session_state.assets.get("stencil_gray_full") is None:
        orig = st.session_state.assets.get("original_img")
        p = cur_preset()
        if orig is not None:
            gproc = preprocess_image(orig, p.contrast)
            if p.stencil_type.startswith("Simple"):
                srgb = create_simple_stencil(gproc, p.threshold, p.blur_amount, p.edge_enhance)
                st.session_state.assets["stencil_gray_full"] = srgb.convert("L")
            else:
                srgb = create_multilayer_stencil(gproc, p.blur_amount)
                st.session_state.assets["stencil_gray_full"] = srgb.convert("L")

    sg = st.session_state.assets.get("stencil_gray_full")
    if sg is None:
        st.info("Upload an image on the left and create a stencil first.")
    else:
        bg_disp, scale = fit_for_display(sg.convert("RGB"), max_w=780)
        bg_rgba = bg_disp.convert("RGBA")  # drawable-canvas prefers RGBA

        st.caption("Paint **white** to keep/bridge. Paint **black** to cut. Strokes map back to the full-resolution stencil.")

        col_canvas, col_tools = st.columns([4, 1])
        with col_tools:
            apply_mode = st.radio("Apply mode", ["Paint White (keep)", "Paint Black (cut)"], index=0, key="applymode")
            # Enforce correct visual brush color by mode (prevents user confusion)
            draw_color = "#FFFFFF" if apply_mode.startswith("Paint White") else "#000000"
            st.color_picker("Brush color (visual only)", draw_color, key="brushcolor_fixed", help="Auto-set by mode.")
            stroke_width = st.slider("Brush size", 1, 60, 12, key="brushsize")

        with col_canvas:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            canvas_res = st_canvas(
                fill_color=draw_color + "80",
                stroke_width=stroke_width,
                stroke_color=draw_color,
                background_image=bg_rgba,
                background_color=None,
                update_streamlit=True,
                height=bg_rgba.size[1],
                width=bg_rgba.size[0],
                drawing_mode="freedraw",
                display_toolbar=True,
                key="canvas_draw_fixed_once",  # persistent key so it doesn't reset while tweaking tools
            )
            st.markdown('</div>', unsafe_allow_html=True)

        apply = st.button("Apply drawing to stencil", key="applydraw")
        if apply and canvas_res.image_data is not None:
            canvas_rgba = np.array(canvas_res.image_data, dtype=np.uint8)
            base_rgba = np.array(bg_rgba, dtype=np.uint8)
            # Detect where the user drew (any pixel differing from the background)
            mask_disp = np.any(canvas_rgba[:, :, :3] != base_rgba[:, :, :3], axis=2).astype(np.uint8) * 255

            if mask_disp.max() == 0:
                st.info("No strokes detected.")
            else:
                full_w, full_h = sg.size
                # Map display-space mask back to full-res
                mask_full = cv2.resize(mask_disp, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

                base = np.array(sg, dtype=np.uint8)
                if apply_mode.startswith("Paint White"):
                    base[mask_full > 0] = 255
                else:
                    base[mask_full > 0] = 0

                merged = Image.fromarray(base, mode="L")
                st.session_state.assets["stencil_gray_full"] = merged
                st.success("Applied.")
                md, _ = fit_for_display(merged, 560)
                st.image(md, width=md.size[0], caption="Updated stencil")

        if st.session_state.assets.get("stencil_gray_full") is not None:
            buf = io.BytesIO()
            out = st.session_state.assets["stencil_gray_full"].point(lambda x: 255 if x > 127 else 0).convert("1")
            out.save(buf, format='PNG'); buf.seek(0)
            st.download_button("📥 Download Current (B&W)", data=buf, file_name="stencil_current_bw.png", mime="image/png")


# ── Tab 5: Presets ──
with tabs[4]:
    st.subheader("Presets & Defaults")
    cS, cL = st.columns([1, 1])

    with cS:
        name = st.text_input("Preset name", value="", placeholder="e.g., High Contrast Pumpkin", key="pname")
        if st.button("Save current as preset", key="psave") and name.strip():
            st.session_state.presets[name.strip()] = asdict(cur_preset())
            st.success(f"Saved preset: {name.strip()}")

    with cL:
        if st.session_state.presets:
            pick = st.selectbox("Load preset", list(st.session_state.presets.keys()), key="pload")
            a, b = st.columns(2)
            if a.button("Load selected preset", key="ploadbtn"):
                set_preset(Preset(**st.session_state.presets[pick]))
                st.success(f"Loaded preset: {pick}")
            if b.button("Delete selected preset", key="pdelbtn"):
                if pick in st.session_state.presets:
                    del st.session_state.presets[pick]
                    st.success(f"Deleted preset: {pick}")

    st.markdown("---")
    st.markdown("### Export / Import")
    exp = io.BytesIO()
    exp.write(json.dumps(st.session_state.presets, indent=2).encode("utf-8")); exp.seek(0)
    st.download_button("⬇️ Export all presets (JSON)", data=exp, file_name="stencil_presets.json",
                       mime="application/json", key="pexport")

    up_js = st.file_uploader("Import presets JSON", type=["json"], key="pimport")
    if up_js is not None:
        try:
            data = json.load(up_js)
            if isinstance(data, dict):
                st.session_state.presets.update(data)
                st.success(f"Imported {len(data)} preset(s).")
            else:
                st.error("Invalid preset file.")
        except Exception as e:
            st.error(f"Failed to import: {e}")

    st.markdown("---")
    if st.button("Make current the session default", key="pmakedef"):
        st.session_state.presets["Default"] = asdict(cur_preset())
        st.success("Default updated for this session.")