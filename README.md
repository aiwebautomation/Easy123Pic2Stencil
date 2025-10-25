# 🎃 Easy123Pic2Stencil — The Ultimate Halloween Stencil Generator (v1.0.0)

Turn **any image** into a **ready-to-print pumpkin carving stencil PDF** — to scale, tiled, and sized perfectly for your pumpkin, wall, or window.
No Photoshop needed. Just upload → adjust → print → carve.

**Made for Halloween creators, DIYers, makers, stencil-heads, graffiti artists, and anyone who loves a spooky shortcut.**

---
### 🧠 What It Does
- 🖼️ Upload *any* image (ghosts, logos, artwork, your face, you name it)
- 🪄 Auto-convert to **black/white stencil** ready for carving
- 🎯 Enter **pumpkin circumference** (round mode) or **flat surface size**
- 📐 Generate **true-to-scale** printable PDF
- 📄 Supports **poster tiling** (1×3, 2×2, 4×4 grids)
- ✂️ Optional **crop marks + overlap** for taping large designs
- 🕸️ Perfect for pumpkin carving, wall stencils, T-shirt art, vinyl cutting, laser engraving

---

### 💡 SEO-friendly tags (add to repo topics)

##============================================================
##============================================================

### 🧱 Quick Deploy (Streamlit Cloud)
1. **Fork / Push** this repo to your GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → “New App”
3. Set main file path to `easy123_pic2stencil_v1_0_0.py`
4. Click *Deploy!* and share your link 🎃

---

### 📸 Use Cases
- **Pumpkin carving templates** — resize to match your pumpkin circumference.
- **Wall or window stencils** — flat mode gives exact printable size.
- **Kids’ craft projects** — quick black/white printouts for coloring or tracing.
- **Maker-lab gear** — laser cutter prep (export PDF at 100% scale).

---

### 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) for instant web UI  
- [Pillow (PIL)](https://python-pillow.org) for image processing  
- [ReportLab](https://www.reportlab.com/dev/docs/reportlab-userguide.pdf) for multi-page PDF generation  
- [NumPy](https://numpy.org) for efficient math

---

### ⚙️ Run Locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run easy123_pic2stencil_v1_0_0.py
