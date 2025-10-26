# 🎃 Easy123Pic2Stencil  
*by [aiwebautomation](https://github.com/aiwebautomation)*  

Turn any image into a **printable pumpkin carving stencil** — simple, fast, and beginner-friendly.  
Live on **Streamlit Cloud** below 👇

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://easy123pic2stencil.streamlit.app)

---

## 🖥️ App Preview

### Interface
| Upload | Configure | Generate |
|:--:|:--:|:--:|
| ![Upload an image](https://raw.githubusercontent.com/aiwebautomation/Easy123Pic2Stencil/main/docs/preview_upload.png) | ![Set size & simplicity](https://raw.githubusercontent.com/aiwebautomation/Easy123Pic2Stencil/main/docs/preview_controls.png) | ![Stencil output](https://raw.githubusercontent.com/aiwebautomation/Easy123Pic2Stencil/main/docs/preview_stencil.png) |

> *Example: turn any image into a printable pumpkin stencil.*

---

### 🪄 Description

Turn **any image** into a **ready-to-print pumpkin carving stencil PDF** — to scale, tiled, and sized perfectly for your **pumpkin, wall, or window**.  
No Photoshop needed. Just **upload → adjust → print → carve.**  
**Made for Halloween creators, DIYers, makers, and anyone who loves a spooky shortcut.**

---

### 🔍 GitHub Topics
`streamlit` `stencil-maker` `pumpkin-carving` `halloween` `photo-to-stencil` `graffiti-art` `spray-paint` `diy-art` `vinyl-cutting` `laser-cutting` `aiwebautomation` `open-source` `maker-tools` `halloween-stencils` `art-projects` `pdf-generator` `image-processing`

---

🎃 **Easy123Pic2Stencil** is a free, open-source web app that instantly converts any image into a clean, printable stencil — perfect for:
- Pumpkin carving templates  
- Graffiti or spray-paint art  
- Laser or vinyl cutting  
- DIY and classroom art projects  

Upload → tweak → preview → print → carve or paint.  
**Simple mode** is instant. **Advanced tabs** give precision control.

---

## 🧩 Run (macOS / Linux / Windows)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run easy123_pic2stencil.py
```

**Debug mode (verbose logs):**
```bash
streamlit run easy123_pic2stencil.py --logger.level=debug
```

---

## 📦 requirements.txt

```txt
streamlit==1.39.0
streamlit-drawable-canvas==0.9.0
pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
```

> ⚠️ **Note:** The **Draw** tab requires `streamlit-drawable-canvas==0.9.0`.  
> Versions **≥ 0.9.2** break background image rendering in Streamlit multi-tab apps, causing the draw editor not to load.  
> ✅ Confirmed working fix: pin to `0.9.0`.

If you previously installed another version, clean up first:

```bash
pip uninstall -y streamlit-drawable-canvas
rm -rf ~/.streamlit/cache ~/.cache/streamlit
```

---

## 🪄 Simple Mode (Default)

1. Upload any image (photo, logo, etc.)  
2. Choose **Round (pumpkin)** or **Flat (wall)**  
3. Enter two numbers:  
   - **Round:** horizontal + vertical circumference  
   - **Flat:** width + height  
4. Click **Generate Stencil / Download PNG** → **Print at 100% scale**

Tiny reference diagrams show exactly where to measure.

---

## ⚙️ Advanced Tabs

**🖼️ Stencil**  
- Threshold, blur/smooth, contrast, invert  
- Edge enhancement for crisper cuts  
- 2-color (carving) or 3–4-tone (painting) output  

**📏 Sizing**  
- Round or flat targets; inches / centimeters  
- Letter usable area ≈ **8.0 × 10.5 in** (≈ **2550×3300 px @ 300 DPI**)  

**🧱 Floating Islands**  
- Detects “floating” pieces that would fall out  
- One-click **Auto-Bridge** connects islands cleanly  

**✏️ Draw / Touch-Up**  
- Paint white → keep area  
- Paint black → cut area  
- Adjustable brush width  
- Live preview + instant download  

🪲 If the canvas doesn’t appear, refresh and upload the image inside the Draw tab.  
✅ Confirmed fix: `streamlit-drawable-canvas==0.9.0`.

---

## 💾 Presets

- Save, load, export, or import full configurations as JSON.

---

## 💡 Tips

- Round pumpkin face ≈ circumference ÷ π per axis (≈ 80% coverage)  
- Oversized stencils automatically scale to fit printable area  
- Always print at 100% scale for accurate size

---

## 🧪 Debug Mode

Run with debug logging for deeper trace info:

```bash
streamlit run easy123_pic2stencil.py --logger.level=debug
```

Logs show file paths, session state, and applet initialization — useful for Draw tab debugging.

---

## 📜 License

MIT License — free for personal, educational, and commercial use.  
© 2025 aiwebautomation.

---

## 🧠 Keywords for Discovery

pumpkin stencil generator, halloween stencil maker, pumpkin template app, graffiti stencil creator, diy spray paint stencil, svg stencil converter, photo to stencil online, pumpkin carving pattern tool, art project templates, aiwebautomation stencil tools, streamlit stencil app, open source stencil maker

---

✅ Ready for GitHub, Streamlit Cloud, or local use.
