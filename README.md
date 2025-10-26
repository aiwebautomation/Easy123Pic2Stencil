
# 🎃 Easy123Pic2Stencil — v1.2.0

**Ultra-simple** pumpkin stencil maker for humans who just want to print and carve.
Default flow is one screen, minimal inputs. Advanced tab is there if you want power tools.

## Run (macOS/Linux/Windows)
```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run easy123_pic2stencil_v1_2_0.py
```

## Simple mode (default)
1. Upload any image.
2. Pick **Round (pumpkin)** or **Flat (wall)**.
3. Enter **two numbers**:
   - Round: **Horizontal** + **Vertical circumference**
   - Flat: **Width** + **Height**
4. Click **Generate PDF**. Print at **100%** scale.

The app includes tiny diagrams that show what to measure.

## Advanced mode
- Threshold / invert / smoothing
- Paper size (letter/a4/custom), margins
- Poster tiling (1x3, 2x2, 4x4…), overlap, crop marks
- Coverage control for curved surfaces

## Tips
- Round size ≈ circumference ÷ π along each axis. Default coverage = 0.8.
- If target is bigger than printable area, the app scales **down** to fit.
- PDF has a footer reminder to print at 100%.

MIT License.
