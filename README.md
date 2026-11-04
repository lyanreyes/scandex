# ScanDex

Scan a Pokémon TCG card with your phone camera or your PC/laptop webcam, see its current market value, and keep it in a digital binder.

* **Web app** (`web/`): HTML, CSS and JavaScript with no build step, with a Pokédex-style theme and a Poké Ball shutter. The layout is a responsive grid that works on a phone and on a PC. Camera access uses `getUserMedia`. You can also upload a photo.
* **AI / data-science pipeline** (`ml/`, Python): collects the dataset, cleans it, runs EDA, builds the visual index, trains the fusion model and evaluates it.

## How it works

```
camera frame ─► crop to on-screen guide ─► fingerprints (aHash, dHash, pHash-64, pHash-256, colour layout, HSV histogram)
            ─► k-NN search over 20,617 indexed cards, ranked by a pairwise logistic-regression fusion model
            ─► OCR text check of the printed collector number (Tesseract.js) confirms or corrects the result
            ─► live price: Pokémon TCG API (TCGplayer USD + Cardmarket EUR)
            ─► add to binder (stored on the device: value totals, refresh prices, CSV export, backup)
```

## Run the app

```bash
python serve.py
```
Open http://localhost:8000 on the PC. The webcam works on `localhost`.

For a **phone** on the same Wi-Fi, the page has to be served over HTTPS because browsers block the camera on plain HTTP:

```bash
python serve.py --https
```
Open the `https://<LAN-IP>:8443` address the server prints. The certificate is self-signed, so the phone shows a warning once. Tap **Advanced → Proceed**. Windows Firewall may ask you to allow Python on private networks.

The `web/` folder is fully static. You can also deploy it to GitHub Pages or Netlify for a real HTTPS link with no warning.

## Rebuild the model (Python 3.11+)

```bash
pip install -r requirements.txt
cd ml
python 01_collect_dataset.py      # metadata + prices + images from the Pokémon TCG API (~20.6k cards, ~600 MB)
python 02_preprocess_eda.py       # cleaning, duplicates, outliers, figures -> outputs/
python 03_build_index.py          # fingerprints -> data/processed/features.npz + web/model/*.json
python 04_evaluate.py             # experiments, LR fusion training -> outputs/, web/model/fusion.json
python 05_evaluate_real.py        # optional: accuracy on your own phone photos (see below)
node ../tests/parity_test.mjs     # JS fingerprints must equal the Python ones
```

The Pokémon TCG API sometimes returns 500/502 errors. The collector retries with back-off and caches each set, so you can rerun it and it will resume. Set `POKEMONTCG_API_KEY` (free at pokemontcg.io) to get higher rate limits.

### Real-photo test
Put your own photos in `ml/data/real/` and name each one after its Pokémon TCG API card id, for example `sv3pt5-25.jpg`. Crop each photo to the card, the same way the app's guide frame does. Then run `05_evaluate_real.py`.

## Data scope
* Every English Pokémon TCG set in the API: 20,617 cards from 172 sets (1999 – 2026). Set `POKEMON_SERIES` in `ml/01_collect_dataset.py` to index fewer eras.

If a scan is not recognised confidently, the app reads the card's printed number and name and searches the online catalogue.

## Data sources and licences
Card data, images and prices come from the [Pokémon TCG API](https://pokemontcg.io). They are used for non-commercial, educational purposes. Pokémon card images and names are trademarks of their owners. ScanDex is not affiliated with The Pokémon Company, Nintendo or TCGplayer. Prices are market estimates and do not account for card condition or grading.

## Paper
`paper/build_paper.py` builds `paper/ScanDex-IMRaD.docx` (Klarify IMRaD layout) from the numbers in `ml/outputs/`. Rerun it after changing the model. `paper/make_diagram.py` draws Figure 1.
