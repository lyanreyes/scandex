"""Builds paper/ScanDex-IMRaD.docx (Klarify IMRaD layout) from the pipeline outputs."""
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

from docfmt import Paper

HERE = Path(__file__).parent
OUT = HERE.parent / "ml" / "outputs"

S = pd.read_csv(OUT / "eval_summary.csv", index_col=0)
SEV = pd.read_csv(OUT / "eval_by_severity.csv", index_col=0)
THR = pd.read_csv(OUT / "eval_threshold.csv")
LAT_MEAN, LAT_P95 = map(float, (OUT / "eval_latency.txt").read_text().split(","))
LR, SUM = "Pairwise-LR fusion", "Equal-weight sum"
ORDER = ["aHash-64", "dHash-64", "pHash-64", "pHash-256", "Colour layout", "HSV histogram", SUM, LR]
t07 = THR[THR.threshold == 0.7].iloc[0]
f1 = lambda v: f"{v:.1f}"

P = Paper()
P.title_block(
    "SCANDEX: A CAMERA-BASED POKÉMON TRADING CARD RECOGNITION AND MARKET VALUATION "
    "WEB APPLICATION USING PERCEPTUAL HASHING AND LEARNED SCORE FUSION",
    "Lyan Reyes, [Member 2], [Member 3]",
    ["First Asia Institute of Technology and Humanities",
     "School of Technology, College of Computing and Information Technology",
     "Bachelor of Science in Computer Science"])

P.abstract(
    "Pokémon Trading Card Game (TCG) collectors routinely need to know what a card is worth, yet "
    "identifying the exact printing and looking up its price by hand is slow and error-prone. "
    "This study develops and evaluates ScanDex, a responsive web application that recognizes a "
    "Pokémon card through a phone camera or a PC webcam, retrieves its current market value from "
    "the Pokémon TCG API, and stores it in a digital binder. A dataset of 4,799 cards from 26 "
    "Scarlet & Violet and Mega Evolution era sets (2022–2026) was collected, cleaned and explored. "
    "Each card is represented by six visual fingerprints (aHash, dHash, 64- and 256-bit pHash, a "
    "colour layout and an HSV colour histogram), and k-nearest-neighbour retrieval ranks the "
    "gallery with a pairwise logistic-regression fusion model whose top answer receives a "
    "calibrated confidence. On 3,600 simulated camera captures of 1,200 held-out cards, the fusion "
    f"model reached {f1(S.loc[LR, 'top1'])}% top-1 and {f1(S.loc[LR, 'top5'])}% top-5 accuracy, "
    f"against {f1(S.loc['pHash-64', 'top1'])}% for the best single hash and {f1(S.loc[SUM, 'top1'])}% "
    f"for an equal-weight baseline. Accuracy fell from {f1(SEV.loc[LR, '1'])}% under mild to "
    f"{f1(SEV.loc[LR, '3'])}% under harsh capture conditions. Scans accepted at a 0.70 confidence "
    f"threshold were {t07.precision * 100:.1f}% correct, and a scan took about 4 ms in the browser. "
    "An optical character recognition (OCR) check of the printed collector number corrects many "
    "remaining errors. The results show that lightweight, explainable retrieval can identify cards "
    "fast enough for real-time use on ordinary devices. Gains under harsh conditions and an "
    "evaluation on real photographs remain future work.",
    "Trading Card Recognition, Image Retrieval, Perceptual Hashing, Learning to Rank, Web Application")

P.two_columns()

# ============================== I. INTRODUCTION ==============================
P.heading("I", "Introduction")
P.para(
    "Collectible trading cards have grown into a large secondary market. Recent media reports "
    "describe Pokémon cards as an alternative asset whose average value rose by almost half in a "
    "single year [2], while an academic repeat-sales analysis of 2021–2023 sales found that "
    "returns are volatile and depend strongly on the individual card [4]. In both readings, the "
    "price of a card depends on its exact identity: the same Pokémon can be printed in many sets, "
    "numbers and rarities whose prices differ by several orders of magnitude. Collectors, and "
    "especially new collectors and small sellers, therefore have to identify each card precisely "
    "and then search a marketplace for its price, which is slow when done card by card.")
P.para(
    "Computer vision offers a way to automate this step. Instance-level image retrieval, finding "
    "the one gallery image that shows the same object as a query photo, is a mature research "
    "area [1], and compact binary codes such as perceptual hashes allow millions of comparisons "
    "per second [5]. Perceptual hashes are designed to survive benign changes such as resizing, "
    "compression and mild lighting differences [6], [10]. Commercial card scanners exist [11], "
    "but they are closed systems whose accuracy is not reported in a way that others can "
    "reproduce, and most require an account or a paid plan.")
P.para(
    "This study aims to develop and evaluate ScanDex, a web application that recognizes a Pokémon "
    "card from a phone camera, webcam or uploaded photo, shows its current market price, and keeps "
    "it in a digital binder. Specifically, the study evaluates (1) the top-1 and top-5 retrieval "
    "accuracy of individual visual fingerprints compared with a learned fusion of all of them, "
    "(2) the effect of hash size (64 versus 256 bits), (3) the robustness of the system to "
    "increasingly difficult capture conditions, (4) the trade-off between a confidence threshold "
    "and the share of scans that can be accepted automatically, and (5) the recognition latency "
    "in relation to real-time use.")
P.para(
    "The study covers English Pokémon cards from the Scarlet & Violet and Mega Evolution eras, "
    "the 26 sets released from November 2022 to September 2026 that are listed in the Pokémon "
    "TCG API [9]. Older cards can still be found through the collector-number lookup and the "
    "search page, but they are not part of the visual index or the evaluation. Prices are raw "
    "(ungraded) market prices from TCGplayer and Cardmarket, and the system does not assess card "
    "condition, grading or authenticity. ScanDex is intended as a convenience tool for "
    "collectors and is not a professional appraisal service.")

# ============================== II. METHODOLOGY ==============================
P.heading("II", "Methodology")
P.sub("Research Design")
P.para(
    "The study used a developmental and experimental research design. ScanDex has an offline "
    "pipeline written in Python and an online web application written in HTML, CSS and "
    "JavaScript (Figure 1). The offline pipeline collects the dataset, cleans and explores it, "
    "extracts visual fingerprints and trains the ranking model. Its outputs are a card index and "
    "a set of model weights, saved as two JSON files. The web application loads these files and "
    "performs recognition entirely in the browser, so camera images never leave the device.")
P.figure(HERE / "fig_architecture.png", "ScanDex system architecture")
P.para(
    "When the user presses the Poké Ball shutter, the app crops the region inside the on-screen "
    "card guide. It computes the same fingerprints as the Python pipeline and ranks every card "
    "in the index. Then it fetches the live price of the best match. In parallel, Tesseract.js "
    "reads the printed collector number (for example, 025/165) to confirm or correct the visual "
    "result. A parity test confirmed that the JavaScript and Python implementations produce "
    "bit-identical hashes on the same pixels.")

P.sub("Data Collection")
P.para(
    "Card metadata, images and prices were downloaded from the public Pokémon TCG API [9] on "
    "4 November 2026. For every card, the API provides its set, collector number, rarity, "
    "supertype, HP, an official card image, and TCGplayer (USD) and Cardmarket (EUR) price "
    "summaries. The free API frequently returned HTTP 500 and 502 errors under load. The "
    "collection script therefore requested small pages, retried with increasing back-off, and "
    "cached each set so that an interrupted run could resume. All 26 sets in scope were "
    "retrieved, giving 4,800 card records. Card images were stored as 200 × 280 pixel "
    "thumbnails. Of these, 4,799 were available; one promotional card returned HTTP 404.")

P.sub("Hardware and Software Requirements")
P.table("Software Requirements", [
    ["Item", "Minimum", "Recommended"],
    ["Operating system (dev.)", "Windows 10", "Windows 11"],
    ["Python", "3.11", "3.12 or higher"],
    ["IDE", "VS Code / PyCharm", "VS Code / PyCharm"],
    ["Client browser", "Chrome 110 / Safari 16", "Latest Chrome / Safari"],
    ["Client OS", "Android 10 / iOS 16", "Android 13 / iOS 17"],
], widths=[Inches(1.25), Inches(0.95), Inches(0.95)])
P.para(
    "Table 1 lists the software requirements. Python with NumPy, pandas, OpenCV, scikit-learn and "
    "Matplotlib is used for the data science pipeline. The web application needs only a modern "
    "browser with camera access through the getUserMedia interface, which requires HTTPS or "
    "localhost [7]. A small Python server with a self-signed certificate is included, so "
    "phones on the same network can use the camera.")
P.table("Hardware Requirements", [
    ["Item", "Minimum", "Recommended"],
    ["Dev. CPU", "Intel Core i5 (8th gen)", "Intel Core i7 / Ryzen 7"],
    ["Dev. RAM", "8 GB", "16 GB"],
    ["Storage", "1 GB free", "5 GB free (SSD)"],
    ["Phone camera", "8 MP rear camera", "12 MP with autofocus"],
    ["PC webcam", "720p", "1080p"],
], widths=[Inches(1.05), Inches(1.05), Inches(1.05)])
P.para(
    "Table 2 shows the hardware requirements. No GPU is needed, because none of the models are "
    "deep neural networks. The whole pipeline, including the evaluation, runs on a laptop CPU "
    "in a few minutes.")

P.sub("Data Preprocessing")
P.para(
    "The records were flattened into one table per card. No duplicate card identifiers were "
    "found, and the card without an image was dropped, leaving 4,799 clean cards. Text fields "
    "were standardised: HP was converted to a number, release dates to dates, and missing "
    "rarities to \"Unknown\" (8 cards). Non-positive prices were treated as missing. A card's "
    "market value was defined as the TCGplayer market price of its first available printing in "
    "the order normal, holofoil, reverse holofoil. Of the 4,799 cards, 886 (18.5%) had no "
    "price. Most of these (852) belong to the six 2026 sets, which TCGplayer had not yet priced "
    "in the API. Missing prices were not imputed, because an invented price is worse than no "
    "price in a valuation tool. These cards remain in the visual index but are excluded from "
    "the price statistics. On a log scale, 21 cards lie above the interquartile-range fence "
    "($232.93). They were kept because they are genuine high-value chase cards, not errors. "
    "No two card images were pixel-identical.")

P.sub("Exploratory Data Analysis")
P.para(
    "Card prices are extremely right-skewed (Figure 2). The median card is worth $0.21 and the "
    "90th percentile $14.38, while the most valuable card, Umbreon ex from Prismatic Evolutions, "
    "is listed at $1,269.69. This is why log10(price) is used in every analysis. Price follows "
    "rarity closely (Figure 3). Special Illustration Rares have a median of $27.93, against "
    "$0.15 for Commons and Uncommons. Being numbered above the printed set total (a \"secret "
    "rare\") has the strongest correlation with log price, r = 0.83. HP correlates moderately "
    "(r = 0.38), and set age barely at all (r = 0.02) over this four-year window (Figure 4).")
P.figure(OUT / "eda_price_distribution.png", "Distribution of card market prices (log scale)")
P.figure(OUT / "eda_price_by_rarity.png", "Median market price by rarity")
P.figure(OUT / "eda_correlation.png", "Correlation of log price with card attributes", width=Inches(2.6))

P.sub("Feature Extraction")
P.para(
    "Each card image is trimmed by 5% on every side, which removes the border and reduces the "
    "effect of small alignment errors. It is then resized to a 64 × 88 base image with a "
    "box filter. Six fingerprints are computed from this base image. The aHash thresholds an "
    "8 × 8 grey image at its mean. The dHash encodes the sign of horizontal gradients in a "
    "9 × 8 image. Two pHash variants threshold the low-frequency block of a 2-D discrete cosine "
    "transform at its median: 8 × 8 from a 32 × 32 image (64 bits) and 16 × 16 from a 64 × 64 "
    "image (256 bits). The colour layout is a zero-mean, unit-norm 4 × 6 RGB thumbnail "
    "(72 values). The HSV histogram has 51 bins: 8 hues × 2 saturations × 3 values, plus 3 grey "
    "bins. It is square-rooted (the Hellinger mapping) and does not depend on where the colours "
    "appear in the frame. For each fingerprint, the distance between a query and a card is the "
    "normalised Hamming distance (hashes) or half of one minus the cosine similarity "
    "(vectors), so every distance lies between 0 and 1.")

P.sub("Capture Simulation")
P.para(
    "Collecting thousands of labelled phone photographs was beyond the scope of the project. "
    "Query images were therefore produced by a camera-capture simulator, a form of data "
    "augmentation [8]. Each card is placed on a random background and warped by a random "
    "perspective and rotation. It is then cropped with a random framing error, as a user would "
    "cause when lining the card up with the guide. Finally, the simulator alters exposure, "
    "contrast, gamma and white balance, adds a specular glare blob, Gaussian blur, sensor noise "
    "and JPEG compression. Three severity levels were defined (Table 3, Figure 5).")
P.table("Capture Simulator Parameters (maximum values)", [
    ["Parameter", "Mild", "Moderate", "Harsh"],
    ["Corner jitter", "2%", "5%", "9%"],
    ["Rotation", "±2°", "±5°", "±9°"],
    ["Framing error", "2%", "4%", "7%"],
    ["Blur σ (px)", "0.5", "1.2", "2.2"],
    ["Noise σ", "2", "5", "9"],
    ["Lighting change", "10%", "20%", "32%"],
    ["Glare probability", "0", "0.4", "0.7"],
    ["JPEG quality", "90", "75", "60"],
], widths=[Inches(1.15), Inches(0.65), Inches(0.7), Inches(0.65)])
P.figure(OUT / "eval_capture_examples.png", "Reference images and simulated captures at three severities")

P.sub("Model Training")
P.para(
    "Recognition is k-nearest-neighbour retrieval: the query is compared with all 4,799 cards "
    "and the closest card is returned. To combine the six distances, a pairwise logistic "
    "regression ranker was trained, in the style of RankNet. For each training query, it "
    "receives the difference between the distance vector of a wrong card and that of the "
    "correct card. It learns weights w such that w · D is smaller for the correct card. Wrong "
    "cards were sampled as 30 hard negatives (the closest look-alikes under the baseline score) "
    "plus 20 random negatives. A second logistic regression, the calibrator, maps the best "
    "score s1 and the margin s2 − s1 to the probability that the top answer is correct. An "
    "earlier version that trained a single logistic regression to classify matching and "
    "non-matching pairs ranked worse than the simple baseline. Pair classification does not "
    "directly optimise ranking, so the pairwise formulation was adopted.")
P.para(
    "The cards were split 70/30 at random (seed 42). From the 70% portion, 600 cards (1,800 "
    "captures, one per severity) trained the ranker, and 300 different cards (900 captures) "
    "fitted the calibrator. The 30% test portion supplied 1,200 cards (3,600 captures) that "
    "were never used for fitting. The gallery always contains every card, so each query must "
    "be found among all 4,799 cards.")

P.sub("Evaluation Metrics")
P.para(
    "Performance was measured with top-1 accuracy (the correct card is ranked first), top-5 "
    "accuracy (the correct card appears among the five candidates the app shows), and mean "
    "reciprocal rank (MRR). Two further measures were the precision and coverage of "
    "automatically accepted scans at different confidence thresholds, and latency. Six single "
    "fingerprints and an equal-weight sum (mean pHash distance + 0.5 × dHash + colour layout) "
    "served as baselines.")

P.sub("Ethical Considerations")
P.para(
    "The dataset contains no personal data. Camera frames are processed in the browser and are "
    "never uploaded or stored. The binder is kept in the browser's local storage on the user's "
    "device and can be exported or deleted by the user. Card images, names and prices come from "
    "the public Pokémon TCG API and are used for non-commercial educational purposes. The app "
    "states that it is not affiliated with The Pokémon Company, Nintendo or TCGplayer. Prices "
    "are presented as market estimates, not as offers to buy or sell, because a wrong valuation "
    "could cause a user to under- or over-pay.")

# ============================== III. RESULT ==============================
P.heading("III", "Result")
P.sub("Comparison of Methods")
rows = [["Method", "Top-1 %", "Top-5 %", "MRR"]]
for m in ORDER:
    rows.append([m, f1(S.loc[m, "top1"]), f1(S.loc[m, "top5"]), f"{S.loc[m, 'MRR']:.3f}"])
P.table("Retrieval Accuracy on 3,600 Test Captures", rows,
        widths=[Inches(1.35), Inches(0.6), Inches(0.6), Inches(0.6)], bold_last=True)
P.para(
    f"Table 4 and Figure 6 compare the methods. The pairwise-LR fusion achieved the highest "
    f"accuracy on every metric: {f1(S.loc[LR, 'top1'])}% top-1, {f1(S.loc[LR, 'top5'])}% top-5, "
    f"and an MRR of {S.loc[LR, 'MRR']:.3f}. This is {S.loc[LR, 'top1'] - S.loc[SUM, 'top1']:.1f} "
    f"percentage points above the hand-tuned equal-weight sum and "
    f"{S.loc[LR, 'top1'] - S.loc['Colour layout', 'top1']:.1f} points above the strongest single "
    f"fingerprint, the colour layout ({f1(S.loc['Colour layout', 'top1'])}%). Among the hashes, "
    f"pHash-64 ({f1(S.loc['pHash-64', 'top1'])}%) outperformed dHash "
    f"({f1(S.loc['dHash-64', 'top1'])}%) and aHash ({f1(S.loc['aHash-64', 'top1'])}%).")
P.figure(OUT / "eval_methods.png", "Top-1 and top-5 accuracy per method")
P.para(
    f"In the hash-size experiment, the 256-bit pHash was more accurate than the 64-bit pHash "
    f"under mild conditions ({f1(SEV.loc['pHash-256', '1'])}% vs {f1(SEV.loc['pHash-64', '1'])}%). "
    f"Overall, however, it was less accurate ({f1(S.loc['pHash-256', 'top1'])}% vs "
    f"{f1(S.loc['pHash-64', 'top1'])}%), because its extra high-frequency bits are the first to "
    f"be destroyed by blur and noise. The learned ranker relied mostly on the HSV histogram and "
    f"the colour layout. These two features complement the hashes: they describe colour rather "
    f"than structure, and the histogram does not depend on framing.")

P.sub("Robustness to Capture Conditions")
rows = [["Method", "Mild", "Moderate", "Harsh"]]
for m in ["pHash-64", "Colour layout", "HSV histogram", SUM, LR]:
    rows.append([m] + [f1(SEV.loc[m, str(k)]) for k in (1, 2, 3)])
P.table("Top-1 Accuracy (%) by Capture Severity", rows,
        widths=[Inches(1.35), Inches(0.6), Inches(0.7), Inches(0.6)], bold_last=True)
P.para(
    f"Accuracy decreased as capture conditions worsened (Table 5, Figure 7). Under mild "
    f"conditions, every method except aHash exceeded 89%, and the fusion reached "
    f"{f1(SEV.loc[LR, '1'])}%. The methods separated under moderate conditions, where the fusion "
    f"kept {f1(SEV.loc[LR, '2'])}% while pHash-64 dropped to {f1(SEV.loc['pHash-64', '2'])}%. "
    f"Under harsh conditions, all methods degraded strongly, and the fusion reached "
    f"{f1(SEV.loc[LR, '3'])}%. Of the test captures the fusion did not rank first, 38% still "
    f"had the correct card among the top five candidates shown to the user.")
P.figure(OUT / "eval_severity.png", "Top-1 accuracy versus capture severity")

P.sub("Confidence Threshold and Latency")
rows = [["Threshold", "Auto-accepted", "Precision"]]
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    r = THR[THR.threshold.round(2) == t].iloc[0]
    rows.append([f"{t:.2f}", f"{r.coverage * 100:.1f}%", f"{r.precision * 100:.1f}%"])
P.table("Calibrated Confidence Threshold Trade-off", rows,
        widths=[Inches(0.9), Inches(1.1), Inches(1.1)])
P.para(
    f"Table 6 and Figure 8 show that the calibrated confidence is informative. At a threshold "
    f"of 0.70, {t07.coverage * 100:.1f}% of all scans were accepted automatically, and "
    f"{t07.precision * 100:.1f}% of those were correct. This is the lowest threshold that reached "
    f"97% precision, so it was adopted as the app's \"Confident match\" level. Below it, the app "
    f"labels the answer a best guess and asks the user to confirm it or pick one of the "
    f"alternatives.")
P.figure(OUT / "eval_threshold.png", "Precision and coverage of auto-accepted scans")
P.para(
    f"In Python, extracting the fingerprints, comparing them with all 4,799 cards and ranking "
    f"took {LAT_MEAN:.1f} ms per query on average (95th percentile {LAT_P95:.1f} ms). In the "
    f"browser, the same JavaScript pipeline ran on 90 additional simulated captures. It took "
    f"3.9 ms on average (95th percentile 5.8 ms) and reached 77.8% top-1 and 84.4% top-5 "
    f"accuracy, consistent with the Python results. Retrieving the live price from the API "
    f"usually takes under a second, and the OCR check finishes in the background a few seconds "
    f"later.")

P.sub("System Output")
P.figure(HERE / "fig_app_scan.jpg",
         "ScanDex scan screen: Poké Ball shutter, matched card, OCR verification and prices")
P.para(
    "Figure 9 shows a scan result. The matched card appears next to the user's own capture with "
    "the calibrated confidence, the TCGplayer and Cardmarket prices of each printing, and "
    "alternatives. Here the OCR check read 025/165 and verified the visual match. Prices come "
    "live from the API. During testing, the API sometimes returned HTTP 500 errors. When "
    "that happened, the app fell back to the price snapshot stored in its index and "
    "labelled it as a snapshot. The Binder page totals the collection value, refreshes "
    "prices and exports a CSV file. The layout adapts from a single column on phones to two "
    "columns on desktop screens.")

# ============================== IV. DISCUSSION ==============================
P.heading("IV", "Discussion")
P.para(
    "The findings show that a combination of simple, explainable fingerprints identifies "
    "Pokémon cards reliably when the photo is reasonably clear. Learned score fusion, rather "
    "than any single hash, makes the difference under realistic conditions. Colour features "
    "were more robust than structural hashes, because framing errors and blur shift edges "
    "but leave the overall colour of the artwork largely intact. The pairwise ranking objective "
    "proved essential. A pair classifier fitted to the same features performed worse than the "
    "hand-made baseline, which agrees with the retrieval literature's emphasis on "
    "ranking-oriented training [1]. The results also confirm that perceptual hashes are "
    "sensitive to geometric change [6], [10]. Under harsh captures, no fingerprint alone "
    "exceeded 35% top-1.")
P.para(
    "The calibrated confidence makes the system safe to use despite its errors: about three "
    "in five scans can be accepted automatically with roughly 97% precision, and the rest are "
    "shown as candidates for the user to confirm. The OCR step adds an independent source of "
    "evidence. Tesseract is known to be sensitive to noise [3], so the app treats OCR only as a "
    "check: it re-ranks candidates when a printed number is found and ignores set codes that "
    "it misreads.")
P.para(
    "The study has several limitations. First, the accuracy figures come from simulated "
    "captures of official card scans. Real photos add reflections from card sleeves, holofoil "
    "patterns and backgrounds that the simulator only approximates, so an evaluation on real "
    "photographs is needed. The project includes a script for this purpose. Second, the "
    "visual index covers only the 2022–2026 sets, and newer sets must be added by re-running "
    "the pipeline. Third, the free API was unstable and had not yet priced the newest sets, "
    "which shows that the app depends on a third-party data source. Finally, perceptual hashes "
    "can be deliberately fooled [10], although this is a minor concern for a collector tool.")
P.para(
    "Future work should evaluate the system on a labelled set of real phone photos and add "
    "automatic card-edge detection with perspective correction, which would address the main "
    "cause of harsh-condition errors. It should also compare the fingerprints with compact deep "
    "embeddings or learned hash codes [5] that can still run on a phone. Condition and grading "
    "estimates, and support for other trading card games, would make the valuation more "
    "complete. Overall, ScanDex shows that a lightweight, transparent recognition pipeline can "
    "turn an ordinary phone or webcam into a fast card-pricing tool.")

# ============================== REFERENCES ==============================
P.para("REFERENCES", WD_ALIGN_PARAGRAPH.CENTER, bold=True, indent=False, before=8, after=4)
P.references([
    "Chen, W., Liu, Y., Wang, W., Bakker, E. M., Georgiou, T., Fieguth, P., Liu, L., & Lew, M. S. "
    "(2023). Deep learning for instance retrieval: A survey. ~IEEE Transactions on Pattern Analysis "
    "and Machine Intelligence, 45~(6), 7270–7292. https://doi.org/10.1109/TPAMI.2022.3218591",
    "Fore, P. (2025, July 12). Gen Z men are obsessed with Pokémon cards—they're using 'boy math' "
    "to argue that they'll beat Nvidia stock and the S&P 500. ~Fortune.~",
    "Hegghammer, T. (2022). OCR with Tesseract, Amazon Textract, and Google Document AI: A "
    "benchmarking experiment. ~Journal of Computational Social Science, 5~(1), 861–882. "
    "https://doi.org/10.1007/s42001-021-00149-1",
    "Hilbert, T. (2024). ~Playground to portfolio: Uncovering the potential for investment-grade "
    "returns in collectible sports and Pokémon cards~ [Senior Independent Study thesis, The "
    "College of Wooster]. Open Works.",
    "Luo, X., Wang, H., Wu, D., Chen, C., Deng, M., Huang, J., & Hua, X.-S. (2023). A survey on "
    "deep hashing methods. ~ACM Transactions on Knowledge Discovery from Data, 17~(1), Article 15.",
    "Madden, J., Bhavsar, M., Dorje, L., & Li, X. (2024). Robustness of practical perceptual "
    "hashing algorithms to hash-evasion and hash-inversion attacks. ~arXiv~. "
    "https://arxiv.org/abs/2406.00918",
    "MDN Web Docs. (2026). ~MediaDevices: getUserMedia() method.~ Mozilla. "
    "https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia",
    "Mumuni, A., & Mumuni, F. (2022). Data augmentation: A comprehensive survey of modern "
    "approaches. ~Array, 16~, Article 100258.",
    "Pokémon TCG API. (2026). ~Pokémon TCG Developers API v2 documentation.~ "
    "https://docs.pokemontcg.io",
    "Prokos, J., Fendley, N., Green, M., Schuster, R., Tromer, E., Jois, T., & Cao, Y. (2023). "
    "Squint hard enough: Attacking perceptual hashing with adversarial machine learning. In "
    "~Proceedings of the 32nd USENIX Security Symposium~ (pp. 211–228). USENIX Association.",
    "Ximilar. (2026). ~Korean and Thai Pokémon cards now supported in our AI recognition system~ "
    "[Blog post]. https://www.ximilar.com/blog/",
])

P.save(HERE / "ScanDex-IMRaD.docx")
print("saved", HERE / "ScanDex-IMRaD.docx")
