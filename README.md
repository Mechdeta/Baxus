# 🥃 Whisky Goggles

A visual price checker for whisky bottles using image matching and OCR.

## 🚀 How It Works

1. **Image Preprocessing**  
   Uses OpenCV to grayscale, threshold, and denoise input images.

2. **SIFT Feature Extraction**  
   Detects key visual features using SIFT.

3. **OCR Text Extraction**  
   Uses Tesseract to extract text from the label for fuzzy text matching.

4. **Matching & Scoring**  
   Compares SIFT descriptors and text to every image in the dataset:

   ◦ Uses `BFMatcher` + Lowe's ratio test  
   ◦ Computes confidence score from feature match + text similarity

5. **Price Logging**  
   User inputs observed price at the store, which is recorded along with the most likely match.

---

## 🗂️ Project Structure

```
.
├── 501 Bottle Dataset - Sheet1.csv     # Raw dataset with names, links, image paths
├── convert_csv_to_json.py              # Script to convert CSV to JSON
├── baxus_dataset.json                  # Final JSON dataset used in matching
├── whisky_goggles.py                   # Main whisky matching script
├── feature_cache.pkl                   # Cached SIFT descriptors to save processing time
├── README.md                           # Documentation
```

---

## 📦 Requirements

Install dependencies with:

```bash
pip install opencv-python pytesseract numpy scikit-learn pandas tqdm
```

Also install Tesseract-OCR:
- On Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
- On Linux: `sudo apt install tesseract-ocr`

---

## 💡 Sample Usage

```bash
python whisky_goggles.py
```

You’ll be prompted to:
- Upload an image of a bottle
- Enter observed price (e.g., 49.99)

The script returns the most likely match based on feature descriptors + OCR text.

---

## 🔮 Future Improvements

- Use CLIP or deep learning-based image encoders for better accuracy
- Web interface using Flask or Streamlit
- Save past price logs for crowd-sourced pricing data
- Add bottle region detection (label only crop)

---

## 📝 License

MIT License. Use and modify freely!

---

## 👀 Example Output

```
Enter the price (e.g., 49.99):  55
Top Match: Blanton's Original Single Barrel
Confidence: 0.4239
Logged Price: $55.00
```

