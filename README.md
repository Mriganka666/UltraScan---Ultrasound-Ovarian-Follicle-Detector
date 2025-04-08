
# 🩺 Ultrasound Ovarian Follicle Detection  
An image processing project built using **Python** and **OpenCV** to detect ovarian follicles from ultrasound scans. This system uses clustering, enhancement, edge detection, and contour analysis to estimate follicle count with precision.

---

## 🚀 Features

🖼️ **Image Upload & Visualization**  
- Loads and displays ultrasound scan images  
- Automatic resizing for consistency  

🧪 **Image Preprocessing**  
- Grayscale conversion  
- Contrast enhancement  
- Region-focused cropping  

🧠 **Follicle Detection & Clustering**  
- KMeans clustering with adjustable clusters  
- Silhouette Score and Davies-Bouldin Index evaluation  
- Binary thresholding and Canny edge detection  
- Cropped analysis for contour-based follicle count  

📊 **Real-Time Results**  
- Processed image display (enhanced, clustered, edge-detected)  
- Highlighted contours over follicles  
- Live count of detected follicles  

🔁 **Modular Design**  
- Split into `script1.py` and `script2.py` for clarity and flexibility  
- `main.py` integrates the GUI and connects to script functions  

---

## 🧮 Additional Tool: Contour Area Calculator

🧊 **Contours_Area.py** provides a standalone GUI-based tool to:
- Upload ultrasound or binary images  
- Detect and compute areas of contours  
- Convert pixel area to mm² using DPI  
- Scrollable result viewer  
- Useful for manual verification or validation of follicle regions  

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Libraries:** OpenCV, NumPy, PIL, scikit-learn, Tkinter  
- **Interface:** Tkinter & OpenCV GUI Windows  

---

## ⚙️ Setup Instructions

1. ✅ Clone the repository  
2. 📦 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. ▶️ Run the main follicle detection interface:
   ```bash
   python main.py
   ```
4. 🧮 To run the separate contour area calculator:
   ```bash
   python Contours_Area.py
   ```

---

## 📁 Project Structure

```
ultrasound-follicle-detection/
├── main.py               # Main interface and flow
├── script1.py            # General processing & clustering with metrics
├── script2.py            # Follicle enhancement, edge detection, contour counting
├── Contours_Area.py      # Standalone tool to calculate contour sizes in mm²
├── assets/               # Optional: folder for test ultrasound images
├── requirements.txt      # Dependency list
└── README.md             # This file
```

---

## 📷 Sample Results _(Optional Section)_

> *👉 [Click here to view sample result](images/sample_result.png) — clustered view, edge detection output, contour stats, etc.*

---

## 🙌 Contributing

Pull requests and suggestions are welcome! Feel free to report issues or improvements.

---

## 📄 License

This project is for educational and research purposes. You are free to use, modify, and enhance it.

---

## 👤 Author & Developer

**Mriganka Ghosh**
