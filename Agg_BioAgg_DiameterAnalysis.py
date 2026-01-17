import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def segment(gray: np.ndarray) -> np.ndarray:
    # Otsu (se inverter, troque THRESH_BINARY por THRESH_BINARY_INV)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    return bw


def extract_particles(bw: np.ndarray, min_area: int = 200):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def measure_contour(c: np.ndarray):
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    eq_diam = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(c)

    circularity = float(4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0

    major, minor = None, None
    if len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))

    return {
        "area_px": area,
        "perimeter_px": perim,
        "eq_diameter_px": eq_diam,
        "circularity": circularity,
        "major_axis_px": major,
        "minor_axis_px": minor,
        "bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(w), "bbox_h": int(h),
    }


def analyze_image(image_path: str, out_dir: str = "out", min_area: int = 200) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir a imagem: {image_path}")

    gray = preprocess(img)
    bw = segment(gray)
    contours = extract_particles(bw, min_area=min_area)

    rows = []
    overlay = img.copy()

    for i, c in enumerate(contours, start=1):
        m = measure_contour(c)
        m["id"] = i
        rows.append(m)

        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
        x, y = m["bbox_x"], m["bbox_y"]
        cv2.putText(overlay, str(i), (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    df = pd.DataFrame(rows).sort_values("id")

    base = Path(image_path).stem
    csv_path = out_dir / f"{base}_measures.csv"
    df.to_csv(csv_path, index=False)

    cv2.imwrite(str(out_dir / f"{base}_binary.png"), bw)
    cv2.imwrite(str(out_dir / f"{base}_overlay.png"), overlay)

    return csv_path


def select_file() -> str | None:
    root = tk.Tk()
    root.withdraw()  # esconde a janelinha principal

    file_path = filedialog.askopenfilename(
        title="Selecione a imagem",
        filetypes=[
            ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("Todos os arquivos", "*.*"),
        ]
    )

    root.destroy()
    return file_path if file_path else None


def main():
    image_path = select_file()
    if not image_path:
        print("Nenhum arquivo selecionado. Encerrando.")
        return

    # Ajuste esse parâmetro conforme sua escala (remove sujeirinha/ruído)
    MIN_AREA = 200

    try:
        csv_path = analyze_image(image_path, out_dir="out", min_area=MIN_AREA)
        messagebox.showinfo("Concluído", f"Análise finalizada!\nCSV salvo em:\n{csv_path}")
        print(f"OK! CSV: {csv_path}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))
        raise


if __name__ == "__main__":
    main()