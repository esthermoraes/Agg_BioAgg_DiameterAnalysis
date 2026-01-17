import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Pré-processamento
def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# Segmentação (auto-inverte se necessário)
def segment_auto(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = (bw > 0).mean()
    if white_ratio > 0.5:
        bw = cv2.bitwise_not(bw)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    return bw

# Separar objetos colados (distance transform + watershed)
def split_touching_objects(bw: np.ndarray, fg_thresh: float = 0.45) -> np.ndarray:
    bw = (bw > 0).astype(np.uint8) * 255

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, fg_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(bw, k, iterations=2)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img3 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img3, markers)

    separated = np.zeros_like(bw)
    separated[markers > 1] = 255
    return separated

# Contornos + filtro
def extract_particles(bw: np.ndarray, min_area: int = 50):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]

# Medidas: D1/D2/D3 (px)
def feret_diameters(contour: np.ndarray) -> tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    return float(max(w, h)), float(min(w, h))

def measure_diameters(contour: np.ndarray) -> dict:
    area = float(cv2.contourArea(contour))
    d1_eq = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0
    d2_feret_max, d3_feret_min = feret_diameters(contour)
    return {
        "D1_eq_px": d1_eq,
        "D2_feret_max_px": d2_feret_max,
        "D3_feret_min_px": d3_feret_min,
    }

# Analisar imagem + salvar o output
def analyze_image(
    image_path: str,
    results_dir: str = "results",
    min_area: int = 50,
    fg_thresh: float = 0.45
):
    results_dir = Path(results_dir)
    csv_dir = results_dir / "csv"
    id_dir = results_dir / "identification"
    csv_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    gray = preprocess(img)
    bw = segment_auto(gray)
    bw_sep = split_touching_objects(bw, fg_thresh=fg_thresh)

    contours = extract_particles(bw_sep, min_area=min_area)

    # Ordenar por posição (top->bottom, left->right)
    def contour_key(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y, x)
    contours = sorted(contours, key=contour_key)

    overlay = img.copy()
    rows = []

    for idx, contour in enumerate(contours, start=1):
        # diâmetros
        d = measure_diameters(contour)
        rows.append({"id": idx, **d})

        # desenhar contorno + texto
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)

        label = str(idx)

        # posição do ID (um pouco acima do objeto)
        y_id = max(y - 10, 20)

        cv2.putText(
            overlay,
            label,
            (x, y_id),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

    # CSV só com diâmetros
    df = pd.DataFrame(rows).sort_values("id")

    base = Path(image_path).stem
    csv_path = csv_dir / f"{base}_diameters_px.csv"
    ann_path = id_dir / f"{base}_identified.png"

    df.to_csv(csv_path, index=False)
    cv2.imwrite(str(ann_path), overlay)

    return {
        "csv": csv_path,
        "annotated": ann_path,
        "count": len(contours),
        "csv_folder": csv_dir,
        "identification_folder": id_dir
    }

# Selecionar arquivo 
def select_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Selecione a imagem",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return path if path else None

# Função principal
def main():
    image_path = select_file()
    if not image_path:
        print("Nenhuma imagem selecionada.")
        return

    MIN_AREA = 30
    FG_THRESH = 0.45

    try:
        result = analyze_image(
            image_path,
            results_dir="results",
            min_area=MIN_AREA,
            fg_thresh=FG_THRESH
        )

        msg = (
            f"Concluído!\n"
            f"Objetos detectados: {result['count']}\n\n"
            f"Imagem identificada: {result['annotated']}\n"
            f"CSV (diâmetros): {result['csv']}\n\n"
            f"Pastas:\n"
            f"- {result['identification_folder']}\n"
            f"- {result['csv_folder']}"
        )
        messagebox.showinfo("OK", msg)
        print(msg)
    except Exception as e:
        messagebox.showerror("Erro", str(e))
        raise

if __name__ == "__main__":
    main()