import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# 1) Pré-processamento
# =========================
def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


# =========================
# 2) Segmentação (auto-inverte se necessário)
# =========================
def segment_auto(gray: np.ndarray) -> np.ndarray:
    """
    Otsu + auto-inversão:
    - se a maior parte ficar branca, provavelmente o fundo ficou "objeto"
    - então inverte
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = (bw > 0).mean()
    if white_ratio > 0.5:
        bw = cv2.bitwise_not(bw)

    # Morfologia LEVE pra remover ruído sem grudar partículas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    return bw


# =========================
# 3) Separar objetos colados (distance transform + watershed)
# =========================
def split_touching_objects(bw: np.ndarray, fg_thresh: float = 0.45) -> np.ndarray:
    """
    Separa partículas que encostam.
    fg_thresh:
      - diminua (0.35~0.42) se estiver pegando poucos objetos
      - aumente (0.50~0.60) se estiver quebrando demais
    """
    bw = (bw > 0).astype(np.uint8) * 255

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, fg_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(bw, k, iterations=2)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img3 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img3, markers)

    separated = np.zeros_like(bw)
    separated[markers > 1] = 255
    return separated


# =========================
# 4) Contornos + filtro
# =========================
def extract_particles(bw: np.ndarray, min_area: int = 50):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            good.append(c)
    return good


# =========================
# 5) Medidas: D1/D2/D3 (px)
# =========================
def feret_diameters(contour: np.ndarray) -> tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    return float(max(w, h)), float(min(w, h))


def measure_contour(contour: np.ndarray) -> dict:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    d1_eq = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0
    d2_feret_max, d3_feret_min = feret_diameters(contour)

    x, y, w, h = cv2.boundingRect(contour)

    circularity = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "circularity": circularity,
        "D1_eq_px": d1_eq,
        "D2_feret_max_px": d2_feret_max,
        "D3_feret_min_px": d3_feret_min,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
    }


# =========================
# 6) Analisar imagem + salvar outputs
# =========================
def analyze_image(image_path: str, out_dir: str = "out", min_area: int = 50, fg_thresh: float = 0.45):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    gray = preprocess(img)

    bw = segment_auto(gray)
    bw_sep = split_touching_objects(bw, fg_thresh=fg_thresh)

    contours = extract_particles(bw_sep, min_area=min_area)

    # Ordenar por posição (top->bottom, left->right) pra ID ficar mais “bonito”
    def contour_key(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y, x)
    contours = sorted(contours, key=contour_key)

    overlay = img.copy()
    rows = []

    for idx, contour in enumerate(contours, start=1):
        m = measure_contour(contour)
        m["id"] = idx
        rows.append(m)

        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

        x, y = m["bbox_x"], m["bbox_y"]

        label1 = f"Agregado {idx}"
        label2 = (
            f"D1: {m['D1_eq_px']:.1f}px | "
            f"D2: {m['D2_feret_max_px']:.1f}px | "
            f"D3: {m['D3_feret_min_px']:.1f}px"
        )

        # “Alt melhor”: duas linhas claras, sempre visíveis
        y1 = max(y - 28, 20)
        y2 = max(y - 10, 40)

        cv2.putText(overlay, label1, (x, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

        cv2.putText(overlay, label2, (x, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    df = pd.DataFrame(rows).sort_values("id")

    base = Path(image_path).stem
    csv_path = out_dir / f"{base}_dados_px.csv"
    bin_path = out_dir / f"{base}_binary.png"
    sep_path = out_dir / f"{base}_binary_separated.png"
    ann_path = out_dir / f"{base}_anotada.png"

    df.to_csv(csv_path, index=False)
    cv2.imwrite(str(bin_path), bw)
    cv2.imwrite(str(sep_path), bw_sep)
    cv2.imwrite(str(ann_path), overlay)

    return {
        "csv": csv_path,
        "binary": bin_path,
        "binary_separated": sep_path,
        "annotated": ann_path,
        "count": len(contours),
    }


# =========================
# 7) Selecionar arquivo + MAIN
# =========================
def select_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Selecione a imagem",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return path if path else None


def main():
    image_path = select_file()
    if not image_path:
        print("Nenhuma imagem selecionada.")
        return

    # Ajustes principais:
    MIN_AREA = 40 # se ainda estiver pegando só 1, tente 20 ou 30
    FG_THRESH = 0.45    # se estiver juntando, tente 0.40; se quebrando demais, 0.55

    try:
        result = analyze_image(image_path, out_dir="results", min_area=MIN_AREA, fg_thresh=FG_THRESH)
        msg = (
            f"Concluído!\n"
            f"Objetos detectados: {result['count']}\n\n"
            f"Anotada: {result['annotated']}\n"
            f"CSV: {result['csv']}\n"
            f"Binary: {result['binary']}\n"
            f"Binary separated: {result['binary_separated']}"
        )
        messagebox.showinfo("OK", msg)
        print(msg)
    except Exception as e:
        messagebox.showerror("Erro", str(e))
        raise


if __name__ == "__main__":
    main()