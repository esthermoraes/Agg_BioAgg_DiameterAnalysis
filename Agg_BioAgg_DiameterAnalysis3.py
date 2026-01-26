import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# IO robusto (Windows + acentos / OneDrive)
# =========================
def imread_unicode(path: str):
    p = Path(path)
    if not p.exists():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# =========================
# Segmentação: pega o agregado (marrom) e ignora furos internos
# =========================
def segment_brown_object(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segmenta o agregado (marrom/terroso) em fundo claro usando HSV.
    Retorna máscara binária 0/255.
    Ajuste os limites HSV se necessário.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Faixa típica de tons "marrom/terroso"
    # Ajuste fino aqui se precisar:
    # - Se estiver faltando pedaço do agregado: diminua S_min/V_min (30 -> 15)
    # - Se estiver pegando fundo: aumente S_min (30 -> 50)
    lower = np.array([5, 30, 30], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Limpa ruído sem grudar demais
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Fecha falhas pequenas na borda
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    return mask


def fill_holes(mask_255: np.ndarray) -> np.ndarray:
    """
    Preenche buracos internos (furos) da máscara:
    faz o contorno "abraçar" o agregado, ignorando pontinhos pretos internos.
    """
    mask = (mask_255 > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # preenche o fundo a partir da borda (0,0)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)

    # invertendo, sobra só os "buracos" internos
    flood_inv = cv2.bitwise_not(flood)

    # une máscara original com buracos preenchidos
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def get_external_contours(mask_255: np.ndarray, min_area: int = 200):
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


# =========================
# Medidas: D1/D2/D3 (px)
# =========================
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


# =========================
# Analisar imagem + salvar: (1) annotated com IDs + (2) CSV diâmetros
# =========================
def analyze_image(
    image_path: str,
    results_dir: str = "results",
    min_area: int = 200,
    save_debug: bool = False
):
    results_dir = Path(results_dir)
    csv_dir = results_dir / "csv"
    id_dir = results_dir / "identification"
    debug_dir = results_dir / "debug"

    csv_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    # 1) Segmenta marrom
    bw = segment_brown_object(img)

    # 2) Preenche furos internos (para contorno externo “abraçar” o agregado)
    bw_filled = fill_holes(bw)

    # 3) Contorno EXTERNO apenas
    contours = get_external_contours(bw_filled, min_area=min_area)

    # Ordenar por posição (top->bottom, left->right) para ID consistente
    def contour_key(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y, x)
    contours = sorted(contours, key=contour_key)

    overlay = img.copy()
    rows = []

    for idx, contour in enumerate(contours, start=1):
        d = measure_diameters(contour)
        rows.append({"id": idx, **d})

        # desenha contorno + APENAS o ID
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        y_id = max(y - 10, 20)

        cv2.putText(
            overlay,
            str(idx),
            (x, y_id),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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

    if save_debug:
        cv2.imwrite(str(debug_dir / f"{base}_01_mask.png"), bw)
        cv2.imwrite(str(debug_dir / f"{base}_02_mask_filled.png"), bw_filled)

    return {
        "csv": csv_path,
        "annotated": ann_path,
        "count": len(contours),
        "csv_folder": csv_dir,
        "identification_folder": id_dir
    }


# =========================
# Selecionar arquivo
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


# =========================
# MAIN
# =========================
def main():
    image_path = select_file()
    if not image_path:
        print("Nenhuma imagem selecionada.")
        return

    # Ajuste esse min_area conforme o tamanho típico dos seus agregados na imagem:
    # - Se estiver detectando sujeirinhas/pontos: AUMENTE (ex: 300, 500)
    # - Se estiver perdendo agregados pequenos: DIMINUA (ex: 80, 120)
    MIN_AREA = 200

    try:
        result = analyze_image(
            image_path,
            results_dir="results",
            min_area=MIN_AREA,
            save_debug=False  # mude para True se quiser salvar máscaras para checar
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