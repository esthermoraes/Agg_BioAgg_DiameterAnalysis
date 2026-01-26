import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# IO robusto (Windows + acentos)
# =========================
def imread_unicode(path: str):
    """Lê imagem mesmo com acentos/apóstrofos no caminho (Windows/OneDrive)."""
    p = Path(path)
    if not p.exists():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# =========================
# 1) Pré-processamento
# =========================
def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


# =========================
# 2) Segmentação (Otsu ou Adaptive) + auto-inversão
# =========================
def segment(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    method:
      - "otsu"     -> bom para iluminação uniforme
      - "adaptive" -> melhor para foto com iluminação desigual
    """
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        # blockSize deve ser ímpar (ex: 31, 51, 71)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            2
        )
    else:
        raise ValueError("method deve ser 'otsu' ou 'adaptive'")

    # auto-inversão (se fundo virar objeto)
    white_ratio = (bw > 0).mean()
    if white_ratio > 0.5:
        bw = cv2.bitwise_not(bw)

    # morfologia LEVE pra tirar ruído sem grudar partículas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    return bw


# =========================
# 3) Separar objetos colados (distance transform + watershed)
# =========================
def split_touching_objects(bw: np.ndarray, fg_thresh: float = 0.45, bg_dilate_iter: int = 2) -> np.ndarray:
    """
    fg_thresh:
      - diminua (0.35~0.42) se estiver juntando e pegando poucos objetos
      - aumente (0.50~0.60) se estiver quebrando demais
    bg_dilate_iter:
      - normalmente 1 ou 2
    """
    bw = (bw > 0).astype(np.uint8) * 255

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, fg_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(bw, k, iterations=bg_dilate_iter)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
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
def extract_particles(bw: np.ndarray, min_area: int = 30):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


# =========================
# 5) Diâmetros D1/D2/D3 (px)
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
# 6) Processar (salva SOMENTE: imagem IDs + CSV diâmetros)
# =========================
def analyze_image(
    image_path: str,
    results_dir: str = "results",
    min_area: int = 30,
    fg_thresh: float = 0.45,
    method: str = "adaptive",
    bg_dilate_iter: int = 2,
    save_debug: bool = True
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

    gray = preprocess(img)
    bw = segment(gray, method=method)
    bw_sep = split_touching_objects(bw, fg_thresh=fg_thresh, bg_dilate_iter=bg_dilate_iter)

    contours = extract_particles(bw_sep, min_area=min_area)

    # ordenar por posição
    def contour_key(c):
        x, y, w, h = cv2.boundingRect(c)
        return (y, x)
    contours = sorted(contours, key=contour_key)

    overlay = img.copy()
    rows = []

    for idx, contour in enumerate(contours, start=1):
        d = measure_diameters(contour)
        rows.append({"id": idx, **d})

        # contorno + APENAS ID
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        y_id = max(y - 10, 20)
        cv2.putText(
            overlay, str(idx), (x, y_id),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
        )

    df = pd.DataFrame(rows).sort_values("id")

    base = Path(image_path).stem
    csv_path = csv_dir / f"{base}_diameters_px.csv"
    ann_path = id_dir / f"{base}_identified.png"

    df.to_csv(csv_path, index=False)
    cv2.imwrite(str(ann_path), overlay)

    if save_debug:
        cv2.imwrite(str(debug_dir / f"{base}_02_bw.png"), bw)
        cv2.imwrite(str(debug_dir / f"{base}_03_bw_sep.png"), bw_sep)

    return {
        "csv": csv_path,
        "annotated": ann_path,
        "count": len(contours),
        "method": method,
        "min_area": min_area,
        "fg_thresh": fg_thresh,
        "bg_dilate_iter": bg_dilate_iter
    }


# =========================
# 7) Calibração interativa (sliders)
# =========================
def calibrate_and_process(image_path: str, results_dir: str = "results"):
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    gray = preprocess(img)

    cv2.namedWindow("CALIBRATION", cv2.WINDOW_NORMAL)

    def nothing(x): pass

    # sliders
    cv2.createTrackbar("min_area", "CALIBRATION", 30, 800, nothing)
    cv2.createTrackbar("fg_thresh_x100", "CALIBRATION", 45, 90, nothing)  # 0.45
    cv2.createTrackbar("bg_dilate_iter", "CALIBRATION", 2, 5, nothing)
    cv2.createTrackbar("method (0=otsu,1=adapt)", "CALIBRATION", 1, 1, nothing)

    last_preview = None

    while True:
        min_area = max(1, cv2.getTrackbarPos("min_area", "CALIBRATION"))
        fg = cv2.getTrackbarPos("fg_thresh_x100", "CALIBRATION") / 100.0
        bg_iter = max(1, cv2.getTrackbarPos("bg_dilate_iter", "CALIBRATION"))
        method_flag = cv2.getTrackbarPos("method (0=otsu,1=adapt)", "CALIBRATION")
        method = "adaptive" if method_flag == 1 else "otsu"

        bw = segment(gray, method=method)
        bw_sep = split_touching_objects(bw, fg_thresh=fg, bg_dilate_iter=bg_iter)
        contours = extract_particles(bw_sep, min_area=min_area)

        # preview: máscara separada com info
        vis = cv2.cvtColor(bw_sep, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            vis,
            f"method={method}  fg={fg:.2f}  min_area={min_area}  bg_iter={bg_iter}  count={len(contours)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )
        cv2.putText(
            vis,
            "Ajuste sliders. Pressione S para processar e salvar. Pressione Q para sair.",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        last_preview = (min_area, fg, method, bg_iter)
        cv2.imshow("CALIBRATION", vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            return None

        if key in (ord("s"), ord("S")):
            cv2.destroyAllWindows()
            # processa com os parâmetros escolhidos
            return analyze_image(
                image_path=image_path,
                results_dir=results_dir,
                min_area=min_area,
                fg_thresh=fg,
                method=method,
                bg_dilate_iter=bg_iter,
                save_debug=True
            )


# =========================
# 8) Selecionar arquivo + MAIN
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

    try:
        result = calibrate_and_process(image_path, results_dir="results")
        if result is None:
            print("Calibração cancelada.")
            return

        msg = (
            f"Concluído!\n"
            f"Objetos detectados: {result['count']}\n"
            f"Parâmetros usados: method={result['method']}, fg={result['fg_thresh']}, min_area={result['min_area']}, bg_iter={result['bg_dilate_iter']}\n\n"
            f"Imagem identificada: {result['annotated']}\n"
            f"CSV (diâmetros): {result['csv']}\n"
        )
        messagebox.showinfo("OK", msg)
        print(msg)

    except Exception as e:
        messagebox.showerror("Erro", str(e))
        raise


if __name__ == "__main__":
    main()