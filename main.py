import cv2
import numpy as np

PATCH_RADIUS = 3
COURONNE = 4

def overlay_mask(img, mask):
    """
    Affiche le masque en rouge sur l'image
    """
    overlay = img.copy()
    overlay[mask == 255] = (0, 0, 255)
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

def resize_for_display(img, target_width=1000):
    """
    Redimensionne l'image pour l'affichage
    """
    h, w = img.shape[:2]
    if w > target_width:
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
        return cv2.resize(img, (target_width, target_height))
    if w <= target_width:
        new_width = max(w, target_width)
        aspect_ratio = h / w
        target_height = int(new_width * aspect_ratio)
        return cv2.resize(img, (new_width, target_height))
    return img

def compute_gradients(gray):
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy

def compute_isophote(Ix, Iy):
    return np.dstack((-Iy, Ix))

def compute_normals(mask):
    mask_float = mask.astype(np.float32)/255.0
    Nx = cv2.Sobel(mask_float, cv2.CV_64F, 1, 0, ksize=3)
    Ny = cv2.Sobel(mask_float, cv2.CV_64F, 0, 1, ksize=3)
    N = np.dstack((Nx, Ny))
    norm = np.linalg.norm(N, axis=2, keepdims=True) + 1e-8
    return N / norm

def get_next_point(mask):
    # Fonction par défaut si compute_priority ne donne rien
    border = cv2.Canny(mask, 100, 200)
    border_points = np.where(border > 0)
    if len(border_points[0]) > 0:
        idx = np.random.randint(0, len(border_points[0]))
        y, x = border_points[0][idx], border_points[1][idx]
        return y, x
    return None, None

def compute_priority(img_gray, mask, C, patch_radius=PATCH_RADIUS, alpha=255.0):

    Ix, Iy = compute_gradients(img_gray)
    isophote = compute_isophote(Ix, Iy)
    N = compute_normals(mask)

    priorities = np.zeros_like(img_gray, dtype=np.float32)

    # Calcul de la bordure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    border_points = np.where(border > 0)
    h, w = img_gray.shape
    r = patch_radius

    for y, x in zip(*border_points):

        # Calcul de C
        x1, x2 = max(0, x - r), min(w, x + r + 1)
        y1, y2 = max(0, y - r), min(h, y + r + 1)
        patch_conf = C[y1:y2, x1:x2]

        # Empêcher les priorités fantômes : si presque aucun pixel n'est connu -> priorité = 0
        if np.sum(patch_conf > 0) < 4:
            continue

        C_p = np.mean(patch_conf)

        # Calcul de D

        # Utiliser la valeur normalisée du gradient pour le calcul D_raw
        D_raw = np.dot(isophote[y, x], N[y, x])
        if np.isnan(D_raw):
            continue

        D_p = abs(D_raw) / alpha # Alpha = 255.0 est la borne max du gradient

        # Calcul de P(p) = C(p) * D(p)
        priorities[y, x] = C_p * D_p

    return priorities


def find_best_patch(img_inpaint_lab, mask, C, target_patch, patch_radius=PATCH_RADIUS, C_threshold=0.9, w_L=2.0):

    h, w = img_inpaint_lab.shape[:2]
    y, x = target_patch
    r = patch_radius + COURONNE
    
    t_y1 = max(0, y-r); t_y2 = min(h, y+r+1)
    t_x1 = max(0, x-r); t_x2 = min(w, x+r+1)
    t_h = t_y2 - t_y1; t_w = t_x2 - t_x1

    # Patch CIELab
    tgt_patch_lab = img_inpaint_lab[t_y1:t_y2, t_x1:t_x2].astype(np.float32)
    tgt_mask = mask[t_y1:t_y2, t_x1:t_x2]
    
    valid = (tgt_mask == 0)
    n_valid = np.sum(valid)
    if n_valid == 0:
        return None

    best_patch = None
    best_dist = float("inf")

    y_min, y_max = r, h - r - 1
    x_min, x_max = r, w - r - 1

    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):

            if mask[yy, xx] != 0:
                continue

            s_y1 = yy - t_h//2; s_y2 = yy + (t_h - t_h//2)
            s_x1 = xx - t_w//2; s_x2 = xx + (t_w - t_w//2)

            src_conf_patch = C[s_y1:s_y2, s_x1:s_x2]

            if np.min(src_conf_patch) < C_threshold: # Seuil de confiance sur le patch source
                continue

            src_patch_lab = img_inpaint_lab[s_y1:s_y2, s_x1:s_x2].astype(np.float32)

            if src_patch_lab.shape != tgt_patch_lab.shape:
                continue

            src_valid = src_patch_lab[valid]
            tgt_valid = tgt_patch_lab[valid]
            
            # Calcul de la distance de texture pondérée
            diff_L = (src_valid[:, 0] - tgt_valid[:, 0])**2 * w_L
            diff_a = (src_valid[:, 1] - tgt_valid[:, 1])**2
            diff_b = (src_valid[:, 2] - tgt_valid[:, 2])**2

            total_diff = diff_L + diff_a + diff_b
            
            if n_valid == 0:
                continue
            
            # MSD sur 3 canaux (normalisation par n_valid * 3)
            dist = np.sum(total_diff) / (n_valid * 3)

            if dist < best_dist:
                best_dist = dist
                best_patch = (yy, xx)

    return best_patch

def run_inpainting(img, mask, C):

    img_inpaint = img.copy()
    r = PATCH_RADIUS

    # Conversion CIELAB
    img_inpaint_lab = cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2LAB)
    h, w = img_inpaint_lab.shape[:2]

    cv2.namedWindow("Inpainting en cours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inpainting en cours", 800, 600)

    while np.any(mask == 255):
        
        # Calcul des priorités
        gray = cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2GRAY)
        priorities = compute_priority(gray, mask, C)

        if np.max(priorities) > 0:
            y, x = np.unravel_index(np.argmax(priorities), priorities.shape)
        else:
            y, x = get_next_point(mask)
        if y is None:
            break
        
        # Calcul de confiance pour la mise à jour
        y1, y2 = max(0, y - r), min(C.shape[0], y + r + 1)
        x1, x2 = max(0, x - r), min(C.shape[1], x + r + 1)
        C_p_value = np.mean(C[y1:y2, x1:x2])

        # Recherche du meilleur patch source
        best = find_best_patch(img_inpaint_lab, mask, C, (y, x))
        if best is None:
            continue

        yy, xx = best

        # Copie du patch source dans le patch cible
        t_y1 = max(0, y-r); t_y2 = min(h, y+r+1)
        t_x1 = max(0, x-r); t_x2 = min(w, x+r+1)
        t_h = t_y2 - t_y1; t_w = t_x2 - t_x1
        s_y1 = yy - t_h//2; s_y2 = yy + (t_h - t_h//2)
        s_x1 = xx - t_w//2; s_x2 = xx + (t_w - t_w//2)

        tgt_mask = mask[t_y1:t_y2, t_x1:t_x2]
        src_patch_bgr = img_inpaint[s_y1:s_y2, s_x1:s_x2]
        img_inpaint[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = src_patch_bgr[tgt_mask == 255]

        src_patch_lab = img_inpaint_lab[s_y1:s_y2, s_x1:s_x2]
        img_inpaint_lab[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = src_patch_lab[tgt_mask == 255]

        # Mise à jour du masque et de la carte de confiance
        mask[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = 0
        C[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = C_p_value

        cv2.imshow("Inpainting en cours", resize_for_display(overlay_mask(img_inpaint, mask)))
        cv2.waitKey(1)

    cv2.destroyWindow("Inpainting en cours")
    return img_inpaint.copy()


def select_mask(img):
    points = []
    img_display = img.copy()

    def select_polygon(event, x, y, flags, param):
        nonlocal points, img_display # Chargement des variables de select_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) > 1:
                cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)
            cv2.circle(img_display, (x, y), 1, (255, 0, 0), -1)

    cv2.namedWindow("Selection")
    cv2.setMouseCallback("Selection", select_polygon)

    while True:
        cv2.imshow("Selection", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # Entrée = validation du masque
            break
        elif key == 27: # Échap = réinitialisation du masque
            points = []
            img_display = img.copy()

    cv2.destroyWindow("Selection")

    if len(points) < 3:
        return np.zeros(img.shape[:2], dtype=np.uint8) # Masque vide

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    return mask


if __name__ == "__main__":
    img_path = "img/Kanizsa_triangle.png"
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Impossible de charger l'image: {img_path}")

    img_working = img.copy()

    print("\n Tracez le masque (Entrée pour valider, Échap pour réinitialiser) \n")
    mask_initial = select_mask(img_working)

    if np.all(mask_initial == 0):
        print("Pas de zone masquée")
        cv2.destroyAllWindows()
        exit()
        
    # Boucle d'amélioration itérative sur la MÊME zone
    iteration_count = 0
    while True:
        iteration_count += 1
        print(f"Itération {iteration_count}")

        # Réinitialisation des masques
        mask_current = mask_initial.copy()
        C = np.ones_like(mask_initial, dtype=np.float32)
        C[mask_current == 255] = 0.0

        # Boucle d'inpainting
        img_result = run_inpainting(img_working.copy(), mask_current, C)

        # Mise à jour de l'image de travail
        img_working = img_result.copy()

        cv2.imshow("Resultat", resize_for_display(img_working))
        print("Appuyez sur 'q' pour quitter, ou sur une autre touche pour lancer une nouvelle itération.")

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

    cv2.imwrite("img/result.png", img_working)
    cv2.destroyAllWindows()
