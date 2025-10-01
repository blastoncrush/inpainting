import cv2
import numpy as np

PATCH_RADIUS = 3

# Sélection de la zone à inpaint
drawing = False
points = []

def select_polygon(event, x, y, flags, param):
    """Callback pour dessiner le polygone de sélection"""
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN: # clic gauche
        points.append((x, y))
        if len(points) > 1:
            cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)
        cv2.circle(img_display, (x, y), 1, (255, 0, 0), -1)

def overlay_mask(img, mask):
    """Affiche la zone à inpaint (mask=255)"""
    overlay = img.copy()
    overlay[mask == 255] = (0, 0, 255)  # rouge pur
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

def resize_for_display(img, target_width=1000):
    """Redimensionne l'image pour l'affichage (car les images sont petites car basse résolution)"""
    h, w = img.shape[:2]
    if w != target_width:
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
        return cv2.resize(img, (target_width, target_height))
    return img

# Sélection de la zone à inpaint
img = cv2.imread("img/fontaine.jpg")
img_display = img.copy()
cv2.namedWindow("Selection")
cv2.setMouseCallback("Selection", select_polygon)

while True:
    cv2.imshow("Selection", img_display)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Entrée = valider
        break
    elif key == 27:  # Échap = reset
        points = []
        img_display = img.copy()

cv2.destroyAllWindows()

# Créer un masque de la zone à inpaint
mask = np.zeros(img.shape[:2], dtype=np.uint8)
if len(points) > 2:
    cv2.fillPoly(mask, [np.array(points)], 255)

# Fonctions mathématiques
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
    """Retourne un point aléatoire sur le bord de oméga"""
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    border_points = np.where(border > 0)
    if len(border_points[0]) > 0:
        idx = np.random.randint(0, len(border_points[0]))
        y, x = border_points[0][idx], border_points[1][idx]
        return y, x
    else:
        return None, None

def compute_priority(img_gray, mask, C, patch_radius=PATCH_RADIUS, alpha=255.0):
    """Applique l'algo pour calculer les priorités"""
    Ix, Iy = compute_gradients(img_gray)
    isophote = compute_isophote(Ix, Iy)
    N = compute_normals(mask)

    priorities = np.zeros_like(img_gray, dtype=np.float32)
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    h, w = img_gray.shape
    for y in range(h):
        for x in range(w):
            if border[y, x] > 0:
                # limites du patch (on fait attention aux bords...)
                x1, x2 = max(0, x-patch_radius), min(w, x+patch_radius+1)
                y1, y2 = max(0, y-patch_radius), min(h, y+patch_radius+1)

                patch_conf = C[y1:y2, x1:x2]
                C_p = np.mean(patch_conf)

                D_p = abs(np.dot(isophote[y, x], N[y, x])) / alpha

                priorities[y, x] = C_p * D_p

    return priorities

def find_best_patch(img, mask, target_patch, patch_radius=PATCH_RADIUS):
    """Calcule le patch source le plus proche du patch cible"""
    h, w = img.shape[:2]
    y, x = target_patch
    t_y1, t_y2 = max(0, y-patch_radius), min(h, y+patch_radius+1)
    t_x1, t_x2 = max(0, x-patch_radius), min(w, x+patch_radius+1)
    
    # Adapte à la taille du patch si on est proche des bords
    patch_h1, patch_h2 = y - t_y1, t_y2 - y - 1
    patch_w1, patch_w2 = x - t_x1, t_x2 - x - 1

    best_patch = None
    best_dist = float("inf")

    for yy in range(patch_h1, h-patch_h2):
        for xx in range(patch_w1, w-patch_w2):
            # pas de problème de bords ici
            s_y1, s_y2 = yy-patch_h1, yy+patch_h2+1
            s_x1, s_x2 = xx-patch_w1, xx+patch_w2+1

            # Vérifier que tout le patch source est en dehors de la zone à inpaint
            src_mask = mask[s_y1:s_y2, s_x1:s_x2]
            if np.any(src_mask != 0):  # Si une partie du patch source est à inpainter
                continue

            src_patch = img[s_y1:s_y2, s_x1:s_x2]
            tgt_patch = img[t_y1:t_y2, t_x1:t_x2]
            tgt_mask = mask[t_y1:t_y2, t_x1:t_x2]

            valid = (tgt_mask == 0)
            if np.sum(valid) == 0:
                continue

            diff = (src_patch[valid] - tgt_patch[valid])**2
            dist = np.sum(diff)

            if dist < best_dist:
                best_dist = dist
                best_patch = (yy, xx)

    return best_patch, (patch_h1, patch_h2), (patch_w1, patch_w2)

# Boucle d’inpainting

# Masque de confiance
C = np.ones_like(mask, dtype=np.float32)
C[mask == 255] = 0.0

iteration = 0
while np.any(mask == 255):
    iteration += 1
    # Conversion en CIELAB
    img_inpaint = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # ==== Etape 1: Calcul des priorités ====
    gray = cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2GRAY)
    priorities = compute_priority(gray, mask, C)
    
    # vérifier si les priorités sont toutes nulles
    if np.all(priorities == 0):
        # Passage en mode oignon (choix aléatoire)
        y, x = get_next_point(mask)
    else:
        y, x = np.unravel_index(np.argmax(priorities), priorities.shape)

    if y is None:  # fin de l'algo
        img = cv2.cvtColor(img_inpaint, cv2.COLOR_LAB2BGR)
        break

    # ==== Etape 2: Trouver le meilleur patch ====
    (yy, xx), (patch_h1, patch_h2), (patch_w1, patch_w2) = find_best_patch(img_inpaint, mask, (y, x))
    t_y1, t_y2 = y-patch_h1, y+patch_h2+1
    t_x1, t_x2 = x-patch_w1, x+patch_w2+1
    tgt_mask = mask[t_y1:t_y2, t_x1:t_x2]

    s_y1, s_y2 = yy-patch_h1, yy+patch_h2+1
    s_x1, s_x2 = xx-patch_w1, xx+patch_w2+1
    src_patch = img_inpaint[s_y1:s_y2, s_x1:s_x2]
    
    # ==== Etape 3: Copier le patch et mettre à jour les infos ====
    img_inpaint[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = src_patch[tgt_mask == 255]
    mask[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = 0
    C[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = C[y, x]

    # Update affichage
    display_vis = img_inpaint.copy()
    display_vis = cv2.cvtColor(display_vis, cv2.COLOR_LAB2BGR)
    
    # Affiche le patch copié en vert
    patch_mask = np.zeros_like(mask)
    s_y1, s_y2 = yy-PATCH_RADIUS, yy+PATCH_RADIUS+1
    s_x1, s_x2 = xx-PATCH_RADIUS, xx+PATCH_RADIUS+1
    patch_mask[s_y1:s_y2, s_x1:s_x2] = 255
    display_vis[patch_mask == 255] = [0, 255, 0]
    
    # Affiche la zone à inpaint en rouge
    display_vis[mask == 255] = [0, 0, 255]
    
    img_bgr = cv2.cvtColor(img_inpaint, cv2.COLOR_LAB2BGR)
    img = img_bgr.copy()
    
    display_vis = resize_for_display(cv2.addWeighted(img_bgr, 0.7, display_vis, 0.3, 0))
    cv2.imshow("Algo en cours", display_vis)
    cv2.waitKey(30)


display_result = resize_for_display(img)
cv2.imshow("Resultat final", display_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

