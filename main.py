import cv2
import numpy as np

PATCH_RADIUS = 1

# Sélection de la zone à inpaint
drawing = False
points = []

def select_polygon(event, x, y, flags, param):
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
    h, w = img.shape[:2]
    if w != target_width:
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
        return cv2.resize(img, (target_width, target_height))
    return img

# Charger l'image
img = cv2.imread("img/inpainting.png")
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

# Fonctions utilitaires
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
    # Retourne un point aléatoire sur le bord de oméga
    border = cv2.Canny(mask, 100, 200)
    border_points = np.where(border > 0)
    if len(border_points[0]) > 0:
        idx = np.random.randint(0, len(border_points[0]))
        y, x = border_points[0][idx], border_points[1][idx]
        return y, x
    else:
        return None, None

def compute_priority(img_gray, mask, C, patch_radius=PATCH_RADIUS, alpha=255.0):
    Ix, Iy = compute_gradients(img_gray)
    isophote = compute_isophote(Ix, Iy)
    N = compute_normals(mask)

    priorities = np.zeros_like(img_gray, dtype=np.float32)
    border = cv2.Canny(mask, 100, 200)

    h, w = img_gray.shape
    for y in range(h):
        for x in range(w):
            if border[y, x] > 0:
                x1, x2 = max(0, x-patch_radius), min(w, x+patch_radius+1)
                y1, y2 = max(0, y-patch_radius), min(h, y+patch_radius+1)

                patch_conf = C[y1:y2, x1:x2]
                C_p = np.mean(patch_conf)

                D_p = abs(np.dot(isophote[y, x], N[y, x])) / alpha

                priorities[y, x] = C_p * D_p

    return priorities

def find_best_patch(img, mask, target_patch, patch_radius=PATCH_RADIUS):
    h, w = img.shape[:2]
    y, x = target_patch
    t_y1, t_y2 = y-patch_radius, y+patch_radius+1
    t_x1, t_x2 = x-patch_radius, x+patch_radius+1

    best_patch = None
    best_dist = float("inf")

    for yy in range(patch_radius, h-patch_radius):
        for xx in range(patch_radius, w-patch_radius):
            if mask[yy, xx] != 0:
                continue

            s_y1, s_y2 = yy-patch_radius, yy+patch_radius+1
            s_x1, s_x2 = xx-patch_radius, xx+patch_radius+1

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

    return best_patch

# Boucle d’inpainting

# Masque de confiance
C = np.ones_like(mask, dtype=np.float32)
C[mask == 255] = 0.0

img_inpaint = img.copy()

error_count = 0
iteration = 0
while np.any(mask == 255):
    iteration += 1
    gray = cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2GRAY)
    priorities = compute_priority(gray, mask, C)
    # vérifier si les priorités sont toutes nulles
    if np.all(priorities == 0):
        print("Passage en monde oignon")
        
        y, x = get_next_point(mask)
    else:
        y, x = np.unravel_index(np.argmax(priorities), priorities.shape)

    if y is None:  # fin de l'algo
        break

    try:
        yy, xx = find_best_patch(img_inpaint, mask, (y, x))
    except Exception as e:
        print("Aucun patch trouvé: ", e)
        error_count += 1
        mask[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = 0
        
        # Affiche en rouge le patch qui a posé problème
        img_inpaint[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = [0, 0, 255]
        C[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = C[y, x]
        if error_count > 100:
            break
        continue

    t_y1, t_y2 = y-PATCH_RADIUS, y+PATCH_RADIUS+1
    t_x1, t_x2 = x-PATCH_RADIUS, x+PATCH_RADIUS+1
    tgt_mask = mask[t_y1:t_y2, t_x1:t_x2]        
    
    s_y1, s_y2 = yy-PATCH_RADIUS, yy+PATCH_RADIUS+1
    s_x1, s_x2 = xx-PATCH_RADIUS, xx+PATCH_RADIUS+1
    src_patch = img_inpaint[s_y1:s_y2, s_x1:s_x2]
    
    img_inpaint[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = src_patch[tgt_mask == 255]
    mask[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = 0
    C[t_y1:t_y2, t_x1:t_x2][tgt_mask == 255] = C[y, x]

    # Update affichage
    vis = overlay_mask(img_inpaint, mask)
    display_vis = resize_for_display(vis)
    cv2.imshow("Algo en cours", display_vis)
    cv2.waitKey(30)

display_result = resize_for_display(img_inpaint)
print(error_count)
cv2.imshow("Resultat final", display_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

