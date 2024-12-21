import cv2
import numpy as np

# Загрузка изображений
img_object = cv2.imread(r'C:\Users\Asus\Downloads\kot_kr\kotiki_poisk\object.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread(r'C:\Users\Asus\Downloads\kot_kr\kotiki_poisk\scene.jpg', cv2.IMREAD_GRAYSCALE)

if img_object is None or img_scene is None:
    raise ValueError("Не удалось загрузить изображения.")

# Инициализация детектора SIFT
sift = cv2.SIFT_create()

# Поиск ключевых точек и вычисление дескрипторов
keypoints_obj, descriptors_obj = sift.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = sift.detectAndCompute(img_scene, None)

# Инициализация матчера
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Поиск соответствий
matches = bf.knnMatch(descriptors_obj, descriptors_scene, k=2)

# Отбор лучших совпадений с использованием критерия Лоу
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Минимальное количество совпадений для надежного определения объекта
MIN_MATCH_COUNT = 10

if len(good_matches) > MIN_MATCH_COUNT:
    # Координаты совпадающих точек
    src_pts = np.float32([keypoints_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Вычисление матрицы гомографии
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Определение углов объекта
    h, w = img_object.shape
    obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(obj_corners, M)

    # Рисование контура объекта на изображении сцены
    img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_scene_color, [np.int32(scene_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Отображение результата
    cv2.imshow('Detected Object', img_scene_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Недостаточно совпадений для определения объекта.")
