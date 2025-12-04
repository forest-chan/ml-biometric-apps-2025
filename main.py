import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

def preprocess_image(image_path, target_size=(100, 100)):
    """
    Предобработка изображения для биометрической идентификации.
    
    Параметры:
    -----------
    image_path : str
        Путь к изображению
    target_size : tuple
        Размер выходного изображения (ширина, высота)
    
    Возвращает:
    -----------
    numpy.ndarray или None
        Вектор признаков (одномерный массив) или None, если лицо не найдено
    """
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None
    
    # Конвертация в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Обнаружение лица с помощью каскада Хаара
    # Загружаем предобученный классификатор
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Детектируем лица
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print(f"Предупреждение: лицо не обнаружено на {image_path}")
        return None
    
    # Берем первое обнаруженное лицо (самое крупное)
    if len(faces) > 1:
        # Сортируем по площади и берем самое большое
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    
    x, y, w, h = faces[0]
    
    # 4. Обрезка изображения (только лицо)
    face_roi = gray[y:y+h, x:x+w]
    
    # 5. Изменение размера до target_size
    face_resized = cv2.resize(face_roi, target_size)
    
    # 6. Гистограммная нормализация (выравнивание гистограммы)
    face_normalized = cv2.equalizeHist(face_resized)
    
    # 7. Преобразование в вектор
    feature_vector = face_normalized.reshape(-1)
    
    return feature_vector


def visualize_preprocessing(image_path, target_size=(100, 100)):
    """
    Визуализация процесса предобработки изображения.
    
    Параметры:
    -----------
    image_path : str
        Путь к изображению
    target_size : tuple
        Размер выходного изображения
    """
    
    # Загрузка исходного изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return
    
    # Конвертация в RGB для matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Обнаружение лица
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"Лицо не найдено на {image_path}")
        return
    
    # Берем первое лицо
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Обрезка и обработка
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, target_size)
    face_normalized = cv2.equalizeHist(face_resized)
    
    # Визуализация
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Исходное изображение
    axes[0].imshow(image_rgb)
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    # Обнаруженное лицо
    image_with_box = image_rgb.copy()
    cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    axes[1].imshow(image_with_box)
    axes[1].set_title('Обнаруженное лицо')
    axes[1].axis('off')
    
    # Обрезанное и масштабированное лицо
    axes[2].imshow(face_resized, cmap='gray')
    axes[2].set_title(f'Обрезка и ресайз {target_size}')
    axes[2].axis('off')
    
    # Нормализованное лицо
    axes[3].imshow(face_normalized, cmap='gray')
    axes[3].set_title('После гистограммной нормализации')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_feature_matrix(user_vectors, imposter_vectors):
    """
    Создание матрицы объекты-признаки и меток классов.
    
    Параметры:
    -----------
    user_vectors : numpy.ndarray
        Векторы признаков пользователя (N_user x 10000)
    imposter_vectors : numpy.ndarray
        Векторы признаков злоумышленников (N_imposter x 10000)
    
    Возвращает:
    -----------
    tuple : (X, y)
        X - матрица признаков (N_total x 10000)
        y - метки классов (N_total,), где 1 = свой, 0 = чужой
    """
    
    # Объединяем все векторы в одну матрицу
    X = np.vstack([user_vectors, imposter_vectors])
    
    # Создаём метки: 1 для своих, 0 для чужих
    y = np.array([1] * len(user_vectors) + [0] * len(imposter_vectors))
    
    print(f"\n=== Создана матрица признаков ===")
    print(f"Форма матрицы X: {X.shape}")
    print(f"Форма меток y: {y.shape}")
    print(f"Метки: {y}")
    
    return X, y


def apply_pca(X, variance_threshold=0.95):
    """
    Применение PCA для снижения размерности.
    
    Параметры:
    -----------
    X : numpy.ndarray
        Матрица признаков (N x D)
    variance_threshold : float
        Доля объясняемой дисперсии (по умолчанию 0.95 = 95%)
    
    Возвращает:
    -----------
    tuple : (X_pca, pca_model)
        X_pca - преобразованная матрица (N x n_components)
        pca_model - обученная модель PCA
    """
    
    # Создаём и обучаем PCA
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X)
    
    print(f"\n=== Применение PCA ===")
    print(f"Исходная размерность: {X.shape[1]}")
    print(f"Размерность после PCA: {X_pca.shape[1]}")
    print(f"Объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    print(f"Число главных компонент: {pca.n_components_}")
    
    return X_pca, pca


def visualize_pca_variance(pca_model):
    """
    Визуализация объяснённой дисперсии главными компонентами.
    
    Параметры:
    -----------
    pca_model : PCA
        Обученная модель PCA
    """
    
    cumsum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График объяснённой дисперсии по компонентам
    ax1.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
            pca_model.explained_variance_ratio_)
    ax1.set_xlabel('Номер главной компоненты')
    ax1.set_ylabel('Объяснённая дисперсия')
    ax1.set_title('Дисперсия по каждой компоненте')
    ax1.grid(True, alpha=0.3)
    
    # График накопленной дисперсии
    ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, marker='o')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% порог')
    ax2.set_xlabel('Число компонент')
    ax2.set_ylabel('Накопленная объяснённая дисперсия')
    ax2.set_title('Накопленная дисперсия')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def train_and_evaluate_biometric_system(X_pca, y):
    """
    Обучение и оценка биометрической системы с использованием Leave-One-Out.
    
    Параметры:
    -----------
    X_pca : numpy.ndarray
        Матрица признаков после PCA (N x n_components)
    y : numpy.ndarray
        Метки классов (N,)
    
    Возвращает:
    -----------
    dict : словарь с результатами оценки
    """
    
    loo = LeaveOneOut()
    
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    print(f"\n=== Обучение и тестирование (Leave-One-Out) ===")
    print(f"Количество итераций: {X_pca.shape[0]}")
    
    # Leave-One-Out кросс-валидация
    for i, (train_index, test_index) in enumerate(loo.split(X_pca)):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Обучаем логистическую регрессию с L2-регуляризацией
        model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Предсказание класса
        y_pred = model.predict(X_test)
        
        # Предсказание вероятности принадлежности к классу "1" (свой)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])
        y_proba_all.append(y_proba[0])
        
        print(f"Итерация {i+1}: истинный={y_test[0]}, предсказанный={y_pred[0]}, вероятность={y_proba[0]:.4f}")
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all)
    
    # Оценка качества
    accuracy = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    # Вычисление FAR и FRR
    # cm[0,0] = True Negatives (TN) - правильно отклонённые чужие
    # cm[0,1] = False Positives (FP) - неправильно принятые чужие (FAR)
    # cm[1,0] = False Negatives (FN) - неправильно отклонённые свои (FRR)
    # cm[1,1] = True Positives (TP) - правильно принятые свои
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    # FAR (False Acceptance Rate) - доля чужих, которых система приняла
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # FRR (False Rejection Rate) - доля своих, которых система отклонила
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'far': far,
        'frr': frr,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_proba': y_proba_all,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    print(f"\n=== Результаты оценки ===")
    print(f"Точность (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nМатрица ошибок:")
    print(f"              Предсказано 0  Предсказано 1")
    print(f"Истинно 0 (чужие)    {tn}              {fp}")
    print(f"Истинно 1 (свои)     {fn}              {tp}")
    print(f"\nFAR (False Acceptance Rate): {far:.4f} ({far*100:.2f}%)")
    print(f"   - Доля чужих, которых система ошибочно пропустила")
    print(f"FRR (False Rejection Rate): {frr:.4f} ({frr*100:.2f}%)")
    print(f"   - Доля своих, которых система ошибочно отклонила")
    
    return results


def visualize_results(results, y):
    """
    Визуализация результатов классификации.
    
    Параметры:
    -----------
    results : dict
        Результаты оценки от train_and_evaluate_biometric_system
    y : numpy.ndarray
        Исходные метки классов
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Матрица ошибок
    cm = results['confusion_matrix']
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Чужой (0)', 'Свой (1)'])
    ax1.set_yticklabels(['Чужой (0)', 'Свой (1)'])
    ax1.set_xlabel('Предсказанный класс')
    ax1.set_ylabel('Истинный класс')
    ax1.set_title('Матрица ошибок')
    
    # Добавляем значения в ячейки
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=20)
    
    plt.colorbar(im, ax=ax1)
    
    # График вероятностей
    n_users = np.sum(y == 1)
    n_imposters = np.sum(y == 0)
    
    user_probas = results['y_proba'][:n_users]
    imposter_probas = results['y_proba'][n_users:]
    
    ax2.scatter(range(n_users), user_probas, c='green', label='Свои', s=100, alpha=0.7)
    ax2.scatter(range(n_users, n_users + n_imposters), imposter_probas, 
                c='red', label='Чужие', s=100, alpha=0.7)
    ax2.axhline(y=0.5, color='black', linestyle='--', label='Порог (0.5)')
    ax2.set_xlabel('Номер образца')
    ax2.set_ylabel('Вероятность класса "Свой"')
    ax2.set_title('Предсказанные вероятности')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()


def process_dataset(user_folder, imposter_folder, target_size=(100, 100)):
    """
    Обработка всего датасета (пользователь + злоумышленники).
    
    Параметры:
    -----------
    user_folder : str
        Путь к папке с фотографиями пользователя
    imposter_folder : str
        Путь к папке с фотографиями злоумышленников
    target_size : tuple
        Размер выходного изображения
    
    Возвращает:
    -----------
    tuple : (user_vectors, imposter_vectors, user_files, imposter_files)
        Векторы признаков и имена файлов
    """
    
    user_vectors = []
    imposter_vectors = []
    user_files = []
    imposter_files = []
    
    # Обработка фото пользователя
    print("Обработка фотографий пользователя...")
    for filename in sorted(os.listdir(user_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(user_folder, filename)
            vector = preprocess_image(filepath, target_size)
            if vector is not None:
                user_vectors.append(vector)
                user_files.append(filename)
                print(f"✓ {filename}")
    
    # Обработка фото злоумышленников
    print("\nОбработка фотографий злоумышленников...")
    for filename in sorted(os.listdir(imposter_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(imposter_folder, filename)
            vector = preprocess_image(filepath, target_size)
            if vector is not None:
                imposter_vectors.append(vector)
                imposter_files.append(filename)
                print(f"✓ {filename}")
    
    return (np.array(user_vectors), np.array(imposter_vectors), 
            user_files, imposter_files)


def visualize_all_faces(user_folder, imposter_folder, target_size=(100, 100)):
    """
    Визуализация всех обработанных лиц из обеих папок.
    
    Параметры:
    -----------
    user_folder : str
        Путь к папке с фотографиями пользователя
    imposter_folder : str
        Путь к папке с фотографиями злоумышленников
    target_size : tuple
        Размер выходного изображения
    """
    
    # Получаем все изображения
    user_images = []
    user_names = []
    
    for filename in sorted(os.listdir(user_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(user_folder, filename)
            vector = preprocess_image(filepath, target_size)
            if vector is not None:
                # Преобразуем вектор обратно в изображение для визуализации
                image = vector.reshape(target_size)
                user_images.append(image)
                user_names.append(filename)
    
    imposter_images = []
    imposter_names = []
    
    for filename in sorted(os.listdir(imposter_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(imposter_folder, filename)
            vector = preprocess_image(filepath, target_size)
            if vector is not None:
                image = vector.reshape(target_size)
                imposter_images.append(image)
                imposter_names.append(filename)
    
    # Визуализация
    total_images = len(user_images) + len(imposter_images)
    fig, axes = plt.subplots(2, max(len(user_images), len(imposter_images)), 
                             figsize=(15, 6))
    
    # Если всего один столбец, преобразуем axes
    if total_images <= 2:
        axes = axes.reshape(2, -1)
    
    # Отображаем фото пользователя
    for i, (img, name) in enumerate(zip(user_images, user_names)):
        if i < axes.shape[1]:
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'User: {name}', fontsize=8)
            axes[0, i].axis('off')
    
    # Скрываем пустые ячейки в первом ряду
    for i in range(len(user_images), axes.shape[1]):
        axes[0, i].axis('off')
    
    # Отображаем фото злоумышленников
    for i, (img, name) in enumerate(zip(imposter_images, imposter_names)):
        if i < axes.shape[1]:
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Imposter: {name}', fontsize=8)
            axes[1, i].axis('off')
    
    # Скрываем пустые ячейки во втором ряду
    for i in range(len(imposter_images), axes.shape[1]):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    USER_FOLDER = "./photos/user"      # Папка с селфи
    IMPOSTER_FOLDER = "./photos/imposter"  # Папка с фото других людей
    
    print("="*70)
    print("БИОМЕТРИЧЕСКАЯ СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ")
    print("="*70)
    
    # Шаг 1: Обработка всего датасета
    print("\n[ШАГ 1] Предобработка изображений")
    print("-"*70)
    user_vectors, imposter_vectors, user_files, imposter_files = process_dataset(
        USER_FOLDER, IMPOSTER_FOLDER
    )
    
    print(f"\n✓ Обработано {len(user_vectors)} фото пользователя")
    print(f"✓ Обработано {len(imposter_vectors)} фото злоумышленников")
    print(f"✓ Размер вектора признаков: {user_vectors.shape[1] if len(user_vectors) > 0 else 0}")
    
    # Визуализация всех обработанных лиц
    print("\n[ВИЗУАЛИЗАЦИЯ] Обработанные лица")
    print("-"*70)
    visualize_all_faces(USER_FOLDER, IMPOSTER_FOLDER)
    
    # Шаг 2: Создание матрицы признаков
    print("\n[ШАГ 2] Создание матрицы объекты-признаки")
    print("-"*70)
    X, y = create_feature_matrix(user_vectors, imposter_vectors)
    
    # Шаг 3: Применение PCA
    print("\n[ШАГ 3] Снижение размерности с помощью PCA")
    print("-"*70)
    X_pca, pca_model = apply_pca(X, variance_threshold=0.95)
    
    # Визуализация PCA
    print("\n[ВИЗУАЛИЗАЦИЯ] Дисперсия главных компонент")
    print("-"*70)
    visualize_pca_variance(pca_model)
    
    # Шаг 4: Обучение и оценка системы
    print("\n[ШАГ 4] Обучение и оценка биометрической системы")
    print("-"*70)
    results = train_and_evaluate_biometric_system(X_pca, y)
    
    # Визуализация результатов
    print("\n[ВИЗУАЛИЗАЦИЯ] Результаты классификации")
    print("-"*70)
    visualize_results(results, y)
    
    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)
