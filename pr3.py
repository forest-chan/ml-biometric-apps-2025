import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

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
    
    # 1. Загрузка изображения
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None
    
    # 2. Конвертация в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Обнаружение лица с помощью каскада Хаара
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
    ЛИНЕЙНАЯ МОДЕЛЬ (Логистическая регрессия).
    
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
    
    print(f"\n=== Обучение и тестирование ЛИНЕЙНОЙ МОДЕЛИ (Leave-One-Out) ===")
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
    results = calculate_metrics(y_true_all, y_pred_all, y_proba_all)
    results['model_name'] = 'Логистическая регрессия (линейная)'
    
    print_results(results)
    
    return results


def train_and_evaluate_ensemble(X, y, use_pca_features=False, model_type='random_forest'):
    """
    Обучение и оценка ансамблевой модели с использованием Leave-One-Out.
    
    Параметры:
    -----------
    X : numpy.ndarray
        Матрица признаков (N x D) - исходные пиксели или PCA-признаки
    y : numpy.ndarray
        Метки классов (N,)
    use_pca_features : bool
        Использовать ли PCA-признаки вместо исходных пикселей
    model_type : str
        Тип модели: 'random_forest'
    
    Возвращает:
    -----------
    dict : словарь с результатами оценки
    """
    
    loo = LeaveOneOut()
    
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    feature_type = "PCA-признаки" if use_pca_features else "исходные пиксели"
    model_name = "Random Forest" if model_type == 'random_forest' else "Undefined"
    
    print(f"\n=== Обучение и тестирование АНСАМБЛЯ {model_name} (Leave-One-Out) ===")
    print(f"Тип признаков: {feature_type}")
    print(f"Количество итераций: {X.shape[0]}")
    
    # Leave-One-Out кросс-валидация
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Обучаем модель
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
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
    results = calculate_metrics(y_true_all, y_pred_all, y_proba_all)
    results['model_name'] = f'{model_name} ({feature_type})'
    
    print_results(results)
    
    return results


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Вычисление всех метрик качества: Accuracy, FAR, FRR, EER, Precision, Recall, AUC-ROC, AUC-PR.
    
    Параметры:
    -----------
    y_true : numpy.ndarray
        Истинные метки
    y_pred : numpy.ndarray
        Предсказанные метки
    y_proba : numpy.ndarray
        Вероятности класса "1" (клиент)
    
    Возвращает:
    -----------
    dict : словарь с метриками
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Извлекаем компоненты матрицы ошибок
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    # FAR (False Acceptance Rate) - доля злоумышленников, которых система приняла
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # FRR (False Rejection Rate) - доля клиентов, которых система отклонила
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Precision и Recall для класса "клиент" (класс 1)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Вычисление EER (Equal Error Rate)
    eer = calculate_eer(y_true, y_proba)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # PR-AUC (Precision-Recall AUC)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    # Альтернативный расчёт PR-AUC через average_precision_score
    avg_precision = average_precision_score(y_true, y_proba)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'far': far,
        'frr': frr,
        'eer': eer,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'avg_precision': avg_precision,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def calculate_eer(y_true, y_proba):
    """
    Вычисление EER (Equal Error Rate).
    
    Параметры:
    -----------
    y_true : numpy.ndarray
        Истинные метки
    y_proba : numpy.ndarray
        Вероятности класса "1"
    
    Возвращает:
    -----------
    float : значение EER
    """
    
    # Вычисляем FPR и FNR для разных порогов
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    fnr = 1 - tpr
    
    # EER - точка, где FPR ≈ FNR (или FAR ≈ FRR)
    # Находим индекс минимальной разности между FPR и FNR
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    
    return eer


def print_results(results):
    """
    Вывод результатов оценки модели.
    
    Параметры:
    -----------
    results : dict
        Результаты оценки
    """
    
    print(f"\n=== Результаты: {results['model_name']} ===")
    print(f"Точность (Accuracy): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\nМатрица ошибок:")
    print(f"                    Предсказано 0  Предсказано 1")
    print(f"Истинно 0 (злоумышленники)    {results['tn']}              {results['fp']}")
    print(f"Истинно 1 (клиенты)           {results['fn']}              {results['tp']}")
    print(f"\nБиометрические метрики:")
    print(f"FAR (False Acceptance Rate): {results['far']:.4f} ({results['far']*100:.2f}%)")
    print(f"   - Доля злоумышленников, которых система ошибочно пропустила")
    print(f"FRR (False Rejection Rate): {results['frr']:.4f} ({results['frr']*100:.2f}%)")
    print(f"   - Доля клиентов, которых система ошибочно отклонила")
    print(f"EER (Equal Error Rate): {results['eer']:.4f} ({results['eer']*100:.2f}%)")
    print(f"   - Точка, где FAR = FRR (чем меньше, тем лучше)")
    print(f"\nМетрики классификации для класса 'клиент':")
    print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   - Доля правильно классифицированных среди всех предсказанных как 'клиент'")
    print(f"Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   - Доля найденных клиентов среди всех реальных клиентов")
    print(f"\nМетрики качества модели:")
    print(f"AUC-ROC: {results['roc_auc']:.4f}")
    print(f"   - Площадь под ROC-кривой (чем ближе к 1, тем лучше)")
    print(f"AUC-PR: {results['pr_auc']:.4f}")
    print(f"   - Площадь под Precision-Recall кривой")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"   - Средняя точность (альтернативная метрика для PR-кривой)")


def compare_models(results_list):
    """
    Сравнение результатов нескольких моделей.
    
    Параметры:
    -----------
    results_list : list
        Список словарей с результатами
    """
    
    print("\n" + "="*100)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ")
    print("="*100)
    print(f"{'Модель':<40} {'Acc':<8} {'FAR':<8} {'FRR':<8} {'EER':<8} {'Prec':<8} {'Rec':<8} {'AUC-ROC':<9} {'AUC-PR':<9}")
    print("-"*100)
    
    for res in results_list:
        print(f"{res['model_name']:<40} "
              f"{res['accuracy']*100:>5.2f}%  "
              f"{res['far']*100:>5.2f}%  "
              f"{res['frr']*100:>5.2f}%  "
              f"{res['eer']*100:>5.2f}%  "
              f"{res['precision']*100:>5.2f}%  "
              f"{res['recall']*100:>5.2f}%  "
              f"{res['roc_auc']:>7.4f}   "
              f"{res['pr_auc']:>7.4f}")
    
    print("="*100)
    
    # Определяем лучшую модель по разным критериям
    best_by_eer = min(results_list, key=lambda x: x['eer'])
    best_by_roc = max(results_list, key=lambda x: x['roc_auc'])
    best_by_pr = max(results_list, key=lambda x: x['pr_auc'])
    
    print(f"\n🏆 Лучшая модель по EER: {best_by_eer['model_name']} (EER = {best_by_eer['eer']*100:.2f}%)")
    print(f"🏆 Лучшая модель по AUC-ROC: {best_by_roc['model_name']} (AUC-ROC = {best_by_roc['roc_auc']:.4f})")
    print(f"🏆 Лучшая модель по AUC-PR: {best_by_pr['model_name']} (AUC-PR = {best_by_pr['pr_auc']:.4f})")


def visualize_model_comparison(results_list):
    """
    Визуализация сравнения моделей.
    
    Параметры:
    -----------
    results_list : list
        Список словарей с результатами
    """
    
    model_names = [res['model_name'] for res in results_list]
    accuracies = [res['accuracy'] * 100 for res in results_list]
    fars = [res['far'] * 100 for res in results_list]
    frrs = [res['frr'] * 100 for res in results_list]
    eers = [res['eer'] * 100 for res in results_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График 1: Точность
    axes[0, 0].bar(range(len(model_names)), accuracies, color='skyblue')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('Точность (%)')
    axes[0, 0].set_title('Accuracy (Точность)')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: FAR vs FRR
    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, fars, width, label='FAR', color='coral')
    axes[0, 1].bar(x + width/2, frrs, width, label='FRR', color='lightgreen')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Ошибка (%)')
    axes[0, 1].set_title('FAR vs FRR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: EER
    axes[1, 0].bar(range(len(model_names)), eers, color='mediumpurple')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('EER (%)')
    axes[1, 0].set_title('EER (Equal Error Rate) - чем меньше, тем лучше')
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: ROC-подобное сравнение
    for res in results_list:
        y_true = res['y_true']
        y_proba = res['y_proba']
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 1].plot(fpr, tpr, marker='o', markersize=3, 
                       label=f"{res['model_name'][:20]}... (AUC={roc_auc:.3f})")
    
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Случайная модель')
    axes[1, 1].set_xlabel('False Positive Rate (FAR)')
    axes[1, 1].set_ylabel('True Positive Rate (1-FRR)')
    axes[1, 1].set_title('ROC Curves')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(results_list):
    """
    Построение ROC и PR кривых для всех моделей.
    
    Параметры:
    -----------
    results_list : list
        Список словарей с результатами
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC кривая
    for res in results_list:
        y_true = res['y_true']
        y_proba = res['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = res['roc_auc']
        
        ax1.plot(fpr, tpr, marker='o', markersize=4, linewidth=2,
                label=f"{res['model_name']} (AUC={roc_auc:.4f})")
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Случайный классификатор (AUC=0.5)')
    ax1.set_xlabel('False Positive Rate (FPR) = FAR', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR) = 1 - FRR', fontsize=11)
    ax1.set_title('ROC Curve (Receiver Operating Characteristic)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    
    # PR кривая
    for res in results_list:
        y_true = res['y_true']
        y_proba = res['y_proba']
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = res['pr_auc']
        avg_prec = res['avg_precision']
        
        ax2.plot(recall, precision, marker='o', markersize=4, linewidth=2,
                label=f"{res['model_name']} (AUC={pr_auc:.4f}, AP={avg_prec:.4f})")
    
    # Базовая линия для PR-кривой (доля положительных примеров)
    n_positive = np.sum(results_list[0]['y_true'] == 1)
    n_total = len(results_list[0]['y_true'])
    baseline = n_positive / n_total
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                label=f'Базовый уровень (доля клиентов={baseline:.2f})')
    
    ax2.set_xlabel('Recall (Полнота)', fontsize=11)
    ax2.set_ylabel('Precision (Точность)', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.show()
    
    # Пояснения
    print("\n" + "="*100)
    print("ИНТЕРПРЕТАЦИЯ ROC И PR КРИВЫХ")
    print("="*100)
    
    print("\n📊 ROC-кривая (Receiver Operating Characteristic):")
    print("   • Показывает зависимость между TPR (True Positive Rate) и FPR (False Positive Rate)")
    print("   • TPR = Recall = TP/(TP+FN) - доля правильно распознанных клиентов")
    print("   • FPR = FAR = FP/(FP+TN) - доля злоумышленников, принятых за клиентов")
    print("   • AUC-ROC близкий к 1.0 = отличная модель")
    print("   • AUC-ROC = 0.5 = случайное угадывание")
    print("   • Идеальная модель: проходит через точку (0, 1) - 0% ложных срабатываний, 100% верных")
    
    print("\n📊 PR-кривая (Precision-Recall):")
    print("   • Показывает зависимость между Precision и Recall")
    print("   • Precision = TP/(TP+FP) - доля реальных клиентов среди всех, кого модель назвала клиентами")
    print("   • Recall = TP/(TP+FN) - доля найденных клиентов среди всех реальных клиентов")
    print("   • PR-кривая более информативна при несбалансированных классах")
    print("   • Базовая линия = доля положительных примеров (в нашем случае 0.5, так как 5 клиентов из 10)")
    print(f"   • Соотношение клиенты/злоумышленники: {n_positive}:{n_total-n_positive} (1:1)")
    
    print("\n💡 Для биометрических систем:")
    print("   • ROC-AUC хорош для общей оценки разделимости классов")
    print("   • PR-AUC важнее при несбалансированных данных (например, злоумышленников гораздо больше)")
    print("   • В нашем случае классы сбалансированы (5:5), поэтому обе метрики одинаково важны")
    print("   • EER (Equal Error Rate) - специфичная для биометрии метрика, показывает компромисс FAR/FRR")
    
    # Анализ для конкретных моделей
    print("\n📈 Результаты для ваших моделей:")
    for res in results_list:
        print(f"\n   {res['model_name']}:")
        print(f"      ROC-AUC = {res['roc_auc']:.4f} ", end="")
        if res['roc_auc'] >= 0.9:
            print("(отлично)")
        elif res['roc_auc'] >= 0.8:
            print("(хорошо)")
        elif res['roc_auc'] >= 0.7:
            print("(удовлетворительно)")
        else:
            print("(требует улучшения)")
        
        print(f"      PR-AUC = {res['pr_auc']:.4f}, Average Precision = {res['avg_precision']:.4f}")
        print(f"      При пороге 0.5: Precision={res['precision']:.4f}, Recall={res['recall']:.4f}")
    
    print("\n" + "="*100)


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


def visualize_eigenfaces(pca_model, n_components=5, target_size=(100, 100)):
    """
    Визуализация eigenfaces (главных компонент PCA).
    
    Параметры:
    -----------
    pca_model : PCA
        Обученная модель PCA
    n_components : int
        Количество компонент для визуализации
    target_size : tuple
        Размер изображения лица
    """
    
    n_to_show = min(n_components, pca_model.n_components_)
    
    fig, axes = plt.subplots(1, n_to_show, figsize=(3*n_to_show, 3))
    
    if n_to_show == 1:
        axes = [axes]
    
    for i in range(n_to_show):
        # Преобразуем вектор главной компоненты обратно в изображение
        eigenface = pca_model.components_[i].reshape(target_size)
        
        # Нормализуем для лучшей визуализации
        eigenface_normalized = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        
        axes[i].imshow(eigenface_normalized, cmap='gray')
        axes[i].set_title(f'Eigenface {i+1}\n({pca_model.explained_variance_ratio_[i]*100:.2f}% дисперсии)')
        axes[i].axis('off')
    
    plt.suptitle('Eigenfaces: Главные компоненты лиц', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n=== Интерпретация Eigenfaces ===")
    print(f"Первый eigenface объясняет {pca_model.explained_variance_ratio_[0]*100:.2f}% вариации в данных.")
    print("Это 'усреднённая' структура лица, которая наиболее сильно варьируется между разными людьми.")
    print("Каждое лицо в датасете можно представить как линейную комбинацию этих eigenfaces.")


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


# Пример использования
if __name__ == "__main__":
    USER_FOLDER = "./photos/user"      # Папка с селфи
    IMPOSTER_FOLDER = "./photos/imposter"  # Папка с фото других людей
    
    print("="*80)
    print("ЗАДАНИЕ 2: СРАВНЕНИЕ ЛИНЕЙНОЙ МОДЕЛИ И АНСАМБЛЯ")
    print("="*80)
    
    # Шаг 1: Обработка всего датасета
    print("\n[ШАГ 1] Предобработка изображений")
    print("-"*80)
    user_vectors, imposter_vectors, user_files, imposter_files = process_dataset(
        USER_FOLDER, IMPOSTER_FOLDER
    )
    
    print(f"\n✓ Обработано {len(user_vectors)} фото пользователя")
    print(f"✓ Обработано {len(imposter_vectors)} фото злоумышленников")
    print(f"✓ Размер вектора признаков: {user_vectors.shape[1] if len(user_vectors) > 0 else 0}")
    
    # Визуализация всех обработанных лиц
    print("\n[ВИЗУАЛИЗАЦИЯ] Обработанные лица")
    print("-"*80)
    visualize_all_faces(USER_FOLDER, IMPOSTER_FOLDER)
    
    # Шаг 2: Создание матрицы признаков
    print("\n[ШАГ 2] Создание матрицы объекты-признаки")
    print("-"*80)
    X, y = create_feature_matrix(user_vectors, imposter_vectors)
    X_raw = X.copy()  # Сохраняем исходные пиксельные векторы для ансамблей
    
    # Шаг 3: Применение PCA
    print("\n[ШАГ 3] Снижение размерности с помощью PCA")
    print("-"*80)
    X_pca, pca_model = apply_pca(X, variance_threshold=0.95)
    
    # Визуализация PCA
    print("\n[ВИЗУАЛИЗАЦИЯ] Дисперсия главных компонент")
    print("-"*80)
    visualize_pca_variance(pca_model)
    
    # Визуализация eigenfaces
    print("\n[ВИЗУАЛИЗАЦИЯ] Eigenfaces (главные компоненты)")
    print("-"*80)
    visualize_eigenfaces(pca_model, n_components=5)
    
    # Шаг 4: Обучение и оценка моделей
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("="*80)
    
    results_list = []
    
    # Модель 1: Логистическая регрессия с PCA (линейная модель)
    print("\n[МОДЕЛЬ 1] Логистическая регрессия с PCA-признаками")
    print("-"*80)
    results_lr = train_and_evaluate_biometric_system(X_pca, y)
    results_list.append(results_lr)
    
    # Модель 2: Random Forest с исходными пикселями
    print("\n[МОДЕЛЬ 2] Random Forest с исходными пикселями")
    print("-"*80)
    results_rf_raw = train_and_evaluate_ensemble(X_raw, y, use_pca_features=False, model_type='random_forest')
    results_list.append(results_rf_raw)
    
    # Модель 3: Random Forest с PCA-признаками
    print("\n[МОДЕЛЬ 3] Random Forest с PCA-признаками")
    print("-"*80)
    results_rf_pca = train_and_evaluate_ensemble(X_pca, y, use_pca_features=True, model_type='random_forest')
    results_list.append(results_rf_pca)
    
    # Сравнение моделей
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    compare_models(results_list)
    
    # Визуализация сравнения
    print("\n[ВИЗУАЛИЗАЦИЯ] Сравнение моделей")
    print("-"*80)
    visualize_model_comparison(results_list)
    
    # НОВОЕ: ROC и PR кривые
    print("\n[ВИЗУАЛИЗАЦИЯ] ROC и Precision-Recall кривые")
    print("-"*80)
    plot_roc_pr_curves(results_list)
    
    # Визуализация результатов лучшей модели
    best_model = min(results_list, key=lambda x: x['eer'])
    print("\n[ВИЗУАЛИЗАЦИЯ] Детальные результаты лучшей модели")
    print("-"*80)
    visualize_results(best_model, y)
    
    print("\n" + "="*80)
    print("ЗАВЕРШЕНО")
    print("="*80)
    
    # Выводы для анализа
    print("\n" + "="*80)
    print("ВЫВОДЫ ПО СРАВНЕНИЮ МОДЕЛЕЙ")
    print("="*80)
    print("\n1. Сравнение линейной модели и ансамблей:")
    print(f"   Линейная модель (Логистическая регрессия):")
    print(f"   - Accuracy: {results_lr['accuracy']*100:.2f}%, EER: {results_lr['eer']*100:.2f}%")
    print(f"   - Простая, интерпретируемая, быстрая")
    print(f"   - Предполагает линейную разделимость классов")
    
    print(f"\n   Ансамбли (Random Forest):")
    print(f"   - С исходными пикселями: EER: {results_rf_raw['eer']*100:.2f}%")
    print(f"   - С PCA-признаками: EER: {results_rf_pca['eer']*100:.2f}%")
    print(f"   - Могут захватывать нелинейные зависимости")
    print(f"   - Более устойчивы к переобучению за счёт усреднения")
    
    print("\n2. Влияние размера датасета:")
    print("   При малом датасете (10 объектов):")
    print("   - Ансамбли могут переобучаться на конкретных примерах")
    print("   - Линейные модели более стабильны")
    print("   - Разница в метриках может быть незначительной")
    print("   - С увеличением данных ансамбли обычно показывают лучше результат")
    
    print("\n3. Рекомендации:")
    if best_model['model_name'].startswith('Логистическая'):
        print("   ✓ На текущем датасете линейная модель показала лучший результат")
        print("   • Это типично для малых датасетов")
        print("   • Ансамбли выигрывают на больших и более разнообразных данных")
    else:
        print("   ✓ Ансамбль показал лучший результат даже на малом датасете")
        print("   • Это говорит о наличии нелинейных зависимостей в данных")
        print("   • С увеличением данных преимущество ансамблей возрастёт")
    
    print("\n" + "="*80)
