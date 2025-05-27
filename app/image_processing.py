"""
Image Processing Module
Містить функції для обробки зображень: sharpening, preprocessing та інші фільтри
"""

import cv2
import numpy as np


def sharpen_image(image, strength=1):
    """
    Підвищення різкості зображення з можливістю регулювання сили ефекту.
    
    Args:
        image: Вхідне зображення
        strength: Рівень різкості (1-3)
        
    Returns:
        Оброблене зображення
    """
    result = image.copy()
    
    if strength == 1:
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        result = cv2.filter2D(image, -1, kernel)
    
    elif strength == 2:
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(image, -1, kernel)
    
    elif strength == 3:
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        usm = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(usm, -1, kernel)
        
        if len(image.shape) == 2:
            lab = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    return result


def apply_preprocessing(image, preprocessing_method):
    """
    Застосування різних методів попередньої обробки зображення.
    
    Args:
        image: Вхідне зображення (grayscale)
        preprocessing_method: Номер методу (0-6)
        
    Returns:
        Оброблене зображення
    """
    if preprocessing_method == 0:
        return image
    
    elif preprocessing_method == 1:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif preprocessing_method == 2:
        # Bilateral Filter (зберігає краї при згладжуванні)
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif preprocessing_method == 3:
        # Adaptive Threshold з змішуванням
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        return cv2.addWeighted(image, 0.7, binary, 0.3, 0)
    
    elif preprocessing_method == 4:
        # Sobel Edge Detection з змішуванням
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.addWeighted(image, 0.7, sobel, 0.3, 0)
    
    elif preprocessing_method == 5:
        # Morphological Operations (відкриття + закриття)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing
    
    elif preprocessing_method == 6:
        # Histogram Normalization
        norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return norm_img
    
    return image


def apply_color_preprocessing(image, preprocessing_method):
    """
    Застосування різних методів попередньої обробки для кольорових зображень.
    
    Args:
        image: Вхідне кольорове зображення (BGR)
        preprocessing_method: Номер методу (0-6)
        
    Returns:
        Оброблене кольорове зображення
    """
    if preprocessing_method == 0:
        return image
    
    elif preprocessing_method == 1:
        # CLAHE для кольорових зображень (в LAB space)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif preprocessing_method == 2:
        # Bilateral Filter
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif preprocessing_method == 3:
        # Adaptive Threshold (конвертуємо в grayscale, обробляємо, повертаємо в кольорове)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.7, binary_3ch, 0.3, 0)
    
    elif preprocessing_method == 4:
        # Sobel Edge Detection для кольорових зображень
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        sobel_3ch = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.7, sobel_3ch, 0.3, 0)
    
    elif preprocessing_method == 5:
        # Morphological Operations для кожного каналу
        kernel = np.ones((3, 3), np.uint8)
        channels = cv2.split(image)
        processed_channels = []
        for channel in channels:
            opening = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            processed_channels.append(closing)
        return cv2.merge(processed_channels)
    
    elif preprocessing_method == 6:
        # Histogram Normalization для кожного каналу
        channels = cv2.split(image)
        normalized_channels = []
        for channel in channels:
            norm_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
            normalized_channels.append(norm_channel)
        return cv2.merge(normalized_channels)
    
    return image


def sharpen_color_image(image, strength=1):
    """
    Підвищення різкості кольорового зображення з можливістю регулювання сили ефекту.
    
    Args:
        image: Вхідне кольорове зображення
        strength: Рівень різкості (1-3)
        
    Returns:
        Оброблене зображення
    """
    if strength == 1:
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    elif strength == 2:
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        return cv2.filter2D(image, -1, kernel)
    
    elif strength == 3:
        # Unsharp masking для кольорових зображень
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        usm = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        kernel = np.array([[-2,-2,-2], 
                           [-2, 17,-2],
                           [-2,-2,-2]])
        result = cv2.filter2D(usm, -1, kernel)
        
        # Додаткове покращення в LAB space
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return image


def enhance_color_image(image, method='auto'):
    """
    Покращення кольорового зображення різними методами.
    
    Args:
        image: Вхідне кольорове зображення
        method: Метод покращення ('auto', 'vibrant', 'contrast', 'bright')
        
    Returns:
        Покращене зображення
    """
    if method == 'vibrant':
        # Підвищення насиченості
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.3)  # Збільшуємо насиченість
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif method == 'contrast':
        # Підвищення контрасту
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif method == 'bright':
        # Підвищення яскравості
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)  # Збільшуємо яскравість
        v = np.clip(v, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif method == 'auto':
        # Автоматичне покращення
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Легке підвищення насиченості
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.1)
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image


def apply_noise_reduction(image, method='gaussian'):
    """
    Застосування зменшення шуму.
    
    Args:
        image: Вхідне зображення
        method: Метод зменшення шуму ('gaussian', 'median', 'bilateral')
        
    Returns:
        Оброблене зображення
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image


def apply_gamma_correction(image, gamma=1.0):
    """
    Застосування гамма-корекції для покращення контрасту.
    
    Args:
        image: Вхідне зображення
        gamma: Значення гамми (< 1.0 = темніше, > 1.0 = світліше)
        
    Returns:
        Оброблене зображення
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_histogram_equalization(image, method='standard'):
    """
    Застосування вирівнювання гістограми.
    
    Args:
        image: Вхідне зображення (grayscale)
        method: Метод ('standard', 'clahe')
        
    Returns:
        Оброблене зображення
    """
    if method == 'standard':
        return cv2.equalizeHist(image)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    else:
        return image


# Словник з назвами методів preprocessing для зручності
PREPROCESSING_METHODS = {
    0: "None",
    1: "CLAHE", 
    2: "Bilateral",
    3: "Adaptive Threshold",
    4: "Sobel",
    5: "Morphological",
    6: "Normalization"
}


def get_preprocessing_name(method_id):
    """
    Отримання назви методу preprocessing за його ID.
    
    Args:
        method_id: ID методу (0-6)
        
    Returns:
        Назва методу
    """
    return PREPROCESSING_METHODS.get(method_id, "Unknown")