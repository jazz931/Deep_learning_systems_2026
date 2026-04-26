import os
import sys
import json
import numpy as np
import logging

# ==========================================
# СКРЫТИЕ ПОВТОРНЫХ ВАРНИНГОВ TENSORFLOW (ДЛЯ ЧИСТОТЫ ВЫВОДА)
# ==========================================
# 0 = все сообщения, 1 = info, 2 = info+warning (оставляем только ошибки)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Отключаем предупреждения Python (например, про deprecated методы в Keras)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# ==========================================

# Фиксируем seed для воспроизводимости (требование из задания)
np.random.seed(42)

# ЭТАЛОННЫЕ ЗНАЧЕНИЯ (для my_face.jpg)
EXPECTED_AGE = 29
EXPECTED_GENDER = "Woman"
EXPECTED_EMOTION = "neutral"
AGE_TOLERANCE = 5  # Допустимая погрешность в возрасте

# Вспомогательная функция: превращает numpy-числа в обычные, чтобы JSON не падал
def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    return obj

def check_dependencies():
    """Проверяет наличие критических библиотек"""
    print("[0/5] Проверка зависимостей и структуры...")
    
    # Проверяем только базовые либы, deepface проверим позже при импорте
    libs = ['tensorflow', 'cv2', 'numpy', 'PIL']
    missing = []
    for lib in libs:
        try:
            if lib == 'cv2': __import__('cv2')
            elif lib == 'PIL': __import__('PIL')
            else: __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print(f"       Не установлены: {', '.join(missing)}")
        print("       Запусти: pip install tensorflow tf-keras opencv-python pillow")
        return False

    # Проверяем папки
    if not os.path.isdir("test_input"): os.makedirs("test_input", exist_ok=True)
    if not os.path.isdir("output"): os.makedirs("output", exist_ok=True)
        
    print("      Зависимости и папки готовы")
    return True

def run_tests():
    print("="*60)
    print("ЗАПУСК ТЕСТА КОРРЕКТНОСТИ (RunCheckerTest)")
    print("="*60)

    if not check_dependencies():
        return False

    TEST_IMG_PATH = os.path.join("test_input", "my_face.jpg")
    RESULT_FILE = os.path.join("output", "deepface_result.json")

    try:
        # 1. Строгая проверка НАЛИЧИЯ фото
        print("[1/5] Проверка входных данных...")
        if not os.path.isfile(TEST_IMG_PATH):
            print(f"       КРИТИЧЕСКАЯ ОШИБКА: Файл {TEST_IMG_PATH} не найден!")
            print(f"       Положи свою фотографию в папку 'test_input/' и переименуй в 'my_face.jpg'")
            return False
        print(f"      Найдено пользовательское фото: {TEST_IMG_PATH}")

        # 2. Импорт
        print("[2/5] Импорт DeepFace...")
        from deepface import DeepFace
        print("      Библиотека загружена")

        # 3. Анализ
        print("[3/5] Анализ изображения...")
        result = DeepFace.analyze(
            img_path=TEST_IMG_PATH,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            silent=True
        )

        # 4. Сохранение результата (с конвертацией типов для JSON)
        print("[4/5] Сохранение отчёта в output/...")
        safe_result = convert_numpy_types(result)
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(safe_result, f, indent=4, ensure_ascii=False)
        print(f"      Отчёт сохранён: {RESULT_FILE}")

        # 5. Сравнение с эталоном
        print("[5/5] Сверка результатов...")
        face_data = result[0]
        actual_age = face_data.get('age')
        actual_gender = face_data.get('dominant_gender')
        actual_emotion = face_data.get('dominant_emotion')

        print(f"      Эталон:    Возраст ~{EXPECTED_AGE}, Пол: {EXPECTED_GENDER}, Эмоция: {EXPECTED_EMOTION}")
        print(f"      Получено:  Возраст {actual_age}, Пол: {actual_gender}, Эмоция: {actual_emotion}")

        checks_passed = True

        # Возраст (с допуском)
        if actual_age and (EXPECTED_AGE - AGE_TOLERANCE <= actual_age <= EXPECTED_AGE + AGE_TOLERANCE):
            print(f"      Возраст: в диапазоне (±{AGE_TOLERANCE})")
        else:
            print(f"      Возраст: не совпадает")
            checks_passed = False

        # Пол (точное совпадение)
        if actual_gender and actual_gender.lower() == EXPECTED_GENDER.lower():
            print(f"      Пол: точное совпадение")
        else:
            print(f"      Пол: не совпадает")
            checks_passed = False

        # Эмоция (точное совпадение)
        if actual_emotion and actual_emotion.lower() == EXPECTED_EMOTION.lower():
            print(f"      Эмоция: точное совпадение")
        else:
            print(f"      Эмоция: не совпадает")
            checks_passed = False

        return checks_passed

    except Exception as e:
        print(f"       Ошибка выполнения: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    print("="*60)
    if success:
        print("TEST PASSED SUCCESSFULLY")
        print("Проект работает корректно. Полный отчёт в output/deepface_result.json")
        sys.exit(0)  # Код 0 = успех (стандарт для автотестов)
    else:
        print("TEST FAILED")
        sys.exit(1)  # Код 1 = провал