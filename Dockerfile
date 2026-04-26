FROM python:3.10-bookworm

WORKDIR /app

# Устанавливаем библиотеки
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Создаём папки заранее
RUN mkdir -p /app/output \
    && mkdir -p /root/.deepface/weights

# Копируем проект
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir tf-keras
RUN pip install --no-cache-dir -e .

# Объявляем volumes (подсказка для пользователя)
VOLUME ["/app/output", "/root/.deepface/weights"]

# Запускаем тест
CMD ["python", "RunCheckerTest.py"]