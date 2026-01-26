"""
Точка входа для Telegram-бота.
Запускает stage_1 (Telegram-логика), которая вызывает stage_2_5 при необходимости.
"""

import logging
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def main():
    """Основная функция запуска бота."""
    try:
        logger.info("🚀 Запуск Telegram-бота...")

        # Импортируем основной модуль бота
        from stage_1 import main as bot_main
        bot_main()

    except Exception as e:
        logger.exception("❌ Критическая ошибка при запуске бота: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
