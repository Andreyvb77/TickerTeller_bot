# stage_1.py
"""
Этап 1. Взаимодействие с пользователем и загрузка котировок за 2 года (yfinance).

Что делает:
- Диалог: /getdata → бот просит тикер → затем сумму → загружает котировки за 2 года → присылает CSV и краткую сводку.
- Сохраняет CSV в data/ под именем user_<id>_<TICKER>_last_2y.csv.
- Подсказывает команду анализа: /analyze <TICKER> <AMOUNT> (для модулей Этапов 2–5).
- Работает в обычном скрипте и в средах с активным event loop (Jupyter/Spyder) — поддержан фоновый запуск.

Зависимости:
- python-telegram-bot==20.6
- yfinance
- pandas
- python-dotenv

Окружение:
- Создайте файл my_bot.env (или .env) рядом со скриптом:
  TELEGRAM_BOT_TOKEN=ВАШ_ТОКЕН

Подключение Этапов 2–5:
- Модуль stage_2_5.py должен лежать рядом. Регистрация его хэндлеров выполняется внутри build_app().
"""

from __future__ import annotations
import io
import os
import asyncio
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from dotenv import load_dotenv
from telegram import InputFile, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

# === TELEGRAM_BOT_TOKEN ===

# 1. Загружаем
from dotenv import load_dotenv
load_dotenv("my_bot.env")
token = os.getenv("TELEGRAM_BOT_TOKEN")

print("Токен в Colab:", token)

# === Определение состояний для ConversationHandler ===

# ASK_TICKER — ожидание ввода тикера акции
# ASK_AMOUNT — ожидание ввода суммы инвестиций
ASK_TICKER, ASK_AMOUNT = range(2)

# Количество лет исторических данных для загрузки
LOOKBACK_YEARS: int = 2

# Директория для хранения временных файлов (графиков, логов и т.д.)
DATA_DIR: Path = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Создаём папку, если её нет

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_script_dir() -> Path:
    """
    Возвращает абсолютный путь к директории, в которой находится текущий скрипт.

    Используется для корректного поиска файлов (.env, логов и т.д.)
    независимо от того, откуда запущен скрипт.

    Обрабатывает случай запуска в интерактивных средах (например, Jupyter),
    где __file__ может быть недоступен.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def load_token_from_env() -> str:
    """
    Загружает токен Telegram-бота из нескольких возможных источников в порядке приоритета:

    1. Файл `my_bot.env` в директории скрипта (специфичный для проекта)
    2. Файл `.env` в директории скрипта (стандартное соглашение)
    3. Переменная окружения ОС `TELEGRAM_BOT_TOKEN`

    Выбрасывает исключение, если токен не найден ни в одном из источников.

    Returns:
        str: Валидный токен бота.

    Raises:
        RuntimeError: Если токен не обнаружен.
    """
    script_dir = get_script_dir()

    # 1) Проверяем наличие специфичного файла my_bot.env
    my_env = script_dir / "my_bot.env"
    if my_env.exists():
        load_dotenv(dotenv_path=my_env, override=True)
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            logging.info("Токен загружен из my_bot.env")
            return token

    # 2) Проверяем стандартный .env
    default_env = script_dir / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            logging.info("Токен загружен из .env")
            return token

    # 3) Проверяем системные переменные окружения
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token:
        logging.info("Токен загружен из переменных окружения ОС")
        return token

    raise RuntimeError(
        "Не найден TELEGRAM_BOT_TOKEN. Укажите его в файле my_bot.env, .env "
        "или как переменную окружения ОС."
    )

def is_event_loop_running() -> bool:
    """
    Проверяет, запущен ли асинхронный event loop в текущем потоке.

    Используется для корректной обработки запуска бота в разных средах:
    - В обычном скрипте: loop ещё не запущен → нужно использовать asyncio.run()
    - В Jupyter/Colab: loop уже запущен → нельзя вызывать asyncio.run()

    Returns:
        bool: True, если event loop активен; иначе False.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.is_running()
    except RuntimeError:
        return False

async def run_bot_async(app: Application) -> None:
    """
    Асинхронно запускает Telegram-бота в режиме polling.

    Выполняет инициализацию приложения, запуск обработчика обновлений
    и блокирует выполнение до остановки (в реальных проектах обычно используется
    сигнал завершения, но для учебного примера достаточно логирования).

    Args:
        app (Application): Инициализированное приложение python-telegram-bot.
    """
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logging.info("Бот запущен и ожидает сообщения...")

    # === Загрузка данных. Глобальная константа LOOKBACK_YEARS ===

@dataclass
class PriceDataSummary:
    """
   Краткая сводка по загруженным историческим данным акций,
   для логирования, отладки и формирования информативных сообщений пользователю.
    """
    ticker: str              # Тикер акции (в верхнем регистре)
    rows: int                # Количество торговых дней в выборке
    start_date: datetime     # Самая ранняя дата в данных
    end_date: datetime       # Самая поздняя дата в данных
    last_close: float        # Цена закрытия на последнюю доступную дату ($)


def load_last_two_years_history(ticker: str) -> Tuple[pd.DataFrame, PriceDataSummary]:
    """
    Загружает исторические данные о цене акций за последние N лет (по умолчанию — 2 года)
    с использованием библиотеки yfinance.

    Args:
        ticker (str): Тикер акции (например, 'AAPL', 'MSFT').

    Returns:
        Tuple[pd.DataFrame, PriceDataSummary]:
            - DataFrame с колонками: Open, High, Low, Close, Adj Close, Volume.
            - Объект PriceDataSummary с метаданными.

    Raises:
        ValueError: Если данные не найдены или их недостаточно (< 30 дней).
    """
    # Определяем временной диапазон и небольшой запас (7 дней на праздники/нерабочие дни)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * LOOKBACK_YEARS + 7)

    # Загружаем данные с Yahoo Finance
    df = yf.download(
        tickers=ticker,
        start=start.date(),
        end=end.date(),
        interval="1d",          # Дневные интервалы
        auto_adjust=False,      # Не корректируем цены автоматически (сохраняем исходные Close/Adj Close)
        progress=False,         # Отключаем прогресс-бар
        threads=True,           # Ускоряем загрузку через многопоточность
    )

    # Проверка: данные получены?
    if df.empty:
        raise ValueError("Не удалось загрузить данные. Проверьте тикер или соединение.")

    # Очистка: удаляем полностью пустые строки и нормализуем индекс времени
    df = df.dropna(how="all").copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # Убираем временную зону для упрощения

    # Минимальное требование: хотя бы 30 торговых дней для анализа
    if df.shape[0] < 30:
        raise ValueError("Недостаточно данных для анализа (требуется минимум 30 торговых дней).")

    # Используем скорректированную цену закрытия (Adj Close), если доступна;
    # иначе — обычную цену закрытия (Close)
    price_series = df.get("Adj Close")
    if price_series is None or price_series.empty:
        price_series = df["Close"]
    last_close = float(price_series.iloc[-1])

    # Формируем сводку
    summary = PriceDataSummary(
        ticker=ticker.upper(),
        rows=df.shape[0],
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        last_close=last_close,
    )

    return df, summary

# === Валидация тикера ===

def is_valid_ticker(text: str) -> bool:
    """
    Функция выполняет базовую валидацию тикера перед обращением к API (yfinance).
    Корректный тикер с критериями:
    - Состоять только из букв и цифр
    - Иметь длину от 1 до 10 символов
    """
    # Нормализация: удаляем пробелы по краям и приводим к единому регистру
    normalized_ticker = text.strip().upper()

    # Проверка 1: строка содержит только буквы и цифры (без '_', '-', '.', пробелов и т.д.)
    # Проверка 2: длина в допустимом диапазоне (1-10 символов)
    return normalized_ticker.isalnum() and (1 <= len(normalized_ticker) <= 10)

# === Обработчики состояний для ConversationHandler ===

async def cmd_start(update: Update, context: CallbackContext) -> None:
    """Приветственное сообщение при запуске бота."""
    if update.message:
        await update.message.reply_text(
            "Привет! Я загружу котировки за последние два года.\n"
            "Начните командой /getdata"
        )

async def getdata_entry(update: Update, context: CallbackContext) -> int:
    """Начало диалога: запрос тикера компании."""
    if update.message:
        await update.message.reply_text(
            "Введите тикер компании (например, AMD):",
            reply_markup=ReplyKeyboardRemove(),
        )
    return ASK_TICKER

async def ask_amount(update: Update, context: CallbackContext) -> int:
    """Валидация тикера и запрос суммы инвестиций."""
    if not update.message:
        return ASK_TICKER

    ticker = update.message.text.strip().upper()
    if not is_valid_ticker(ticker):
        await update.message.reply_text("Некорректный тикер. Используйте 1-10 букв/цифр (например, AMD):")
        return ASK_TICKER

    context.user_data["ticker"] = ticker
    await update.message.reply_text(
        f"Тикер: {ticker}. Введите сумму для условной инвестиции в USD (например, 1000):"
    )
    return ASK_AMOUNT

async def load_and_reply(update: Update, context: CallbackContext) -> int:
    """Загрузка данных, сохранение CSV и отправка пользователю."""
    if not update.message:
        return ConversationHandler.END

    # Обработка суммы с поддержкой запятой как десятичного разделителя
    text_val = update.message.text.strip().replace(",", ".")
    try:
        amount = float(text_val)
        if not pd.notna(amount) or amount <= 0:
            raise ValueError
    except Exception:
        await update.message.reply_text("Сумма должна быть положительным числом. Попробуйте снова:")
        return ASK_AMOUNT

    ticker: Optional[str] = context.user_data.get("ticker")
    if not ticker:
        await update.message.reply_text("Не удалось определить тикер. Начните заново: /getdata")
        return ConversationHandler.END

    await update.message.reply_text("Загружаю данные с Yahoo Finance...")

    try:
        # Загрузка в фоновом потоке, чтобы не блокировать event loop
        df, summary = await asyncio.to_thread(load_last_two_years_history, ticker)
    except Exception as e:
        logging.exception("Ошибка загрузки данных: %s", e)
        await update.message.reply_text(f"Не удалось загрузить данные: {e}")
        return ConversationHandler.END

    # Сохранение данных в CSV
    user_id = update.effective_user.id if update.effective_user else 0
    filename = f"user_{user_id}_{summary.ticker}_last_{LOOKBACK_YEARS}y.csv"
    local_path = DATA_DIR / filename
    df.to_csv(local_path, index=True)

    # Подготовка CSV для отправки
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=True)
    csv_bytes.seek(0)

    caption = (
        f" Данные загружены для {summary.ticker}\n"
        f"- Дней: {summary.rows}\n"
        f"- Период: {summary.start_date.date()} — {summary.end_date.date()}\n"
        f"- Последняя цена: {summary.last_close:.2f} USD\n"
        f"- Условная сумма: {amount:.2f} USD\n\n"
        f"Теперь запустите анализ: /analyze {summary.ticker} {amount}"
    )

    await update.message.reply_document(
        document=InputFile(csv_bytes, filename=filename),
        caption=caption,
    )

    return ConversationHandler.END

async def cancel(update: Update, context: CallbackContext) -> int:
    """Обработчик отмены диалога по команде /cancel."""
    if update.message:
        await update.message.reply_text("Диалог отменён.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# === Обработчик ошибок для бота ===

async def on_error(update: object, context) -> None:
    """Логирует необработанные исключения с полной информацией об ошибке."""
    logging.exception("Unhandled error", exc_info=context.error)

    # === Инициализация и запуск Telegram-бота ===

def build_app(token: str) -> Application:
    """Создаёт и настраивает приложение бота с обработчиками."""
    app: Application = ApplicationBuilder().token(token).build()

    # Диалог для получения тикера и суммы инвестиций
    conv = ConversationHandler(
        entry_points=[CommandHandler("getdata", getdata_entry)],
        states={
            ASK_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_amount)],
            ASK_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, load_and_reply)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )

    # Основные команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(conv)
    app.add_handler(CommandHandler("cancel", cancel))

    # Динамическая загрузка дополнительных обработчиков (этапы 2-5)
    try:
        from stage_2_5 import register_stage2_5_handlers, register_optional_csv_handler
        register_stage2_5_handlers(app)        # Регистрация команд анализа
        register_optional_csv_handler(app)     # Обработка CSV-файлов
        logging.info("Этапы 2-5 успешно подключены")
    except Exception as e:
        logging.warning("Не удалось подключить Этапы 2–5: %s", e)

    # Глобальный обработчик ошибок
    app.add_error_handler(on_error)

    return app

def main() -> None:
    """Точка входа: загружает токен, создаёт приложение и запускает polling."""
    token: str = load_token_from_env()
    app: Application = build_app(token)

    # Совместимость с Jupyter/Colab (где event loop уже запущен)
    if is_event_loop_running():
        asyncio.get_running_loop().create_task(run_bot_async(app))
        logging.info("Бот запущен в фоновом режиме (асинхронно)")
    else:
        logging.info("Запуск бота в режиме polling...")
        app.run_polling(close_loop=True)

if __name__ == "__main__":
    main()
