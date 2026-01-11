from datetime import datetime
import pandas as pd

from logger import logger
from yahoo_api import get_ticket_filepath, get_ticket_plot_filepath, is_valid_ticker, save_ticker_history, file_exists
from models import forecasting_pipeline, plot_forecast
from utils import find_trade_points, simulate_trading

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.error import TelegramError

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для анализа и прогнозирования американских акций.\n"
        "Введите тикер компании (например, AAPL) и сумму для инвестиций в долларах через пробел.\n"
        "Пример 1: AAPL 1000\n"
        "Пример 2: NVDA 325"
        "Бот не является финансовым инструментом, результаты носят исключительно учебный характер."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        text = update.message.text.strip()
        ticker, amount = text.split(' ')
        ticker = ticker.upper()
        # семантическая валидация ввода    
        try:
            amount = int(amount)
        except ValueError as e:
            await update.message.reply_text(f"Проверьте формат: не получилось перевести в число '{amount}'")
            log_line = (
                f"{user_id}\t{dt}\t{ticker}"
            )
            logger.info(f"LOG_SESSION VALIDATION ERROR\t invalid amount: {log_line}")
            return 
        # логическая валидация ввода + сохранение данных
        if is_valid_ticker(ticker):
            if not file_exists(ticker):
                save_ticker_history(ticker)
                print(f"Данные для {ticker} сохранены.")
            else:
                print(f"Файл для {ticker} уже существует.")
        else:
            await update.message.reply_text(f"Тикет '{ticker}' не найден")
            log_line = (
                f"{user_id}\t{dt}\t{ticker}"
            )
            logger.info(f"LOG_SESSION VALIDATION ERROR\t tiker not found: {log_line}")
            return 
        
        await update.message.reply_text(f"Тикет: {ticker}\nСумма денег: {amount}$")
        wait_msg = await update.message.reply_text("Проводится расчет стратегии...")
        result, metrics, best_model, best_metric, forecast_df, df = forecasting_pipeline(ticker)
 
        last_actual = df['close'].iloc[-1]
        last_pred = forecast_df['best_pred'].iloc[-1]
        abs_change = last_pred - last_actual
        rel_change = (abs_change / last_actual) * 100
        if abs_change > 0:
            trend = "вырастут"
        else:
            trend = "упадут"
        msg = (f"Ожидается, что через 30 дней акции {ticker} {trend} на "
               f"{abs(abs_change):.2f}$ ({abs(rel_change):.2f}%) относительно текущей цены.\n"
               f"Текущая цена: {last_actual:.2f}\n"
               f"Прогнозируемая цена: {last_pred:.2f}")
        await update.message.reply_text(msg)
        await wait_msg.delete()
        
        plot_forecast(df, forecast_df, ticker)
        with open(get_ticket_plot_filepath(ticker), 'rb') as photo:
            await update.message.reply_photo(photo, caption=f"Прогноз цены {ticker.upper()} на 30 дней")

        prices = forecast_df['best_pred'].values
        dates = forecast_df['date'].values
        profit, actions = simulate_trading(prices, amount, dates)
        
        strategy_msg = "Рекомендации по торговле:\n"
        if len(actions) == 0:
            strategy_msg += "В прогнозе не найдено подходящих точек для покупки и продажи.\n"
        else:
            a = actions[0]
            if len(actions) == 1 and a['buy_date'] == dates[0] and a['sell_date'] == dates[-1]:
                if a['profit'] > 0:
                    strategy_msg += (
                        "Рынок показывает устойчивый рост. "
                        "Оптимальная стратегия — купить в первый день прогноза и продать в последний.\n"
                    )
                    strategy_msg += (
                        f"Покупка: {pd.to_datetime(a['buy_date']).strftime('%Y-%m-%d')} по {a['buy_price']:.2f}, "
                        f"продажа: {pd.to_datetime(a['sell_date']).strftime('%Y-%m-%d')} по {a['sell_price']:.2f}, "
                        f"прибыль: {a['profit']:.2f}$\n"
                        f"\nОриентировочная суммарная прибыль: {profit:.2f}$"
                    )
                else:
                    strategy_msg += (
                        "Рынок прогнозируется как падающий. "
                        "Покупка акций не рекомендуется."
                    )
                    profit = 0
            else:
                for a in actions:
                    strategy_msg += (
                        f"Покупка: {pd.to_datetime(a['buy_date']).strftime('%Y-%m-%d')} по {a['buy_price']:.2f}, "
                        f"продажа: {pd.to_datetime(a['sell_date']).strftime('%Y-%m-%d')} по {a['sell_price']:.2f}, "
                        f"прибыль: {a['profit']:.2f}$\n"
                    )
                strategy_msg += f"\nОриентировочная суммарная прибыль: {profit:.2f}$"
        
        await update.message.reply_text(strategy_msg)
        
        log_line = (
            f"{user_id}\t{dt}\t{ticker}\t{amount}\t{best_model}\t{best_metric:.4f}\t{profit}"
        )
        logger.info(f"LOG_SESSION SUCCESS\t{log_line}")

    except Exception as e:
        logger.exception(f"Непредвиденная ошибка в handle_message: {e}")
        await update.message.reply_text("Произошла ошибка")
        
   
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Извините, я не знаю такой команды. Введите тикер и сумму.")
