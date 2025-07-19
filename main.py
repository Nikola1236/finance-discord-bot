import discord
from discord.ext import commands, tasks
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import json
import sqlite3
from datetime import datetime, timedelta
import logging
import asyncpg
from typing import Dict, List, Optional, Tuple
import ccxt
import MetaTrader5 as mt5
from textblob import TextBlob
import tweepy
import feedparser
import warnings
warnings.filterwarnings('ignore')

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaminarBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='/', intents=intents)
        
        # Inizializzazione componenti
        self.db_pool = None
        self.ml_models = {}
        self.trading_strategies = {}
        self.active_positions = {}
        self.user_subscriptions = {}
        self.price_alerts = {}
        
        # API Keys (da configurare nel tuo environment)
        self.config = {
            'discord_token': 'YOUR_DISCORD_TOKEN',
            'alpha_vantage_key': 'YOUR_ALPHA_VANTAGE_KEY',
            'news_api_key': 'YOUR_NEWS_API_KEY',
            'twitter_bearer_token': 'YOUR_TWITTER_BEARER_TOKEN',
            'mt5_login': 'YOUR_MT5_LOGIN',
            'mt5_password': 'YOUR_MT5_PASSWORD',
            'mt5_server': 'YOUR_MT5_SERVER'
        }
        
        # Inizializzazione exchange
        self.exchange = ccxt.binance({
            'apiKey': 'YOUR_BINANCE_API_KEY',
            'secret': 'YOUR_BINANCE_SECRET',
            'sandbox': True  # Usa True per testing
        })

    async def setup_hook(self):
        """Inizializzazione del bot"""
        await self.setup_database()
        await self.load_ml_models()
        await self.setup_mt5_connection()
        self.market_analysis_task.start()
        self.alert_monitoring_task.start()
        logger.info("Laminar Bot inizializzato con successo!")

    async def setup_database(self):
        """Setup database PostgreSQL"""
        try:
            self.db_pool = await asyncpg.create_pool(
                "postgresql://user:password@localhost/laminar_bot"
            )
            
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20),
                        signal_type VARCHAR(10),
                        price DECIMAL(15, 6),
                        confidence DECIMAL(5, 4),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        strategy VARCHAR(50)
                    )
                ''')
                
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT,
                        symbol VARCHAR(20),
                        side VARCHAR(10),
                        quantity DECIMAL(15, 6),
                        entry_price DECIMAL(15, 6),
                        stop_loss DECIMAL(15, 6),
                        take_profit DECIMAL(15, 6),
                        status VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        user_id BIGINT,
                        symbol VARCHAR(20),
                        channel_id BIGINT,
                        PRIMARY KEY (user_id, symbol)
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Errore setup database: {e}")

    async def load_ml_models(self):
        """Carica i modelli di machine learning"""
        try:
            # Modello LSTM per predizioni di prezzo
            self.ml_models['lstm_predictor'] = self.create_lstm_model()
            
            # Random Forest per classificazione segnali
            self.ml_models['signal_classifier'] = RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
            
            # Isolation Forest per anomaly detection
            self.ml_models['anomaly_detector'] = IsolationForest(
                contamination=0.1, 
                random_state=42
            )
            
            # Scaler per normalizzazione dati
            self.ml_models['scaler'] = StandardScaler()
            
            logger.info("Modelli ML caricati con successo")
            
        except Exception as e:
            logger.error(f"Errore caricamento modelli ML: {e}")

    def create_lstm_model(self, input_shape=(60, 5)):
        """Crea modello LSTM per predizioni"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    async def setup_mt5_connection(self):
        """Inizializza connessione MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            authorized = mt5.login(
                self.config['mt5_login'],
                password=self.config['mt5_password'],
                server=self.config['mt5_server']
            )
            
            if authorized:
                logger.info("MT5 connesso con successo")
                return True
            else:
                logger.error("MT5 login fallito")
                return False
                
        except Exception as e:
            logger.error(f"Errore connessione MT5: {e}")
            return False

    @tasks.loop(minutes=1)
    async def market_analysis_task(self):
        """Task per analisi continua del mercato"""
        try:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
            
            for symbol in symbols:
                # Ottieni dati storici
                data = await self.get_historical_data(symbol, '1h', 100)
                
                if data is not None and len(data) > 0:
                    # Analisi tecnica
                    signals = await self.technical_analysis(symbol, data)
                    
                    # Machine Learning prediction
                    ml_signal = await self.ml_prediction(symbol, data)
                    
                    # Sentiment analysis
                    sentiment = await self.analyze_sentiment(symbol)
                    
                    # Combina tutti i segnali
                    final_signal = await self.combine_signals(
                        signals, ml_signal, sentiment
                    )
                    
                    # Salva segnale se significativo
                    if final_signal['confidence'] > 0.7:
                        await self.save_signal(symbol, final_signal)
                        await self.notify_subscribers(symbol, final_signal)
                        
        except Exception as e:
            logger.error(f"Errore market analysis: {e}")

    @tasks.loop(minutes=5)
    async def alert_monitoring_task(self):
        """Monitora gli alert di prezzo"""
        try:
            for user_id, alerts in self.price_alerts.items():
                for alert in alerts:
                    symbol = alert['symbol']
                    target_price = alert['target_price']
                    
                    current_price = await self.get_current_price(symbol)
                    
                    if current_price:
                        if (alert['direction'] == 'above' and current_price >= target_price) or \
                           (alert['direction'] == 'below' and current_price <= target_price):
                            
                            user = self.get_user(user_id)
                            if user:
                                embed = discord.Embed(
                                    title="🔔 Alert Triggered!",
                                    description=f"{symbol} ha raggiunto {current_price}!",
                                    color=0x00ff00
                                )
                                await user.send(embed=embed)
                                
                            # Rimuovi alert triggerato
                            self.price_alerts[user_id].remove(alert)
                            
        except Exception as e:
            logger.error(f"Errore monitoring alert: {e}")

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Ottiene dati storici da Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Errore fetch dati {symbol}: {e}")
            return None

    async def technical_analysis(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analisi tecnica avanzata"""
        try:
            signals = {
                'rsi_signal': 0,
                'macd_signal': 0,
                'bb_signal': 0,
                'volume_signal': 0,
                'pattern_signal': 0
            }
            
            # RSI
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            if rsi[-1] < 30:
                signals['rsi_signal'] = 1  # Buy
            elif rsi[-1] > 70:
                signals['rsi_signal'] = -1  # Sell
                
            # MACD
            macd, macdsignal, macdhist = talib.MACD(data['close'].values)
            if macd[-1] > macdsignal[-1] and macd[-2] <= macdsignal[-2]:
                signals['macd_signal'] = 1  # Buy
            elif macd[-1] < macdsignal[-1] and macd[-2] >= macdsignal[-2]:
                signals['macd_signal'] = -1  # Sell
                
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(data['close'].values)
            if data['close'].iloc[-1] <= lower[-1]:
                signals['bb_signal'] = 1  # Buy
            elif data['close'].iloc[-1] >= upper[-1]:
                signals['bb_signal'] = -1  # Sell
                
            # Volume analysis
            volume_sma = talib.SMA(data['volume'].values, timeperiod=20)
            if data['volume'].iloc[-1] > volume_sma[-1] * 1.5:
                signals['volume_signal'] = 1  # High volume confirmation
                
            # Pattern recognition
            doji = talib.CDLDOJI(data['open'].values, data['high'].values, 
                               data['low'].values, data['close'].values)
            hammer = talib.CDLHAMMER(data['open'].values, data['high'].values,
                                   data['low'].values, data['close'].values)
            
            if doji[-1] != 0 or hammer[-1] != 0:
                signals['pattern_signal'] = 1
                
            return signals
            
        except Exception as e:
            logger.error(f"Errore analisi tecnica {symbol}: {e}")
            return {}

    async def ml_prediction(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Predizione ML"""
        try:
            # Prepara features
            features = self.prepare_ml_features(data)
            
            if len(features) < 60:  # Dati insufficienti
                return {'signal': 0, 'confidence': 0}
            
            # Normalizza features
            features_scaled = self.ml_models['scaler'].fit_transform(features)
            
            # Predizione LSTM
            X = features_scaled[-60:].reshape(1, 60, -1)
            prediction = self.ml_models['lstm_predictor'].predict(X, verbose=0)[0][0]
            
            current_price = data['close'].iloc[-1]
            predicted_change = (prediction - current_price) / current_price
            
            # Classificazione segnale
            if predicted_change > 0.02:  # >2% aumento previsto
                signal = 1
                confidence = min(abs(predicted_change) * 10, 1.0)
            elif predicted_change < -0.02:  # >2% diminuzione prevista
                signal = -1
                confidence = min(abs(predicted_change) * 10, 1.0)
            else:
                signal = 0
                confidence = 0
                
            return {
                'signal': signal,
                'confidence': confidence,
                'predicted_price': prediction,
                'predicted_change': predicted_change
            }
            
        except Exception as e:
            logger.error(f"Errore ML prediction {symbol}: {e}")
            return {'signal': 0, 'confidence': 0}

    def prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepara features per ML"""
        features = []
        
        # Price features
        features.extend([
            data['close'].pct_change().fillna(0),
            data['high'].pct_change().fillna(0),
            data['low'].pct_change().fillna(0),
            data['volume'].pct_change().fillna(0)
        ])
        
        # Technical indicators
        features.append(pd.Series(talib.RSI(data['close'].values, 14)).fillna(50))
        
        macd, signal, hist = talib.MACD(data['close'].values)
        features.extend([
            pd.Series(macd).fillna(0),
            pd.Series(signal).fillna(0),
            pd.Series(hist).fillna(0)
        ])
        
        # Moving averages
        features.extend([
            data['close'] / data['close'].rolling(20).mean() - 1,
            data['close'] / data['close'].rolling(50).mean() - 1
        ])
        
        return np.column_stack(features)

    async def analyze_sentiment(self, symbol: str) -> Dict:
        """Analizza sentiment da news e social"""
        try:
            sentiment_score = 0
            news_count = 0
            
            # Cerca news
            news_data = await self.fetch_news(symbol)
            
            for article in news_data:
                blob = TextBlob(article['title'] + ' ' + article.get('description', ''))
                sentiment_score += blob.sentiment.polarity
                news_count += 1
            
            if news_count > 0:
                avg_sentiment = sentiment_score / news_count
            else:
                avg_sentiment = 0
                
            return {
                'sentiment_score': avg_sentiment,
                'news_count': news_count,
                'signal': 1 if avg_sentiment > 0.1 else (-1 if avg_sentiment < -0.1 else 0)
            }
            
        except Exception as e:
            logger.error(f"Errore sentiment analysis {symbol}: {e}")
            return {'sentiment_score': 0, 'news_count': 0, 'signal': 0}

    async def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news da API"""
        try:
            # Implementa chiamata a News API o RSS feed
            # Placeholder per ora
            return []
        except Exception as e:
            logger.error(f"Errore fetch news {symbol}: {e}")
            return []

    async def combine_signals(self, technical: Dict, ml: Dict, sentiment: Dict) -> Dict:
        """Combina tutti i segnali con pesi"""
        try:
            weights = {
                'technical': 0.4,
                'ml': 0.4,
                'sentiment': 0.2
            }
            
            # Calcola segnale tecnico medio
            tech_signals = [v for v in technical.values() if isinstance(v, (int, float))]
            tech_signal = sum(tech_signals) / len(tech_signals) if tech_signals else 0
            
            # Combina segnali
            combined_signal = (
                tech_signal * weights['technical'] +
                ml['signal'] * weights['ml'] +
                sentiment['signal'] * weights['sentiment']
            )
            
            # Calcola confidence
            confidence = (
                abs(tech_signal) * weights['technical'] +
                ml['confidence'] * weights['ml'] +
                abs(sentiment['sentiment_score']) * weights['sentiment']
            )
            
            return {
                'signal': 1 if combined_signal > 0.3 else (-1 if combined_signal < -0.3 else 0),
                'confidence': min(confidence, 1.0),
                'technical_signal': tech_signal,
                'ml_signal': ml['signal'],
                'sentiment_signal': sentiment['signal']
            }
            
        except Exception as e:
            logger.error(f"Errore combine signals: {e}")
            return {'signal': 0, 'confidence': 0}

    async def save_signal(self, symbol: str, signal: Dict):
        """Salva segnale nel database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    '''INSERT INTO signals (symbol, signal_type, price, confidence, strategy) 
                       VALUES ($1, $2, $3, $4, $5)''',
                    symbol,
                    'BUY' if signal['signal'] > 0 else 'SELL',
                    await self.get_current_price(symbol),
                    signal['confidence'],
                    'combined'
                )
        except Exception as e:
            logger.error(f"Errore save signal: {e}")

    async def notify_subscribers(self, symbol: str, signal: Dict):
        """Notifica gli utenti iscritti"""
        try:
            if symbol in self.user_subscriptions:
                for user_id, channel_id in self.user_subscriptions[symbol]:
                    channel = self.get_channel(channel_id)
                    if channel:
                        embed = self.create_signal_embed(symbol, signal)
                        await channel.send(f"<@{user_id}>", embed=embed)
        except Exception as e:
            logger.error(f"Errore notify subscribers: {e}")

    def create_signal_embed(self, symbol: str, signal: Dict) -> discord.Embed:
        """Crea embed per segnale"""
        color = 0x00ff00 if signal['signal'] > 0 else 0xff0000
        signal_type = "🟢 BUY" if signal['signal'] > 0 else "🔴 SELL"
        
        embed = discord.Embed(
            title=f"📊 {signal_type} Signal - {symbol}",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(
            name="Confidence",
            value=f"{signal['confidence']:.2%}",
            inline=True
        )
        
        embed.add_field(
            name="Strategy",
            value="AI Combined Analysis",
            inline=True
        )
        
        return embed

    async def get_current_price(self, symbol: str) -> float:
        """Ottiene prezzo attuale"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Errore get price {symbol}: {e}")
            return None

    # === COMANDI DISCORD ===

    @commands.command(name='price')
    async def price_command(self, ctx, symbol: str):
        """Mostra prezzo attuale di un asset"""
        try:
            price = await self.get_current_price(symbol.upper())
            
            if price:
                ticker = self.exchange.fetch_ticker(symbol.upper())
                change_24h = ticker['percentage']
                
                color = 0x00ff00 if change_24h >= 0 else 0xff0000
                
                embed = discord.Embed(
                    title=f"💰 {symbol.upper()} Price",
                    color=color,
                    timestamp=datetime.utcnow()
                )
                
                embed.add_field(name="Current Price", value=f"${price:.6f}", inline=True)
                embed.add_field(name="24h Change", value=f"{change_24h:.2f}%", inline=True)
                embed.add_field(name="High 24h", value=f"${ticker['high']:.6f}", inline=True)
                embed.add_field(name="Low 24h", value=f"${ticker['low']:.6f}", inline=True)
                embed.add_field(name="Volume 24h", value=f"{ticker['baseVolume']:,.0f}", inline=True)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Impossibile ottenere prezzo per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='signal')
    async def signal_command(self, ctx, symbol: str):
        """Mostra ultimo segnale per un asset"""
        try:
            async with self.db_pool.acquire() as conn:
                signal = await conn.fetchrow(
                    '''SELECT * FROM signals WHERE symbol = $1 
                       ORDER BY timestamp DESC LIMIT 1''',
                    symbol.upper()
                )
            
            if signal:
                embed = discord.Embed(
                    title=f"🎯 Latest Signal - {symbol.upper()}",
                    color=0x00ff00 if signal['signal_type'] == 'BUY' else 0xff0000,
                    timestamp=signal['timestamp']
                )
                
                embed.add_field(name="Signal", value=signal['signal_type'], inline=True)
                embed.add_field(name="Confidence", value=f"{signal['confidence']:.2%}", inline=True)
                embed.add_field(name="Price", value=f"${signal['price']:.6f}", inline=True)
                embed.add_field(name="Strategy", value=signal['strategy'], inline=True)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Nessun segnale trovato per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='subscribe')
    async def subscribe_command(self, ctx, symbol: str):
        """Iscriviti alle notifiche per un asset"""
        try:
            symbol = symbol.upper()
            user_id = ctx.author.id
            channel_id = ctx.channel.id
            
            if symbol not in self.user_subscriptions:
                self.user_subscriptions[symbol] = []
                
            if (user_id, channel_id) not in self.user_subscriptions[symbol]:
                self.user_subscriptions[symbol].append((user_id, channel_id))
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        '''INSERT INTO subscriptions (user_id, symbol, channel_id) 
                           VALUES ($1, $2, $3) ON CONFLICT DO NOTHING''',
                        user_id, symbol, channel_id
                    )
                
                await ctx.send(f"✅ Iscritto alle notifiche per {symbol}")
            else:
                await ctx.send(f"ℹ️ Sei già iscritto alle notifiche per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='unsubscribe')
    async def unsubscribe_command(self, ctx, symbol: str):
        """Disiscriviti dalle notifiche per un asset"""
        try:
            symbol = symbol.upper()
            user_id = ctx.author.id
            
            if symbol in self.user_subscriptions:
                self.user_subscriptions[symbol] = [
                    (uid, cid) for uid, cid in self.user_subscriptions[symbol] 
                    if uid != user_id
                ]
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        '''DELETE FROM subscriptions 
                           WHERE user_id = $1 AND symbol = $2''',
                        user_id, symbol
                    )
                
                await ctx.send(f"✅ Disiscritto dalle notifiche per {symbol}")
            else:
                await ctx.send(f"ℹ️ Non eri iscritto alle notifiche per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='alert')
    async def alert_command(self, ctx, symbol: str, target_price: float, direction: str = "above"):
        """Imposta alert di prezzo"""
        try:
            user_id = ctx.author.id
            symbol = symbol.upper()
            
            if user_id not in self.price_alerts:
                self.price_alerts[user_id] = []
            
            alert = {
                'symbol': symbol,
                'target_price': target_price,
                'direction': direction.lower(),
                'created_at': datetime.utcnow()
            }
            
            self.price_alerts[user_id].append(alert)
            
            await ctx.send(
                f"✅ Alert impostato per {symbol} {direction} ${target_price:.6f}"
            )
            
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='trend')
    async def trend_command(self, ctx, symbol: str):
        """Analizza trend di un asset"""
        try:
            symbol = symbol.upper()
            data = await self.get_historical_data(symbol, '1h', 100)
            
            if data is not None:
                # Calcola moving averages
                data['MA20'] = data['close'].rolling(20).mean()
                data['MA50'] = data['close'].rolling(50).mean()
                
                current_price = data['close'].iloc[-1]
                ma20 = data['MA20'].iloc[-1]
                ma50 = data['MA50'].iloc[-1]
                
                # Determina trend
                if current_price > ma20 > ma50:
                    trend = "🟢 Strong Bullish"
                    color = 0x00ff00
                elif current_price > ma20:
                    trend = "🟡 Bullish"
                    color = 0xffff00
                elif current_price < ma20 < ma50:
                    trend = "🔴 Strong Bearish"
                    color = 0xff0000
                elif current_price < ma20:
                    trend = "🟠 Bearish"
                    color = 0xff8000
                else:
                    trend = "⚪ Sideways"
                    color = 0x808080
                
                embed = discord.Embed(
                    title=f"📈 Trend Analysis - {symbol}",
                    color=color,
                    timestamp=datetime.utcnow()
                )
                
                embed.add_field(name="Current Trend", value=trend, inline=True)
                embed.add_field(name="Current Price", value=f"${current_price:.6f}", inline=True)
                embed.add_field(name="MA20", value=f"${ma20:.6f}", inline=True)
                embed.add_field(name="MA50", value=f"${ma50:.6f}", inline=True)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Impossibile ottenere dati per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='performance')
    async def performance_command(self, ctx):
        """Mostra statistiche performance del bot"""
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END) as buy_signals,
                        COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END) as sell_signals,
                        AVG(confidence) as avg_confidence,
                        MAX(timestamp) as last_signal
                    FROM signals
                    WHERE timestamp >= NOW() - INTERVAL '30 days'
                ''')
            
            embed = discord.Embed(
                title="📊 Bot Performance (Last 30 Days)",
                color=0x0099ff,
                timestamp=datetime.utcnow()
            )
            
            if stats['total_signals']:
                embed.add_field(name="Total Signals", value=stats['total_signals'], inline=True)
                embed.add_field(name="Buy Signals", value=stats['buy_signals'], inline=True)
                embed.add_field(name="Sell Signals", value=stats['sell_signals'], inline=True)
                embed.add_field(name="Avg Confidence", value=f"{stats['avg_confidence']:.2%}", inline=True)
                embed.add_field(name="Last Signal", value=stats['last_signal'].strftime('%Y-%m-%d %H:%M'), inline=True)
            else:
                embed.description = "Nessun dato disponibile per gli ultimi 30 giorni"
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='help')
    async def help_command(self, ctx):
        """Mostra tutti i comandi disponibili"""
        embed = discord.Embed(
            title="🤖 Laminar Bot - Comandi Disponibili",
            description="Bot di trading avanzato con AI e Machine Learning",
            color=0x0099ff
        )
        
        # Info & Analisi
        embed.add_field(
            name="📊 Info & Analisi di Mercato",
            value="`/price [symbol]` - Prezzo attuale\n"
                  "`/trend [symbol]` - Analisi trend\n"
                  "`/volume [symbol]` - Analisi volume\n"
                  "`/news [symbol]` - News recenti\n"
                  "`/sentiment [symbol]` - Sentiment mercato",
            inline=False
        )
        
        # Segnali & Notifiche
        embed.add_field(
            name="🔔 Segnali & Notifiche",
            value="`/signal [symbol]` - Ultimo segnale\n"
                  "`/subscribe [symbol]` - Attiva notifiche\n"
                  "`/unsubscribe [symbol]` - Disattiva notifiche\n"
                  "`/alert [symbol] [price] [direction]` - Alert prezzo",
            inline=False
        )
        
        # Trading & Gestione
        embed.add_field(
            name="⚙️ Gestione & Strategie",
            value="`/performance` - Statistiche bot\n"
                  "`/positions` - Posizioni aperte\n"
                  "`/backtest [symbol] [days]` - Test storico\n"
                  "`/risk [percent]` - Imposta rischio",
            inline=False
        )
        
        # AI Features
        embed.add_field(
            name="🧠 AI & Machine Learning",
            value="`/ai-suggest [symbol]` - Suggerimento AI\n"
                  "`/correlation [symbol1] [symbol2]` - Analisi correlazione\n"
                  "`/anomaly [symbol]` - Rilevamento anomalie\n"
                  "`/pattern [symbol]` - Pattern recognition",
            inline=False
        )
        
        await ctx.send(embed=embed)

    @commands.command(name='volume')
    async def volume_command(self, ctx, symbol: str):
        """Analisi volume di un asset"""
        try:
            symbol = symbol.upper()
            data = await self.get_historical_data(symbol, '1h', 48)
            
            if data is not None:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].mean()
                volume_ratio = current_volume / avg_volume
                
                # Determina stato volume
                if volume_ratio > 2.0:
                    volume_status = "🔥 Extremely High"
                    color = 0xff0000
                elif volume_ratio > 1.5:
                    volume_status = "📈 High"
                    color = 0xff8000
                elif volume_ratio > 0.8:
                    volume_status = "✅ Normal"
                    color = 0x00ff00
                else:
                    volume_status = "📉 Low"
                    color = 0x808080
                
                embed = discord.Embed(
                    title=f"📊 Volume Analysis - {symbol}",
                    color=color,
                    timestamp=datetime.utcnow()
                )
                
                embed.add_field(name="Current Volume", value=f"{current_volume:,.0f}", inline=True)
                embed.add_field(name="24h Average", value=f"{avg_volume:,.0f}", inline=True)
                embed.add_field(name="Volume Ratio", value=f"{volume_ratio:.2f}x", inline=True)
                embed.add_field(name="Status", value=volume_status, inline=False)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Impossibile ottenere dati volume per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='news')
    async def news_command(self, ctx, symbol: str):
        """Mostra news recenti per un asset"""
        try:
            symbol = symbol.upper()
            news_data = await self.fetch_comprehensive_news(symbol)
            
            if news_data:
                embed = discord.Embed(
                    title=f"📰 Latest News - {symbol}",
                    color=0x0099ff,
                    timestamp=datetime.utcnow()
                )
                
                for i, article in enumerate(news_data[:5]):
                    embed.add_field(
                        name=f"📄 {article['title'][:50]}...",
                        value=f"[Read More]({article['url']})\n"
                              f"*{article['published']}*",
                        inline=False
                    )
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Nessuna news trovata per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='sentiment')
    async def sentiment_command(self, ctx, symbol: str):
        """Analisi sentiment per un asset"""
        try:
            symbol = symbol.upper()
            sentiment_data = await self.analyze_sentiment(symbol)
            
            score = sentiment_data['sentiment_score']
            
            if score > 0.3:
                sentiment_status = "😍 Very Bullish"
                color = 0x00ff00
            elif score > 0.1:
                sentiment_status = "😊 Bullish"
                color = 0x90EE90
            elif score > -0.1:
                sentiment_status = "😐 Neutral"
                color = 0x808080
            elif score > -0.3:
                sentiment_status = "😞 Bearish"
                color = 0xff8000
            else:
                sentiment_status = "😱 Very Bearish"
                color = 0xff0000
            
            embed = discord.Embed(
                title=f"💭 Sentiment Analysis - {symbol}",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(name="Sentiment Score", value=f"{score:.3f}", inline=True)
            embed.add_field(name="Status", value=sentiment_status, inline=True)
            embed.add_field(name="News Analyzed", value=sentiment_data['news_count'], inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='positions')
    async def positions_command(self, ctx):
        """Mostra posizioni aperte dell'utente"""
        try:
            user_id = ctx.author.id
            
            async with self.db_pool.acquire() as conn:
                positions = await conn.fetch('''
                    SELECT * FROM positions 
                    WHERE user_id = $1 AND status = 'OPEN'
                    ORDER BY created_at DESC
                ''', user_id)
            
            if positions:
                embed = discord.Embed(
                    title="💼 Your Open Positions",
                    color=0x0099ff,
                    timestamp=datetime.utcnow()
                )
                
                total_pnl = 0
                for pos in positions:
                    current_price = await self.get_current_price(pos['symbol'])
                    if current_price:
                        if pos['side'] == 'BUY':
                            pnl = (current_price - pos['entry_price']) * pos['quantity']
                        else:
                            pnl = (pos['entry_price'] - current_price) * pos['quantity']
                        
                        total_pnl += pnl
                        pnl_percent = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
                        
                        embed.add_field(
                            name=f"{pos['side']} {pos['symbol']}",
                            value=f"Qty: {pos['quantity']}\n"
                                  f"Entry: ${pos['entry_price']:.6f}\n"
                                  f"Current: ${current_price:.6f}\n"
                                  f"PnL: ${pnl:.2f} ({pnl_percent:+.2f}%)",
                            inline=True
                        )
                
                embed.add_field(
                    name="💰 Total PnL",
                    value=f"${total_pnl:.2f}",
                    inline=False
                )
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("ℹ️ Nessuna posizione aperta")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='backtest')
    async def backtest_command(self, ctx, symbol: str, days: int = 30):
        """Esegue backtest della strategia"""
        try:
            symbol = symbol.upper()
            
            # Ottieni dati storici più estesi
            data = await self.get_historical_data(symbol, '1h', days * 24)
            
            if data is not None and len(data) > 100:
                results = await self.run_backtest(symbol, data)
                
                embed = discord.Embed(
                    title=f"📊 Backtest Results - {symbol}",
                    description=f"Period: {days} days",
                    color=0x00ff00 if results['total_return'] > 0 else 0xff0000,
                    timestamp=datetime.utcnow()
                )
                
                embed.add_field(name="Total Return", value=f"{results['total_return']:.2f}%", inline=True)
                embed.add_field(name="Win Rate", value=f"{results['win_rate']:.1f}%", inline=True)
                embed.add_field(name="Total Trades", value=results['total_trades'], inline=True)
                embed.add_field(name="Sharpe Ratio", value=f"{results['sharpe_ratio']:.2f}", inline=True)
                embed.add_field(name="Max Drawdown", value=f"{results['max_drawdown']:.2f}%", inline=True)
                embed.add_field(name="Avg Trade", value=f"{results['avg_trade']:.2f}%", inline=True)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Dati insufficienti per backtest di {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    async def run_backtest(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Esegue backtest della strategia AI"""
        try:
            trades = []
            position = None
            portfolio_value = 10000  # Starting capital
            initial_value = portfolio_value
            
            for i in range(100, len(data)):  # Inizia dopo 100 candle per indicatori
                current_data = data.iloc[:i+1]
                
                # Genera segnale per questo punto storico
                technical_signals = await self.technical_analysis(symbol, current_data.tail(100))
                ml_signal = {'signal': 0, 'confidence': 0.5}  # Simplified for backtest
                sentiment = {'signal': 0, 'sentiment_score': 0}
                
                combined = await self.combine_signals(technical_signals, ml_signal, sentiment)
                
                current_price = data.iloc[i]['close']
                
                # Logic di trading
                if combined['confidence'] > 0.6:
                    if combined['signal'] > 0 and position is None:
                        # Buy signal
                        position = {
                            'type': 'BUY',
                            'entry_price': current_price,
                            'entry_time': data.iloc[i]['timestamp'],
                            'quantity': portfolio_value * 0.02 / current_price  # 2% risk per trade
                        }
                    elif combined['signal'] < 0 and position is not None:
                        # Sell signal
                        pnl_percent = (current_price - position['entry_price']) / position['entry_price'] * 100
                        portfolio_value += portfolio_value * 0.02 * (pnl_percent / 100)
                        
                        trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl_percent': pnl_percent,
                            'duration': (data.iloc[i]['timestamp'] - position['entry_time']).total_seconds() / 3600
                        })
                        
                        position = None
            
            # Calcola statistiche
            if trades:
                winning_trades = [t for t in trades if t['pnl_percent'] > 0]
                win_rate = len(winning_trades) / len(trades) * 100
                
                total_return = (portfolio_value - initial_value) / initial_value * 100
                avg_trade = sum([t['pnl_percent'] for t in trades]) / len(trades)
                
                # Calcola Sharpe ratio semplificato
                returns = [t['pnl_percent'] for t in trades]
                if len(returns) > 1:
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Max drawdown
                running_max = initial_value
                max_drawdown = 0
                for trade in trades:
                    portfolio_value_at_trade = initial_value * (1 + sum([t['pnl_percent']/100 for t in trades[:trades.index(trade)+1]]))
                    if portfolio_value_at_trade > running_max:
                        running_max = portfolio_value_at_trade
                    else:
                        drawdown = (running_max - portfolio_value_at_trade) / running_max * 100
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                
                return {
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'total_trades': len(trades),
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'avg_trade': avg_trade
                }
            else:
                return {
                    'total_return': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'avg_trade': 0
                }
                
        except Exception as e:
            logger.error(f"Errore backtest: {e}")
            return {}

    @commands.command(name='ai-suggest')
    async def ai_suggest_command(self, ctx, symbol: str):
        """Suggerimento AI per un asset"""
        try:
            symbol = symbol.upper()
            data = await self.get_historical_data(symbol, '1h', 100)
            
            if data is not None:
                # Analisi completa
                technical_signals = await self.technical_analysis(symbol, data)
                ml_signal = await self.ml_prediction(symbol, data)
                sentiment = await self.analyze_sentiment(symbol)
                combined = await self.combine_signals(technical_signals, ml_signal, sentiment)
                
                # Risk assessment
                volatility = data['close'].pct_change().std() * 100
                
                if volatility > 5:
                    risk_level = "🔴 High Risk"
                    risk_color = 0xff0000
                elif volatility > 3:
                    risk_level = "🟡 Medium Risk"
                    risk_color = 0xffff00
                else:
                    risk_level = "🟢 Low Risk"
                    risk_color = 0x00ff00
                
                # Generate AI suggestion
                if combined['confidence'] > 0.7:
                    if combined['signal'] > 0:
                        suggestion = "🚀 STRONG BUY"
                        action_color = 0x00ff00
                        reasoning = "Multiple indicators align for bullish momentum"
                    else:
                        suggestion = "📉 STRONG SELL"
                        action_color = 0xff0000
                        reasoning = "Multiple indicators suggest bearish pressure"
                elif combined['confidence'] > 0.5:
                    if combined['signal'] > 0:
                        suggestion = "📈 MODERATE BUY"
                        action_color = 0x90EE90
                        reasoning = "Some positive signals detected"
                    else:
                        suggestion = "📉 MODERATE SELL"
                        action_color = 0xff8000
                        reasoning = "Some negative signals detected"
                else:
                    suggestion = "⏸️ WAIT"
                    action_color = 0x808080
                    reasoning = "Mixed signals, wait for clearer direction"
                
                embed = discord.Embed(
                    title=f"🧠 AI Suggestion - {symbol}",
                    color=action_color,
                    timestamp=datetime.utcnow()
                )
                
                embed.add_field(name="🎯 Recommendation", value=suggestion, inline=True)
                embed.add_field(name="🎲 Confidence", value=f"{combined['confidence']:.1%}", inline=True)
                embed.add_field(name="⚠️ Risk Level", value=risk_level, inline=True)
                
                embed.add_field(name="📊 Technical Score", value=f"{combined.get('technical_signal', 0):.2f}", inline=True)
                embed.add_field(name="🤖 ML Score", value=f"{ml_signal['signal']}", inline=True)
                embed.add_field(name="💭 Sentiment", value=f"{sentiment['signal']}", inline=True)
                
                embed.add_field(name="📝 Reasoning", value=reasoning, inline=False)
                
                # Add stop loss and take profit suggestions
                current_price = await self.get_current_price(symbol)
                if current_price and combined['signal'] != 0:
                    if combined['signal'] > 0:  # Buy suggestion
                        stop_loss = current_price * 0.98  # 2% stop loss
                        take_profit = current_price * 1.04  # 4% take profit
                    else:  # Sell suggestion
                        stop_loss = current_price * 1.02  # 2% stop loss
                        take_profit = current_price * 0.96  # 4% take profit
                    
                    embed.add_field(name="🛡️ Suggested SL", value=f"${stop_loss:.6f}", inline=True)
                    embed.add_field(name="🎯 Suggested TP", value=f"${take_profit:.6f}", inline=True)
                    embed.add_field(name="💰 Current Price", value=f"${current_price:.6f}", inline=True)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Impossibile analizzare {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='correlation')
    async def correlation_command(self, ctx, symbol1: str, symbol2: str):
        """Analizza correlazione tra due asset"""
        try:
            symbol1 = symbol1.upper()
            symbol2 = symbol2.upper()
            
            # Ottieni dati per entrambi gli asset
            data1 = await self.get_historical_data(symbol1, '1h', 100)
            data2 = await self.get_historical_data(symbol2, '1h', 100)
            
            if data1 is not None and data2 is not None:
                # Allinea i timestamp
                merged = pd.merge(
                    data1[['timestamp', 'close']].rename(columns={'close': 'close1'}),
                    data2[['timestamp', 'close']].rename(columns={'close': 'close2'}),
                    on='timestamp'
                )
                
                if len(merged) > 20:
                    # Calcola correlazione sui returns
                    merged['return1'] = merged['close1'].pct_change()
                    merged['return2'] = merged['close2'].pct_change()
                    
                    correlation = merged['return1'].corr(merged['return2'])
                    
                    # Interpreta correlazione
                    if correlation > 0.8:
                        corr_strength = "🔥 Very Strong Positive"
                        color = 0x00ff00
                    elif correlation > 0.5:
                        corr_strength = "💪 Strong Positive"
                        color = 0x90EE90
                    elif correlation > 0.3:
                        corr_strength = "📈 Moderate Positive"
                        color = 0xffff00
                    elif correlation > -0.3:
                        corr_strength = "😐 Weak/No Correlation"
                        color = 0x808080
                    elif correlation > -0.5:
                        corr_strength = "📉 Moderate Negative"
                        color = 0xff8000
                    elif correlation > -0.8:
                        corr_strength = "💀 Strong Negative"
                        color = 0xff4500
                    else:
                        corr_strength = "🔥 Very Strong Negative"
                        color = 0xff0000
                    
                    embed = discord.Embed(
                        title=f"🔗 Correlation Analysis",
                        description=f"{symbol1} vs {symbol2}",
                        color=color,
                        timestamp=datetime.utcnow()
                    )
                    
                    embed.add_field(name="Correlation Coefficient", value=f"{correlation:.3f}", inline=True)
                    embed.add_field(name="Relationship Strength", value=corr_strength, inline=True)
                    embed.add_field(name="Data Points", value=len(merged), inline=True)
                    
                    # Aggiungi interpretazione
                    if abs(correlation) > 0.7:
                        interpretation = f"Asset highly {'correlated' if correlation > 0 else 'inversely correlated'} - consider for {'diversification' if correlation < 0 else 'momentum strategies'}"
                    elif abs(correlation) > 0.3:
                        interpretation = "Moderate relationship - can be useful for pair trading strategies"
                    else:
                        interpretation = "Low correlation - good for portfolio diversification"
                    
                    embed.add_field(name="💡 Trading Insight", value=interpretation, inline=False)
                    
                    await ctx.send(embed=embed)
                else:
                    await ctx.send("❌ Dati insufficienti per calcolare correlazione")
            else:
                await ctx.send("❌ Impossibile ottenere dati per uno o entrambi gli asset")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='anomaly')
    async def anomaly_command(self, ctx, symbol: str):
        """Rileva anomalie nei prezzi usando ML"""
        try:
            symbol = symbol.upper()
            data = await self.get_historical_data(symbol, '1h', 200)
            
            if data is not None and len(data) > 50:
                # Prepara features per anomaly detection
                features = []
                
                # Price features
                data['returns'] = data['close'].pct_change()
                data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                data['volatility'] = data['returns'].rolling(20).std()
                data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                
                # Technical features
                data['rsi'] = pd.Series(talib.RSI(data['close'].values, 14))
                macd, signal, hist = talib.MACD(data['close'].values)
                data['macd'] = pd.Series(macd)
                
                feature_cols = ['returns', 'log_returns', 'volatility', 'volume_ratio', 'rsi', 'macd']
                features_df = data[feature_cols].dropna()
                
                if len(features_df) > 30:
                    # Fit Isolation Forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_scores = iso_forest.fit_predict(features_df)
                    
                    # Get recent anomalies
                    recent_data = features_df.tail(24)  # Last 24 hours
                    recent_scores = anomaly_scores[-24:]
                    recent_anomalies = recent_data[recent_scores == -1]
                    
                    embed = discord.Embed(
                        title=f"🔍 Anomaly Detection - {symbol}",
                        color=0xff0000 if len(recent_anomalies) > 0 else 0x00ff00,
                        timestamp=datetime.utcnow()
                    )
                    
                    total_anomalies = sum(1 for score in anomaly_scores if score == -1)
                    anomaly_rate = total_anomalies / len(anomaly_scores) * 100
                    
                    embed.add_field(name="Recent Anomalies (24h)", value=len(recent_anomalies), inline=True)
                    embed.add_field(name="Total Anomaly Rate", value=f"{anomaly_rate:.1f}%", inline=True)
                    embed.add_field(name="Data Points Analyzed", value=len(features_df), inline=True)
                    
                    if len(recent_anomalies) > 0:
                        status = "⚠️ ANOMALIES DETECTED"
                        warning = "Unusual market behavior detected in recent data. Exercise caution."
                    else:
                        status = "✅ NORMAL BEHAVIOR"
                        warning = "No significant anomalies in recent trading patterns."
                    
                    embed.add_field(name="Status", value=status, inline=False)
                    embed.add_field(name="⚠️ Warning", value=warning, inline=False)
                    
                    # Current market state
                    current_volatility = data['volatility'].iloc[-1]
                    avg_volatility = data['volatility'].mean()
                    
                    if current_volatility > avg_volatility * 2:
                        market_state = "🔥 High Volatility"
                    elif current_volatility < avg_volatility * 0.5:
                        market_state = "😴 Low Volatility"
                    else:
                        market_state = "📊 Normal Volatility"
                    
                    embed.add_field(name="Market State", value=market_state, inline=True)
                    
                    await ctx.send(embed=embed)
                else:
                    await ctx.send("❌ Dati insufficienti per anomaly detection")
            else:
                await ctx.send(f"❌ Impossibile ottenere dati sufficienti per {symbol}")
                
        except Exception as e:
            await ctx.send(f"❌ Errore: {str(e)}")

    @commands.command(name='pattern')
    async def pattern_command(self, ctx, symbol: str):
        """Riconosce pattern di candlestick"""
        try:
            symbol = symbol.upper()
            data = await self.get_historical_data(symbol, '1h', 50)
            
            if data is not None and len(data) > 20:
                # Pattern recognition usando TA-Lib
                patterns = {}
                
                # Pattern bullish
                patterns['Hammer'] = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])
                patterns['Morning Star'] = talib.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'])
                patterns['Bullish Engulfing'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])
                patterns['Piercing Pattern'] = talib.CDLPIERCING(data['open'], data['high'], data['low'], data['close'])
                
                # Pattern bearish  
                patterns['Shooting Star'] = talib.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])
                patterns['Evening Star'] = talib.CDLEVENINGSTAR(data['open'], data['high'], data['low'], data['close'])
                patterns['Dark Cloud Cover'] = talib.CDLDARKCLOUDCOVER(data['open'], data['high'], data['low'], data['close'])
                patterns['Hanging Man'] = talib.CDLHANGINGMAN(data['open'], data['high'], data['low'], data['close'])
                
                # Pattern neutri/indecisione
                patterns['Doji'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
                patterns['Spinning Top'] = talib.CDLSPINNINGTOP(data['open'], data['high'], data['low'], data['close'])
                
                # Trova pattern attivi
                active_patterns = []
                recent_patterns = []
                
                for pattern_name, pattern_data in patterns.items():
                    if len(pattern_data) > 0:
                        # Controlla ultimi 5 periodi
                        recent_signals = pattern_data[-5:]
                        if any(signal != 0 for signal in recent_signals):
                            # Trova la candela più recente con il pattern
                            for i in range(len(recent_signals)-1, -1, -1):
                                if recent_signals[i] != 0:
                                    strength = "Strong" if abs(recent_signals[i]) >= 100 else "Moderate"
                                    direction = "Bullish" if recent_signals[i] > 0 else "Bearish"
                                    periods_ago = len(recent_signals) - 1 - i
                                    
                                    recent_patterns.append({
                                        'name': pattern_name,
                                        'direction': direction,
                                        'strength': strength,
                                        'periods_ago': periods_ago,
                                        'value': recent_signals[i]
                                    })
                                    break
