#!/usr/bin/env python3
"""
Name of Service: TRADING SYSTEM NEWS COLLECTION SERVICE
Version: 2.0.0
Last Updated: 2025-06-27
Purpose: Raw news data collection from multiple sources for future ML analysis

REVISION HISTORY:
v2.0.0 (2025-06-27) - Complete rewrite for raw data collection
- Multiple news source integration
- Raw data storage (no processing)
- Cloud-ready architecture
- Prepared for DigitalOcean migration

This service collects news without interpretation, building a data lake
for future pattern analysis.

KEY FEATURES:
- Multiple sources: NewsAPI, AlphaVantage, RSS feeds
- Raw data collection with rich metadata
- Market state tracking (pre-market, regular, after-hours)
- Headline keyword extraction (earnings, FDA, merger, etc.)
- Ticker extraction from article text
- Breaking news detection
- Update tracking (how often stories are updated)
- Trending news identification
- Search API for analysis services
- Pre-market focused collection scheduling

FUTURE ML SUPPORT:
All data is stored raw for future ML processing:
- Pattern discovery: Which keywords lead to price movements
- Timing analysis: When news has maximum impact
- Update patterns: How story evolution affects trading
- Ticker correlations: Multi-symbol impact analysis
"""

import os
import json
import time
import sqlite3
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import database utilities if available
try:
    from database_utils_old import DatabaseServiceMixin
    USE_DB_UTILS = True
except ImportError:
    USE_DB_UTILS = False
    print("Warning: database_utils not found, using direct SQLite connections")


class NewsCollectionService(DatabaseServiceMixin if USE_DB_UTILS else object):
    """
    News Collection Service - Gathers raw news data from multiple sources
    No analysis, no sentiment, just pure data collection
    """
    
    def __init__(self, db_path='/tmp//tmp/trading_system.db'):
        if USE_DB_UTILS:
            super().__init__(db_path)
        else:
            self.db_path = db_path
            
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # API Keys (will be environment variables in production)
        self.api_keys = {
            'newsapi': os.getenv('NEWSAPI_KEY', ''),
            'alphavantage': os.getenv('ALPHAVANTAGE_KEY', ''),
            'finnhub': os.getenv('FINNHUB_KEY', '')
        }
        
        # Free RSS feeds (no API key needed)
        self.rss_feeds = {
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'yahoo_finance': 'https://finance.yahoo.com/rss/',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'investing_com': 'https://www.investing.com/rss/news.rss',
            'reuters_business': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best'
        }
        
        # Initialize database schema
        self._init_database_schema()
        
        self.logger.info("News Collection Service v2.0.0 initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('news_collection')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # File handler for persistent logs
        fh = logging.FileHandler('/tmp//tmp/logs/news_collection.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def _init_database_schema(self):
        """Initialize database tables for raw news storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Raw news data table - Enhanced for maximum data collection
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id TEXT UNIQUE NOT NULL,  -- Hash of headline+source+timestamp
                    symbol TEXT,
                    headline TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_url TEXT,
                    published_timestamp TIMESTAMP NOT NULL,
                    collected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_snippet TEXT,
                    full_url TEXT,
                    metadata JSON,  -- Store any extra fields from APIs
                    is_pre_market BOOLEAN DEFAULT FALSE,
                    market_state TEXT,  -- pre-market, regular, after-hours, weekend
                    headline_keywords JSON,  -- ["earnings", "fda", "merger", etc]
                    mentioned_tickers JSON,  -- Other tickers found in article
                    article_length INTEGER,  -- Character count
                    is_breaking_news BOOLEAN DEFAULT FALSE,
                    update_count INTEGER DEFAULT 0,  -- How many times we've seen updates
                    first_seen_timestamp TIMESTAMP,  -- When we first saw this story
                    UNIQUE(headline, source, published_timestamp)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_news_symbol 
                ON news_raw(symbol)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_news_timestamp 
                ON news_raw(published_timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_news_premarket 
                ON news_raw(is_pre_market)
            ''')
            
            # Collection stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_collection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL,
                    articles_collected INTEGER DEFAULT 0,
                    articles_new INTEGER DEFAULT 0,
                    articles_duplicate INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    metadata JSON
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database schema: {e}")
            raise
            
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy", 
                "service": "news_collection",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat()
            })
            
        @self.app.route('/collect_news', methods=['POST'])
        def collect_news():
            """Manually trigger news collection"""
            data = request.json or {}
            symbols = data.get('symbols', None)
            sources = data.get('sources', 'all')
            
            result = self.collect_all_news(symbols, sources)
            return jsonify(result)
            
        @self.app.route('/news_stats', methods=['GET'])
        def news_stats():
            """Get collection statistics"""
            hours = request.args.get('hours', 24, type=int)
            stats = self._get_collection_stats(hours)
            return jsonify(stats)
            
        @self.app.route('/search_news', methods=['GET'])
        def search_news():
            """Search news by various criteria"""
            params = {
                'symbol': request.args.get('symbol'),
                'keywords': request.args.getlist('keywords'),
                'market_state': request.args.get('market_state'),
                'breaking_only': request.args.get('breaking_only', 'false').lower() == 'true',
                'hours': request.args.get('hours', 24, type=int)
            }
            results = self._search_news(params)
            return jsonify(results)
            
        @self.app.route('/trending_news', methods=['GET'])
        def trending_news():
            """Get trending news (most updated stories)"""
            hours = request.args.get('hours', 4, type=int)
            limit = request.args.get('limit', 20, type=int)
            results = self._get_trending_news(hours, limit)
            return jsonify(results)
            
    def generate_news_id(self, headline: str, source: str, timestamp: str) -> str:
        """Generate unique ID for news article"""
        content = f"{headline}_{source}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def is_pre_market_news(self, timestamp: datetime) -> bool:
        """Check if news was published during pre-market hours (4 AM - 9:30 AM EST)"""
        # Convert to EST (assuming system is in UTC)
        est_hour = timestamp.hour - 5  # Rough EST conversion
        if est_hour < 0:
            est_hour += 24
            
        return 4 <= est_hour < 9.5
        
    def get_market_state(self, timestamp: datetime) -> str:
        """Determine market state when news was published"""
        # Convert to EST
        est_hour = timestamp.hour - 5
        if est_hour < 0:
            est_hour += 24
        
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend
        if weekday >= 5:  # Saturday or Sunday
            return "weekend"
        
        # Weekday market states
        if 4 <= est_hour < 9.5:
            return "pre-market"
        elif 9.5 <= est_hour < 16:
            return "regular"
        elif 16 <= est_hour < 20:
            return "after-hours"
        else:
            return "closed"
            
    def extract_headline_keywords(self, headline: str) -> List[str]:
        """Extract important keywords from headline (raw detection, no analysis)"""
        keywords = []
        headline_lower = headline.lower()
        
        # Financial event keywords
        keyword_patterns = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'beat', 'miss', 'eps'],
            'fda': ['fda', 'approval', 'drug', 'clinical', 'trial', 'phase'],
            'merger': ['merger', 'acquisition', 'acquire', 'buyout', 'takeover', 'deal'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'price target', 'analyst'],
            'insider': ['insider', 'ceo', 'cfo', 'director', 'executive'],
            'legal': ['lawsuit', 'settlement', 'investigation', 'sec', 'fraud'],
            'product': ['launch', 'release', 'announce', 'unveil', 'introduce'],
            'guidance': ['guidance', 'forecast', 'outlook', 'warns', 'expects'],
            'partnership': ['partnership', 'collaboration', 'joint venture', 'agreement'],
            'ipo': ['ipo', 'public offering', 'listing', 'debut'],
            'bankruptcy': ['bankruptcy', 'chapter 11', 'restructuring', 'default'],
            'dividend': ['dividend', 'yield', 'payout', 'distribution']
        }
        
        for category, patterns in keyword_patterns.items():
            if any(pattern in headline_lower for pattern in patterns):
                keywords.append(category)
                
        return keywords
        
    def extract_mentioned_tickers(self, text: str) -> List[str]:
        """Extract stock tickers mentioned in text (basic pattern matching)"""
        import re
        
        # Pattern for stock tickers: 1-5 uppercase letters, possibly preceded by $
        ticker_pattern = r'\$?[A-Z]{1,5}\b'
        
        # Common words to exclude that might match pattern
        exclusions = {'I', 'A', 'THE', 'AND', 'OR', 'TO', 'IN', 'OF', 'FOR', 
                     'CEO', 'CFO', 'IPO', 'FDA', 'SEC', 'NYSE', 'ETF'}
        
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter out common words and return unique tickers
        tickers = []
        for ticker in potential_tickers:
            ticker = ticker.replace('
        
    def collect_newsapi_data(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Collect news from NewsAPI.org"""
        if not self.api_keys['newsapi']:
            self.logger.warning("NewsAPI key not configured")
            return []
            
        collected_news = []
        base_url = "https://newsapi.org/v2/everything"
        
        # If no symbols specified, get general market news
        queries = symbols if symbols else ['stock market', 'trading', 'NYSE', 'NASDAQ']
        
        for query in queries[:5]:  # Limit to conserve free tier
            try:
                params = {
                    'apiKey': self.api_keys['newsapi'],
                    'q': query,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'from': (datetime.now() - timedelta(days=1)).isoformat()
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        published = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                        
                        news_item = {
                            'symbol': query if query in symbols else None,
                            'headline': article.get('title', ''),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'source_url': article.get('url'),
                            'published_timestamp': published,
                            'content_snippet': article.get('description', '')[:500],
                            'full_url': article.get('url'),
                            'is_pre_market': self.is_pre_market_news(published),
                            'market_state': self.get_market_state(published),
                            'headline_keywords': self.extract_headline_keywords(article.get('title', '')),
                            'mentioned_tickers': self.extract_mentioned_tickers(
                                article.get('title', '') + ' ' + article.get('description', '')
                            ),
                            'article_length': len(article.get('content', article.get('description', ''))),
                            'is_breaking_news': self.is_breaking_news(article.get('title', ''), published),
                            'metadata': {
                                'author': article.get('author'),
                                'image_url': article.get('urlToImage'),
                                'content_length': len(article.get('content', ''))
                            }
                        }
                        collected_news.append(news_item)
                        
                else:
                    self.logger.error(f"NewsAPI error: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting from NewsAPI: {e}")
                
        return collected_news
        
    def collect_rss_feeds(self) -> List[Dict]:
        """Collect news from RSS feeds"""
        collected_news = []
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit entries per feed
                    # Parse published date
                    published = None
                    if hasattr(entry, 'published_parsed'):
                        published = datetime.fromtimestamp(
                            time.mktime(entry.published_parsed)
                        )
                    else:
                        published = datetime.now()
                        
                    news_item = {
                        'symbol': None,  # RSS feeds don't typically have symbols
                        'headline': entry.get('title', ''),
                        'source': source_name,
                        'source_url': feed_url,
                        'published_timestamp': published,
                        'content_snippet': entry.get('summary', '')[:500],
                        'full_url': entry.get('link'),
                        'is_pre_market': self.is_pre_market_news(published),
                        'market_state': self.get_market_state(published),
                        'headline_keywords': self.extract_headline_keywords(entry.get('title', '')),
                        'mentioned_tickers': self.extract_mentioned_tickers(
                            entry.get('title', '') + ' ' + entry.get('summary', '')
                        ),
                        'article_length': len(entry.get('summary', '')),
                        'is_breaking_news': self.is_breaking_news(entry.get('title', ''), published),
                        'metadata': {
                            'tags': [tag.term for tag in entry.get('tags', [])] if hasattr(entry, 'tags') else []
                        }
                    }
                    collected_news.append(news_item)
                    
            except Exception as e:
                self.logger.error(f"Error collecting from {source_name}: {e}")
                
        return collected_news
        
    def collect_alphavantage_news(self, symbols: List[str]) -> List[Dict]:
        """Collect news from Alpha Vantage"""
        if not self.api_keys['alphavantage'] or not symbols:
            return []
            
        collected_news = []
        base_url = "https://www.alphavantage.co/query"
        
        for symbol in symbols[:5]:  # Limit API calls
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.api_keys['alphavantage']
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('feed', []):
                        published = datetime.strptime(
                            article['time_published'], 
                            '%Y%m%dT%H%M%S'
                        )
                        
                        news_item = {
                            'symbol': symbol,
                            'headline': article.get('title', ''),
                            'source': article.get('source', 'AlphaVantage'),
                            'source_url': article.get('source_domain'),
                            'published_timestamp': published,
                            'content_snippet': article.get('summary', '')[:500],
                            'full_url': article.get('url'),
                            'is_pre_market': self.is_pre_market_news(published),
                            'market_state': self.get_market_state(published),
                            'headline_keywords': self.extract_headline_keywords(article.get('title', '')),
                            'mentioned_tickers': self.extract_mentioned_tickers(
                                article.get('title', '') + ' ' + article.get('summary', '')
                            ),
                            'article_length': len(article.get('summary', '')),
                            'is_breaking_news': self.is_breaking_news(article.get('title', ''), published),
                            'metadata': {
                                'authors': article.get('authors', []),
                                'topics': article.get('topics', []),
                                'ticker_sentiment': article.get('ticker_sentiment', {})
                            }
                        }
                        collected_news.append(news_item)
                        
            except Exception as e:
                self.logger.error(f"Error collecting AlphaVantage news for {symbol}: {e}")
                
        return collected_news
        
    def save_news_items(self, news_items: List[Dict]) -> Dict[str, int]:
        """Save news items to database"""
        stats = {'total': len(news_items), 'saved': 0, 'duplicates': 0, 'errors': 0}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in news_items:
            try:
                # Generate unique ID
                news_id = self.generate_news_id(
                    item['headline'],
                    item['source'],
                    str(item['published_timestamp'])
                )
                
                # Prepare JSON fields
                metadata_json = json.dumps(item.get('metadata', {}))
                keywords_json = json.dumps(item.get('headline_keywords', []))
                tickers_json = json.dumps(item.get('mentioned_tickers', []))
                
                # Check if this is an update to existing news
                cursor.execute('''
                    SELECT id, update_count, first_seen_timestamp 
                    FROM news_raw 
                    WHERE headline LIKE ? AND source = ?
                    LIMIT 1
                ''', (item['headline'][:50] + '%', item['source']))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute('''
                        UPDATE news_raw 
                        SET update_count = update_count + 1,
                            metadata = ?,
                            article_length = ?
                        WHERE id = ?
                    ''', (metadata_json, item.get('article_length', 0), existing[0]))
                else:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO news_raw 
                        (news_id, symbol, headline, source, source_url, 
                         published_timestamp, content_snippet, full_url, 
                         metadata, is_pre_market, market_state, headline_keywords,
                         mentioned_tickers, article_length, is_breaking_news,
                         update_count, first_seen_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        news_id,
                        item.get('symbol'),
                        item['headline'],
                        item['source'],
                        item.get('source_url'),
                        item['published_timestamp'],
                        item.get('content_snippet'),
                        item.get('full_url'),
                        metadata_json,
                        item.get('is_pre_market', False),
                        item.get('market_state', 'unknown'),
                        keywords_json,
                        tickers_json,
                        item.get('article_length', 0),
                        item.get('is_breaking_news', False),
                        0,  # update_count starts at 0
                        datetime.now()  # first_seen_timestamp
                    ))
                
                if cursor.rowcount > 0:
                    stats['saved'] += 1
                else:
                    stats['duplicates'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error saving news item: {e}")
                stats['errors'] += 1
                
        conn.commit()
        conn.close()
        
        return stats
        
    def collect_all_news(self, symbols: Optional[List[str]] = None, 
                        sources: str = 'all') -> Dict:
        """Collect news from all configured sources"""
        start_time = datetime.now()
        all_news = []
        collection_stats = {}
        
        # Use ThreadPoolExecutor for concurrent collection
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            if sources in ['all', 'newsapi']:
                futures['newsapi'] = executor.submit(self.collect_newsapi_data, symbols)
                
            if sources in ['all', 'rss']:
                futures['rss'] = executor.submit(self.collect_rss_feeds)
                
            if sources in ['all', 'alphavantage'] and symbols:
                futures['alphavantage'] = executor.submit(
                    self.collect_alphavantage_news, symbols
                )
                
            # Collect results
            for source, future in futures.items():
                try:
                    news_items = future.result(timeout=30)
                    all_news.extend(news_items)
                    collection_stats[source] = len(news_items)
                except Exception as e:
                    self.logger.error(f"Error collecting from {source}: {e}")
                    collection_stats[source] = 0
                    
        # Save all collected news
        save_stats = self.save_news_items(all_news)
        
        # Log collection statistics
        self._log_collection_stats(collection_stats, save_stats)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'status': 'success',
            'execution_time': execution_time,
            'articles_collected': save_stats['total'],
            'articles_saved': save_stats['saved'],
            'duplicates': save_stats['duplicates'],
            'errors': save_stats['errors'],
            'sources': collection_stats,
            'timestamp': datetime.now().isoformat()
        }
        
    def _log_collection_stats(self, collection_stats: Dict, save_stats: Dict):
        """Log collection statistics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for source, count in collection_stats.items():
                cursor.execute('''
                    INSERT INTO news_collection_stats 
                    (source, articles_collected, articles_new, 
                     articles_duplicate, error_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    source,
                    count,
                    save_stats['saved'] if source == 'total' else 0,
                    save_stats['duplicates'] if source == 'total' else 0,
                    save_stats['errors'] if source == 'total' else 0,
                    json.dumps(save_stats)
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging statistics: {e}")
            
    def _get_collection_stats(self, hours: int = 24) -> Dict:
        """Get collection statistics for the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get stats by source
            cursor.execute('''
                SELECT 
                    source,
                    SUM(articles_collected) as total_collected,
                    SUM(articles_new) as total_new,
                    SUM(articles_duplicate) as total_duplicate,
                    COUNT(*) as collection_runs
                FROM news_collection_stats
                WHERE collection_timestamp > datetime('now', '-{} hours')
                GROUP BY source
            '''.format(hours))
            
            source_stats = {}
            for row in cursor.fetchall():
                source_stats[row[0]] = {
                    'collected': row[1] or 0,
                    'new': row[2] or 0,
                    'duplicate': row[3] or 0,
                    'runs': row[4] or 0
                }
                
            # Get total news count
            cursor.execute('''
                SELECT COUNT(*) FROM news_raw
                WHERE collected_timestamp > datetime('now', '-{} hours')
            '''.format(hours))
            
            total_news = cursor.fetchone()[0] or 0
            
            # Get pre-market news count
            cursor.execute('''
                SELECT COUNT(*) FROM news_raw
                WHERE collected_timestamp > datetime('now', '-{} hours')
                AND is_pre_market = 1
            '''.format(hours))
            
            pre_market_news = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'period_hours': hours,
                'total_articles': total_news,
                'pre_market_articles': pre_market_news,
                'sources': source_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
            
    def _search_news(self, params: Dict) -> Dict:
        """Search news by various criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = '''
                SELECT 
                    news_id, symbol, headline, source, published_timestamp,
                    content_snippet, full_url, is_pre_market, market_state,
                    headline_keywords, mentioned_tickers, article_length,
                    is_breaking_news, update_count
                FROM news_raw
                WHERE 1=1
            '''
            
            query_params = []
            
            # Add filters
            if params['symbol']:
                query += ' AND (symbol = ? OR mentioned_tickers LIKE ?)'
                query_params.extend([params['symbol'], f'%"{params["symbol"]}"%'])
                
            if params['keywords']:
                for keyword in params['keywords']:
                    query += ' AND headline_keywords LIKE ?'
                    query_params.append(f'%"{keyword}"%')
                    
            if params['market_state']:
                query += ' AND market_state = ?'
                query_params.append(params['market_state'])
                
            if params['breaking_only']:
                query += ' AND is_breaking_news = 1'
                
            # Time filter
            query += ' AND published_timestamp > datetime("now", "-{} hours")'.format(params['hours'])
            
            # Order by most recent
            query += ' ORDER BY published_timestamp DESC LIMIT 100'
            
            cursor.execute(query, query_params)
            
            # Format results
            results = []
            for row in cursor.fetchall():
                results.append({
                    'news_id': row['news_id'],
                    'symbol': row['symbol'],
                    'headline': row['headline'],
                    'source': row['source'],
                    'published': row['published_timestamp'],
                    'snippet': row['content_snippet'],
                    'url': row['full_url'],
                    'pre_market': bool(row['is_pre_market']),
                    'market_state': row['market_state'],
                    'keywords': json.loads(row['headline_keywords']),
                    'mentioned_tickers': json.loads(row['mentioned_tickers']),
                    'article_length': row['article_length'],
                    'breaking': bool(row['is_breaking_news']),
                    'updates': row['update_count']
                })
                
            conn.close()
            
            return {
                'count': len(results),
                'results': results,
                'query_params': params,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error searching news: {e}")
            return {'error': str(e), 'results': []}
            
    def _get_trending_news(self, hours: int, limit: int) -> Dict:
        """Get trending news based on update frequency"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get stories with most updates
            cursor.execute('''
                SELECT 
                    symbol, headline, source, MAX(published_timestamp) as latest_update,
                    MIN(first_seen_timestamp) as first_seen,
                    COUNT(*) as mention_count,
                    SUM(update_count) as total_updates,
                    GROUP_CONCAT(DISTINCT market_state) as market_states,
                    MAX(is_breaking_news) as has_breaking
                FROM news_raw
                WHERE published_timestamp > datetime('now', '-{} hours')
                GROUP BY substr(headline, 1, 50), source
                HAVING mention_count > 1 OR total_updates > 0
                ORDER BY mention_count DESC, total_updates DESC
                LIMIT ?
            '''.format(hours), (limit,))
            
            trending = []
            for row in cursor.fetchall():
                trending.append({
                    'symbol': row['symbol'],
                    'headline': row['headline'],
                    'source': row['source'],
                    'first_seen': row['first_seen'],
                    'latest_update': row['latest_update'],
                    'mention_count': row['mention_count'],
                    'total_updates': row['total_updates'],
                    'market_states': row['market_states'].split(','),
                    'has_breaking': bool(row['has_breaking'])
                })
                
            conn.close()
            
            return {
                'count': len(trending),
                'hours_window': hours,
                'trending': trending,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trending news: {e}")
            return {'error': str(e), 'trending': []}
            
    def run_scheduled_collection(self):
        """Run scheduled news collection - can be called by external scheduler"""
        self.logger.info("Starting scheduled news collection")
        
        # During pre-market (4 AM - 9:30 AM EST), collect more aggressively
        current_time = datetime.now()
        market_state = self.get_market_state(current_time)
        
        if market_state == 'pre-market':
            self.logger.info("Pre-market hours - aggressive collection mode")
            # Get top movers/pre-market actives if available
            symbols = self._get_active_symbols()  # TODO: Implement based on scanner
        else:
            symbols = None  # General market news
            
        # Collect from all sources
        result = self.collect_all_news(symbols, 'all')
        
        self.logger.info(f"Scheduled collection complete: {result}")
        return result
        
    def _get_active_symbols(self) -> List[str]:
        """Get active symbols from scanner service (placeholder for integration)"""
        # TODO: This will integrate with security_scanner service
        # For now, return empty list
        return []
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting News Collection Service v2.0.0 on port 5008")
        self.logger.info(f"Database path: {self.db_path}")
        self.logger.info(f"APIs configured: {[k for k, v in self.api_keys.items() if v]}")
        
        self.app.run(host='0.0.0.0', port=5008, debug=False)


if __name__ == "__main__":
    service = NewsCollectionService()
    service.run()
, '')
            if ticker not in exclusions and len(ticker) >= 2:
                tickers.append(ticker)
                
        return list(set(tickers))  # Remove duplicates
        
    def is_breaking_news(self, headline: str, published_time: datetime) -> bool:
        """Detect if this appears to be breaking news"""
        breaking_indicators = ['breaking', 'alert', 'urgent', 'just in', 
                              'developing', 'exclusive', 'flash']
        
        headline_lower = headline.lower()
        
        # Check for breaking news indicators
        has_breaking_word = any(indicator in headline_lower for indicator in breaking_indicators)
        
        # Check if very recent (within last hour)
        time_diff = datetime.now() - published_time
        is_very_recent = time_diff.total_seconds() < 3600
        
        return has_breaking_word or is_very_recent
        
    def collect_newsapi_data(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Collect news from NewsAPI.org"""
        if not self.api_keys['newsapi']:
            self.logger.warning("NewsAPI key not configured")
            return []
            
        collected_news = []
        base_url = "https://newsapi.org/v2/everything"
        
        # If no symbols specified, get general market news
        queries = symbols if symbols else ['stock market', 'trading', 'NYSE', 'NASDAQ']
        
        for query in queries[:5]:  # Limit to conserve free tier
            try:
                params = {
                    'apiKey': self.api_keys['newsapi'],
                    'q': query,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'from': (datetime.now() - timedelta(days=1)).isoformat()
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        published = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                        
                        news_item = {
                            'symbol': query if query in symbols else None,
                            'headline': article.get('title', ''),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'source_url': article.get('url'),
                            'published_timestamp': published,
                            'content_snippet': article.get('description', '')[:500],
                            'full_url': article.get('url'),
                            'is_pre_market': self.is_pre_market_news(published),
                            'metadata': {
                                'author': article.get('author'),
                                'image_url': article.get('urlToImage')
                            }
                        }
                        collected_news.append(news_item)
                        
                else:
                    self.logger.error(f"NewsAPI error: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting from NewsAPI: {e}")
                
        return collected_news
        
    def collect_rss_feeds(self) -> List[Dict]:
        """Collect news from RSS feeds"""
        collected_news = []
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit entries per feed
                    # Parse published date
                    published = None
                    if hasattr(entry, 'published_parsed'):
                        published = datetime.fromtimestamp(
                            time.mktime(entry.published_parsed)
                        )
                    else:
                        published = datetime.now()
                        
                    news_item = {
                        'symbol': None,  # RSS feeds don't typically have symbols
                        'headline': entry.get('title', ''),
                        'source': source_name,
                        'source_url': feed_url,
                        'published_timestamp': published,
                        'content_snippet': entry.get('summary', '')[:500],
                        'full_url': entry.get('link'),
                        'is_pre_market': self.is_pre_market_news(published),
                        'metadata': {
                            'tags': entry.get('tags', [])
                        }
                    }
                    collected_news.append(news_item)
                    
            except Exception as e:
                self.logger.error(f"Error collecting from {source_name}: {e}")
                
        return collected_news
        
    def collect_alphavantage_news(self, symbols: List[str]) -> List[Dict]:
        """Collect news from Alpha Vantage"""
        if not self.api_keys['alphavantage'] or not symbols:
            return []
            
        collected_news = []
        base_url = "https://www.alphavantage.co/query"
        
        for symbol in symbols[:5]:  # Limit API calls
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.api_keys['alphavantage']
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('feed', []):
                        published = datetime.strptime(
                            article['time_published'], 
                            '%Y%m%dT%H%M%S'
                        )
                        
                        news_item = {
                            'symbol': symbol,
                            'headline': article.get('title', ''),
                            'source': article.get('source', 'AlphaVantage'),
                            'source_url': article.get('source_domain'),
                            'published_timestamp': published,
                            'content_snippet': article.get('summary', '')[:500],
                            'full_url': article.get('url'),
                            'is_pre_market': self.is_pre_market_news(published),
                            'metadata': {
                                'authors': article.get('authors', []),
                                'topics': article.get('topics', [])
                            }
                        }
                        collected_news.append(news_item)
                        
            except Exception as e:
                self.logger.error(f"Error collecting AlphaVantage news for {symbol}: {e}")
                
        return collected_news
        
    def save_news_items(self, news_items: List[Dict]) -> Dict[str, int]:
        """Save news items to database"""
        stats = {'total': len(news_items), 'saved': 0, 'duplicates': 0, 'errors': 0}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in news_items:
            try:
                # Generate unique ID
                news_id = self.generate_news_id(
                    item['headline'],
                    item['source'],
                    str(item['published_timestamp'])
                )
                
                # Prepare metadata as JSON
                metadata_json = json.dumps(item.get('metadata', {}))
                
                # Insert news item
                cursor.execute('''
                    INSERT OR IGNORE INTO news_raw 
                    (news_id, symbol, headline, source, source_url, 
                     published_timestamp, content_snippet, full_url, 
                     metadata, is_pre_market)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    news_id,
                    item.get('symbol'),
                    item['headline'],
                    item['source'],
                    item.get('source_url'),
                    item['published_timestamp'],
                    item.get('content_snippet'),
                    item.get('full_url'),
                    metadata_json,
                    item.get('is_pre_market', False)
                ))
                
                if cursor.rowcount > 0:
                    stats['saved'] += 1
                else:
                    stats['duplicates'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error saving news item: {e}")
                stats['errors'] += 1
                
        conn.commit()
        conn.close()
        
        return stats
        
    def collect_all_news(self, symbols: Optional[List[str]] = None, 
                        sources: str = 'all') -> Dict:
        """Collect news from all configured sources"""
        start_time = datetime.now()
        all_news = []
        collection_stats = {}
        
        # Use ThreadPoolExecutor for concurrent collection
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            if sources in ['all', 'newsapi']:
                futures['newsapi'] = executor.submit(self.collect_newsapi_data, symbols)
                
            if sources in ['all', 'rss']:
                futures['rss'] = executor.submit(self.collect_rss_feeds)
                
            if sources in ['all', 'alphavantage'] and symbols:
                futures['alphavantage'] = executor.submit(
                    self.collect_alphavantage_news, symbols
                )
                
            # Collect results
            for source, future in futures.items():
                try:
                    news_items = future.result(timeout=30)
                    all_news.extend(news_items)
                    collection_stats[source] = len(news_items)
                except Exception as e:
                    self.logger.error(f"Error collecting from {source}: {e}")
                    collection_stats[source] = 0
                    
        # Save all collected news
        save_stats = self.save_news_items(all_news)
        
        # Log collection statistics
        self._log_collection_stats(collection_stats, save_stats)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'status': 'success',
            'execution_time': execution_time,
            'articles_collected': save_stats['total'],
            'articles_saved': save_stats['saved'],
            'duplicates': save_stats['duplicates'],
            'errors': save_stats['errors'],
            'sources': collection_stats,
            'timestamp': datetime.now().isoformat()
        }
        
    def _log_collection_stats(self, collection_stats: Dict, save_stats: Dict):
        """Log collection statistics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for source, count in collection_stats.items():
                cursor.execute('''
                    INSERT INTO news_collection_stats 
                    (source, articles_collected, articles_new, 
                     articles_duplicate, error_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    source,
                    count,
                    save_stats['saved'] if source == 'total' else 0,
                    save_stats['duplicates'] if source == 'total' else 0,
                    save_stats['errors'] if source == 'total' else 0,
                    json.dumps(save_stats)
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging statistics: {e}")
            
    def _get_collection_stats(self, hours: int = 24) -> Dict:
        """Get collection statistics for the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get stats by source
            cursor.execute('''
                SELECT 
                    source,
                    SUM(articles_collected) as total_collected,
                    SUM(articles_new) as total_new,
                    SUM(articles_duplicate) as total_duplicate,
                    COUNT(*) as collection_runs
                FROM news_collection_stats
                WHERE collection_timestamp > datetime('now', '-{} hours')
                GROUP BY source
            '''.format(hours))
            
            source_stats = {}
            for row in cursor.fetchall():
                source_stats[row[0]] = {
                    'collected': row[1] or 0,
                    'new': row[2] or 0,
                    'duplicate': row[3] or 0,
                    'runs': row[4] or 0
                }
                
            # Get total news count
            cursor.execute('''
                SELECT COUNT(*) FROM news_raw
                WHERE collected_timestamp > datetime('now', '-{} hours')
            '''.format(hours))
            
            total_news = cursor.fetchone()[0] or 0
            
            # Get pre-market news count
            cursor.execute('''
                SELECT COUNT(*) FROM news_raw
                WHERE collected_timestamp > datetime('now', '-{} hours')
                AND is_pre_market = 1
            '''.format(hours))
            
            pre_market_news = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'period_hours': hours,
                'total_articles': total_news,
                'pre_market_articles': pre_market_news,
                'sources': source_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting News Collection Service v2.0.0 on port 5008")
        self.logger.info(f"Database path: {self.db_path}")
        self.logger.info(f"APIs configured: {[k for k, v in self.api_keys.items() if v]}")
        
        self.app.run(host='0.0.0.0', port=5008, debug=False)


if __name__ == "__main__":
    service = NewsCollectionService()
    service.run()
