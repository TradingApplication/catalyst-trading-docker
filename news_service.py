#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: news_service.py
Version: 2.1.0
Last Updated: 2025-07-01
Purpose: Raw news data collection from multiple sources with PostgreSQL integration

REVISION HISTORY:
v2.1.0 (2025-07-01) - Production-ready refactor
- Migrated from SQLite to PostgreSQL
- All configuration via environment variables
- Proper database connection pooling
- Enhanced error handling and retry logic
- Added source tier classification

v2.0.0 (2025-06-27) - Complete rewrite for raw data collection
- Multiple news source integration
- Raw data storage (no processing)
- Cloud-ready architecture
- Prepared for DigitalOcean migration

Description of Service:
This service collects news without interpretation, building a data lake
for future pattern analysis.

KEY FEATURES:
- Multiple sources: NewsAPI, AlphaVantage, RSS feeds, Finnhub
- Raw data collection with rich metadata
- Market state tracking (pre-market, regular, after-hours)
- Headline keyword extraction (earnings, FDA, merger, etc.)
- Ticker extraction from article text
- Breaking news detection
- Update tracking (how often stories are updated)
- Trending news identification
- Search API for analysis services
- Pre-market focused collection scheduling
- Source tier classification for reliability
"""

import os
import json
import time
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from structlog import get_logger
import redis

# Import database utilities
from database_utils import (
    get_db_connection,
    insert_news_article,
    get_recent_news,
    get_redis,
    get_configuration
)


class NewsCollectionService:
    """
    News Collection Service - Gathers raw news data from multiple sources
    No analysis, no sentiment, just pure data collection
    """
    
    def __init__(self):
        # Initialize environment
        self.setup_environment()
        
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_routes()
        
        # Initialize Redis client
        self.redis_client = get_redis()
        
        # API Keys from environment
        self.api_keys = {
            'newsapi': os.getenv('NEWSAPI_KEY', ''),
            'alphavantage': os.getenv('ALPHAVANTAGE_KEY', ''),
            'finnhub': os.getenv('FINNHUB_KEY', '')
        }
        
        # Validate API keys
        self._validate_api_keys()
        
        # RSS feeds with tier classification
        self.rss_feeds = {
            'marketwatch': {
                'url': 'https://feeds.marketwatch.com/marketwatch/topstories/',
                'tier': 1
            },
            'yahoo_finance': {
                'url': 'https://finance.yahoo.com/rss/',
                'tier': 1
            },
            'seeking_alpha': {
                'url': 'https://seekingalpha.com/feed.xml',
                'tier': 2
            },
            'investing_com': {
                'url': 'https://www.investing.com/rss/news.rss',
                'tier': 2
            },
            'reuters_business': {
                'url': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best',
                'tier': 1
            }
        }
        
        # Collection configuration
        self.collection_config = {
            'max_articles_per_source': int(os.getenv('MAX_ARTICLES_PER_SOURCE', '20')),
            'collection_timeout': int(os.getenv('COLLECTION_TIMEOUT', '30')),
            'concurrent_sources': int(os.getenv('CONCURRENT_SOURCES', '3')),
            'cache_ttl': int(os.getenv('NEWS_CACHE_TTL', '300'))  # 5 minutes
        }
        
        self.logger.info("News Collection Service v2.1.0 initialized",
                        environment=os.getenv('ENVIRONMENT', 'development'),
                        apis_configured=list(k for k, v in self.api_keys.items() if v))
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        # Paths
        self.log_path = os.getenv('LOG_PATH', '/app/logs')
        self.data_path = os.getenv('DATA_PATH', '/app/data')
        
        # Create directories
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Service configuration
        self.service_name = 'news_collection'
        self.port = int(os.getenv('PORT', '5008'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
    def setup_logging(self):
        """Setup structured logging"""
        self.logger = get_logger()
        self.logger = self.logger.bind(service=self.service_name)
        
    def _validate_api_keys(self):
        """Validate and warn about missing API keys"""
        for api_name, key in self.api_keys.items():
            if not key:
                self.logger.warning(f"{api_name} API key not configured",
                                  api=api_name)
                
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy", 
                "service": "news_collection",
                "version": "2.1.0",
                "timestamp": datetime.now().isoformat(),
                "apis_available": [k for k, v in self.api_keys.items() if v]
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
                'hours': request.args.get('hours', 24, type=int),
                'limit': request.args.get('limit', 100, type=int)
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
            
        @self.app.route('/news/<symbol>', methods=['GET'])
        def get_symbol_news(symbol):
            """Get news for specific symbol"""
            hours = request.args.get('hours', 24, type=int)
            news = get_recent_news(symbol, hours)
            return jsonify({
                'symbol': symbol,
                'count': len(news),
                'articles': news
            })
            
    def generate_news_id(self, headline: str, source: str, timestamp: str) -> str:
        """Generate unique ID for news article"""
        content = f"{headline}_{source}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def is_pre_market_news(self, timestamp: datetime) -> bool:
        """Check if news was published during pre-market hours (4 AM - 9:30 AM EST)"""
        # Convert to EST (accounting for timezone from environment)
        timezone_offset = int(os.getenv('TIMEZONE_OFFSET', '-5'))
        est_hour = timestamp.hour + timezone_offset
        if est_hour < 0:
            est_hour += 24
        elif est_hour >= 24:
            est_hour -= 24
            
        return 4 <= est_hour < 9.5
        
    def get_market_state(self, timestamp: datetime) -> str:
        """Determine market state when news was published"""
        # Convert to EST
        timezone_offset = int(os.getenv('TIMEZONE_OFFSET', '-5'))
        est_hour = timestamp.hour + timezone_offset
        if est_hour < 0:
            est_hour += 24
        elif est_hour >= 24:
            est_hour -= 24
        
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
        
        # Financial event keywords - loaded from configuration
        keyword_patterns = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'beat', 'miss', 'eps', 'quarterly'],
            'fda': ['fda', 'approval', 'drug', 'clinical', 'trial', 'phase', 'biotech'],
            'merger': ['merger', 'acquisition', 'acquire', 'buyout', 'takeover', 'deal', 'bid'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'price target', 'analyst', 'outperform'],
            'insider': ['insider', 'ceo', 'cfo', 'director', 'executive', 'sold', 'bought'],
            'legal': ['lawsuit', 'settlement', 'investigation', 'sec', 'fraud', 'probe', 'subpoena'],
            'product': ['launch', 'release', 'announce', 'unveil', 'introduce', 'new product'],
            'guidance': ['guidance', 'forecast', 'outlook', 'warns', 'expects', 'raises', 'lowers'],
            'partnership': ['partnership', 'collaboration', 'joint venture', 'agreement', 'contract'],
            'ipo': ['ipo', 'public offering', 'listing', 'debut', 'direct listing'],
            'bankruptcy': ['bankruptcy', 'chapter 11', 'restructuring', 'default', 'liquidation'],
            'dividend': ['dividend', 'yield', 'payout', 'distribution', 'special dividend']
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
                     'CEO', 'CFO', 'IPO', 'FDA', 'SEC', 'NYSE', 'ETF', 'AI', 'IT',
                     'US', 'UK', 'EU', 'GDP', 'API', 'URL', 'HTML', 'JSON'}
        
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter out common words and return unique tickers
        tickers = []
        for ticker in potential_tickers:
            ticker = ticker.replace('$', '')
            if ticker not in exclusions and len(ticker) >= 2:
                tickers.append(ticker)
                
        return list(set(tickers))  # Remove duplicates
        
    def is_breaking_news(self, headline: str, published_time: datetime) -> bool:
        """Detect if this appears to be breaking news"""
        breaking_indicators = ['breaking', 'alert', 'urgent', 'just in', 
                              'developing', 'exclusive', 'flash', 'update']
        
        headline_lower = headline.lower()
        
        # Check for breaking news indicators
        has_breaking_word = any(indicator in headline_lower for indicator in breaking_indicators)
        
        # Check if very recent (within last hour)
        time_diff = datetime.now() - published_time
        is_very_recent = time_diff.total_seconds() < 3600
        
        return has_breaking_word or is_very_recent
        
    def determine_source_tier(self, source: str) -> int:
        """Determine reliability tier of news source (1=highest, 5=lowest)"""
        tier_mapping = {
            # Tier 1 - Most reliable
            'Reuters': 1, 'Bloomberg': 1, 'Wall Street Journal': 1, 'Financial Times': 1,
            'MarketWatch': 1, 'CNBC': 1, 'Yahoo Finance': 1,
            
            # Tier 2 - Reliable
            'Seeking Alpha': 2, 'Investing.com': 2, 'TheStreet': 2, 'Barron\'s': 2,
            
            # Tier 3 - Moderate
            'Benzinga': 3, 'InvestorPlace': 3, 'Motley Fool': 3,
            
            # Tier 4 - Variable
            'StockTwits': 4, 'Reddit': 4,
            
            # Tier 5 - Least reliable/Unknown
            'Unknown': 5
        }
        
        # Check source against mapping
        for key, tier in tier_mapping.items():
            if key.lower() in source.lower():
                return tier
                
        return 5  # Default to lowest tier for unknown sources
        
    def collect_newsapi_data(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Collect news from NewsAPI.org"""
        if not self.api_keys['newsapi']:
            self.logger.warning("NewsAPI key not configured")
            return []
            
        collected_news = []
        base_url = "https://newsapi.org/v2/everything"
        
        # If no symbols specified, get general market news
        queries = symbols if symbols else ['stock market', 'S&P 500', 'NYSE', 'NASDAQ', 'trading']
        
        for query in queries[:5]:  # Limit to conserve API calls
            try:
                # Check cache first
                cache_key = f"newsapi:{query}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.logger.debug("Using cached NewsAPI data", query=query)
                    collected_news.extend(json.loads(cached_data))
                    continue
                
                params = {
                    'apiKey': self.api_keys['newsapi'],
                    'q': query,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': self.collection_config['max_articles_per_source'],
                    'from': (datetime.now() - timedelta(days=1)).isoformat()
                }
                
                response = requests.get(base_url, params=params, 
                                      timeout=self.collection_config['collection_timeout'])
                
                if response.status_code == 200:
                    data = response.json()
                    articles = []
                    
                    for article in data.get('articles', []):
                        published = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                        
                        news_item = {
                            'news_id': self.generate_news_id(
                                article.get('title', ''),
                                article.get('source', {}).get('name', 'NewsAPI'),
                                str(published)
                            ),
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
                            'is_breaking_news': self.is_breaking_news(article.get('title', ''), published),
                            'source_tier': self.determine_source_tier(
                                article.get('source', {}).get('name', 'Unknown')
                            ),
                            'metadata': {
                                'author': article.get('author'),
                                'image_url': article.get('urlToImage'),
                                'content_length': len(article.get('content', ''))
                            }
                        }
                        articles.append(news_item)
                        collected_news.append(news_item)
                    
                    # Cache the results
                    self.redis_client.setex(
                        cache_key, 
                        self.collection_config['cache_ttl'],
                        json.dumps(articles)
                    )
                        
                else:
                    self.logger.error("NewsAPI error",
                                    status_code=response.status_code,
                                    query=query)
                    
            except Exception as e:
                self.logger.error("Error collecting from NewsAPI",
                                error=str(e),
                                query=query)
                
        return collected_news
        
    def collect_rss_feeds(self) -> List[Dict]:
        """Collect news from RSS feeds"""
        collected_news = []
        
        for source_name, feed_info in self.rss_feeds.items():
            try:
                # Check cache
                cache_key = f"rss:{source_name}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.logger.debug("Using cached RSS data", source=source_name)
                    collected_news.extend(json.loads(cached_data))
                    continue
                
                feed = feedparser.parse(feed_info['url'])
                articles = []
                
                for entry in feed.entries[:self.collection_config['max_articles_per_source']]:
                    # Parse published date
                    published = None
                    if hasattr(entry, 'published_parsed'):
                        published = datetime.fromtimestamp(
                            time.mktime(entry.published_parsed)
                        )
                    else:
                        published = datetime.now()
                        
                    news_item = {
                        'news_id': self.generate_news_id(
                            entry.get('title', ''),
                            source_name,
                            str(published)
                        ),
                        'symbol': None,  # RSS feeds don't typically have symbols
                        'headline': entry.get('title', ''),
                        'source': source_name,
                        'source_url': feed_info['url'],
                        'published_timestamp': published,
                        'content_snippet': entry.get('summary', '')[:500],
                        'full_url': entry.get('link'),
                        'is_pre_market': self.is_pre_market_news(published),
                        'market_state': self.get_market_state(published),
                        'headline_keywords': self.extract_headline_keywords(entry.get('title', '')),
                        'mentioned_tickers': self.extract_mentioned_tickers(
                            entry.get('title', '') + ' ' + entry.get('summary', '')
                        ),
                        'is_breaking_news': self.is_breaking_news(entry.get('title', ''), published),
                        'source_tier': feed_info['tier'],
                        'metadata': {
                            'tags': [tag.term for tag in entry.get('tags', [])] if hasattr(entry, 'tags') else []
                        }
                    }
                    articles.append(news_item)
                    collected_news.append(news_item)
                
                # Cache the results
                self.redis_client.setex(
                    cache_key,
                    self.collection_config['cache_ttl'],
                    json.dumps(articles)
                )
                    
            except Exception as e:
                self.logger.error("Error collecting RSS feed",
                                source=source_name,
                                error=str(e))
                
        return collected_news
        
    def collect_alphavantage_news(self, symbols: List[str]) -> List[Dict]:
        """Collect news from Alpha Vantage"""
        if not self.api_keys['alphavantage'] or not symbols:
            return []
            
        collected_news = []
        base_url = "https://www.alphavantage.co/query"
        
        for symbol in symbols[:5]:  # Limit API calls
            try:
                # Check cache
                cache_key = f"alphavantage:{symbol}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.logger.debug("Using cached AlphaVantage data", symbol=symbol)
                    collected_news.extend(json.loads(cached_data))
                    continue
                
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.api_keys['alphavantage']
                }
                
                response = requests.get(base_url, params=params, 
                                      timeout=self.collection_config['collection_timeout'])
                
                if response.status_code == 200:
                    data = response.json()
                    articles = []
                    
                    for article in data.get('feed', []):
                        published = datetime.strptime(
                            article['time_published'], 
                            '%Y%m%dT%H%M%S'
                        )
                        
                        news_item = {
                            'news_id': self.generate_news_id(
                                article.get('title', ''),
                                article.get('source', 'AlphaVantage'),
                                str(published)
                            ),
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
                            'is_breaking_news': self.is_breaking_news(article.get('title', ''), published),
                            'source_tier': self.determine_source_tier(article.get('source', 'Unknown')),
                            'metadata': {
                                'authors': article.get('authors', []),
                                'topics': article.get('topics', []),
                                'ticker_sentiment': article.get('ticker_sentiment', {}),
                                'overall_sentiment_score': article.get('overall_sentiment_score'),
                                'overall_sentiment_label': article.get('overall_sentiment_label')
                            }
                        }
                        articles.append(news_item)
                        collected_news.append(news_item)
                    
                    # Cache the results
                    self.redis_client.setex(
                        cache_key,
                        self.collection_config['cache_ttl'],
                        json.dumps(articles)
                    )
                        
            except Exception as e:
                self.logger.error("Error collecting AlphaVantage news",
                                symbol=symbol,
                                error=str(e))
                
        return collected_news
        
    def collect_finnhub_news(self, symbols: List[str]) -> List[Dict]:
        """Collect news from Finnhub"""
        if not self.api_keys['finnhub'] or not symbols:
            return []
            
        collected_news = []
        base_url = "https://finnhub.io/api/v1/company-news"
        
        for symbol in symbols[:5]:  # Limit API calls
            try:
                # Check cache
                cache_key = f"finnhub:{symbol}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.logger.debug("Using cached Finnhub data", symbol=symbol)
                    collected_news.extend(json.loads(cached_data))
                    continue
                
                # Date range for news
                date_from = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                date_to = datetime.now().strftime('%Y-%m-%d')
                
                params = {
                    'symbol': symbol,
                    'from': date_from,
                    'to': date_to,
                    'token': self.api_keys['finnhub']
                }
                
                response = requests.get(base_url, params=params,
                                      timeout=self.collection_config['collection_timeout'])
                
                if response.status_code == 200:
                    data = response.json()
                    articles = []
                    
                    for article in data[:self.collection_config['max_articles_per_source']]:
                        published = datetime.fromtimestamp(article.get('datetime', time.time()))
                        
                        news_item = {
                            'news_id': self.generate_news_id(
                                article.get('headline', ''),
                                article.get('source', 'Finnhub'),
                                str(published)
                            ),
                            'symbol': symbol,
                            'headline': article.get('headline', ''),
                            'source': article.get('source', 'Finnhub'),
                            'source_url': article.get('url'),
                            'published_timestamp': published,
                            'content_snippet': article.get('summary', '')[:500],
                            'full_url': article.get('url'),
                            'is_pre_market': self.is_pre_market_news(published),
                            'market_state': self.get_market_state(published),
                            'headline_keywords': self.extract_headline_keywords(article.get('headline', '')),
                            'mentioned_tickers': self.extract_mentioned_tickers(
                                article.get('headline', '') + ' ' + article.get('summary', '')
                            ),
                            'is_breaking_news': self.is_breaking_news(article.get('headline', ''), published),
                            'source_tier': self.determine_source_tier(article.get('source', 'Unknown')),
                            'metadata': {
                                'category': article.get('category'),
                                'id': article.get('id'),
                                'related': article.get('related', symbol)
                            }
                        }
                        articles.append(news_item)
                        collected_news.append(news_item)
                    
                    # Cache the results
                    self.redis_client.setex(
                        cache_key,
                        self.collection_config['cache_ttl'],
                        json.dumps(articles)
                    )
                        
            except Exception as e:
                self.logger.error("Error collecting Finnhub news",
                                symbol=symbol,
                                error=str(e))
                
        return collected_news
        
    def save_news_items(self, news_items: List[Dict]) -> Dict[str, int]:
        """Save news items to database"""
        stats = {'total': len(news_items), 'saved': 0, 'duplicates': 0, 'errors': 0}
        
        for item in news_items:
            try:
                # Use database utility function
                news_id = insert_news_article(item)
                
                if news_id:
                    stats['saved'] += 1
                else:
                    stats['duplicates'] += 1
                    
            except Exception as e:
                self.logger.error("Error saving news item",
                                headline=item.get('headline'),
                                error=str(e))
                stats['errors'] += 1
                
        return stats
        
    def collect_all_news(self, symbols: Optional[List[str]] = None, 
                        sources: str = 'all') -> Dict:
        """Collect news from all configured sources"""
        start_time = datetime.now()
        all_news = []
        collection_stats = {}
        
        # Use ThreadPoolExecutor for concurrent collection
        with ThreadPoolExecutor(max_workers=self.collection_config['concurrent_sources']) as executor:
            futures = {}
            
            if sources in ['all', 'newsapi'] and self.api_keys['newsapi']:
                futures['newsapi'] = executor.submit(self.collect_newsapi_data, symbols)
                
            if sources in ['all', 'rss']:
                futures['rss'] = executor.submit(self.collect_rss_feeds)
                
            if sources in ['all', 'alphavantage'] and self.api_keys['alphavantage'] and symbols:
                futures['alphavantage'] = executor.submit(
                    self.collect_alphavantage_news, symbols
                )
                
            if sources in ['all', 'finnhub'] and self.api_keys['finnhub'] and symbols:
                futures['finnhub'] = executor.submit(
                    self.collect_finnhub_news, symbols
                )
                
            # Collect results
            for source, future in futures.items():
                try:
                    news_items = future.result(timeout=self.collection_config['collection_timeout'])
                    all_news.extend(news_items)
                    collection_stats[source] = len(news_items)
                    self.logger.info("Collected news from source",
                                   source=source,
                                   count=len(news_items))
                except Exception as e:
                    self.logger.error("Error collecting from source",
                                    source=source,
                                    error=str(e))
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
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    for source, count in collection_stats.items():
                        cur.execute('''
                            INSERT INTO news_collection_stats 
                            (source, articles_collected, articles_new, 
                             articles_duplicate, error_count, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        ''', (
                            source,
                            count,
                            save_stats['saved'] if source == 'total' else 0,
                            save_stats['duplicates'] if source == 'total' else 0,
                            save_stats['errors'] if source == 'total' else 0,
                            json.dumps(save_stats)
                        ))
                        
        except Exception as e:
            self.logger.error("Error logging statistics", error=str(e))
            
    def _get_collection_stats(self, hours: int = 24) -> Dict:
        """Get collection statistics for the last N hours"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get stats by source
                    cur.execute('''
                        SELECT 
                            source,
                            SUM(articles_collected) as total_collected,
                            SUM(articles_new) as total_new,
                            SUM(articles_duplicate) as total_duplicate,
                            COUNT(*) as collection_runs
                        FROM news_collection_stats
                        WHERE collection_timestamp > NOW() - INTERVAL '%s hours'
                        GROUP BY source
                    ''', [hours])
                    
                    source_stats = {}
                    for row in cur.fetchall():
                        source_stats[row['source']] = {
                            'collected': row['total_collected'] or 0,
                            'new': row['total_new'] or 0,
                            'duplicate': row['total_duplicate'] or 0,
                            'runs': row['collection_runs'] or 0
                        }
                        
                    # Get total news count
                    cur.execute('''
                        SELECT COUNT(*) as total FROM news_raw
                        WHERE collected_timestamp > NOW() - INTERVAL '%s hours'
                    ''', [hours])
                    
                    total_news = cur.fetchone()['total'] or 0
                    
                    # Get pre-market news count
                    cur.execute('''
                        SELECT COUNT(*) as pre_market FROM news_raw
                        WHERE collected_timestamp > NOW() - INTERVAL '%s hours'
                        AND is_pre_market = true
                    ''', [hours])
                    
                    pre_market_news = cur.fetchone()['pre_market'] or 0
                    
                    return {
                        'period_hours': hours,
                        'total_articles': total_news,
                        'pre_market_articles': pre_market_news,
                        'sources': source_stats,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.error("Error getting statistics", error=str(e))
            return {'error': str(e)}
            
    def _search_news(self, params: Dict) -> Dict:
        """Search news by various criteria"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Build query
                    query = '''
                        SELECT 
                            news_id, symbol, headline, source, published_timestamp,
                            content_snippet, full_url, is_pre_market, market_state,
                            headline_keywords, mentioned_tickers,
                            is_breaking_news, update_count, source_tier
                        FROM news_raw
                        WHERE published_timestamp > NOW() - INTERVAL '%s hours'
                    '''
                    
                    query_params = [params['hours']]
                    
                    # Add filters
                    if params['symbol']:
                        query += ' AND (symbol = %s OR %s = ANY(mentioned_tickers))'
                        query_params.extend([params['symbol'], params['symbol']])
                        
                    if params['keywords']:
                        for keyword in params['keywords']:
                            query += ' AND %s = ANY(headline_keywords)'
                            query_params.append(keyword)
                            
                    if params['market_state']:
                        query += ' AND market_state = %s'
                        query_params.append(params['market_state'])
                        
                    if params['breaking_only']:
                        query += ' AND is_breaking_news = true'
                        
                    # Order by most recent
                    query += ' ORDER BY published_timestamp DESC LIMIT %s'
                    query_params.append(params['limit'])
                    
                    cur.execute(query, query_params)
                    
                    # Format results
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            'news_id': row['news_id'],
                            'symbol': row['symbol'],
                            'headline': row['headline'],
                            'source': row['source'],
                            'published': row['published_timestamp'].isoformat(),
                            'snippet': row['content_snippet'],
                            'url': row['full_url'],
                            'pre_market': row['is_pre_market'],
                            'market_state': row['market_state'],
                            'keywords': row['headline_keywords'],
                            'mentioned_tickers': row['mentioned_tickers'],
                            'breaking': row['is_breaking_news'],
                            'updates': row['update_count'],
                            'source_tier': row['source_tier']
                        })
                        
                    return {
                        'count': len(results),
                        'results': results,
                        'query_params': params,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.error("Error searching news", error=str(e))
            return {'error': str(e), 'results': []}
            
    def _get_trending_news(self, hours: int, limit: int) -> Dict:
        """Get trending news based on update frequency"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get stories with most updates
                    cur.execute('''
                        SELECT 
                            symbol, headline, source, 
                            MAX(published_timestamp) as latest_update,
                            MIN(first_seen_timestamp) as first_seen,
                            COUNT(*) as mention_count,
                            SUM(update_count) as total_updates,
                            STRING_AGG(DISTINCT market_state, ',') as market_states,
                            bool_or(is_breaking_news) as has_breaking,
                            MIN(source_tier) as best_source_tier
                        FROM news_raw
                        WHERE published_timestamp > NOW() - INTERVAL '%s hours'
                        GROUP BY SUBSTRING(headline, 1, 50), source
                        HAVING COUNT(*) > 1 OR SUM(update_count) > 0
                        ORDER BY mention_count DESC, total_updates DESC
                        LIMIT %s
                    ''', [hours, limit])
                    
                    trending = []
                    for row in cur.fetchall():
                        trending.append({
                            'symbol': row['symbol'],
                            'headline': row['headline'],
                            'source': row['source'],
                            'first_seen': row['first_seen'].isoformat(),
                            'latest_update': row['latest_update'].isoformat(),
                            'mention_count': row['mention_count'],
                            'total_updates': row['total_updates'],
                            'market_states': row['market_states'].split(',') if row['market_states'] else [],
                            'has_breaking': row['has_breaking'],
                            'best_source_tier': row['best_source_tier']
                        })
                        
                    return {
                        'count': len(trending),
                        'hours_window': hours,
                        'trending': trending,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.error("Error getting trending news", error=str(e))
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
            symbols = self._get_active_symbols()
        else:
            symbols = None  # General market news
            
        # Collect from all sources
        result = self.collect_all_news(symbols, 'all')
        
        self.logger.info("Scheduled collection complete",
                        collected=result.get('articles_collected', 0),
                        saved=result.get('articles_saved', 0))
        return result
        
    def _get_active_symbols(self) -> List[str]:
        """Get active symbols from scanner service"""
        try:
            scanner_url = os.getenv('SCANNER_SERVICE_URL', 'http://scanner-service:5001')
            response = requests.get(f"{scanner_url}/active_symbols", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('symbols', [])
        except Exception as e:
            self.logger.warning("Could not get active symbols from scanner",
                              error=str(e))
            
        return []
            
    def run(self):
        """Start the Flask application"""
        self.logger.info("Starting News Collection Service",
                        version="2.1.0",
                        port=self.port,
                        environment=os.getenv('ENVIRONMENT', 'development'))
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    service = NewsCollectionService()
    service.run()