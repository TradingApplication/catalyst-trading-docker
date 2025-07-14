#!/usr/bin/env python3
"""
Name of Application: Catalyst Trading System
Name of file: database_utils.py (Trading Functions Addition)
Version: 2.3.4
Last Updated: 2025-07-08
Purpose: Add missing trading service functions to database_utils.py

REVISION HISTORY:
v2.3.4 (2025-07-08) - Added missing trading service functions
- Added insert_trade_record function
- Added update_trade_exit function  
- Added get_open_positions function
- Added get_pending_signals function
- Added supporting trade management functions

Description:
Add these functions to the end of database_utils.py BEFORE the module import section
"""

# =============================================================================
# TRADING SERVICE FUNCTIONS - Add to database_utils.py v2.3.4
# =============================================================================

def insert_trade_record(trade_data: Dict) -> int:
    """
    Insert a new trade record into the database
    
    Args:
        trade_data: Dictionary containing trade information including:
            - symbol: Stock symbol
            - signal_id: ID of the signal that triggered this trade
            - entry_price: Entry price
            - quantity: Number of shares
            - order_type: Type of order (market, limit, etc.)
            - side: buy/sell
            - stop_loss: Stop loss price
            - take_profit: Take profit price
            
    Returns:
        ID of the inserted trade record
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_records (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        signal_id INTEGER,
                        order_id VARCHAR(100),
                        side VARCHAR(10) NOT NULL,
                        order_type VARCHAR(20),
                        quantity INTEGER NOT NULL,
                        entry_price DECIMAL(10,2),
                        exit_price DECIMAL(10,2),
                        stop_loss DECIMAL(10,2),
                        take_profit DECIMAL(10,2),
                        entry_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        exit_timestamp TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'open',
                        pnl_amount DECIMAL(10,2),
                        pnl_percentage DECIMAL(5,2),
                        commission DECIMAL(10,2),
                        entry_reason TEXT,
                        exit_reason TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                    ON trade_records(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_status 
                    ON trade_records(status)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_entry_timestamp 
                    ON trade_records(entry_timestamp DESC)
                """)
                
                # Insert trade record
                cur.execute("""
                    INSERT INTO trade_records (
                        symbol, signal_id, order_id, side, order_type,
                        quantity, entry_price, stop_loss, take_profit,
                        entry_reason, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    trade_data.get('symbol'),
                    trade_data.get('signal_id'),
                    trade_data.get('order_id'),
                    trade_data.get('side', 'buy'),
                    trade_data.get('order_type', 'market'),
                    trade_data.get('quantity'),
                    trade_data.get('entry_price'),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('entry_reason'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                trade_id = cur.fetchone()['id']
                conn.commit()
                
                logger.info(f"Inserted trade record", 
                           trade_id=trade_id, 
                           symbol=trade_data.get('symbol'))
                
                return trade_id
                
    except Exception as e:
        logger.error("Failed to insert trade record", error=str(e))
        raise DatabaseError(f"Failed to insert trade record: {str(e)}")


def update_trade_exit(trade_id: int, exit_data: Dict) -> bool:
    """
    Update trade record with exit information
    
    Args:
        trade_id: ID of the trade to update
        exit_data: Dictionary containing exit information:
            - exit_price: Exit price
            - exit_reason: Reason for exit (stop_loss, take_profit, manual, etc.)
            - exit_timestamp: Optional exit timestamp
            
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get current trade info for P&L calculation
                cur.execute("""
                    SELECT entry_price, quantity, side 
                    FROM trade_records 
                    WHERE id = %s AND status = 'open'
                """, (trade_id,))
                
                trade = cur.fetchone()
                if not trade:
                    logger.warning(f"Trade not found or already closed", trade_id=trade_id)
                    return False
                
                # Calculate P&L
                entry_price = float(trade['entry_price'])
                exit_price = float(exit_data.get('exit_price'))
                quantity = int(trade['quantity'])
                side = trade['side']
                
                if side == 'buy':
                    pnl_amount = (exit_price - entry_price) * quantity
                else:  # sell/short
                    pnl_amount = (entry_price - exit_price) * quantity
                
                pnl_percentage = (pnl_amount / (entry_price * quantity)) * 100
                
                # Update trade record
                cur.execute("""
                    UPDATE trade_records 
                    SET exit_price = %s,
                        exit_timestamp = %s,
                        exit_reason = %s,
                        pnl_amount = %s,
                        pnl_percentage = %s,
                        status = 'closed'
                    WHERE id = %s
                """, (
                    exit_data.get('exit_price'),
                    exit_data.get('exit_timestamp') or datetime.now(),
                    exit_data.get('exit_reason'),
                    pnl_amount,
                    pnl_percentage,
                    trade_id
                ))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Updated trade exit", 
                               trade_id=trade_id, 
                               pnl=pnl_amount)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to update trade exit", error=str(e))
        return False


def get_open_positions() -> List[Dict]:
    """
    Get all currently open trading positions
    
    Returns:
        List of open position dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        id, symbol, signal_id, order_id, side, order_type,
                        quantity, entry_price, stop_loss, take_profit,
                        entry_timestamp, entry_reason, metadata,
                        CURRENT_TIMESTAMP - entry_timestamp as position_age
                    FROM trade_records
                    WHERE status = 'open'
                    ORDER BY entry_timestamp DESC
                """)
                
                positions = []
                for row in cur.fetchall():
                    position = dict(row)
                    # Convert timestamps to ISO format
                    if position.get('entry_timestamp'):
                        position['entry_timestamp'] = position['entry_timestamp'].isoformat()
                    # Convert timedelta to seconds
                    if position.get('position_age'):
                        position['position_age_seconds'] = position['position_age'].total_seconds()
                        del position['position_age']
                    positions.append(position)
                
                logger.info(f"Retrieved {len(positions)} open positions")
                return positions
                
    except Exception as e:
        logger.error("Failed to get open positions", error=str(e))
        return []


def get_pending_signals(limit: int = 10) -> List[Dict]:
    """
    Get pending trading signals that haven't been executed yet
    
    Args:
        limit: Maximum number of signals to return
        
    Returns:
        List of pending signal dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create signals table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        signal_type VARCHAR(20) NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(10,2),
                        stop_loss DECIMAL(10,2),
                        take_profit DECIMAL(10,2),
                        confidence DECIMAL(5,2),
                        risk_reward_ratio DECIMAL(5,2),
                        catalyst_info JSONB,
                        technical_info JSONB,
                        executed BOOLEAN DEFAULT FALSE,
                        execution_time TIMESTAMP,
                        trade_id INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol 
                    ON trading_signals(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_executed 
                    ON trading_signals(executed)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_created 
                    ON trading_signals(created_at DESC)
                """)
                
                # Get pending signals
                cur.execute("""
                    SELECT 
                        id, symbol, signal_type, action, entry_price,
                        stop_loss, take_profit, confidence, risk_reward_ratio,
                        catalyst_info, technical_info, created_at, expires_at
                    FROM trading_signals
                    WHERE executed = FALSE
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT %s
                """, (limit,))
                
                signals = []
                for row in cur.fetchall():
                    signal = dict(row)
                    # Convert timestamps to ISO format
                    if signal.get('created_at'):
                        signal['created_at'] = signal['created_at'].isoformat()
                    if signal.get('expires_at'):
                        signal['expires_at'] = signal['expires_at'].isoformat()
                    signals.append(signal)
                
                logger.info(f"Retrieved {len(signals)} pending signals")
                return signals
                
    except Exception as e:
        logger.error("Failed to get pending signals", error=str(e))
        return []


# =============================================================================
# ADDITIONAL TRADING HELPER FUNCTIONS
# =============================================================================

def mark_signal_executed(signal_id: int, trade_id: int) -> bool:
    """
    Mark a trading signal as executed
    
    Args:
        signal_id: ID of the signal
        trade_id: ID of the resulting trade
        
    Returns:
        True if updated successfully
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trading_signals
                    SET executed = TRUE,
                        execution_time = CURRENT_TIMESTAMP,
                        trade_id = %s
                    WHERE id = %s
                """, (trade_id, signal_id))
                
                updated = cur.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.info(f"Marked signal as executed", 
                               signal_id=signal_id, 
                               trade_id=trade_id)
                
                return updated
                
    except Exception as e:
        logger.error("Failed to mark signal executed", error=str(e))
        return False


def get_position_by_symbol(symbol: str) -> Optional[Dict]:
    """
    Get open position for a specific symbol
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Position dictionary or None if no open position
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trade_records
                    WHERE symbol = %s AND status = 'open'
                    ORDER BY entry_timestamp DESC
                    LIMIT 1
                """, (symbol,))
                
                result = cur.fetchone()
                if result:
                    position = dict(result)
                    if position.get('entry_timestamp'):
                        position['entry_timestamp'] = position['entry_timestamp'].isoformat()
                    return position
                
                return None
                
    except Exception as e:
        logger.error("Failed to get position by symbol", error=str(e))
        return None


def get_trade_history(symbol: Optional[str] = None, days: int = 30) -> List[Dict]:
    """
    Get historical trades
    
    Args:
        symbol: Optional symbol to filter by
        days: Number of days to look back
        
    Returns:
        List of historical trades
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if symbol:
                    cur.execute("""
                        SELECT * FROM trade_records
                        WHERE symbol = %s
                        AND entry_timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        ORDER BY entry_timestamp DESC
                    """, (symbol, days))
                else:
                    cur.execute("""
                        SELECT * FROM trade_records
                        WHERE entry_timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        ORDER BY entry_timestamp DESC
                        LIMIT 100
                    """, (days,))
                
                trades = []
                for row in cur.fetchall():
                    trade = dict(row)
                    # Convert timestamps
                    for field in ['entry_timestamp', 'exit_timestamp', 'created_at']:
                        if trade.get(field):
                            trade[field] = trade[field].isoformat()
                    trades.append(trade)
                
                return trades
                
    except Exception as e:
        logger.error("Failed to get trade history", error=str(e))
        return []


def calculate_portfolio_metrics() -> Dict:
    """
    Calculate portfolio performance metrics
    
    Returns:
        Dictionary with portfolio metrics
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get closed trades for metrics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl_amount > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl_amount < 0 THEN 1 END) as losing_trades,
                        SUM(pnl_amount) as total_pnl,
                        AVG(pnl_percentage) as avg_pnl_pct,
                        MAX(pnl_amount) as best_trade,
                        MIN(pnl_amount) as worst_trade
                    FROM trade_records
                    WHERE status = 'closed'
                    AND exit_timestamp > CURRENT_TIMESTAMP - INTERVAL '30 days'
                """)
                
                metrics = dict(cur.fetchone())
                
                # Calculate win rate
                if metrics['total_trades'] > 0:
                    metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
                else:
                    metrics['win_rate'] = 0
                
                # Get open positions value
                cur.execute("""
                    SELECT 
                        COUNT(*) as open_positions,
                        SUM(quantity * entry_price) as open_value
                    FROM trade_records
                    WHERE status = 'open'
                """)
                
                open_info = dict(cur.fetchone())
                metrics.update(open_info)
                
                return metrics
                
    except Exception as e:
        logger.error("Failed to calculate portfolio metrics", error=str(e))
        return {}


# Note: Place this code BEFORE the module import section at the end of database_utils.py