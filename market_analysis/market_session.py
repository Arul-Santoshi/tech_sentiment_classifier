"""
Market session classification utilities.

Classifies article timestamps into market sessions relative to US market hours.
US Market Hours: 9:30 AM - 4:00 PM Eastern Time

Sessions:
- pre_market: 12:00 AM - 9:29 AM ET (same trading day)
- market_hours: 9:30 AM - 4:00 PM ET (same trading day)
- after_hours: 4:01 PM - 11:59 PM ET (next trading day)
- weekend: Saturday/Sunday (next Monday)
"""

from datetime import datetime, timedelta, time
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# US Eastern timezone
ET = ZoneInfo("America/New_York")

# Market hours in Eastern Time
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# US Market holidays 2026 (add more years as needed)
MARKET_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}

# Combined holidays set
MARKET_HOLIDAYS = MARKET_HOLIDAYS_2026


def is_market_holiday(date_str: str) -> bool:
    """Check if a date is a US market holiday."""
    return date_str in MARKET_HOLIDAYS


def is_trading_day(dt: datetime) -> bool:
    """Check if a datetime falls on a trading day (weekday, not holiday)."""
    date_str = dt.strftime("%Y-%m-%d")
    return dt.weekday() < 5 and not is_market_holiday(date_str)


def get_next_trading_day(dt: datetime) -> datetime:
    """Get the next trading day from a given datetime."""
    next_day = dt + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day.replace(hour=0, minute=0, second=0, microsecond=0)


def get_current_or_next_trading_day(dt: datetime) -> datetime:
    """Get current day if trading day, otherwise next trading day."""
    if is_trading_day(dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return get_next_trading_day(dt)


def classify_market_session(
    published_at: str,
    source_timezone: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Classify an article's publication time into a market session.

    Args:
        published_at: ISO format timestamp (e.g., "2026-02-03T14:30:00Z")
        source_timezone: Optional timezone of the source (defaults to UTC)

    Returns:
        Tuple of (market_session, trading_date)
        - market_session: 'pre_market', 'market_hours', 'after_hours', or 'weekend'
        - trading_date: The trading date this article's sentiment should map to (YYYY-MM-DD)
    """
    # Parse the timestamp
    try:
        if published_at.endswith("Z"):
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        elif "+" in published_at or "-" in published_at[10:]:
            dt = datetime.fromisoformat(published_at)
        else:
            # Assume UTC if no timezone
            dt = datetime.fromisoformat(published_at).replace(tzinfo=ZoneInfo("UTC"))
    except (ValueError, TypeError):
        # If parsing fails, return unknown with today's date
        return "unknown", datetime.now().strftime("%Y-%m-%d")

    # Convert to Eastern Time
    dt_et = dt.astimezone(ET)
    current_time = dt_et.time()
    current_date = dt_et.date()

    # Check if weekend
    if dt_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        # Weekend news maps to next Monday (or next trading day if Monday is holiday)
        next_trading = get_next_trading_day(dt_et)
        return "weekend", next_trading.strftime("%Y-%m-%d")

    # Check if market holiday
    if is_market_holiday(current_date.isoformat()):
        next_trading = get_next_trading_day(dt_et)
        return "weekend", next_trading.strftime("%Y-%m-%d")  # Treat as weekend

    # Weekday, not holiday - classify by time
    if current_time < MARKET_OPEN:
        # Pre-market: before 9:30 AM ET → same trading day
        return "pre_market", current_date.isoformat()
    elif current_time <= MARKET_CLOSE:
        # Market hours: 9:30 AM - 4:00 PM ET → same trading day
        return "market_hours", current_date.isoformat()
    else:
        # After hours: after 4:00 PM ET → next trading day
        next_trading = get_next_trading_day(dt_et)
        return "after_hours", next_trading.strftime("%Y-%m-%d")


def classify_by_date_only(
    date_str: str,
    assume_session: str = "pre_market",
) -> Tuple[str, str]:
    """
    Classify when only date is available (no timestamp).

    This is a fallback for data sources that don't provide timestamps.
    We assume pre_market by default as most news aggregation happens overnight.

    Args:
        date_str: Date string (YYYY-MM-DD)
        assume_session: Session to assume when time is unknown

    Returns:
        Tuple of (market_session, trading_date)
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ET)
    except ValueError:
        return "unknown", date_str

    # Check if weekend
    if dt.weekday() >= 5:
        next_trading = get_next_trading_day(dt)
        return "weekend", next_trading.strftime("%Y-%m-%d")

    # Check if holiday
    if is_market_holiday(date_str):
        next_trading = get_next_trading_day(dt)
        return "weekend", next_trading.strftime("%Y-%m-%d")

    # Weekday - use assumed session
    if assume_session == "after_hours":
        next_trading = get_next_trading_day(dt)
        return "after_hours", next_trading.strftime("%Y-%m-%d")
    else:
        return assume_session, date_str


def get_session_description(session: str) -> str:
    """Get human-readable description of a market session."""
    descriptions = {
        "pre_market": "Pre-market (12:00 AM - 9:29 AM ET)",
        "market_hours": "Market hours (9:30 AM - 4:00 PM ET)",
        "after_hours": "After hours (4:01 PM - 11:59 PM ET)",
        "weekend": "Weekend/Holiday",
        "unknown": "Unknown timing",
    }
    return descriptions.get(session, session)


if __name__ == "__main__":
    # Test the classification
    test_cases = [
        "2026-02-03T08:00:00-05:00",  # Pre-market Tuesday
        "2026-02-03T10:30:00-05:00",  # Market hours Tuesday
        "2026-02-03T17:00:00-05:00",  # After hours Tuesday
        "2026-02-07T10:00:00-05:00",  # Saturday
        "2026-02-08T15:00:00-05:00",  # Sunday
        "2026-01-01T12:00:00-05:00",  # New Year's Day (holiday)
        "2026-02-03T14:30:00Z",       # UTC time (9:30 AM ET)
    ]

    print("Market Session Classification Tests")
    print("=" * 60)

    for ts in test_cases:
        session, trading_date = classify_market_session(ts)
        print(f"{ts}")
        print(f"  → Session: {session}, Trading Date: {trading_date}")
        print()
