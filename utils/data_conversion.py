import datetime
import decimal
import logging
import uuid
from typing import Any, Optional, Union, TypeVar, Generic, Type

logger = logging.getLogger(__name__)

def parse_date(value: Any) -> Optional[datetime.date]:
    """
    Centralized date parsing with multiple strategies.
    
    Args:
        value: Input value to parse as date
        
    Returns:
        Parsed datetime.date object or None if parsing fails
    """
    if value is None:
        return None
    
    if isinstance(value, datetime.date):
        return value
    
    if isinstance(value, datetime.datetime):
        return value.date()
    
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("none", "null", "na", "n/a"):
            return None
            
        # Strategy 1: Try dateutil parser (robust)
        try:
            from dateutil import parser
            try:
                # Try with dayfirst=True (DD/MM/YYYY format)
                return parser.parse(value, dayfirst=True).date()
            except Exception:
                # Fall back to parser's default behavior
                return parser.parse(value).date()
        except ImportError:
            logger.debug("dateutil not installed, falling back to fixed formats")
        except Exception:
            pass
            
        # Strategy 2: Try fixed formats
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"]
        for fmt in formats:
            try:
                return datetime.datetime.strptime(value, fmt).date()
            except ValueError:
                continue
                
        # Strategy 3: Try dateparser (most flexible, but slowest)
        try:
            import dateparser
            dt = dateparser.parse(value)
            if dt:
                return dt.date()
        except ImportError:
            logger.debug("dateparser not installed, cannot use advanced parsing")
        except Exception:
            pass
    
    logger.warning(f"Could not parse date from: {value} ({type(value)})")
    return None

def parse_decimal(value: Any) -> Optional[decimal.Decimal]:
    """
    Centralized decimal parsing with accounting format support.
    
    Args:
        value: Input value to parse as decimal
        
    Returns:
        Parsed decimal.Decimal object or None if parsing fails
    """
    if value is None:
        return None
        
    if isinstance(value, decimal.Decimal):
        return value
        
    if isinstance(value, (int, float)):
        return decimal.Decimal(str(value))
        
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("none", "null", "na", "n/a"):
            return None
            
        # Handle accounting format (parentheses for negative)
        if '(' in value and ')' in value:
            value = "-" + value.replace('(', '').replace(')', '')
            
        # Remove currency symbols and formatting
        for char in ['$', '£', '€', '¥', '₹', '฿', '₽', '₩', ',']:
            value = value.replace(char, '')
            
        try:
            return decimal.Decimal(value)
        except Exception:
            pass
    
    logger.warning(f"Could not parse decimal from: {value} ({type(value)})")
    return None

def parse_integer(value: Any) -> Optional[int]:
    """
    Centralized integer parsing with support for various formats.
    
    Args:
        value: Input value to parse as integer
        
    Returns:
        Parsed integer or None if parsing fails
    """
    if value is None:
        return None
        
    if isinstance(value, int):
        return value
        
    if isinstance(value, float):
        return int(value)
        
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("none", "null", "na", "n/a"):
            return None
            
        # Remove commas from formatted numbers
        value = value.replace(',', '')
        
        try:
            # Handle floating point strings (e.g., "123.0")
            if '.' in value:
                return int(float(value))
            return int(value)
        except Exception:
            pass
    
    logger.warning(f"Could not parse integer from: {value} ({type(value)})")
    return None

def parse_boolean(value: Any) -> Optional[bool]:
    """
    Centralized boolean parsing with support for various formats.
    
    Args:
        value: Input value to parse as boolean
        
    Returns:
        Parsed boolean or None if parsing fails
    """
    if value is None:
        return None
        
    if isinstance(value, bool):
        return value
        
    if isinstance(value, (int, float)):
        return bool(value)
        
    if isinstance(value, str):
        value = value.lower().strip()
        if value == "" or value in ("none", "null", "na", "n/a"):
            return None
            
        if value in ('true', 'yes', 't', 'y', '1', 'on'):
            return True
        if value in ('false', 'no', 'f', 'n', '0', 'off'):
            return False
    
    logger.warning(f"Could not parse boolean from: {value} ({type(value)})")
    return None

def truncate_string(value: Any, max_length: int = 255) -> Optional[str]:
    """
    Convert value to string and truncate if necessary.
    
    Args:
        value: Value to convert to string
        max_length: Maximum allowed length
        
    Returns:
        Truncated string or None
    """
    if value is None:
        return None
        
    if value == "":
        return ""
        
    str_value = str(value)
    if len(str_value) > max_length:
        logger.debug(f"Truncating string from {len(str_value)} to {max_length} characters")
        return str_value[:max_length]
        
    return str_value