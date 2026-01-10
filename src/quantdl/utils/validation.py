"""
Input validation utilities for SQL injection prevention and data integrity.
"""

import re
import datetime as dt


def validate_date_string(date_str: str) -> str:
    """
    Validate and sanitize date string to prevent SQL injection.

    :param date_str: Date string in 'YYYY-MM-DD' format
    :return: Validated date string
    :raises ValueError: If date string is invalid or malformed

    Example:
        >>> validate_date_string('2024-01-15')
        '2024-01-15'
        >>> validate_date_string('2024-13-01')  # Invalid month
        ValueError: Invalid date value: 2024-13-01
        >>> validate_date_string("'; DROP TABLE--")  # SQL injection attempt
        ValueError: Invalid date format
    """
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    # Also validate it's a valid date
    try:
        dt.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date value: {date_str}. {e}")

    return date_str


def validate_permno(permno: int) -> int:
    """
    Validate CRSP permanent number to ensure it's a positive integer.

    :param permno: CRSP permanent number (unique security identifier)
    :return: Validated permno
    :raises ValueError: If permno is invalid (not an integer or non-positive)

    Example:
        >>> validate_permno(10516)
        10516
        >>> validate_permno(-100)
        ValueError: Invalid permno: -100
        >>> validate_permno("10516")  # String instead of int
        ValueError: Invalid permno: "10516"
    """
    if not isinstance(permno, int) or permno <= 0:
        raise ValueError(f"Invalid permno: {permno}. Must be a positive integer")
    return permno


def validate_year(year: int, min_year: int = 1900, max_year: int = 2100) -> int:
    """
    Validate year parameter to ensure it's within reasonable bounds.

    :param year: Year to validate
    :param min_year: Minimum valid year (default: 1900)
    :param max_year: Maximum valid year (default: 2100)
    :return: Validated year
    :raises ValueError: If year is invalid

    Example:
        >>> validate_year(2024)
        2024
        >>> validate_year(1800)
        ValueError: Invalid year: 1800
        >>> validate_year("2024")  # String instead of int
        ValueError: Invalid year: "2024"
    """
    if not isinstance(year, int) or year < min_year or year > max_year:
        raise ValueError(
            f"Invalid year: {year}. Must be an integer between {min_year} and {max_year}"
        )
    return year


def validate_month(month: int) -> int:
    """
    Validate month parameter to ensure it's between 1 and 12.

    :param month: Month to validate (1-12)
    :return: Validated month
    :raises ValueError: If month is invalid

    Example:
        >>> validate_month(6)
        6
        >>> validate_month(13)
        ValueError: Invalid month: 13
        >>> validate_month("6")  # String instead of int
        ValueError: Invalid month: "6"
    """
    if not isinstance(month, int) or month < 1 or month > 12:
        raise ValueError(f"Invalid month: {month}. Must be an integer between 1 and 12")
    return month
