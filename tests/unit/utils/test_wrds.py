import pandas as pd
import pytest
from sqlalchemy.exc import PendingRollbackError, OperationalError

from quantdl.utils.wrds import raw_sql_with_retry


def test_raw_sql_with_retry_success():
    db = type("Db", (), {})()
    db.raw_sql = lambda sql, **kwargs: pd.DataFrame({"a": [1]})

    result = raw_sql_with_retry(db, "select 1")

    assert result["a"][0] == 1


def test_raw_sql_with_retry_rolls_back_and_retries():
    calls = {"count": 0}

    class Conn:
        def __init__(self) -> None:
            self.rolled_back = False

        def rollback(self) -> None:
            self.rolled_back = True

    class Db:
        def __init__(self) -> None:
            self.connection = Conn()

        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PendingRollbackError("pending rollback")
            return pd.DataFrame({"a": [2]})

    db = Db()

    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert db.connection.rolled_back is True
    assert result["a"][0] == 2


def test_raw_sql_with_retry_without_connection_attribute():
    calls = {"count": 0}

    class Db:
        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PendingRollbackError("pending rollback")
            return pd.DataFrame({"a": [3]})

    db = Db()

    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert result["a"][0] == 3


def test_raw_sql_with_retry_rollback_fails():
    """Test rollback failure is handled gracefully (bare except at line 34-35)"""
    calls = {"count": 0}

    class Conn:
        def rollback(self) -> None:
            raise Exception("Rollback failed")

    class Db:
        def __init__(self) -> None:
            self.connection = Conn()

        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PendingRollbackError("pending rollback")
            return pd.DataFrame({"a": [4]})

    db = Db()

    # Should succeed despite rollback failure
    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert result["a"][0] == 4


def test_raw_sql_with_retry_pending_rollback_max_retries():
    """Test PendingRollbackError exceeds max retries (line 39)"""
    calls = {"count": 0}

    class Conn:
        def rollback(self) -> None:
            pass

    class Db:
        def __init__(self) -> None:
            self.connection = Conn()

        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            raise PendingRollbackError("pending rollback")

    db = Db()

    with pytest.raises(PendingRollbackError, match="pending rollback"):
        raw_sql_with_retry(db, "select 1", max_retries=2)

    assert calls["count"] == 3  # Initial + 2 retries


def test_raw_sql_with_retry_operational_error_connection_closed_success():
    """Test OperationalError with connection closed - retry succeeds (lines 40-45)"""
    calls = {"count": 0}

    class Db:
        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OperationalError("statement", "params", "server closed the connection unexpectedly")
            return pd.DataFrame({"a": [5]})

    db = Db()

    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert result["a"][0] == 5


def test_raw_sql_with_retry_operational_error_closed_connection_variant():
    """Test OperationalError with 'closed the connection' variant (lines 43-45)"""
    calls = {"count": 0}

    class Db:
        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OperationalError("statement", "params", "connection was closed the connection")
            return pd.DataFrame({"a": [6]})

    db = Db()

    result = raw_sql_with_retry(db, "select 1")

    assert calls["count"] == 2
    assert result["a"][0] == 6


def test_raw_sql_with_retry_operational_error_max_retries():
    """Test OperationalError connection closed exceeds max retries (line 47)"""
    calls = {"count": 0}

    class Db:
        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            raise OperationalError("statement", "params", "server closed the connection")

    db = Db()

    with pytest.raises(OperationalError, match="closed the connection"):
        raw_sql_with_retry(db, "select 1", max_retries=2)

    assert calls["count"] == 3  # Initial + 2 retries


def test_raw_sql_with_retry_operational_error_other():
    """Test OperationalError with non-connection error raises immediately (lines 48-50)"""
    calls = {"count": 0}

    class Db:
        def raw_sql(self, sql, **kwargs):
            calls["count"] += 1
            raise OperationalError("statement", "params", "some other operational error")

    db = Db()

    with pytest.raises(OperationalError, match="other operational error"):
        raw_sql_with_retry(db, "select 1", max_retries=2)

    # Should not retry for other operational errors
    assert calls["count"] == 1
