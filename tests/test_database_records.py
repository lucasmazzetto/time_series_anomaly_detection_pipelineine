from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.sql.dml import Insert

from app.database.anomaly_detection import AnomalyDetectionRecord
from app.database.series_version import SeriesVersionRecord


def test_anomaly_detection_record_build_sets_fields_and_timestamps():
    """@brief Verify build initializes fields and timestamps.

    @details Ensures the factory uses a consistent timestamp and sets
    model/data paths and version as provided.
    """
    fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    with patch(
        "app.database.anomaly_detection.AnomalyDetectionRecord._timestamp",
        return_value=fixed_time,
    ):
        record = AnomalyDetectionRecord.build(
            series_id="series_x",
            version=None,
            model_path=None,
            data_path=None,
        )

    assert record.series_id == "series_x"
    assert record.version is None
    assert record.model_path is None
    assert record.data_path is None
    assert record.created_at == fixed_time
    assert record.updated_at == fixed_time


def test_anomaly_detection_record_save_assigns_version_and_persists():
    """@brief Verify save assigns version and persists the record.

    @details Ensures a missing version is allocated via SeriesVersionRecord
    and that session persistence hooks are invoked.
    """
    session = MagicMock()
    record = AnomalyDetectionRecord.build(series_id="series_y")

    with patch(
        "app.database.anomaly_detection.SeriesVersionRecord.next_version",
        return_value=4,
    ) as next_version_mock:
        version = AnomalyDetectionRecord.save(session, record)

    assert version == 4
    assert record.version == 4
    next_version_mock.assert_called_once_with(session, "series_y")
    session.add.assert_called_once_with(record)
    session.flush.assert_called_once()


def test_anomaly_detection_record_save_keeps_existing_version():
    """@brief Verify save keeps an existing version untouched.

    @details Ensures no version allocation occurs when version is already set,
    while persistence still happens.
    """
    session = MagicMock()
    record = AnomalyDetectionRecord.build(series_id="series_z", version=2)

    with patch(
        "app.database.anomaly_detection.SeriesVersionRecord.next_version"
    ) as next_version_mock:
        version = AnomalyDetectionRecord.save(session, record)

    assert version == 2
    assert record.version == 2
    next_version_mock.assert_not_called()
    session.add.assert_called_once_with(record)
    session.flush.assert_called_once()


def test_anomaly_detection_record_update_updates_changes_without_commit():
    """@brief Verify update sets model/data path changes in-memory.

    @details Ensures update writes the new paths and validates session
    attachment without issuing an implicit commit.
    """
    record = AnomalyDetectionRecord.build(series_id="series_u", version=1)
    session = MagicMock()

    with patch(
        "app.database.anomaly_detection.object_session",
        return_value=session,
    ):
        record.update(model_path="/tmp/m.pkl", data_path="/tmp/d.json")

    assert record.model_path == "/tmp/m.pkl"
    assert record.data_path == "/tmp/d.json"
    session.commit.assert_not_called()


def test_anomaly_detection_record_update_raises_without_session():
    """@brief Verify update fails without an attached session.

    @details Ensures a clear RuntimeError is raised when the record is
    detached and cannot be persisted.
    """
    record = AnomalyDetectionRecord.build(series_id="series_n", version=1)

    with patch(
        "app.database.anomaly_detection.object_session", return_value=None
    ):
        with pytest.raises(RuntimeError, match="not attached to a session"):
            record.update(model_path="/tmp/m.pkl", data_path="/tmp/d.json")


def test_anomaly_detection_record_commit_persists_transaction():
    """@brief Verify commit persists through the attached session."""
    record = AnomalyDetectionRecord.build(series_id="series_commit", version=1)
    session = MagicMock()

    with patch(
        "app.database.anomaly_detection.object_session",
        return_value=session,
    ):
        record.commit()

    session.commit.assert_called_once()


def test_anomaly_detection_record_commit_raises_without_session():
    """@brief Verify commit fails when record is detached."""
    record = AnomalyDetectionRecord.build(series_id="series_detached", version=1)

    with patch(
        "app.database.anomaly_detection.object_session", return_value=None
    ):
        with pytest.raises(RuntimeError, match="not attached to a session"):
            record.commit()


def test_series_version_record_next_version_executes_insert():
    """@brief Verify next_version builds and executes an insert statement.

    @details Ensures the method executes a SQLAlchemy Insert and returns
    the scalar version produced by the statement.
    """
    session = MagicMock()
    result = MagicMock()
    result.scalar_one.return_value = 9
    session.execute.return_value = result

    version = SeriesVersionRecord.next_version(session, "series_v")

    assert version == 9
    assert session.execute.call_count == 1
    stmt = session.execute.call_args[0][0]
    assert isinstance(stmt, Insert)
    result.scalar_one.assert_called_once()


def test_series_version_record_count_series_returns_row_count():
    """@brief Verify count_series returns row count from SQL aggregation."""
    session = MagicMock()
    query = session.query.return_value
    query.scalar.return_value = 2

    payload = SeriesVersionRecord.count_series(session)

    assert payload == 2
    assert session.query.call_count == 1
    assert "count(" in str(session.query.call_args[0][0]).lower()
    query.scalar.assert_called_once()


def test_anomaly_detection_record_to_dict_serializes_fields():
    """@brief Verify to_dict serializes row fields in API-friendly format.

    @details Ensures primitive fields are preserved and datetime fields are
    converted to ISO-8601 strings.
    """
    created = datetime(2025, 2, 10, 12, 0, 0, tzinfo=timezone.utc)
    updated = datetime(2025, 2, 11, 13, 30, 0, tzinfo=timezone.utc)

    record = AnomalyDetectionRecord(
        series_id="series_dict",
        version=11,
        model_path="/tmp/m.pkl",
        data_path="/tmp/d.json",
        created_at=created,
        updated_at=updated,
    )

    payload = record.to_dict()

    assert payload["series_id"] == "series_dict"
    assert payload["version"] == 11
    assert payload["model_path"] == "/tmp/m.pkl"
    assert payload["data_path"] == "/tmp/d.json"
    assert payload["created_at"] == created.isoformat()
    assert payload["updated_at"] == updated.isoformat()


def test_anomaly_detection_record_get_last_model_returns_serialized_latest():
    """@brief Verify get_last_model returns the serialized latest row.

    @details Ensures the latest row is read through the query chain and
    exposed as a serialized dictionary payload.
    """
    session = MagicMock()
    created = datetime(2025, 2, 12, 10, 0, 0, tzinfo=timezone.utc)
    updated = datetime(2025, 2, 12, 11, 0, 0, tzinfo=timezone.utc)

    record = AnomalyDetectionRecord(
        series_id="series_latest",
        version=5,
        model_path="/tmp/latest.pkl",
        data_path="/tmp/latest.json",
        created_at=created,
        updated_at=updated,
    )

    query = session.query.return_value
    query.filter.return_value.order_by.return_value.first.return_value = record

    payload = AnomalyDetectionRecord.get_last_model(session, "series_latest")

    assert payload["series_id"] == "series_latest"
    assert payload["version"] == 5
    assert payload["model_path"] == "/tmp/latest.pkl"
    assert payload["created_at"] == created.isoformat()
    session.query.assert_called_once_with(AnomalyDetectionRecord)


def test_anomaly_detection_record_get_last_model_raises_when_missing():
    """@brief Verify get_last_model raises when no series rows exist.

    @details Ensures callers receive a clear ValueError when metadata for
    a target series cannot be found.
    """
    session = MagicMock()
    query = session.query.return_value
    query.filter.return_value.order_by.return_value.first.return_value = None

    with pytest.raises(ValueError, match="No model found for series_id"):
        AnomalyDetectionRecord.get_last_model(session, "missing_series")


def test_anomaly_detection_record_get_model_version_returns_serialized_row():
    """@brief Verify get_model_version returns requested serialized row.

    @details Ensures explicit version lookups resolve through query filters
    and expose serialized model metadata.
    """
    session = MagicMock()
    created = datetime(2025, 2, 13, 10, 0, 0, tzinfo=timezone.utc)
    updated = datetime(2025, 2, 13, 11, 0, 0, tzinfo=timezone.utc)

    record = AnomalyDetectionRecord(
        series_id="series_versioned",
        version=3,
        model_path="/tmp/v3.pkl",
        data_path="/tmp/v3.json",
        created_at=created,
        updated_at=updated,
    )

    query = session.query.return_value
    query.filter.return_value.first.return_value = record

    payload = AnomalyDetectionRecord.get_model_version(session, "series_versioned", 3)

    assert payload["series_id"] == "series_versioned"
    assert payload["version"] == 3
    assert payload["model_path"] == "/tmp/v3.pkl"
    assert payload["updated_at"] == updated.isoformat()
    session.query.assert_called_once_with(AnomalyDetectionRecord)


def test_anomaly_detection_record_get_model_version_raises_when_missing():
    """@brief Verify get_model_version raises when requested row is absent.

    @details Ensures missing explicit version lookups produce a clear
    ValueError for service-layer HTTP mapping.
    """
    session = MagicMock()
    query = session.query.return_value
    query.filter.return_value.first.return_value = None

    with pytest.raises(ValueError, match="Model version '7' not found"):
        AnomalyDetectionRecord.get_model_version(session, "series_missing", 7)
