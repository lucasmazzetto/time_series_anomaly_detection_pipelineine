from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.sql.dml import Insert

from app.database.anomaly_detection_record import AnomalyDetectionRecord
from app.database.series_version_record import SeriesVersionRecord


def test_anomaly_detection_record_build_sets_fields_and_timestamps():
    """@brief Verify build initializes fields and timestamps.

    @details Ensures the factory uses a consistent timestamp and sets
    model/data paths and version as provided.
    """
    fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    with patch(
        "app.database.anomaly_detection_record.AnomalyDetectionRecord._timestamp",
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
        "app.database.anomaly_detection_record.SeriesVersionRecord.next_version",
        return_value=4,
    ) as next_version_mock:
        version = AnomalyDetectionRecord.save(session, record)

    assert version == 4
    assert record.version == 4
    next_version_mock.assert_called_once_with(session, "series_y")
    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(record)


def test_anomaly_detection_record_save_keeps_existing_version():
    """@brief Verify save keeps an existing version untouched.

    @details Ensures no version allocation occurs when version is already set,
    while persistence still happens.
    """
    session = MagicMock()
    record = AnomalyDetectionRecord.build(series_id="series_z", version=2)

    with patch(
        "app.database.anomaly_detection_record.SeriesVersionRecord.next_version"
    ) as next_version_mock:
        version = AnomalyDetectionRecord.save(session, record)

    assert version == 2
    assert record.version == 2
    next_version_mock.assert_not_called()
    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(record)


def test_anomaly_detection_record_update_persists_changes():
    """@brief Verify update persists model/data path changes.

    @details Ensures update writes the new paths and commits through the
    attached session.
    """
    record = AnomalyDetectionRecord.build(series_id="series_u", version=1)
    session = MagicMock()

    with patch(
        "app.database.anomaly_detection_record.object_session",
        return_value=session,
    ):
        record.update(model_path="/tmp/m.pkl", data_path="/tmp/d.json")

    assert record.model_path == "/tmp/m.pkl"
    assert record.data_path == "/tmp/d.json"
    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()


def test_anomaly_detection_record_update_raises_without_session():
    """@brief Verify update fails without an attached session.

    @details Ensures a clear RuntimeError is raised when the record is
    detached and cannot be persisted.
    """
    record = AnomalyDetectionRecord.build(series_id="series_n", version=1)

    with patch(
        "app.database.anomaly_detection_record.object_session", return_value=None
    ):
        with pytest.raises(RuntimeError, match="not attached to a session"):
            record.update(model_path="/tmp/m.pkl", data_path="/tmp/d.json")


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
