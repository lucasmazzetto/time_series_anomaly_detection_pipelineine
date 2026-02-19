import json

import pytest

from app.schemas.data_point import DataPoint
from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries
from app.storage.aws_s3 import AWSS3Storage
from app.storage.local_storage import LocalStorage
from app.utils.storage import get_storage


class _FakeBody:
    def __init__(self, payload: bytes) -> None:
        """@brief Build a fake streaming body used by mocked S3 responses."""
        self._payload = payload

    def read(self) -> bytes:
        """@brief Return full payload bytes, emulating boto3 stream read."""
        return self._payload


class _FakeS3ClientError(Exception):
    def __init__(self, code: str) -> None:
        """@brief Build a fake S3 error exposing AWS-like `response` shape."""
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _FakeS3Client:
    def __init__(self) -> None:
        """@brief Initialize in-memory object store keyed by bucket/key."""
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        """@brief Persist a mocked S3 object payload in memory."""
        assert ContentType == "application/json"
        self._objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        """@brief Fetch mocked S3 object or raise a not-found style error."""
        payload = self._objects.get((Bucket, Key))
        if payload is None:
            raise _FakeS3ClientError("NoSuchKey")
        return {"Body": _FakeBody(payload)}


def _sample_series() -> TimeSeries:
    """@brief Return a valid sample series for storage roundtrip tests."""
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=2.0),
            DataPoint(timestamp=1_700_000_002, value=3.0),
        ]
    )


def test_aws_s3_storage_roundtrip_state_and_data():
    """@brief Ensure AWSS3Storage saves and reloads state/data losslessly."""
    client = _FakeS3Client()
    storage = AWSS3Storage(s3_client=client, bucket="bucket-a", prefix="models")
    state = ModelState(model="mock", parameters={"mean": 2.0, "std": 1.0})
    payload = _sample_series()

    model_uri = storage.save_state("series_01", 1, state)
    data_uri = storage.save_data("series_01", 1, payload)

    assert model_uri == "s3://bucket-a/models/series_01/series_01_model_v1.json"
    assert data_uri == "s3://bucket-a/models/series_01/series_01_data_v1.json"

    loaded_state = storage.load_state(model_uri)
    loaded_data = storage.load_data(data_uri)

    assert loaded_state == state
    assert loaded_data == payload


def test_aws_s3_storage_load_raises_file_not_found_for_missing_object():
    """@brief Ensure missing S3 objects are mapped to FileNotFoundError."""
    storage = AWSS3Storage(s3_client=_FakeS3Client(), bucket="bucket-a", prefix="")

    with pytest.raises(FileNotFoundError):
        storage.load_state("s3://bucket-a/missing/model.json")


def test_aws_s3_storage_load_accepts_plain_object_key():
    """@brief Ensure loader supports plain keys in addition to full S3 URIs."""
    client = _FakeS3Client()
    state = ModelState(model="mock", parameters={"alpha": 1})
    key = "prefix/series_x/series_x_model_v2.json"
    client.put_object(
        Bucket="bucket-b",
        Key=key,
        Body=json.dumps(state.model_dump(mode="json")).encode("utf-8"),
        ContentType="application/json",
    )

    storage = AWSS3Storage(s3_client=client, bucket="bucket-b", prefix="")
    loaded = storage.load_state(key)

    assert loaded == state


def test_get_storage_returns_local_backend(monkeypatch: pytest.MonkeyPatch):
    """@brief Ensure storage factory resolves LocalStorage for local backend."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    storage = get_storage()
    assert isinstance(storage, LocalStorage)


def test_get_storage_returns_s3_backend(monkeypatch: pytest.MonkeyPatch):
    """@brief Ensure storage factory resolves AWSS3Storage for S3 backend."""
    import app.storage.aws_s3 as aws_s3_module

    class DummyS3Storage:
        pass

    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("AWS_S3_BUCKET", "bucket-for-tests")
    monkeypatch.setattr(aws_s3_module, "AWSS3Storage", DummyS3Storage)

    storage = get_storage()
    assert isinstance(storage, DummyS3Storage)
