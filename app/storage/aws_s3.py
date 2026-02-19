from __future__ import annotations

import json
from typing import Any

from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries
from app.storage.storage import Storage
from app.utils.env import (
    get_aws_access_key_id,
    get_aws_s3_bucket,
    get_aws_s3_endpoint_url,
    get_aws_s3_prefix,
    get_aws_s3_region,
    get_aws_secret_access_key,
    get_aws_session_token,
)


class AWSS3Storage(Storage):
    def __init__(
        self,
        s3_client: Any | None = None,
        bucket: str | None = None,
        prefix: str | None = None,
    ) -> None:
        """@brief Initialize S3 storage adapter with client and path settings.

        @param s3_client Optional preconfigured boto3 S3 client.
        @param bucket Optional bucket override.
        @param prefix Optional object key prefix override.
        @throws ValueError If bucket configuration is missing/invalid.
        @throws RuntimeError If boto3 is unavailable for S3 backend initialization.
        """
        self._bucket = (bucket or get_aws_s3_bucket()).strip()
        self._prefix = self._normalize_prefix(
            get_aws_s3_prefix() if prefix is None else prefix
        )

        if s3_client is not None:
            self._s3 = s3_client
            return

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for STORAGE_BACKEND='s3'."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "region_name": get_aws_s3_region(),
        }

        endpoint_url = get_aws_s3_endpoint_url()
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        access_key_id = get_aws_access_key_id()
        secret_access_key = get_aws_secret_access_key()
        session_token = get_aws_session_token()

        if access_key_id:
            client_kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            client_kwargs["aws_secret_access_key"] = secret_access_key
        if session_token:
            client_kwargs["aws_session_token"] = session_token

        self._s3 = boto3.client("s3", **client_kwargs)

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        """@brief Normalize key prefix by trimming spaces and slashes.

        @param prefix Raw configured prefix value.
        @return Prefix without leading/trailing slashes.
        """
        return prefix.strip().strip("/")

    def _object_key(self, filename: str, series_id: str) -> str:
        """@brief Build final object key for the configured bucket/prefix.

        @param filename Artifact file name to be stored.
        @param series_id Series identifier used as key namespace.
        @return Final S3 object key.
        """
        base = f"{series_id}/{filename}"
        if not self._prefix:
            return base
        return f"{self._prefix}/{base}"

    @staticmethod
    def _to_uri(bucket: str, key: str) -> str:
        """@brief Return canonical S3 URI string for stored object.

        @param bucket S3 bucket name.
        @param key S3 object key.
        @return `s3://` URI for the provided object.
        """
        return f"s3://{bucket}/{key}"

    def _resolve_bucket_and_key(self, path: str) -> tuple[str, str]:
        """@brief Resolve bucket/key from either `s3://...` URI or plain key.

        @param path Stored model/data path (`s3://bucket/key` or plain key).
        @return Tuple `(bucket, key)` to be used in S3 operations.
        @throws ValueError If the provided S3 URI has invalid structure.
        """
        value = path.strip()
        if value.startswith("s3://"):
            without_scheme = value[5:]
            bucket, separator, key = without_scheme.partition("/")
            if not bucket or not separator or not key:
                raise ValueError(f"Invalid S3 URI: '{path}'")
            return bucket, key
        return self._bucket, value.lstrip("/")

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        """@brief Best-effort check for not-found responses from S3 clients.

        @param exc Exception raised by the S3 client.
        @return True when the error code maps to object-not-found semantics.
        """
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str(response.get("Error", {}).get("Code", "")).strip()
            return code in {"NoSuchKey", "404", "NotFound"}
        return False

    def _get_json_object(self, path: str) -> dict[str, Any]:
        """@brief Load and parse JSON object from S3 path.

        @param path Stored model/data path (`s3://bucket/key` or plain key).
        @return Parsed JSON object content.
        @throws FileNotFoundError If target object is missing in S3.
        @throws ValueError If the path format is invalid.
        """
        bucket, key = self._resolve_bucket_and_key(path)
        try:
            response = self._s3.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if self._is_not_found_error(exc):
                raise FileNotFoundError(path) from exc
            raise

        content = response["Body"].read().decode("utf-8")
        return json.loads(content)

    def save_state(self, series_id: str, version: int, state: ModelState) -> str:
        """@brief Save model state in S3 as JSON.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param state Serialized model state payload.
        @return S3 URI where the state was stored.
        """
        key = self._object_key(f"{series_id}_model_v{version}.json", series_id)
        body = json.dumps(state.model_dump(mode="json")).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return self._to_uri(self._bucket, key)

    def save_data(self, series_id: str, version: int, payload: TimeSeries) -> str:
        """@brief Save training data in S3 as JSON.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param payload Training data payload.
        @return S3 URI where the training data was stored.
        """
        key = self._object_key(f"{series_id}_data_v{version}.json", series_id)
        body = json.dumps(payload.model_dump(mode="json")).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return self._to_uri(self._bucket, key)

    def load_state(self, model_path: str) -> ModelState:
        """@brief Load model state from S3 JSON object.

        @param model_path Stored model path (`s3://bucket/key` or plain key).
        @return Deserialized model state payload.
        @throws FileNotFoundError If target object does not exist.
        @throws ValidationError If loaded payload does not match `ModelState`.
        """
        raw_state = self._get_json_object(model_path)
        return ModelState.model_validate(raw_state)

    def load_data(self, data_path: str) -> TimeSeries:
        """@brief Load training data from S3 JSON object.

        @param data_path Stored data path (`s3://bucket/key` or plain key).
        @return Deserialized training data payload.
        @throws FileNotFoundError If target object does not exist.
        @throws ValidationError If loaded payload does not match `TimeSeries`.
        """
        raw_data = self._get_json_object(data_path)
        return TimeSeries.model_validate(raw_data)
