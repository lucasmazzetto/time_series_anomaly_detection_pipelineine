from app.storage.local_storage import LocalStorage
from app.storage.storage import Storage
from app.utils.env import get_storage_backend


def get_storage() -> Storage:
    """@brief Resolve configured storage adapter for artifacts.

    @return Storage implementation selected by `STORAGE_BACKEND`.
    @throws ValueError If `STORAGE_BACKEND` has an invalid value.
    @throws RuntimeError If backend is unsupported after validation.
    """
    backend = get_storage_backend()

    if backend == "local":
        return LocalStorage()

    if backend == "s3":
        # Lazy import keeps boto3 optional unless S3 backend is enabled.
        from app.storage.aws_s3 import AWSS3Storage

        return AWSS3Storage()

    # Defensive guard: get_storage_backend already validates accepted values.
    raise RuntimeError(f"Unsupported storage backend: '{backend}'.")
