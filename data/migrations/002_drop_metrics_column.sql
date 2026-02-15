BEGIN;

ALTER TABLE anomaly_detection_models RENAME TO anomaly_detection_models_old;

CREATE TABLE anomaly_detection_models (
    series_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    model_path TEXT,
    data_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (series_id, version)
);

INSERT INTO anomaly_detection_models (series_id, version, model_path, data_path, created_at, updated_at)
SELECT series_id, version, model_path, data_path, created_at, updated_at FROM anomaly_detection_models_old;

DROP TABLE anomaly_detection_models_old;

COMMIT;
