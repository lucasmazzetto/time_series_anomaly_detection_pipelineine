CREATE TABLE IF NOT EXISTS anomaly_detection_models (
    series_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    model_path TEXT,
    data_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (series_id, version)
);
