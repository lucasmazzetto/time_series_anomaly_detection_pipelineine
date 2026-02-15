"""create anomaly detection models table

Revision ID: 0001
Revises: 
Create Date: 2026-02-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "anomaly_detection_models",
        sa.Column("series_id", sa.String(), primary_key=True),
        sa.Column("version", sa.Integer(), primary_key=True),
        sa.Column("model_path", sa.String(), nullable=True),
        sa.Column("data_path", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_anomaly_detection_models_series_id",
        "anomaly_detection_models",
        ["series_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_anomaly_detection_models_series_id",
        table_name="anomaly_detection_models",
    )
    op.drop_table("anomaly_detection_models")
