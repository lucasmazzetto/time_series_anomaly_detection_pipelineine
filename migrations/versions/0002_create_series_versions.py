"""create series versions table

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "series_versions",
        sa.Column("series_id", sa.String(), primary_key=True),
        sa.Column("last_version", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("series_versions")
