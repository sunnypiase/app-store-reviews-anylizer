import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ReviewSample(Base):
    __tablename__ = "review_samples"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, server_default=text("gen_random_uuid()")
    )
    # BigInteger: App Store app ids can exceed the 32-bit INTEGER range.
    app_id: Mapped[int] = mapped_column(BigInteger)
    country_code: Mapped[str] = mapped_column(String(2))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    reviews: Mapped[list["Review"]] = relationship(
        back_populates="sample",
        lazy="raise",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint(
            "sample_id", "store_review_id", name="uq_reviews_sample_id_store_review_id"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, server_default=text("gen_random_uuid()")
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_samples.id", ondelete="CASCADE")
    )
    store_review_id: Mapped[str]
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    user_name: Mapped[str]
    title: Mapped[str]
    content: Mapped[str]
    rating: Mapped[int]
    app_version: Mapped[str]

    sample: Mapped["ReviewSample"] = relationship(back_populates="reviews", lazy="raise")
