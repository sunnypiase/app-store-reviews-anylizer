from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from app.database import get_db_session


DbSession = Annotated[AsyncSession, Depends(get_db_session)]