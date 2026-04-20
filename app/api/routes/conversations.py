"""Conversation CRUD endpoints.

- GET /conversations             : list tất cả (summary)
- GET /conversations/{conv_id}   : chi tiết 1 hội thoại
- DELETE /conversations/{conv_id}: xoá
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import ConversationDetail, ConversationSummary
from app.core.logging_config import get_logger
from app.db.conversation_dal import (
    delete_conversation_db,
    load_all_conversations,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations():
    """Liệt kê tất cả hội thoại (chỉ trả metadata, không trả messages)."""
    convs = load_all_conversations()
    return [
        ConversationSummary(
            conv_id=conv_id,
            thread_id=data["thread_id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            message_count=len(data.get("messages") or []),
        )
        for conv_id, data in convs.items()
    ]


@router.get("/{conv_id}", response_model=ConversationDetail)
def get_conversation(conv_id: str):
    """Chi tiết 1 hội thoại (kèm messages)."""
    convs = load_all_conversations()
    if conv_id not in convs:
        raise HTTPException(status_code=404, detail="Conversation not found")

    data = convs[conv_id]
    return ConversationDetail(
        conv_id=conv_id,
        thread_id=data["thread_id"],
        title=data["title"],
        messages=data.get("messages") or [],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.delete("/{conv_id}")
def delete_conversation(conv_id: str):
    """Xoá 1 hội thoại."""
    try:
        delete_conversation_db(conv_id)
        return {"status": "ok", "conv_id": conv_id}
    except Exception as e:
        logger.exception("Delete conversation failed")
        raise HTTPException(status_code=500, detail=str(e))
