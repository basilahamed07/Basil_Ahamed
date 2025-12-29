import logging
from typing import Dict, Optional
from sqlalchemy.orm import Session
from ...database.models import DiscoveryQuerySession, DiscoveryQueryMessage
from .chatbot import get_chatbot, ChatbotException
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatbotService:
    """Service layer for chatbot with integrated session/message management"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.chatbot = get_chatbot()
    
    def _create_session(
        self, 
        user_id: int, 
        title: str, 
        application_id: int
    ) -> int:
        """Create a new discovery session"""
        try:
            new_session = DiscoveryQuerySession(
                UserId=user_id,
                Title=title,
                IsFavorite=False,
                application_id=application_id,
                CreatedAt=datetime.utcnow(),
                UpdatedAt=datetime.utcnow()
            )
            
            self.db.add(new_session)
            self.db.commit()
            self.db.refresh(new_session)
            
            logger.info(f"✅ Created new session: {new_session.Id}")
            return new_session.Id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Failed to create session: {e}")
            raise ChatbotException(f"Failed to create session: {str(e)}", 500)
    
    def _log_message(
        self, 
        session_id: int, 
        sender: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> int:
        """Log a message to the session"""
        try:
            message = DiscoveryQueryMessage(
                SessionId=session_id,
                Sender=sender,
                Content=content,
                Metadata=metadata or {}
            )
            
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)
            
            logger.info(f"✅ Logged {sender} message to session {session_id}")
            return message.Id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Failed to log message: {e}")
            raise ChatbotException(f"Failed to log message: {str(e)}", 500)
    
    def _validate_session(self, session_id: int) -> bool:
        """Validate that session exists"""
        session = self.db.query(DiscoveryQuerySession).filter(
            DiscoveryQuerySession.Id == session_id
        ).first()
        return session is not None
    
    def process_query(
        self,
        query: str,
        application_id: int,
        user_id: int,
        session_id: Optional[int] = None,
        title: Optional[str] = None,
        customer_id: Optional[int] = None,
        connector_type: Optional[str] = None,
        asset_type_id: Optional[int] = None
    ) -> Dict:
        """
        Process chatbot query with full session/message management.
        
        This is the main entry point that:
        1. Creates session if needed
        2. Logs user message
        3. Gets chatbot response
        4. Logs assistant response
        5. Returns complete response with session_id
        """
        try:
            # ================================================================
            # STEP 1: SESSION MANAGEMENT
            # ================================================================
            
            if not session_id:
                # Create new session
                if not title:
                    raise ChatbotException(
                        "Title is required when creating a new session", 
                        400
                    )
                
                logger.info(f"📝 Creating new session: '{title}'")
                session_id = self._create_session(
                    user_id=user_id,
                    title=title,
                    application_id=application_id
                )
            else:
                # Validate existing session
                logger.info(f"✅ Using existing session: {session_id}")
                if not self._validate_session(session_id):
                    raise ChatbotException(
                        f"Session {session_id} not found", 
                        404
                    )
            
            # ================================================================
            # STEP 2: LOG USER MESSAGE
            # ================================================================
            
            logger.info(f"💬 Logging user message...")
            self._log_message(
                session_id=session_id,
                sender="user",
                content=query,
                metadata={"language": "en"}
            )
            
            # ================================================================
            # STEP 3: GET CHATBOT RESPONSE
            # ================================================================
            
            logger.info(f"🤖 Processing chatbot query...")
            chatbot_result = self.chatbot.ask_question(
                query=query,
                application_id=application_id,
                customer_id=customer_id,
                connector_type=connector_type,
                db=self.db
            )
            
            answer = chatbot_result["answer"]
            sources = chatbot_result["sources"]
            
            logger.info(f"✅ Generated response ({len(answer)} chars)")
            
            # ================================================================
            # STEP 4: LOG ASSISTANT RESPONSE
            # ================================================================
            
            # Extract metadata from chatbot result
            assistant_metadata = {
                "confidence": "high",
                "sources_count": len(sources),
                "source_files": [s.get("file_path") for s in sources[:10]]  # Limit to 10
            }
            
            # Add any additional metadata from chatbot result
            if "metadata" in chatbot_result:
                assistant_metadata.update(chatbot_result["metadata"])
            
            logger.info(f"💬 Logging assistant response...")
            self._log_message(
                session_id=session_id,
                sender="assistant",
                content=answer,
                metadata=assistant_metadata
            )
            
            # ================================================================
            # STEP 5: RETURN COMPLETE RESPONSE
            # ================================================================
            
            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "files_processed": len(set(s.get("file_path") for s in sources)),
                "source_files": list(set(s.get("file_path") for s in sources[:20]))
            }
            
        except ChatbotException:
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in process_query: {e}", exc_info=True)
            raise ChatbotException(f"Failed to process query: {str(e)}", 500)