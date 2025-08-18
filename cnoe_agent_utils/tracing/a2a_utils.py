# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
A2A Protocol Utilities for CNOE Agents

This module provides utilities for working with the A2A (Agent-to-Agent) protocol,
particularly for extracting metadata like trace_id from A2A RequestContext.
"""

import os
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


def extract_trace_id_from_context(context: Any) -> Optional[str]:
    """
    Extract trace_id from A2A RequestContext message metadata.
    
    This utility eliminates the need to duplicate trace_id extraction logic
    across every agent executor implementation. Always returns None if
    tracing is disabled or any error occurs.
    
    Args:
        context: A2A RequestContext object containing the message
        
    Returns:
        trace_id if found in message metadata, None otherwise
        
    Usage:
        from cnoe_agent_utils.tracing import extract_trace_id_from_context
        
        class MyAgentExecutor(AgentExecutor):
            async def execute(self, context: RequestContext, event_queue: EventQueue):
                trace_id = extract_trace_id_from_context(context)
                async for event in self.agent.stream(query, context_id, trace_id):
                    # ... handle events
    """
    try:
        # Return None immediately if tracing is disabled
        if os.getenv("ENABLE_TRACING", "false").lower() != "true":
            return None
            
        if not context or not hasattr(context, 'message'):
            return None
            
        message = context.message
        if not message:
            return None
            
        # Extract metadata from message
        message_metadata = getattr(message, 'metadata', {})
        if not isinstance(message_metadata, dict):
            return None
            
        trace_id = message_metadata.get('trace_id')
        
        if trace_id:
            logger.info(f"🔍 A2A Utils - Extracted trace_id from message metadata: {trace_id}")
        else:
            logger.debug("🔍 A2A Utils - No trace_id found in message metadata")
            
        return trace_id
        
    except Exception as e:
        logger.error(f"❌ A2A Utils - Error extracting trace_id: {e}")
        return None