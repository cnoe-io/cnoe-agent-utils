"""Extended DeepAgents Middleware

This module extends the default DeepAgents middleware with additional specialized
capabilities for advanced agent workflows.
"""

from langgraph.runtime import Runtime
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import AIMessage, ToolMessage, RemoveMessage
from langgraph.types import Command
from typing import Any
import logging
import uuid
from deepagents.fs import FS

logger = logging.getLogger(__name__)

###########################
# Call Tool With File Arg Middleware
###########################

class CallToolWithFileArgMiddleware(AgentMiddleware):
    """
    By default, substitute any tool-call argument values that are file paths
    in the in-memory FS with their contents for all non-filesystem tools.
    """

    FS_TOOL_NAMES = {"ls", "read_file", "search_file", "tool_result_to_file", "edit_file"}

    @staticmethod
    def _replace_fs_content(obj: Any) -> Any:
        """Recursively replace strings that match file paths in FS with their contents."""
        if isinstance(obj, str) and obj in FS:
            try:
                content_len = len(FS[obj]) if isinstance(FS[obj], str) else "n/a"
            except Exception:
                content_len = "n/a"
            print(f"[CallToolWithFileArgMiddleware] _replace_fs_content: replacing file path '{obj}' with file contents (len={content_len})")
            return FS[obj]
        if isinstance(obj, list):
            return [CallToolWithFileArgMiddleware._replace_fs_content(x) for x in obj]
        if isinstance(obj, dict):
            return {k: CallToolWithFileArgMiddleware._replace_fs_content(v) for k, v in obj.items()}
        return obj

    def modify_model_request(self, request: ModelRequest, agent_state: AgentState) -> ModelRequest:
        from deepagents.prompts import CALL_TOOL_WITH_FILE_ARG_SYSTEM_PROMPT
        request.system_prompt = request.system_prompt + "\n\n" + CALL_TOOL_WITH_FILE_ARG_SYSTEM_PROMPT
        return request

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | Command | None:
        """
        Inspect the last assistant message. If it contains a tool call with file arg emit two messages:
        1) a ToolMessage with empty content acknowledging the original call (same tool_call_id),
        2) a rewritten AIMessage that contains the actual target tool calls where file-path
        arguments are replaced with their contents, and each rewritten tool call uses a new id.
        """
        messages = state.get("messages") or []
        if not messages:
            print("[CallToolWithFileArgMiddleware] after_model: no messages; skipping")
            return None

        # Locate the most recent assistant message (AIMessage or dict-style)
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
                last_ai = msg
                break
        if last_ai is None:
            print("[CallToolWithFileArgMiddleware] after_model: no assistant message found; skipping")
            return None
        # Extract tool calls and original content
        if isinstance(last_ai, AIMessage):
            tool_calls = getattr(last_ai, "tool_calls", None) or []
            original_content = getattr(last_ai, "content", "") or ""
        else:
            tool_calls = last_ai.get("tool_calls") or []
            original_content = last_ai.get("content") or ""
        ack_msgs: list[ToolMessage] = []

        # Gather existing ToolMessage IDs to avoid duplicate processing
        existing_tool_call_ids: set[str] = set()
        try:
            for m in messages:
                if isinstance(m, ToolMessage):
                    tid = getattr(m, "tool_call_id", None)
                    if tid:
                        existing_tool_call_ids.add(tid)
                elif isinstance(m, dict):
                    tid = m.get("tool_call_id")
                    if tid:
                        existing_tool_call_ids.add(tid)
        except Exception:
            pass

        if not tool_calls:
            return None

        mutated = False
        new_tool_calls = []
        for call in tool_calls:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args") or {}
                cid = call.get("id")
            else:
                name = getattr(call, "name", None)
                args = getattr(call, "args", {}) or {}
                cid = getattr(call, "id", None)

            # If we already have a ToolMessage acknowledging this call ID, skip mutation/ack
            if cid and cid in existing_tool_call_ids:
                new_tool_calls.append(call)
                continue

            norm_name = name.replace("functions.", "") if name else None

            # Default behavior: mutate args for all non-filesystem tools
            if norm_name in self.FS_TOOL_NAMES:
                # Leave filesystem tools unchanged; they expect file paths
                new_tool_calls.append(call)
                continue

            transformed_args = self._replace_fs_content(args)
            if transformed_args != args:
                if cid:
                    ack_msgs.append(ToolMessage(content="", tool_call_id=cid))
                new_tool_calls.append(
                    {
                        "name": norm_name,
                        "args": transformed_args,
                        "id": f"call_{uuid.uuid4().hex}",
                    }
                )
                mutated = True
            else:
                new_tool_calls.append(call)

        if not mutated:
            return None

        # Construct rewritten AIMessage preserving tool_call_id correlations
        rewritten = AIMessage(content=original_content, tool_calls=new_tool_calls)
        print(f"[CallToolWithFileArgMiddleware] after_model: rewritten assistant with {len(new_tool_calls)} tool_calls")
        if ack_msgs:
            return Command(update={"messages": [*ack_msgs, rewritten]})
        return Command(update={"messages": [rewritten]})

###########################
# Quick Action Middleware
###########################

class RemoveToolsForSubagentMiddleware(AgentMiddleware):
    """Remove write_todos and task tools when an agent is called as a subagent."""
    
    def __init__(self):
        super().__init__()
        self.is_subagent = False
    
    def modify_model_request(self, request: ModelRequest, agent_state: AgentState) -> ModelRequest:
        tasks = agent_state.get("tasks")
        # If tasks exist, we're in quick action mode and marketing is a subagent
        if tasks is not None:
            self.is_subagent = True
            request.tools = [t for t in request.tools if t.name not in ["write_todos", "task"]]
        return request


class QuickActionTasksAnnouncementMiddleware(AgentMiddleware):
    """Announce the next task via AIMessage tool call without executing it."""

    def before_model(self, state: AgentState):
        tasks = state.get("tasks") or []
        if not tasks:
            return None

        # Initialize todos from tasks if needed
        todos = state.get("todos") or []

        # Announce and schedule the next task; mark its todo as in_progress
        task_obj = tasks[0]
        desc = task_obj.get("description", "")
        sub = task_obj.get("subagent", "")

        try:
            for i, td in enumerate(todos):
                if isinstance(td, dict) and td.get("content") == desc and td.get("status") != "completed":
                    todos[i] = {**td, "status": "in_progress"}
                    break
        except Exception:
            pass

        # Find and remove the last write_todos tool call and its response
        messages_to_remove = []
        messages = state.get("messages") or []
        
        # Find the last AIMessage with write_todos tool call
        last_write_todos_ai_msg = None
        last_write_todos_tool_call_id = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "write_todos":
                        last_write_todos_ai_msg = msg
                        last_write_todos_tool_call_id = tc.get("id")
                        break
            if last_write_todos_ai_msg:
                break
                
        if last_write_todos_ai_msg and last_write_todos_tool_call_id:
            messages_to_remove.append(RemoveMessage(id=last_write_todos_ai_msg.id))
            
            # Find the corresponding ToolMessage
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage) and msg.tool_call_id == last_write_todos_tool_call_id:
                    messages_to_remove.append(RemoveMessage(id=msg.id))
                    break

        wt_id = f"call_{uuid.uuid4().hex}"
        wt_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "write_todos",
                "args": {"todos": todos},
                "id": wt_id,
            }],
        )
        wt_ack_msg = ToolMessage(content="", tool_call_id=wt_id)

        tool_call_id = f"call_{uuid.uuid4().hex}"
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "task",
                "args": {"description": desc, "subagent_type": sub},
                "id": tool_call_id,
            }],
        )

        return Command(
            update={
                "messages": [*messages_to_remove, wt_msg, wt_ack_msg, ai_msg],
                "pending_task_tool_call_id": tool_call_id,
                "todos": todos,
            },
        )
