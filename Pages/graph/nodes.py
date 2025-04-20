from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
import json
from typing import Literal
from .tools import complete_python_task
# from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode

# from langgraph.prebuilt import tools_condition
import os


llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [complete_python_task]

model = llm.bind_tools(tools)


with open(os.path.join(os.path.dirname(__file__), "../prompts/main_prompt.md"), "r") as file:
    prompt = file.read()

chat_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("placeholder", "{messages}"),
])
model = chat_template | model

def create_data_summary(state: AgentState) -> str:
    summary = ""
    variables = []
    for d in state["input_data"]:
        variables.append(d.variable_name)
        summary += f"\n\nVariable: {d.variable_name}\n"
        summary += f"Description: {d.data_description}"
    
    if "current_variables" in state:
        remaining_variables = [v for v in state["current_variables"] if v not in variables]
        for v in remaining_variables:
            summary += f"\n\nVariable: {v}"
    return summary

def route_to_tools(
    state: AgentState,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route back to the agent.
    """

    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

def call_model(state: AgentState):

    current_data_template  = """The following data is available:\n{data_summary}"""
    current_data_message = HumanMessage(content=current_data_template.format(data_summary=create_data_summary(state)))
    state["messages"] = [current_data_message] + state["messages"]

    llm_outputs = model.invoke(state)

    return {"messages": [llm_outputs], "intermediate_outputs": [current_data_message.content]}


def call_tools(state: AgentState):
    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", [])
    if not tool_calls:
        return {"messages": []}

    tool_messages = []
    state_updates = {}

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = {**tool_call["args"], "graph_state": state}
        tool_call_id = tool_call["id"]

        # Find the corresponding tool
        tool = next((t for t in tools if getattr(t, 'name', None) == tool_name), None)

        if tool is None:
            tool_messages.append(ToolMessage(
                content=f"Tool '{tool_name}' not found.",
                name=tool_name,
                tool_call_id=tool_call_id
            ))
            continue

        try:
            # If using LangChain tool interface
            if hasattr(tool, "invoke"):
                result = tool.invoke(tool_input)
            else:
                result = tool(**tool_input)

            # Handle (message, updates) or just message
            if isinstance(result, tuple) and len(result) == 2:
                tool_output, updates = result
            else:
                tool_output, updates = result, {}

            tool_messages.append(ToolMessage(
                content=str(tool_output),
                name=tool_name,
                tool_call_id=tool_call_id
            ))

            # Merge updates into state_updates
            for k, v in updates.items():
                if k in state_updates and isinstance(state_updates[k], list):
                    state_updates[k].extend(v)
                else:
                    state_updates[k] = v

        except Exception as e:
            tool_messages.append(ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                name=tool_name,
                tool_call_id=tool_call_id
            ))

    # Attach tool messages to state
    if "messages" not in state_updates:
        state_updates["messages"] = []

    state_updates["messages"].extend(tool_messages)
    return state_updates