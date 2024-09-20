import { ChatOpenAI } from "@langchain/openai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { MemorySaver } from "@langchain/langgraph";
// read the environment variables from .env
import "dotenv/config";

const tools = [new TavilySearchResults({ maxResults: 3 })];
// Create a model and give it access to the tools
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools);

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;

  const response = await model.invoke(messages);

  return { messages: response };
}

function shouldUseTool(state: typeof MessagesAnnotation.State) {
  const lastMessage = state.messages[state.messages.length - 1];

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.additional_kwargs.tool_calls) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// Define the graph and compile it into a runnable
export const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("agent", shouldUseTool)
  .addEdge("tools", "agent")
  .compile();
