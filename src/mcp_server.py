"""
MCP (Model Context Protocol) support for the RAG system.
Exposes RAG tools via MCP-compatible interface.
"""

import json
from typing import Any, Callable


class MCPTool:
    def __init__(self, name: str, description: str, parameters: dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() if v.get("required", False)],
            },
        }


class MCPServer:
    def __init__(self, name: str = "corrective-rag"):
        self.name = name
        self.tools: list[MCPTool] = []

    def register_tool(self, tool: MCPTool):
        self.tools.append(tool)

    def handle_list_tools(self) -> dict:
        return {"tools": [t.to_schema() for t in self.tools]}

    def handle_call_tool(self, name: str, arguments: dict) -> dict:
        tool = next((t for t in self.tools if t.name == name), None)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = tool.handler(**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            return {"error": str(e)}

    def handle_jsonrpc(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": self.handle_list_tools()}
        elif method == "tools/call":
            return {"jsonrpc": "2.0", "id": req_id, "result": self.handle_call_tool(
                params.get("name", ""), params.get("arguments", {})
            )}
        elif method == "initialize":
            return {"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": "0.2.0"},
            }}
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def create_mcp_server(rag_pipeline, graph_store=None) -> MCPServer:
    server = MCPServer()

    def query_rag(question: str, mode: str = "corrective") -> dict:
        result = rag_pipeline.run(question)
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "trace": result.trace.steps,
        }

    def search_documents(query: str, top_k: int = 5) -> dict:
        results = rag_pipeline.retriever.search(query, top_k=top_k)
        return {
            "results": [
                {"chunk_id": c.doc_id, "text": c.text[:200], "score": float(s)}
                for c, s in results
            ]
        }

    def get_graph_neighbors(entity: str, max_hops: int = 1) -> dict:
        if not graph_store:
            return {"error": "Graph store not available"}
        neighbors = graph_store.graph.neighbors(entity) if entity in graph_store.graph else []
        return {"entity": entity, "neighbors": list(neighbors)}

    server.register_tool(MCPTool(
        name="query_rag",
        description="Ask a question using the Corrective RAG pipeline with 3-signal verification",
        parameters={
            "question": {"type": "string", "description": "The question to answer", "required": True},
            "mode": {"type": "string", "enum": ["corrective", "agentic"], "description": "Pipeline mode"},
        },
        handler=query_rag,
    ))

    server.register_tool(MCPTool(
        name="search_documents",
        description="Search indexed documents by semantic similarity",
        parameters={
            "query": {"type": "string", "description": "Search query", "required": True},
            "top_k": {"type": "integer", "description": "Number of results to return"},
        },
        handler=search_documents,
    ))

    if graph_store:
        server.register_tool(MCPTool(
            name="get_graph_neighbors",
            description="Get neighboring entities in the knowledge graph",
            parameters={
                "entity": {"type": "string", "description": "Entity name to look up", "required": True},
                "max_hops": {"type": "integer", "description": "Maximum hop count"},
            },
            handler=get_graph_neighbors,
        ))

    return server
