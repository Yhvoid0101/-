#!/usr/bin/env python3
"""模拟MCP服务器 — 用于测试HermesMCPGateway

实现真正的MCP协议(JSON-RPC 2.0 over stdio):
1. 接收initialize请求 → 返回capabilities
2. 接收tools/list请求 → 返回工具列表
3. 接收tools/call请求 → 执行工具并返回结果
4. 支持notifications
"""
import sys
import json
import time
import math

def log(msg):
    print(f"[MCP-Mock] {msg}", file=sys.stderr)

def send_response(request_id, result):
    msg = {"jsonrpc": "2.0", "id": request_id, "result": result}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def send_error(request_id, code, message):
    msg = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

TOOLS = [
    {
        "name": "echo",
        "description": "回显输入文本",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要回显的文本"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "add",
        "description": "计算两个数的和",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "search",
        "description": "搜索信息",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "结果数量限制", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "读取文件内容",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "slow_tool",
        "description": "模拟慢速工具(用于超时测试)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "delay_seconds": {"type": "number", "description": "延迟秒数", "default": 2}
            }
        }
    },
    {
        "name": "error_tool",
        "description": "模拟错误工具(用于错误处理测试)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_type": {"type": "string", "description": "错误类型", "default": "generic"}
            }
        }
    }
]

def handle_tool_call(name, arguments):
    if name == "echo":
        return {"content": [{"type": "text", "text": arguments.get("text", "")}]}
    elif name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return {"content": [{"type": "text", "text": str(a + b)}]}
    elif name == "search":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        results = [f"结果{i+1}: 关于'{query}'的信息" for i in range(min(limit, 3))]
        return {"content": [{"type": "text", "text": "\n".join(results)}]}
    elif name == "read_file":
        path = arguments.get("path", "")
        return {"content": [{"type": "text", "text": f"[模拟文件内容] {path}"}]}
    elif name == "slow_tool":
        delay = arguments.get("delay_seconds", 2)
        time.sleep(delay)
        return {"content": [{"type": "text", "text": f"延迟{delay}秒后完成"}]}
    elif name == "error_tool":
        error_type = arguments.get("error_type", "generic")
        return {"isError": True, "content": [{"type": "text", "text": f"工具错误: {error_type}"}]}
    else:
        return None

def main():
    log("模拟MCP服务器启动")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue

            request_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params", {})

            if method == "initialize":
                send_response(request_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "mock-mcp-server",
                        "version": "1.0.0"
                    }
                })
                log("握手完成")
            elif method == "notifications/initialized":
                pass
            elif method == "tools/list":
                send_response(request_id, {"tools": TOOLS})
                log(f"返回{len(TOOLS)}个工具")
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = handle_tool_call(tool_name, arguments)
                if result is not None:
                    send_response(request_id, result)
                    log(f"工具调用: {tool_name}")
                else:
                    send_error(request_id, -32601, f"未知工具: {tool_name}")
            elif method == "ping":
                send_response(request_id, {})
            else:
                send_error(request_id, -32601, f"未知方法: {method}")
        except Exception as e:
            log(f"处理异常: {e}")
    log("模拟MCP服务器关闭")

if __name__ == "__main__":
    main()
