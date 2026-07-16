import os
import sys
import json
import time
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()

TOOL_CALL_QUERIES = [
    {"query": "Read the contents of the file config.yaml", "expected_tool": "read_file", "category": "file_ops"},
    {
        "query": "Search for all Python files containing 'import asyncio'",
        "expected_tool": "search",
        "category": "search",
    },
    {"query": "Execute the command 'git status' in the terminal", "expected_tool": "terminal", "category": "execution"},
    {
        "query": "Send a message to the #general channel on Slack",
        "expected_tool": "send_message",
        "category": "communication",
    },
    {
        "query": "Create a new file called test_output.py with a hello world program",
        "expected_tool": "write_file",
        "category": "file_ops",
    },
    {"query": "List all running processes on the system", "expected_tool": "terminal", "category": "execution"},
    {
        "query": "Fetch the latest data from the REST API endpoint /api/users",
        "expected_tool": "http_request",
        "category": "network",
    },
    {"query": "Analyze the code in main.py for potential bugs", "expected_tool": "code_review", "category": "analysis"},
    {
        "query": "Translate the following text from English to Chinese: Hello World",
        "expected_tool": "translate",
        "category": "language",
    },
    {"query": "Calculate the sum of numbers from 1 to 100", "expected_tool": "calculate", "category": "math"},
    {"query": "Compress the directory /tmp/logs into a zip file", "expected_tool": "file_ops", "category": "file_ops"},
    {"query": "Deploy the application to the staging environment", "expected_tool": "deploy", "category": "devops"},
    {
        "query": "Query the database for users created in the last 7 days",
        "expected_tool": "database",
        "category": "data",
    },
    {"query": "Generate a unit test for the function calculate_tax()", "expected_tool": "code_gen", "category": "code"},
    {"query": "Monitor the CPU usage for the next 5 minutes", "expected_tool": "monitor", "category": "observability"},
    {"query": "Roll back the last deployment to production", "expected_tool": "deploy", "category": "devops"},
    {
        "query": "Parse the CSV file and extract all unique email addresses",
        "expected_tool": "data_process",
        "category": "data",
    },
    {
        "query": "Set up a cron job to run the backup script daily at 2am",
        "expected_tool": "schedule",
        "category": "automation",
    },
    {"query": "Check the SSL certificate expiry for example.com", "expected_tool": "network", "category": "network"},
    {"query": "Create a pull request with the current changes", "expected_tool": "git", "category": "devops"},
    {"query": "Find and fix the memory leak in the worker process", "expected_tool": "debug", "category": "analysis"},
    {"query": "Encrypt the sensitive data using AES-256", "expected_tool": "crypto", "category": "security"},
    {"query": "Generate a report of all API calls made this week", "expected_tool": "report", "category": "analysis"},
    {
        "query": "Restart the nginx service on the production server",
        "expected_tool": "service_mgmt",
        "category": "devops",
    },
    {
        "query": "Validate the JSON schema of the configuration file",
        "expected_tool": "validate",
        "category": "validation",
    },
    {"query": "Download the latest dataset from the S3 bucket", "expected_tool": "cloud_storage", "category": "cloud"},
    {"query": "Run the load test with 1000 concurrent users", "expected_tool": "test", "category": "testing"},
    {"query": "Update the DNS record for api.example.com", "expected_tool": "dns", "category": "network"},
    {"query": "Analyze the sentiment of customer reviews", "expected_tool": "nlp", "category": "analysis"},
    {"query": "Optimize the SQL query that takes too long to execute", "expected_tool": "database", "category": "data"},
    {"query": "Create a Docker image from the current project", "expected_tool": "container", "category": "devops"},
    {
        "query": "Scan the codebase for security vulnerabilities",
        "expected_tool": "security_scan",
        "category": "security",
    },
    {
        "query": "Schedule a meeting with the team for tomorrow at 3pm",
        "expected_tool": "calendar",
        "category": "productivity",
    },
    {"query": "Convert the markdown document to PDF format", "expected_tool": "convert", "category": "file_ops"},
    {"query": "Set up monitoring alerts for the microservice", "expected_tool": "monitor", "category": "observability"},
    {"query": "Debug why the WebSocket connection keeps dropping", "expected_tool": "debug", "category": "analysis"},
    {"query": "Migrate the database schema to version 3.0", "expected_tool": "database", "category": "data"},
    {"query": "Generate API documentation from the OpenAPI spec", "expected_tool": "doc_gen", "category": "code"},
    {
        "query": "Configure the firewall rules to allow HTTPS traffic",
        "expected_tool": "security_config",
        "category": "security",
    },
    {"query": "Benchmark the performance of the new algorithm", "expected_tool": "benchmark", "category": "testing"},
    {"query": "Clean up temporary files older than 30 days", "expected_tool": "file_ops", "category": "file_ops"},
    {"query": "Set up a Redis cluster for session caching", "expected_tool": "infra", "category": "devops"},
    {"query": "Extract tables from the PDF financial report", "expected_tool": "data_extract", "category": "data"},
    {"query": "Write a Git hook to run linting before commits", "expected_tool": "git", "category": "devops"},
    {"query": "Analyze the network traffic for anomalies", "expected_tool": "network_analysis", "category": "security"},
    {"query": "Create a Kubernetes deployment manifest", "expected_tool": "container", "category": "devops"},
    {"query": "Sync the local database with the remote API", "expected_tool": "sync", "category": "data"},
    {
        "query": "Generate a random password that meets security requirements",
        "expected_tool": "crypto",
        "category": "security",
    },
    {"query": "Plot a chart showing the monthly revenue trend", "expected_tool": "visualize", "category": "analysis"},
    {"query": "Configure the CI/CD pipeline for automated testing", "expected_tool": "ci_cd", "category": "devops"},
    {
        "query": "Identify and remove duplicate records from the dataset",
        "expected_tool": "data_process",
        "category": "data",
    },
]


async def run_mcp_benchmark():
    print("=" * 70)
    print("MCP-AgentBench Tool Calling Test (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.orchestrator import HermesOrchestrator

    orch = HermesOrchestrator(api_key=os.environ["NVAPI_API_KEY"])
    await orch.initialize()
    await orch.start()

    results = []
    correct = 0
    category_stats = {}

    for i, tc in enumerate(TOOL_CALL_QUERIES):
        print(f"  [{i+1}/{len(TOOL_CALL_QUERIES)}] {tc['category']}: {tc['query'][:50]}...")

        t0 = time.time()
        try:
            r = await orch.intelligent_execute(
                f"For this task, identify the most appropriate tool/action type needed. "
                f"Reply with ONLY the tool name or category, nothing else.\n\n"
                f"Task: {tc['query']}"
            )
            elapsed = time.time() - t0
            content = str(r.data).lower() if r.data else ""

            category_match = tc["category"].lower() in content or tc["expected_tool"].lower() in content
            if category_match:
                correct += 1

            cat = tc["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0}
            category_stats[cat]["total"] += 1
            if category_match:
                category_stats[cat]["correct"] += 1

            result = {
                "query": tc["query"],
                "expected_tool": tc["expected_tool"],
                "category": tc["category"],
                "match": category_match,
                "elapsed": elapsed,
                "response_preview": content[:100],
            }
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "query": tc["query"],
                "expected_tool": tc["expected_tool"],
                "category": tc["category"],
                "match": False,
                "elapsed": elapsed,
                "error": str(e)[:100],
            }
            cat = tc["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0}
            category_stats[cat]["total"] += 1

        results.append(result)

    await orch.stop()

    total = len(TOOL_CALL_QUERIES)
    rate = correct / total * 100

    print(f"\n{'='*70}")
    print(f"MCP-AgentBench Result: {correct}/{total} ({rate:.1f}%)")
    print(f"Target: >= 40%, Status: {'PASS' if rate >= 40 else 'FAIL'}")
    print(f"\nCategory Breakdown:")
    for cat, stats in sorted(category_stats.items()):
        cat_rate = stats["correct"] / stats["total"] * 100
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_rate:.0f}%)")
    print(f"{'='*70}")

    report = {
        "benchmark": "MCP-AgentBench (adapted)",
        "total": total,
        "correct": correct,
        "rate": rate,
        "target": 40,
        "passed": rate >= 40,
        "category_stats": category_stats,
        "results": results,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "mcp_bench_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return rate


if __name__ == "__main__":
    asyncio.run(run_mcp_benchmark())
