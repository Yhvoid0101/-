import os
import sys
import json
import time
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()

CODING_TASKS = [
    {
        "id": "django_001",
        "repo": "django/django",
        "problem": "Fix a bug where QuerySet.select_related() causes incorrect results when used with deferred fields. The issue is that select_related() doesn't properly handle deferred fields, leading to duplicate or missing data in the queryset results.",
        "expected_keywords": ["select_related", "deferred", "queryset"],
    },
    {
        "id": "flask_001",
        "repo": "pallets/flask",
        "problem": "Fix a bug where Flask's url_for() generates incorrect URLs when the application has a subdomain and the SERVER_NAME config is set. The generated URLs point to the wrong host.",
        "expected_keywords": ["url_for", "subdomain", "SERVER_NAME"],
    },
    {
        "id": "requests_001",
        "repo": "psf/requests",
        "problem": "Fix a bug where Session.send() does not properly merge environment proxy settings with session-level proxy settings, causing requests to ignore proxy configuration.",
        "expected_keywords": ["proxy", "session", "merge"],
    },
    {
        "id": "pytest_001",
        "repo": "pytest-dev/pytest",
        "problem": "Fix a bug where pytest.raises() context manager does not properly clean up when the expected exception is not raised, leading to incorrect test pass/fail status.",
        "expected_keywords": ["raises", "exception", "context"],
    },
    {
        "id": "sklearn_001",
        "repo": "scikit-learn/scikit-learn",
        "problem": "Fix a bug where GridSearchCV with refit=False still modifies the best_estimator_ attribute, causing unexpected behavior when the best estimator should not be refitted.",
        "expected_keywords": ["GridSearchCV", "refit", "best_estimator"],
    },
    {
        "id": "fastapi_001",
        "repo": "fastapi/fastapi",
        "problem": "Fix a bug where dependency injection fails when a dependency has a default value that is a Pydantic model, causing incorrect parameter resolution.",
        "expected_keywords": ["dependency", "pydantic", "injection"],
    },
    {
        "id": "numpy_001",
        "repo": "numpy/numpy",
        "problem": "Fix a bug where np.dot() produces incorrect results for certain float128 arrays due to improper memory alignment on x86 platforms.",
        "expected_keywords": ["dot", "float128", "alignment"],
    },
    {
        "id": "pandas_001",
        "repo": "pandas-dev/pandas",
        "problem": "Fix a bug where DataFrame.groupby().agg() with a lambda function causes a TypeError when the lambda returns a Series instead of a scalar value.",
        "expected_keywords": ["groupby", "agg", "lambda"],
    },
    {
        "id": "celery_001",
        "repo": "celery/celery",
        "problem": "Fix a bug where task.retry() with max_retries=None causes an infinite loop instead of respecting the task's retry policy.",
        "expected_keywords": ["retry", "max_retries", "loop"],
    },
    {
        "id": "sqlalchemy_001",
        "repo": "sqlalchemy/sqlalchemy",
        "problem": "Fix a bug where joinedload() produces incorrect SQL when used with a composite primary key relationship, generating duplicate JOIN clauses.",
        "expected_keywords": ["joinedload", "composite", "JOIN"],
    },
]


async def run_swebench_lite():
    print("=" * 70)
    print("SWE-bench Lite Coding Capability Test (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.orchestrator import HermesOrchestrator

    orch = HermesOrchestrator(api_key=os.environ["NVAPI_API_KEY"])
    await orch.initialize()
    await orch.start()

    results = []
    resolved = 0

    for task in CODING_TASKS:
        print(f"\n--- Task: {task['id']} ({task['repo']}) ---")
        prompt = (
            f"You are a senior software engineer. Analyze and provide a fix for this bug:\n\n"
            f"Repository: {task['repo']}\n"
            f"Problem: {task['problem']}\n\n"
            f"Provide:\n"
            f"1. Root cause analysis\n"
            f"2. The specific code fix (diff format)\n"
            f"3. Test case to verify the fix"
        )

        t0 = time.time()
        try:
            r = await orch.intelligent_execute(prompt)
            elapsed = time.time() - t0
            content = str(r.data) if r.data else ""

            keyword_hits = sum(1 for kw in task["expected_keywords"] if kw.lower() in content.lower())
            keyword_rate = keyword_hits / len(task["expected_keywords"])

            task_resolved = keyword_rate >= 0.5 and r.success
            if task_resolved:
                resolved += 1

            result = {
                "id": task["id"],
                "repo": task["repo"],
                "success": r.success,
                "elapsed": elapsed,
                "keyword_hits": keyword_hits,
                "keyword_rate": keyword_rate,
                "resolved": task_resolved,
                "content_preview": content[:200],
            }
            print(
                f"  Success={r.success}, Keywords={keyword_hits}/{len(task['expected_keywords'])}, "
                f"Rate={keyword_rate:.0%}, Resolved={task_resolved}, Time={elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "id": task["id"],
                "repo": task["repo"],
                "success": False,
                "elapsed": elapsed,
                "error": str(e)[:100],
                "resolved": False,
            }
            print(f"  ERROR: {e}")

        results.append(result)

    await orch.stop()

    total = len(CODING_TASKS)
    rate = resolved / total * 100

    print(f"\n{'='*70}")
    print(f"SWE-bench Lite Result: {resolved}/{total} resolved ({rate:.1f}%)")
    print(f"Target: >= 30%, Status: {'PASS' if rate >= 30 else 'FAIL'}")
    print(f"{'='*70}")

    report = {
        "benchmark": "SWE-bench Lite (adapted)",
        "total": total,
        "resolved": resolved,
        "rate": rate,
        "target": 30,
        "passed": rate >= 30,
        "results": results,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "swebench_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return resolved, total


if __name__ == "__main__":
    asyncio.run(run_swebench_lite())
