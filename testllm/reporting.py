"""
Reporting functionality for testLLM Framework
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .core import TestResult, AgentUnderTest, run_test_from_yaml, load_test_file


@dataclass
class TestSuiteStats:
    """Statistics for a test suite run"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    execution_time: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage"""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


@dataclass
class TestSuiteResult:
    """Results from running a test suite"""
    results: List[TestResult] = field(default_factory=list)
    stats: TestSuiteStats = field(default_factory=TestSuiteStats)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def add_result(self, result: TestResult):
        """Add a test result to the suite"""
        self.results.append(result)
        self._update_stats()
    
    def _update_stats(self):
        """Update suite statistics"""
        self.stats.total = len(self.results)
        self.stats.passed = sum(1 for r in self.results if r.passed)
        self.stats.failed = self.stats.total - self.stats.passed
        self.stats.execution_time = sum(r.execution_time for r in self.results)
    
    def get_failures(self) -> List[TestResult]:
        """Get only the failed test results"""
        return [r for r in self.results if not r.passed]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the test suite results"""
        return {
            "timestamp": self.timestamp,
            "total_tests": self.stats.total,
            "passed": self.stats.passed,
            "failed": self.stats.failed,
            "pass_rate": f"{self.stats.pass_rate:.1f}%",
            "execution_time": f"{self.stats.execution_time:.2f}s"
        }


class TestSuiteReport:
    """Test suite runner and reporter"""
    
    def __init__(self):
        self.test_files: List[str] = []
        self.test_directories: List[str] = []
    
    def add_test_file(self, file_path: str):
        """Add a single test file to the suite"""
        self.test_files.append(file_path)
    
    def add_test_directory(self, directory: str, pattern: str = "test_*.yaml"):
        """Add all matching test files from a directory"""
        self.test_directories.append((directory, pattern))
    
    def discover_tests(self) -> List[str]:
        """Discover all test files to run"""
        all_files = self.test_files.copy()
        
        for directory, pattern in self.test_directories:
            path = Path(directory)
            if path.exists():
                all_files.extend(str(f) for f in path.glob(pattern))
        
        return list(set(all_files))  # Remove duplicates
    
    def execute(self, agent: AgentUnderTest) -> TestSuiteResult:
        """Execute all tests in the suite"""
        suite_result = TestSuiteResult()
        test_files = self.discover_tests()
        
        for test_file in test_files:
            try:
                test_def = load_test_file(test_file)
                result = run_test_from_yaml(test_def, agent)
                result.test_id = f"{test_file}::{result.test_id}"  # Include file in ID
                suite_result.add_result(result)
            except Exception as e:
                # Create a failed result for file loading errors
                error_result = TestResult(
                    test_id=f"{test_file}::load_error",
                    description=f"Failed to load test file: {test_file}",
                    passed=False,
                    errors=[str(e)]
                )
                suite_result.add_result(error_result)
        
        return suite_result


def export_report(suite_result: TestSuiteResult, output_path: str, format: str = "html"):
    """Export test results to a file"""
    if format.lower() == "html":
        _export_html_report(suite_result, output_path)
    elif format.lower() == "json":
        _export_json_report(suite_result, output_path)
    else:
        raise ValueError(f"Unsupported report format: {format}")


def _export_html_report(suite_result: TestSuiteResult, output_path: str):
    """Export results as HTML report"""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>testLLM Test Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            overflow: hidden; 
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            text-align: center; 
        }}
        .summary {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-bottom: 1px solid #dee2e6; 
        }}
        .stat-card {{ 
            background: white; 
            padding: 15px; 
            border-radius: 6px; 
            text-align: center; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
        }}
        .stat-value {{ 
            font-size: 2em; 
            font-weight: bold; 
            margin-bottom: 5px; 
        }}
        .stat-label {{ 
            color: #6c757d; 
            font-size: 0.9em; 
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .tests-container {{ 
            padding: 20px; 
        }}
        .test {{ 
            border: 1px solid #dee2e6; 
            margin: 15px 0; 
            border-radius: 6px; 
            overflow: hidden; 
        }}
        .test.passed {{ border-left: 4px solid #28a745; }}
        .test.failed {{ border-left: 4px solid #dc3545; }}
        .test-header {{ 
            padding: 15px; 
            background: #f8f9fa; 
            border-bottom: 1px solid #dee2e6; 
            cursor: pointer; 
        }}
        .test-header:hover {{ background: #e9ecef; }}
        .test-title {{ 
            font-weight: bold; 
            margin-bottom: 5px; 
        }}
        .test-description {{ 
            color: #6c757d; 
            font-size: 0.9em; 
        }}
        .test-details {{ 
            padding: 15px; 
            display: none; 
        }}
        .test-details.show {{ display: block; }}
        .conversation {{ 
            margin: 15px 0; 
            padding: 15px; 
            background: #f8f9fa; 
            border-radius: 4px; 
        }}
        .turn {{ 
            margin: 10px 0; 
            padding: 10px; 
            border-radius: 4px; 
        }}
        .turn.user {{ 
            background: #e3f2fd; 
            border-left: 3px solid #2196f3; 
        }}
        .turn.agent {{ 
            background: #f3e5f5; 
            border-left: 3px solid #9c27b0; 
        }}
        .assertion {{ 
            margin: 5px 0; 
            padding: 8px; 
            border-radius: 4px; 
            font-size: 0.9em; 
        }}
        .assertion.passed {{ background: #d4edda; color: #155724; }}
        .assertion.failed {{ background: #f8d7da; color: #721c24; }}
        .error {{ 
            background: #f8d7da; 
            color: #721c24; 
            padding: 10px; 
            border-radius: 4px; 
            margin: 10px 0; 
        }}
        .toggle-btn {{ 
            float: right; 
            background: none; 
            border: none; 
            font-size: 1.2em; 
            cursor: pointer; 
            color: #6c757d; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>testLLM Test Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value passed">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failed">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pass_rate}</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_time}s</div>
                <div class="stat-label">Execution Time</div>
            </div>
        </div>
        
        <div class="tests-container">
            {test_details}
        </div>
    </div>
    
    <script>
        function toggleTest(testId) {{
            const details = document.getElementById('details-' + testId);
            const btn = document.getElementById('btn-' + testId);
            
            if (details.classList.contains('show')) {{
                details.classList.remove('show');
                btn.textContent = '▼';
            }} else {{
                details.classList.add('show');
                btn.textContent = '▲';
            }}
        }}
    </script>
</body>
</html>
"""
    
    # Generate test details HTML
    test_details_html = ""
    for i, result in enumerate(suite_result.results):
        status_class = "passed" if result.passed else "failed"
        status_text = "PASSED" if result.passed else "FAILED"
        
        # Generate conversations HTML
        conversations_html = ""
        for conv in result.conversations:
            turns_html = ""
            for turn in conv.get("turns", []):
                if turn["role"] == "user":
                    turns_html += f"""
                    <div class="turn user">
                        <strong>User:</strong> {turn["content"]}
                    </div>
                    """
                elif turn["role"] == "agent":
                    assertions_html = ""
                    for assertion in turn.get("assertions", []):
                        assertion_class = "passed" if assertion.get("passed", False) else "failed"
                        assertion_type = assertion.get("assertion_type", "unknown")
                        assertion_message = assertion.get("message", "")
                        assertions_html += f"""
                        <div class="assertion {assertion_class}">
                            <strong>{assertion_type}:</strong> {assertion_message or 'OK'}
                        </div>
                        """
                    
                    turns_html += f"""
                    <div class="turn agent">
                        <strong>Agent:</strong> {turn["content"]}
                        {assertions_html}
                    </div>
                    """
            
            conversations_html += f"""
            <div class="conversation">
                <h4>{conv.get("name", "Conversation")}</h4>
                {turns_html}
            </div>
            """
        
        # Generate errors HTML
        errors_html = ""
        if result.errors:
            errors_html = f"""
            <div class="error">
                <strong>Errors:</strong><br>
                {'<br>'.join(result.errors)}
            </div>
            """
        
        test_details_html += f"""
        <div class="test {status_class}">
            <div class="test-header" onclick="toggleTest({i})">
                <div class="test-title">
                    {result.test_id} - {status_text}
                    <button class="toggle-btn" id="btn-{i}">▼</button>
                </div>
                <div class="test-description">{result.description}</div>
            </div>
            <div class="test-details" id="details-{i}">
                {errors_html}
                {conversations_html}
                <p><small>Execution time: {result.execution_time:.3f}s</small></p>
            </div>
        </div>
        """
    
    summary = suite_result.get_summary()
    html_content = html_template.format(
        timestamp=summary["timestamp"],
        total_tests=summary["total_tests"],
        passed=summary["passed"],
        failed=summary["failed"],
        pass_rate=summary["pass_rate"],
        execution_time=summary["execution_time"],
        test_details=test_details_html
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _export_json_report(suite_result: TestSuiteResult, output_path: str):
    """Export results as JSON report"""
    data = {
        "summary": suite_result.get_summary(),
        "results": []
    }
    
    for result in suite_result.results:
        result_data = {
            "test_id": result.test_id,
            "description": result.description,
            "passed": result.passed,
            "execution_time": result.execution_time,
            "errors": result.errors,
            "conversations": result.conversations
        }
        data["results"].append(result_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)