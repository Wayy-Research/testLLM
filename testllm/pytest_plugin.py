"""
Pytest plugin for testLLM Framework
"""

import pytest
import yaml
import os
from typing import Dict, Any, List
from .core import AgentUnderTest, run_test_from_yaml, TestResult


def pytest_addoption(parser):
    """Add testLLM-specific command line options"""
    group = parser.getgroup("testllm")
    group.addoption(
        "--testllm",
        action="store_true",
        default=False,
        help="Run testLLM tests"
    )
    group.addoption(
        "--testllm-report",
        action="store",
        default=None,
        help="Generate HTML report at specified path"
    )


def pytest_configure(config):
    """Configure pytest for testLLM"""
    config.addinivalue_line(
        "markers", "testllm: mark test as a testLLM agent test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for testLLM"""
    if config.getoption("--testllm"):
        # Only run testLLM tests when --testllm flag is used
        testllm_items = []
        for item in items:
            if item.get_closest_marker("testllm"):
                testllm_items.append(item)
        items[:] = testllm_items
    else:
        # Skip testLLM tests when flag is not used
        skip_testllm = pytest.mark.skip(reason="need --testllm option to run")
        for item in items:
            if item.get_closest_marker("testllm"):
                item.add_marker(skip_testllm)


class TestLLMItem(pytest.Item):
    """Custom pytest item for testLLM YAML tests"""
    
    def __init__(self, name, parent, yaml_file, agent_fixture_name="agent"):
        super().__init__(name, parent)
        self.yaml_file = yaml_file
        self.agent_fixture_name = agent_fixture_name
        self.add_marker(pytest.mark.testllm)
    
    def runtest(self):
        """Run the testLLM test"""
        # Get the agent fixture
        agent = self.session._fixturemanager.getfixturevalue(
            self.agent_fixture_name, self
        )
        
        if not isinstance(agent, AgentUnderTest):
            raise TypeError(f"Agent fixture must return an AgentUnderTest instance, got {type(agent)}")
        
        # Load and run test
        with open(self.yaml_file, 'r') as f:
            test_def = yaml.safe_load(f)
        
        result = run_test_from_yaml(test_def, agent)
        
        if not result.passed:
            error_msg = f"Test {result.test_id} failed"
            if result.errors:
                error_msg += f": {'; '.join(result.errors)}"
            raise AssertionError(error_msg)
    
    def repr_failure(self, excinfo):
        """Represent test failure"""
        if isinstance(excinfo.value, AssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)


class TestLLMFile(pytest.File):
    """Custom pytest file handler for YAML test files"""
    
    def collect(self):
        """Collect tests from YAML file"""
        try:
            with open(self.fspath, 'r') as f:
                test_def = yaml.safe_load(f)
            
            # Create a test item for this YAML file
            yield TestLLMItem(
                name=test_def.get("test_id", self.fspath.basename),
                parent=self,
                yaml_file=str(self.fspath)
            )
        except Exception as e:
            # If YAML parsing fails, create a failing test
            yield TestLLMItem(
                name=f"yaml_parse_error_{self.fspath.basename}",
                parent=self,
                yaml_file=str(self.fspath)
            )


def pytest_collect_file(path, parent):
    """Collect testLLM YAML files"""
    if path.ext == ".yaml" and path.basename.startswith("test_"):
        return TestLLMFile(path, parent)


@pytest.fixture
def testllm_results():
    """Fixture to collect testLLM results for reporting"""
    results = []
    yield results


def pytest_runtest_makereport(item, call):
    """Hook to capture test results for reporting"""
    if call.when == "call" and hasattr(item, 'yaml_file'):
        outcome = "passed" if call.excinfo is None else "failed"
        # Store result for potential HTML report generation
        # This would be enhanced with actual result storage


class TestLLMConfig:
    """Configuration helper for testLLM tests"""
    
    def __init__(self):
        self.test_directories = ["tests/", "test_data/"]
        self.default_timeout = 30
        self.retry_count = 0
    
    @classmethod
    def from_file(cls, config_file: str) -> 'TestLLMConfig':
        """Load configuration from file"""
        config = cls()
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                config.test_directories = data.get('test_directories', config.test_directories)
                config.default_timeout = data.get('default_timeout', config.default_timeout)
                config.retry_count = data.get('retry_count', config.retry_count)
        return config


@pytest.fixture(scope="session")
def testllm_config():
    """Provide testLLM configuration"""
    return TestLLMConfig.from_file("testllm.yaml")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add testLLM summary to terminal output"""
    if config.getoption("--testllm"):
        terminalreporter.write_sep("=", "testLLM Summary")
        
        # Count testLLM tests
        testllm_passed = 0
        testllm_failed = 0
        
        for report in terminalreporter.getreports(''):
            if hasattr(report, 'item') and report.item.get_closest_marker("testllm"):
                if report.passed:
                    testllm_passed += 1
                elif report.failed:
                    testllm_failed += 1
        
        total_testllm = testllm_passed + testllm_failed
        if total_testllm > 0:
            terminalreporter.write_line(
                f"testLLM tests: {testllm_passed} passed, {testllm_failed} failed, {total_testllm} total"
            )
        
        # Generate HTML report if requested
        report_path = config.getoption("--testllm-report")
        if report_path:
            _generate_html_report(terminalreporter.getreports(''), report_path)
            terminalreporter.write_line(f"HTML report generated: {report_path}")


def _generate_html_report(reports: List[Any], output_path: str):
    """Generate HTML report for testLLM tests"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>testLLM Test Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .test { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .passed { border-left: 5px solid #4CAF50; }
            .failed { border-left: 5px solid #f44336; }
            .assertion { margin: 5px 0; padding: 5px; background: #f9f9f9; }
            .assertion.passed { background: #e8f5e8; }
            .assertion.failed { background: #ffeaea; }
        </style>
    </head>
    <body>
        <h1>testLLM Test Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Tests: {total_tests}</p>
            <p>Passed: {passed_tests}</p>
            <p>Failed: {failed_tests}</p>
        </div>
        {test_details}
    </body>
    </html>
    """
    
    testllm_reports = [r for r in reports if hasattr(r, 'item') and r.item.get_closest_marker("testllm")]
    
    passed_count = sum(1 for r in testllm_reports if r.passed)
    failed_count = sum(1 for r in testllm_reports if r.failed)
    total_count = len(testllm_reports)
    
    test_details = ""
    for report in testllm_reports:
        status_class = "passed" if report.passed else "failed"
        status_text = "PASSED" if report.passed else "FAILED"
        
        test_details += f"""
        <div class="test {status_class}">
            <h3>{report.item.name} - {status_text}</h3>
            <p>File: {getattr(report.item, 'yaml_file', 'Unknown')}</p>
            {f'<p>Error: {report.longrepr}</p>' if report.failed else ''}
        </div>
        """
    
    html_content = html_template.format(
        total_tests=total_count,
        passed_tests=passed_count,
        failed_tests=failed_count,
        test_details=test_details
    )
    
    with open(output_path, 'w') as f:
        f.write(html_content)