"""
Telemetry module for testLLM cloud services
Collects and sends test data to Vercel API endpoints
"""

import os
import json
import time
import uuid
import threading
import requests
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime, timezone

from .config import get_config, CloudConfig
from .auth import get_auth_manager, is_authenticated


@dataclass
class TestSession:
    """Test session metadata"""
    session_id: str
    session_name: str
    user_id: Optional[str]
    team_id: Optional[str]
    framework_version: str
    agent_type: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    execution_time: float = 0.0
    created_at: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestResult:
    """Individual test result"""
    result_id: str
    session_id: str
    test_id: str
    test_type: str  # 'semantic', 'flow', 'behavioral'
    passed: bool
    overall_score: float
    execution_time: float
    user_input: Optional[str]
    agent_response: Optional[str]
    criteria: List[str]
    evaluations: List[Dict[str, Any]]
    errors: List[str]
    created_at: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConversationStep:
    """Conversation flow step"""
    step_id: str
    result_id: str
    step_order: int
    step_name: str
    user_input: str
    agent_response: str
    step_passed: bool
    step_score: float
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class TelemetryError(Exception):
    """Telemetry related errors"""
    pass


class TelemetryCollector:
    """Collects and sends telemetry data to cloud services"""
    
    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or get_config()
        self.auth_manager = get_auth_manager()
        
        # Current session
        self.current_session: Optional[TestSession] = None
        
        # Data queues for batching
        self.session_queue = Queue()
        self.result_queue = Queue()
        self.step_queue = Queue()
        
        # Background thread for sending data
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_sender_thread()
        
        # Local storage for offline mode
        self._local_storage_path = Path(self.config.local_storage_path) / "telemetry"
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists"""
        if self.config.local_storage_enabled:
            self._local_storage_path.mkdir(parents=True, exist_ok=True)
    
    def _start_sender_thread(self):
        """Start the background sender thread"""
        if self.config.telemetry_enabled and not self._sender_thread:
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()
    
    def _sender_loop(self):
        """Background loop for sending telemetry data"""
        while not self._stop_event.is_set():
            try:
                self._process_queues()
                self._stop_event.wait(self.config.flush_interval)
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Telemetry sender error: {e}")
    
    def _process_queues(self):
        """Process and send queued telemetry data"""
        # Process sessions
        sessions = self._drain_queue(self.session_queue)
        if sessions:
            self._send_sessions(sessions)
        
        # Process results
        results = self._drain_queue(self.result_queue)
        if results:
            self._send_results(results)
        
        # Process steps
        steps = self._drain_queue(self.step_queue)
        if steps:
            self._send_steps(steps)
    
    def _drain_queue(self, queue: Queue) -> List[Any]:
        """Drain items from a queue"""
        items = []
        while True:
            try:
                item = queue.get_nowait()
                items.append(item)
                if len(items) >= self.config.batch_size:
                    break
            except Empty:
                break
        return items
    
    def start_session(
        self, 
        session_name: str, 
        agent_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new test session
        
        Args:
            session_name: Name of the test session
            agent_type: Type of agent being tested
            metadata: Additional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        user_info = self.auth_manager.get_user_info()
        
        self.current_session = TestSession(
            session_id=session_id,
            session_name=session_name,
            user_id=user_info.user_id if user_info else None,
            team_id=user_info.team_id if user_info else None,
            framework_version=self._get_framework_version(),
            agent_type=agent_type,
            metadata=metadata or {}
        )
        
        if self.config.telemetry_enabled:
            self.session_queue.put(self.current_session)
        
        if self.config.local_storage_enabled:
            self._store_locally("session", asdict(self.current_session))
        
        return session_id
    
    def end_session(self, session_id: str):
        """End a test session and send final data"""
        if self.current_session and self.current_session.session_id == session_id:
            # Update session with final stats
            if self.config.telemetry_enabled:
                self.session_queue.put(self.current_session)
            
            if self.config.local_storage_enabled:
                self._store_locally("session_final", asdict(self.current_session))
            
            self.current_session = None
    
    def record_test_result(
        self,
        test_id: str,
        test_type: str,
        passed: bool,
        overall_score: float,
        execution_time: float,
        user_input: Optional[str] = None,
        agent_response: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        evaluations: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a test result
        
        Args:
            test_id: Test identifier
            test_type: Type of test ('semantic', 'flow', 'behavioral')
            passed: Whether the test passed
            overall_score: Overall test score
            execution_time: Test execution time
            user_input: User input (if collect_inputs is enabled)
            agent_response: Agent response (if collect_outputs is enabled)
            criteria: Test criteria
            evaluations: Evaluation results
            errors: Error messages
            metadata: Additional metadata
            
        Returns:
            Result ID
        """
        if not self.current_session:
            raise TelemetryError("No active session. Call start_session() first.")
        
        result_id = str(uuid.uuid4())
        
        # Filter data based on collection preferences
        filtered_input = user_input if self.config.collect_inputs else None
        filtered_output = agent_response if self.config.collect_outputs else None
        filtered_metadata = metadata if self.config.collect_metadata else {}
        
        result = TestResult(
            result_id=result_id,
            session_id=self.current_session.session_id,
            test_id=test_id,
            test_type=test_type,
            passed=passed,
            overall_score=overall_score,
            execution_time=execution_time,
            user_input=filtered_input,
            agent_response=filtered_output,
            criteria=criteria or [],
            evaluations=evaluations or [],
            errors=errors or [],
            metadata=filtered_metadata
        )
        
        # Update session stats
        self.current_session.total_tests += 1
        if passed:
            self.current_session.passed_tests += 1
        else:
            self.current_session.failed_tests += 1
        self.current_session.execution_time += execution_time
        
        if self.config.telemetry_enabled:
            self.result_queue.put(result)
        
        if self.config.local_storage_enabled:
            self._store_locally("result", asdict(result))
        
        return result_id
    
    def record_conversation_step(
        self,
        result_id: str,
        step_order: int,
        step_name: str,
        user_input: str,
        agent_response: str,
        step_passed: bool,
        step_score: float
    ) -> str:
        """
        Record a conversation flow step
        
        Args:
            result_id: Associated test result ID
            step_order: Order of step in conversation
            step_name: Name of the step
            user_input: User input for this step
            agent_response: Agent response for this step
            step_passed: Whether the step passed
            step_score: Step score
            
        Returns:
            Step ID
        """
        step_id = str(uuid.uuid4())
        
        # Filter data based on collection preferences
        filtered_input = user_input if self.config.collect_inputs else ""
        filtered_output = agent_response if self.config.collect_outputs else ""
        
        step = ConversationStep(
            step_id=step_id,
            result_id=result_id,
            step_order=step_order,
            step_name=step_name,
            user_input=filtered_input,
            agent_response=filtered_output,
            step_passed=step_passed,
            step_score=step_score
        )
        
        if self.config.telemetry_enabled:
            self.step_queue.put(step)
        
        if self.config.local_storage_enabled:
            self._store_locally("step", asdict(step))
        
        return step_id
    
    def flush(self):
        """Flush all queued data immediately"""
        if self.config.telemetry_enabled:
            self._process_queues()
    
    def _send_sessions(self, sessions: List[TestSession]):
        """Send session data to API"""
        if not is_authenticated():
            return
        
        try:
            data = [asdict(session) for session in sessions]
            response = requests.post(
                f"{self.config.api_base_url}/sessions/create",
                headers=self.config.get_api_headers(),
                json={"sessions": data},
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                if self.config.debug_mode:
                    print(f"Failed to send sessions: {response.status_code}")
                # Re-queue for retry
                for session in sessions:
                    self.session_queue.put(session)
                    
        except requests.RequestException as e:
            if self.config.debug_mode:
                print(f"Error sending sessions: {e}")
            # Re-queue for retry
            for session in sessions:
                self.session_queue.put(session)
    
    def _send_results(self, results: List[TestResult]):
        """Send test results to API"""
        if not is_authenticated():
            return
        
        try:
            data = [asdict(result) for result in results]
            response = requests.post(
                f"{self.config.api_base_url}/results/submit",
                headers=self.config.get_api_headers(),
                json={"results": data},
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                if self.config.debug_mode:
                    print(f"Failed to send results: {response.status_code}")
                # Re-queue for retry
                for result in results:
                    self.result_queue.put(result)
                    
        except requests.RequestException as e:
            if self.config.debug_mode:
                print(f"Error sending results: {e}")
            # Re-queue for retry
            for result in results:
                self.result_queue.put(result)
    
    def _send_steps(self, steps: List[ConversationStep]):
        """Send conversation steps to API"""
        if not is_authenticated():
            return
        
        try:
            data = [asdict(step) for step in steps]
            response = requests.post(
                f"{self.config.api_base_url}/steps/submit",
                headers=self.config.get_api_headers(),
                json={"steps": data},
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                if self.config.debug_mode:
                    print(f"Failed to send steps: {response.status_code}")
                # Re-queue for retry
                for step in steps:
                    self.step_queue.put(step)
                    
        except requests.RequestException as e:
            if self.config.debug_mode:
                print(f"Error sending steps: {e}")
            # Re-queue for retry
            for step in steps:
                self.step_queue.put(step)
    
    def _store_locally(self, data_type: str, data: Dict[str, Any]):
        """Store data locally for offline mode"""
        if not self.config.local_storage_enabled:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}_{str(uuid.uuid4())[:8]}.json"
            file_path = self._local_storage_path / filename
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error storing data locally: {e}")
    
    def _get_framework_version(self) -> str:
        """Get testLLM framework version"""
        try:
            from .__version__ import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def get_session_url(self, session_id: str) -> str:
        """Get dashboard URL for a session"""
        return f"{self.config.dashboard_url}/sessions/{session_id}"
    
    def cleanup(self):
        """Cleanup resources"""
        self._stop_event.set()
        if self._sender_thread:
            self._sender_thread.join(timeout=5)
        self.flush()


# Global telemetry collector instance
_telemetry_collector: Optional[TelemetryCollector] = None


def get_telemetry_collector() -> TelemetryCollector:
    """Get the global telemetry collector instance"""
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector


def set_telemetry_collector(collector: TelemetryCollector) -> None:
    """Set the global telemetry collector instance"""
    global _telemetry_collector
    _telemetry_collector = collector


def reset_telemetry_collector() -> None:
    """Reset the global telemetry collector instance"""
    global _telemetry_collector
    if _telemetry_collector:
        _telemetry_collector.cleanup()
    _telemetry_collector = None


# Convenience functions
def start_session(session_name: str, agent_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> str:
    """Start a new test session"""
    return get_telemetry_collector().start_session(session_name, agent_type, metadata)


def end_session(session_id: str) -> None:
    """End a test session"""
    get_telemetry_collector().end_session(session_id)


def record_test_result(
    test_id: str,
    test_type: str,
    passed: bool,
    overall_score: float,
    execution_time: float,
    **kwargs
) -> str:
    """Record a test result"""
    return get_telemetry_collector().record_test_result(
        test_id, test_type, passed, overall_score, execution_time, **kwargs
    )


def record_conversation_step(
    result_id: str,
    step_order: int,
    step_name: str,
    user_input: str,
    agent_response: str,
    step_passed: bool,
    step_score: float
) -> str:
    """Record a conversation flow step"""
    return get_telemetry_collector().record_conversation_step(
        result_id, step_order, step_name, user_input, agent_response, step_passed, step_score
    )


def flush_telemetry() -> None:
    """Flush all queued telemetry data"""
    get_telemetry_collector().flush()


def get_session_url(session_id: str) -> str:
    """Get dashboard URL for a session"""
    return get_telemetry_collector().get_session_url(session_id)


def cleanup_telemetry() -> None:
    """Cleanup telemetry resources"""
    reset_telemetry_collector()