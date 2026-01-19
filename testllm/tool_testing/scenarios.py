"""
Pre-built Response Scenarios - Common tool response patterns for testing.

This module provides ready-to-use response scenarios for common tool types
like search, CRUD operations, API integrations, etc.

Example usage:
    from testllm.tool_testing.scenarios import SearchScenarios, CRUDScenarios

    # Use pre-built flight search scenarios
    flight_sim = SearchScenarios.flight_search()

    # Use pre-built user CRUD scenarios
    user_crud = CRUDScenarios.user_management()
"""

from typing import Any, Dict, List, Optional
from .simulation import ToolSimulator, simulate_tool, ScenarioType
from .types import ToolCall


class SearchScenarios:
    """Pre-built scenarios for search-type tools"""

    @staticmethod
    def flight_search() -> ToolSimulator:
        """
        Pre-configured flight search simulator.

        Scenarios:
        - success: Returns sample flights
        - empty: No flights found
        - failure: Search service unavailable
        - timeout: Search timed out
        - rate_limited: Too many requests
        """
        return (
            simulate_tool("search_flights")
            .on_success({
                "flights": [
                    {"id": "FL001", "airline": "United", "price": 299, "departure": "10:00", "arrival": "13:30"},
                    {"id": "FL002", "airline": "Delta", "price": 349, "departure": "14:00", "arrival": "17:30"},
                    {"id": "FL003", "airline": "American", "price": 279, "departure": "18:00", "arrival": "21:30"},
                ],
                "total_results": 3,
                "currency": "USD"
            })
            .as_default()
            .on_empty({
                "flights": [],
                "total_results": 0,
                "message": "No flights found for the selected criteria"
            })
            .on_failure("Flight search service is temporarily unavailable")
            .on_timeout(30000)
            .on_rate_limited()
        )

    @staticmethod
    def hotel_search() -> ToolSimulator:
        """Pre-configured hotel search simulator"""
        return (
            simulate_tool("search_hotels")
            .on_success({
                "hotels": [
                    {"id": "H001", "name": "Grand Hotel", "stars": 5, "price_per_night": 250, "available": True},
                    {"id": "H002", "name": "City Inn", "stars": 3, "price_per_night": 120, "available": True},
                    {"id": "H003", "name": "Budget Stay", "stars": 2, "price_per_night": 75, "available": True},
                ],
                "total_results": 3,
                "currency": "USD"
            })
            .as_default()
            .on_empty({"hotels": [], "total_results": 0})
            .on_failure("Hotel search service unavailable")
            .on_timeout(25000)
        )

    @staticmethod
    def product_search() -> ToolSimulator:
        """Pre-configured product search simulator"""
        return (
            simulate_tool("search_products")
            .on_success({
                "products": [
                    {"id": "P001", "name": "Laptop", "price": 999.99, "in_stock": True, "rating": 4.5},
                    {"id": "P002", "name": "Tablet", "price": 499.99, "in_stock": True, "rating": 4.2},
                ],
                "total_results": 2,
                "page": 1,
                "total_pages": 1
            })
            .as_default()
            .on_empty({"products": [], "total_results": 0})
            .on_failure("Product search failed")
            .on_partial({
                "products": [{"id": "P001", "name": "Laptop", "price": 999.99}],
                "total_results": "unknown",
                "partial": True,
                "message": "Results may be incomplete"
            })
        )

    @staticmethod
    def web_search() -> ToolSimulator:
        """Pre-configured web search simulator"""
        return (
            simulate_tool("web_search")
            .on_success({
                "results": [
                    {"title": "Example Article", "url": "https://example.com/article", "snippet": "This is a sample article..."},
                    {"title": "Another Result", "url": "https://example.com/other", "snippet": "More content here..."},
                ],
                "total_results": 1000
            })
            .as_default()
            .on_empty({"results": [], "total_results": 0, "message": "No results found"})
            .on_failure("Search engine error")
            .on_rate_limited()
        )

    @staticmethod
    def generic_search(tool_name: str = "search") -> ToolSimulator:
        """Create a generic search simulator with common scenarios"""
        return (
            simulate_tool(tool_name)
            .on_success({"results": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}], "total": 2})
            .as_default()
            .on_empty({"results": [], "total": 0})
            .on_failure("Search failed")
            .on_timeout(30000)
            .on_rate_limited()
        )


class CRUDScenarios:
    """Pre-built scenarios for CRUD operations"""

    @staticmethod
    def user_management() -> Dict[str, ToolSimulator]:
        """
        Pre-configured user CRUD simulators.

        Returns dict with: create_user, get_user, update_user, delete_user
        """
        return {
            "create_user": (
                simulate_tool("create_user")
                .on_success({
                    "id": "usr_123456",
                    "email": "user@example.com",
                    "created_at": "2024-01-01T00:00:00Z",
                    "status": "active"
                })
                .as_default()
                .on_failure("Failed to create user")
                .on_custom(
                    "duplicate",
                    {"error": "User already exists", "code": "DUPLICATE_EMAIL"},
                    "Duplicate user error"
                )
                .on_invalid_input("Invalid email format")
            ),
            "get_user": (
                simulate_tool("get_user")
                .on_success({
                    "id": "usr_123456",
                    "email": "user@example.com",
                    "name": "John Doe",
                    "created_at": "2024-01-01T00:00:00Z"
                })
                .as_default()
                .on_not_found()
                .on_auth_error()
            ),
            "update_user": (
                simulate_tool("update_user")
                .on_success({
                    "id": "usr_123456",
                    "updated": True,
                    "updated_at": "2024-01-02T00:00:00Z"
                })
                .as_default()
                .on_not_found()
                .on_failure("Failed to update user")
                .on_invalid_input("Invalid field value")
            ),
            "delete_user": (
                simulate_tool("delete_user")
                .on_success({"deleted": True, "id": "usr_123456"})
                .as_default()
                .on_not_found()
                .on_failure("Failed to delete user")
                .on_auth_error()
            )
        }

    @staticmethod
    def document_crud() -> Dict[str, ToolSimulator]:
        """Pre-configured document CRUD simulators"""
        return {
            "create_document": (
                simulate_tool("create_document")
                .on_success({"id": "doc_001", "created": True, "url": "https://example.com/docs/doc_001"})
                .as_default()
                .on_failure("Failed to create document")
            ),
            "get_document": (
                simulate_tool("get_document")
                .on_success({"id": "doc_001", "title": "Sample Document", "content": "Document content..."})
                .as_default()
                .on_not_found()
            ),
            "update_document": (
                simulate_tool("update_document")
                .on_success({"id": "doc_001", "updated": True})
                .as_default()
                .on_not_found()
                .on_failure("Failed to update document")
            ),
            "delete_document": (
                simulate_tool("delete_document")
                .on_success({"deleted": True})
                .as_default()
                .on_not_found()
            )
        }


class APIScenarios:
    """Pre-built scenarios for common API integrations"""

    @staticmethod
    def weather_api() -> ToolSimulator:
        """Pre-configured weather API simulator"""
        return (
            simulate_tool("get_weather")
            .on_success({
                "location": "New York, NY",
                "temperature": 72,
                "unit": "fahrenheit",
                "conditions": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 8,
                "forecast": [
                    {"day": "Today", "high": 75, "low": 62, "conditions": "Partly Cloudy"},
                    {"day": "Tomorrow", "high": 78, "low": 65, "conditions": "Sunny"},
                ]
            })
            .as_default()
            .on_not_found()  # Location not found
            .on_failure("Weather service unavailable")
            .on_rate_limited()
        )

    @staticmethod
    def payment_api() -> ToolSimulator:
        """Pre-configured payment API simulator"""
        return (
            simulate_tool("process_payment")
            .on_success({
                "transaction_id": "txn_abc123",
                "status": "completed",
                "amount": 99.99,
                "currency": "USD",
                "receipt_url": "https://example.com/receipts/txn_abc123"
            })
            .as_default()
            .on_failure("Payment declined")
            .on_custom(
                "insufficient_funds",
                {"error": "Insufficient funds", "code": "INSUFFICIENT_FUNDS"},
                "Insufficient funds error"
            )
            .on_custom(
                "card_expired",
                {"error": "Card expired", "code": "CARD_EXPIRED"},
                "Expired card error"
            )
            .on_timeout(10000)  # Payments should be fast
        )

    @staticmethod
    def email_api() -> ToolSimulator:
        """Pre-configured email sending simulator"""
        return (
            simulate_tool("send_email")
            .on_success({
                "message_id": "msg_123",
                "sent": True,
                "recipients": 1
            })
            .as_default()
            .on_failure("Failed to send email")
            .on_custom(
                "invalid_recipient",
                {"error": "Invalid email address", "code": "INVALID_RECIPIENT"},
                "Invalid recipient error"
            )
            .on_rate_limited()
        )

    @staticmethod
    def notification_api() -> ToolSimulator:
        """Pre-configured push notification simulator"""
        return (
            simulate_tool("send_notification")
            .on_success({"sent": True, "notification_id": "notif_001"})
            .as_default()
            .on_failure("Notification service error")
            .on_custom(
                "device_not_registered",
                {"error": "Device not registered", "code": "DEVICE_NOT_FOUND"},
                "Device not registered"
            )
        )


class DatabaseScenarios:
    """Pre-built scenarios for database operations"""

    @staticmethod
    def query_tool() -> ToolSimulator:
        """Pre-configured database query simulator"""
        return (
            simulate_tool("execute_query")
            .on_success({
                "rows": [
                    {"id": 1, "name": "John", "email": "john@example.com"},
                    {"id": 2, "name": "Jane", "email": "jane@example.com"},
                ],
                "row_count": 2,
                "execution_time_ms": 45
            })
            .as_default()
            .on_empty({"rows": [], "row_count": 0, "execution_time_ms": 12})
            .on_failure("Database connection error")
            .on_timeout(30000)
            .on_custom(
                "syntax_error",
                {"error": "SQL syntax error", "code": "SYNTAX_ERROR"},
                "SQL syntax error"
            )
        )

    @staticmethod
    def transaction_tool() -> ToolSimulator:
        """Pre-configured database transaction simulator"""
        return (
            simulate_tool("execute_transaction")
            .on_success({"committed": True, "affected_rows": 5})
            .as_default()
            .on_failure("Transaction failed, rolled back")
            .on_custom(
                "deadlock",
                {"error": "Deadlock detected", "code": "DEADLOCK", "retryable": True},
                "Database deadlock"
            )
            .on_custom(
                "constraint_violation",
                {"error": "Constraint violation", "code": "CONSTRAINT_ERROR"},
                "Constraint violation"
            )
        )


class FileScenarios:
    """Pre-built scenarios for file operations"""

    @staticmethod
    def file_operations() -> Dict[str, ToolSimulator]:
        """Pre-configured file operation simulators"""
        return {
            "read_file": (
                simulate_tool("read_file")
                .on_success({"content": "File content here...", "size_bytes": 1024, "encoding": "utf-8"})
                .as_default()
                .on_not_found()
                .on_auth_error()  # Permission denied
                .on_failure("Error reading file")
            ),
            "write_file": (
                simulate_tool("write_file")
                .on_success({"written": True, "bytes_written": 1024})
                .as_default()
                .on_failure("Error writing file")
                .on_auth_error()  # Permission denied
                .on_custom(
                    "disk_full",
                    {"error": "Disk full", "code": "NO_SPACE"},
                    "Disk full error"
                )
            ),
            "delete_file": (
                simulate_tool("delete_file")
                .on_success({"deleted": True})
                .as_default()
                .on_not_found()
                .on_auth_error()
            ),
            "list_files": (
                simulate_tool("list_files")
                .on_success({
                    "files": [
                        {"name": "document.txt", "size": 1024, "modified": "2024-01-01T00:00:00Z"},
                        {"name": "image.png", "size": 204800, "modified": "2024-01-02T00:00:00Z"},
                    ],
                    "total": 2
                })
                .as_default()
                .on_empty({"files": [], "total": 0})
                .on_not_found()  # Directory not found
                .on_auth_error()
            )
        }

    @staticmethod
    def upload_download() -> Dict[str, ToolSimulator]:
        """Pre-configured upload/download simulators"""
        return {
            "upload_file": (
                simulate_tool("upload_file")
                .on_success({
                    "file_id": "file_abc123",
                    "url": "https://storage.example.com/files/file_abc123",
                    "size_bytes": 102400
                })
                .as_default()
                .on_failure("Upload failed")
                .on_custom(
                    "file_too_large",
                    {"error": "File exceeds maximum size", "max_size": 10485760},
                    "File too large"
                )
                .on_rate_limited()
            ),
            "download_file": (
                simulate_tool("download_file")
                .on_success({
                    "content": "Base64 encoded content...",
                    "filename": "document.pdf",
                    "size_bytes": 204800
                })
                .as_default()
                .on_not_found()
                .on_failure("Download failed")
                .on_timeout(60000)
            )
        }


class CalendarScenarios:
    """Pre-built scenarios for calendar operations"""

    @staticmethod
    def calendar_operations() -> Dict[str, ToolSimulator]:
        """Pre-configured calendar simulators"""
        return {
            "create_event": (
                simulate_tool("create_event")
                .on_success({
                    "event_id": "evt_001",
                    "title": "Meeting",
                    "start": "2024-01-15T10:00:00Z",
                    "end": "2024-01-15T11:00:00Z",
                    "calendar_link": "https://calendar.example.com/evt_001"
                })
                .as_default()
                .on_failure("Failed to create event")
                .on_custom(
                    "conflict",
                    {"error": "Time slot already booked", "code": "CONFLICT"},
                    "Calendar conflict"
                )
            ),
            "get_events": (
                simulate_tool("get_events")
                .on_success({
                    "events": [
                        {"id": "evt_001", "title": "Morning Standup", "start": "2024-01-15T09:00:00Z"},
                        {"id": "evt_002", "title": "Project Review", "start": "2024-01-15T14:00:00Z"},
                    ],
                    "total": 2
                })
                .as_default()
                .on_empty({"events": [], "total": 0, "message": "No events found"})
            ),
            "delete_event": (
                simulate_tool("delete_event")
                .on_success({"deleted": True})
                .as_default()
                .on_not_found()
            ),
            "check_availability": (
                simulate_tool("check_availability")
                .on_success({
                    "available": True,
                    "next_available_slot": "2024-01-15T10:00:00Z"
                })
                .as_default()
                .on_custom(
                    "busy",
                    {"available": False, "conflicts": ["Meeting at 10:00"], "next_available_slot": "2024-01-15T11:00:00Z"},
                    "Time slot busy"
                )
            )
        }


def get_all_scenarios() -> Dict[str, Dict[str, ToolSimulator]]:
    """
    Get all pre-built scenarios organized by category.

    Returns:
        Dict mapping category names to dicts of simulator names and simulators
    """
    return {
        "search": {
            "flight_search": SearchScenarios.flight_search(),
            "hotel_search": SearchScenarios.hotel_search(),
            "product_search": SearchScenarios.product_search(),
            "web_search": SearchScenarios.web_search(),
        },
        "crud_user": CRUDScenarios.user_management(),
        "crud_document": CRUDScenarios.document_crud(),
        "api": {
            "weather": APIScenarios.weather_api(),
            "payment": APIScenarios.payment_api(),
            "email": APIScenarios.email_api(),
            "notification": APIScenarios.notification_api(),
        },
        "database": {
            "query": DatabaseScenarios.query_tool(),
            "transaction": DatabaseScenarios.transaction_tool(),
        },
        "files": FileScenarios.file_operations(),
        "files_transfer": FileScenarios.upload_download(),
        "calendar": CalendarScenarios.calendar_operations(),
    }
