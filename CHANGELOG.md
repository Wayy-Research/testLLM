# Changelog

All notable changes to testLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-XX

### Added
- Initial release of testLLM framework
- Core testing framework for LLM-based agents
- Support for API-based and local agents
- Semantic assertion library with 10+ assertion types
- YAML-based test definitions
- Programmatic test creation
- pytest integration with custom plugin
- Multi-turn conversation testing
- HTML and JSON report generation
- Anonymous telemetry collection system
- Custom freemium license with data collection rights
- Comprehensive documentation and examples

### Assertion Types
- `contains` - Text pattern matching
- `excludes` - Text pattern exclusion
- `regex` - Regular expression matching
- `sentiment` - Emotional tone analysis
- `max_length` / `min_length` - Response length validation
- `json_valid` - JSON format validation
- `json_schema` - JSON schema compliance
- `all_of` / `any_of` - Composite assertions

### Agent Support
- `ApiAgent` - HTTP API endpoint testing
- `LocalAgent` - Local model testing
- `AgentUnderTest` - Custom agent implementations

### Features
- Input/output focused testing (works with any agent)
- Conversation state management
- Test result aggregation and reporting
- Privacy-conscious telemetry collection
- Development tools and utilities

[Unreleased]: [YOUR_GITHUB_URL]/compare/v0.1.0...HEAD
[0.1.0]: [YOUR_GITHUB_URL]/releases/tag/v0.1.0