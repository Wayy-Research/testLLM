from setuptools import setup, find_packages

setup(
    name="testllm",
    version="0.1.0",
    description="Testing Framework for LLM-Based Agents",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="testLLM Team",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "PyYAML>=6.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "pytest11": [
            "testllm = testllm.pytest_plugin",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)