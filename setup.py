import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __version__.py
version_file = os.path.join(this_directory, 'testllm', '__version__.py')
with open(version_file) as f:
    exec(f.read())

setup(
    name="test-llm",
    version=__version__,
    description="Testing Framework for LLM-Based Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="testLLM Team",
    author_email="rcgalbo@gmail.com",
    url="https://github.com/rcgalbo/wayy-research",
    project_urls={
        "Bug Tracker": "https://github.com/rcgalbo/wayy-research/issues",
        "Documentation": "https://github.com/rcgalbo/wayy-research/blob/main/README.md",
        "Source Code": "https://github.com/rcgalbo/wayy-research",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    install_requires=[
        "pytest>=7.0.0",
        "PyYAML>=6.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "twine>=4.0.0",
            "build>=0.10.0",
        ],
        "test": [
            "pytest-xdist>=3.0.0",
            "pytest-mock>=3.10.0",
            "responses>=0.23.0",
        ],
    },
    entry_points={
        "pytest11": [
            "testllm = testllm.pytest_plugin",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="testing llm agents ai machine-learning pytest",
    python_requires=">=3.9",
    zip_safe=False,
)