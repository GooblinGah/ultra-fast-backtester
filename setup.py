#!/usr/bin/env python3
"""
Setup script for Ultra-Fast Backtester
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ultra-fast-backtester",
    version="1.0.0",
    author="Ultra-Fast Backtester Team",
    author_email="contact@ultrafastbacktester.com",
    description="An ultra-fast, event-driven backtesting framework for algorithmic trading strategies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ultra-fast-backtester",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "tensorflow>=2.8.0",
            "torch>=1.10.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "backtester=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 