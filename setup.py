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
    version="0.1.0",
    author="Adi Sivahuma",
    author_email="adi.siv@berkeley.edu",
    description="An event-driven backtesting framework for algorithmic trading strategies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/GooblinGah/ultra-fast-backtester",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "bokeh>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultrafast-backtester=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 