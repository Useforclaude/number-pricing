from setuptools import setup, find_packages

setup(
    name="number_pricing",
    version="1.0.0",
    description="Standalone ML package for Thai phone number price prediction",
    author="Codex AI",
    url="https://github.com/Useforclaude/number-pricing",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
