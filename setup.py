# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agrpo-deepspeed",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Adaptive Group Relative Policy Optimization with DeepSpeed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agrpo-deepspeed",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agrpo-train=scripts.train:main",
            "agrpo-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agrpo_deepspeed": ["config/*.json"],
    },
)