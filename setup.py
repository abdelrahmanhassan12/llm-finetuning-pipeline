from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-finetuning-pipeline",
    version="1.0.0",
    author="Abdelrahman Hassan",
    author_email="ahmostafa00@gmail.com",
    description="End-to-end pipeline for fine-tuning and serving small language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdelrahmanhassan12/llm-finetuning-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "pylint>=2.8",
            "mypy>=0.812",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-pipeline=orchestration.pipeline_orchestrator:main",
            "llm-server=deployment.model_server:main",
            "llm-monitor=deployment.monitoring:main",
            "llm-cicd=orchestration.ci_cd:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords="llm, fine-tuning, nlp, machine-learning, ai, transformers, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/abdelrahmanhassan12/llm-finetuning-pipeline/issues",
        "Source": "https://github.com/abdelrahmanhassan12/llm-finetuning-pipeline",
        "Documentation": "https://github.com/abdelrahmanhassan12/llm-finetuning-pipeline#readme",
    },
)

