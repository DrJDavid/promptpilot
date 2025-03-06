from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Filter requirements to remove problematic packages
with open("requirements.txt", "r", encoding="utf-8") as f:
    all_requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
# Create a list of requirements without the problematic pgvector-python
requirements = [req for req in all_requirements if not req.startswith('pgvector-python')]

setup(
    name="promptpilot",
    version="0.1.0",
    author="PromptPilot Team",
    author_email="info@promptpilot.ai",
    description="An AI-powered tool for repository analysis and intelligent code assistance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/organization/promptpilot",
    packages=find_packages(),
    py_modules=["promptpilot"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "promptpilot=promptpilot:cli",
        ],
    },
) 