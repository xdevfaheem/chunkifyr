from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    requirements = [pkg.split("\n")[0].strip() for pkg in fr.readlines()]

setup(
    name="chunkifyr",
    version="0.1.0",
    author="Mohammed Faheem",
    author_email="xdevfahem@gmail.com",
    description="Your go-to library for text chunking",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    url="https://github.com/xdevfaheem/chunkifyr",  
    project_urls={
        "Bug Tracker": "https://github.com/xdevfaheem/chunkifyr/issues", 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"), 
    python_requires=">=3.9", 
    install_requires=[
        requirements
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
        ],
    },
)
