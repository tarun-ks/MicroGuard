from setuptools import setup, find_packages

setup(
    name="microguard",
    version="0.1.0",
    description="Lightweight RAG faithfulness classifier using sub-billion parameter SLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Author Name",
    author_email="email@example.com",
    url="https://github.com/yourusername/MicroGuard",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "accelerate",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache-2.0",
)
