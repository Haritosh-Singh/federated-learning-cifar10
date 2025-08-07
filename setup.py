from setuptools import setup, find_packages

setup(
    name="federated-cifar10",
    version="1.0.0",
    description="Federated Learning simulation comparing FedAvg and FedProx on CIFAR-10",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "flwr[simulation]>=1.9.0",
        "flwr-datasets[vision]>=0.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.0.0",
        "tqdm>=4.64.0",
        "colorlog>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)