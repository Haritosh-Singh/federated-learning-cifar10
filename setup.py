from setuptools import setup, find_packages

setup(
    name="federated_cifar10",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flwr",
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas"
    ],
)
