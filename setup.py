import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blr-sghmc", # Replace with your own username
    version="0.0.5",
    author="Aaron Liu, Christy Hu",
    author_email="tl254@duke.edu, dh275@duke.edu",
    description="Bayesian Logistic Regression using Stochastic Gradient Hamiltonian Monte Carlo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tienyuliu/blr-dev",
    project_urls={
        "Bug Tracker": "https://github.com/tienyuliu/blr-dev/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)