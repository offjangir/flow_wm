from setuptools import setup, find_packages

setup(
    name="differentiable_robot_rendering",
    version="0.1.0",
    description="A library for differentiable robot rendering with kinematically-aware 3D gaussians",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alper Canberk",
    author_email="alper.tu.canberk@gmail.com",
    url="https://github.com/alpercanberk/differentiable-robot-rendering",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        # Add more dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="robotics, differentiable, simulation, control",
    python_requires=">=3.7",
)