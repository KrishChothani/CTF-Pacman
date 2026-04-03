from setuptools import setup, find_packages

setup(
    name="ctf-pacman",
    version="0.1.0",
    description="Capture the Flag Pacman - Multi-Agent Reinforcement Learning",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "pytest>=7.4.0",
        "tqdm>=4.66.0",
        "scipy>=1.11.0",
    ],
    entry_points={
        "console_scripts": [
            "ctf-train = scripts.train:main",
            "ctf-eval = scripts.evaluate:main",
        ],
    },
)
