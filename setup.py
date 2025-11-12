from setuptools import setup, find_packages

setup(
    name='stress_predictor',
    version='0.1.0',
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        'pandas==2.3.3',
        'matplotlib==3.9.4',
        'scikit-learn==1.6.1',
    ],
    setup_requires=[
        'setuptools==80.9.0',
    ],
)
