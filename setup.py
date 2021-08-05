from setuptools import setup

setup(
    name='dataperf',
    version='0.0.0',
    author='Coactive Systems Inc',
    author_email='exec@coactive.ai',
    description='Initial demo for ML dataperf',
    packages=['dataperf'],
    install_requires=[
        'pyarrow',
        'fastparquet',
        'pandas',
        'jupyter',
        'matplotlib',
        'scikit-learn',
        'tqdm'
    ]
)
