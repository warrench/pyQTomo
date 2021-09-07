from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

with open(here / "requirements.txt", encoding='utf-8') as f:
    reqs = f.read().splitlines()

setup(
    name='pyqtomo',
    version="0.0.1",
    description='pyQTomo | python quantum tomography package',
    author='Christopher Warren',
    author_email='warrenc@chalmers.se',
    license='Apache 2.0',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=reqs,
    project_urls={
        'Source Code': 'https://github.com/warrench/pyQTomo' 
    }
)