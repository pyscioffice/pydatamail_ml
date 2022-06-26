"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
from pathlib import Path
import versioneer


setup(
    name='pydatamail_ml',
    version=versioneer.get_version(),
    description='pydatamail_ml - Machine Learning extension for pydatamail',
    long_description=Path("README.md").read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/pyscioffice/pydatamail_ml',
    author='Jan Janssen',
    author_email='jan.janssen@outlook.com',
    license='BSD',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        'bleach==5.0.0',
        'cssutils==2.4.2',
        'langdetect==1.0.9',
        'numpy==1.23.0',
        'pandas==1.4.3',
        'pydatamail==0.0.5',
        'scikit-learn==1.1.1',
        'sqlalchemy==1.4.39',
        'tqdm==4.64.0',
    ],
    cmdclass=versioneer.get_cmdclass()
)
