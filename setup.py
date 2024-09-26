from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='agents',
    version='0.1.20240926-13',
    description='Yet another langchain-esque package to use language agents and compose agent-systems',
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    url='https://github.com/cdcai/multiagent',
    author='Sean Browning',
    author_email='sbrowning@cdc.gov',
    license='MIT',
    packages=find_packages(include=["agents", "agents.*"]),
    install_requires=requirements,
    zip_safe=False
)