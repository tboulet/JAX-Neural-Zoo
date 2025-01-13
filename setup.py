from setuptools import setup, find_namespace_packages

setup(
    name="jaxzoo",
    url="https://github.com/tboulet/JAX-Neural-Zoo", 
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    
    packages=find_namespace_packages(),

    version="1.0",
    license="MIT",
    description="Highly flexible input/output space agnostic NN models in JAX.",
    long_description=open('README.md').read(),      
    long_description_content_type="text/markdown",  
)