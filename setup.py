"""akro setuptools script."""
from setuptools import find_packages
from setuptools import setup

# Required dependencies
required = [
    # Please keep alphabetized
    'gym>=0.12.4',
    'numpy',
]

# Framework-specific dependencies
extras = {
    'tf': ['tensorflow<2.0'],
    'theano': ['theano'],
}
extras['all'] = list(set(sum(extras.values(), [])))

# Development dependencies (*not* included in "all")
extras['dev'] = [
    # Please keep alphabetized
    'coverage',
    'flake8',
    'flake8-docstrings>=1.5.0',
    'flake8-import-order',
    'pep8-naming',
    'pre-commit',
    'pycodestyle>=2.5.0',
    'pydocstyle>=4.0.0',
    'pylint',
    'pytest>=4.4.0',  # Required for pytest-xdist
    'pytest-cov',
    'pytest-xdist',
    'sphinx',
    'recommonmark',
    'yapf',
]

with open('README.md') as f:
    readme = f.read()

# Get the package version dynamically
with open('VERSION') as v:
    version = v.read().strip()

setup(
    name='akro',
    version=version,
    author='Reinforcement Learning Working Group',
    author_email='akro@noreply.github.com',
    description='Spaces types for reinforcement learning',
    url='https://github.com/rlworkgroup/akro',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.5',
    install_requires=required,
    extras_require=extras,
    license='MIT',
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
)
