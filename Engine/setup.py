#!/usr/bin/env python3
"""
Setup script for OneBitLLM Python package.
Builds and installs the Rust extension module.
"""
import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

class RustExtension(Extension):
    """Custom extension for building Rust crates"""
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildRustExt(build_ext):
    """Build Rust extensions"""
    def build_extension(self, ext):
        if isinstance(ext, RustExtension):
            # Build Rust crate
            cmd = [
                sys.executable, '-m', 'pip', 'install', 
                'maturin'
            ]
            subprocess.check_call(cmd)
            
            # Build with maturin
            cmd = ['maturin', 'build', '--release', '--strip']
            subprocess.check_call(cmd, cwd=ext.sourcedir)
        else:
            super().build_extension(ext)

setup(
    name='onebitllm',
    version='0.1.0',
    description='High-performance 1-bit quantized LLM engine',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='OneBitLLM Team',
    license='LicenseRef-OneBitLLM-Research-Only-1.0',
    license_files=['LICENSE'],
    python_requires='>=3.8',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    install_requires=[
        'numpy>=1.19.0',
        'pydantic>=2.0.0',
    ],
    ext_modules=[
        RustExtension('onebitllm', sourcedir='.')
    ],
    cmdclass={'build_ext': BuildRustExt},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
