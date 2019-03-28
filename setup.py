from setuptools import setup, find_packages
import sys

setup(name='clevr_envs',
      packages=[package for package in find_packages()
                if package.startswith('clevr_envs')],
      install_requires=[],
      description="Mujoco environments inspired by Clevr dataset.",
      author="LB",
      author_email="",
      version="0.1.0")
