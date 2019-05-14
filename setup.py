from setuptools import setup, Command
import os


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='qperceptron',
    version='0.1',
    packages=['qperceptron',],
    license='MIT License',
    author='Roeland Wiersema',
    package_dir={'qperceptron': 'src'},
    description='Minimal code to reproduce the figures in the 2019 paper "PAPERNAME"',
    cmdclass={
        'clean': CleanCommand,
    }
)
