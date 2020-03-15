import setuptools

_name = 'mpebpm'

setuptools.setup(
  name=_name,
  description='Massively Parallel Empirical Bayes Poisson Means',
  version='0.1',
  url=f'https://www.github.com/aksarkar/{_name}',
  author='Abhishek Sarkar',
  author_email='aksarkar@uchicago.edu',
  license='MIT',
  install_requires=[],
  packages=setuptools.find_packages('src'),
  package_dir={'': 'src'},
)
