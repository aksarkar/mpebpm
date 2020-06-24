import setuptools

setuptools.setup(
  name='mpebpm',
  description='Massively Parallel Empirical Bayes Poisson Means',
  version='0.2',
  url='https://www.github.com/aksarkar/mpebpm',
  author='Abhishek Sarkar',
  author_email='aksarkar@uchicago.edu',
  license='MIT',
  install_requires=['numpy', 'scipy', 'torch'],
  packages=setuptools.find_packages('src'),
  package_dir={'': 'src'},
  tests_require=['pytest'],
)
