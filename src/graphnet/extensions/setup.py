from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='mknn',
      version='0.1.0',
      ext_modules=[
          CUDAExtension('mknn', [
              'mknn_cuda.cpp',
              'mknn.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
