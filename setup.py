from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='Pose Estimation for BRICS-MINI',
    name='pose_estimation',
    packages=find_packages(),
    install_requires=[
        'natsort',
        'numpy',
        'opencv-python',
        'pyrender',
        'scikit-image',
        'smplx==0.1.28',
        'torch',
        'torchvision',
        'yacs',
        'ultralytics',
        'chumpy @ git+https://github.com/mattloper/chumpy',
        'mmcv-full==1.5.0',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
        'ujson',
        'ipdb',
        'ffmpeg',
        'ffmpeg-python',
        'tabulate',
        'termcolor'
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
            'webdataset',
        ],
    },
)