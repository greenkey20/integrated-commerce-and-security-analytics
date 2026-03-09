"""
Customer Segmentation 프로젝트 설치 스크립트
"""
from setuptools import setup, find_packages

setup(
    name="customer-segmentation",
    version="1.0.0",
    description="고객 세그멘테이션 및 이상 탐지 시스템",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "pandas>=2.0.3",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "seaborn>=0.12.2",
        "matplotlib>=3.7.2",
        "python-multipart>=0.0.5",
        "httpx>=0.24.0",
        "scipy>=1.11.0",
        "typing-extensions>=4.5.0",
        "protobuf>=3.20.0,<4.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ]
    },
    entry_points={
        "console_scripts": [
            "customer-api=api.customer_api:main",
            "customer-app=app:main",
        ]
    },
    package_data={
        "": ["*.csv", "*.pkl", "*.joblib", "*.h5"],
    },
    include_package_data=True,
)
