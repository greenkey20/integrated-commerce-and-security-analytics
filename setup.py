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
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "tensorflow>=2.10.0",
        "streamlit>=1.25.0",
        "plotly>=5.10.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.5.0",
        "python-multipart>=0.0.5",
        "httpx>=0.24.0",
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
