from setuptools import setup, find_packages
from llm_inference import __version__


def read_req_file(req_type: str = None):
    """Permet de récupérer les dépendances pour chaque environnement: package, dev, api, etc."""
    file = f"requirements-{req_type}.txt" if req_type != None else "requirements.txt"
    with open(file) as f:
        requires = (line.strip() for line in f)
        return [req for req in requires if req and not req.startswith(("#", "-f"))]


def read_dependency_links():
    """Permet de récupérer les url de dépendances pour chaque environnement."""
    with open("requirements.txt") as f:
        requires = (line.strip() for line in f)
        return [req[:3] for req in requires if req and req.startswith("-f")]


keywords = "machine learning, api, python"

setup(
    name="llm-inference",
    version=__version__,
    description="Serveur d'inférence pour les modèles Bloomz",
    long_description=open("index.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Killian Mahé",
    author_email="killianmahe.pro@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    keywords=[key for key in keywords.split(", ")],
    install_requires=read_req_file(),  # Ajout de dépendances principales dans requirements.txt
    dependency_links=read_dependency_links(),
    extra_requires={
        "dev": read_req_file("dev"),  # Ajout de dépendances dev dans requirements-dev.txt
    },
    python_requires=">=3.9",
    project_urls={
        "Documentation": "",
    },
)
