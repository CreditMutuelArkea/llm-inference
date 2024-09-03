# Bloomz Inference Server

This project implements an inference server for Bloomz Large Language Models (LLMs). The server allows for efficient and scalable inference of Bloomz models, making it suitable for various NLP applications.

## Features

- **Flexible:** Easily configurable to support different model sizes and configurations.
- **REST API:** Provides a RESTful API for easy integration with other applications.

## Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Clone the Repository

```bash
git clone <github.com>
cd llm-inference
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Before running the server, you need to configure it and set some environment variables.

```shell
HUGGING_FACE_HUB_TOKEN="<YOUR HF HUB TOKEN>"
```

### Running the Server

```bash
python -m llm_inference --model "cmarkea/bloomz-3b-retriever-v2" --task EMBEDDING
```

### API Endpoints

You can access server documentation through this endpoint : `/docs`

## Testing

To run the tests, use the following command:

```bash
pytest tests/
```

## Contributing

We welcome contributions to this project! Please open an issue or submit a pull request on GitHub.

### Guidelines

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Make your changes.
- Ensure all tests pass.
- Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Bloomz](https://bloomz.ai) for providing the pre-trained models.
- [Your Organization](https://yourorganization.com) for supporting this project.

## Contact

For any inquiries or support, please contact [your email](mailto:youremail@example.com).
