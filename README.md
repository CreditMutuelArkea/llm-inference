# Bloomz Inference Server

This project implements an inference server for Bloomz Large Language Models (LLMs). The server allows for efficient and scalable inference of Bloomz models, making it suitable for various NLP applications.

## Features

- **Flexible:** Easily configurable to support different model sizes and configurations.
- **REST API:** Provides a RESTful API for easy integration with other applications.

## Installation

### Prerequisites

- Python 3.10 or higher
- `poetry` package manager

### Clone the Repository

```bash
git clone <github.com>
cd llm-inference
```

### Install Dependencies

```bash
poetry install
```

## Usage

### Configuration

Before running the server, you need to configure it and set some environment variables.

```shell
HUGGING_FACE_HUB_TOKEN="<YOUR HF HUB TOKEN>"
```

### Running the Server

For each kind of task goes a specific model here are some models based on bloomz architecture, Open Sourced that you
could use you will find the latest model on [Credit Mutuel Arkea's hugginface ODQA collection](https://huggingface.co/collections/cmarkea/odqa-65f56ecd2b3e8e993a9982d6).

If you are using PyCharm find run configuration in `.run/**.yaml` they should appear direcly in PyCharm, those configurations uses the samllest models.

#### Embedding server
Use to vectorise documents search for `*-retriever` [models](https://huggingface.co/collections/cmarkea/odqa-65f56ecd2b3e8e993a9982d6), then start the inference server like this (smallest model) :
```bash
python -m llm_inference --task EMBEDDING --port 8081 --model cmarkea/bloomz-560m-retriever-v2
```

Then go to http://localhost:8081/docs.

#### Reranking / Scoring server
Use to rank severeal context according to a specific query, search for `*-reranking` [models](https://huggingface.co/collections/cmarkea/odqa-65f56ecd2b3e8e993a9982d6), then start the inference server like this (smallest model) :
```bash
python -m llm_inference --task SCORING --port 8082 --model cmarkea/bloomz-560m-reranking
```

Then go to http://localhost:8082/docs.

Be aware to check the examples in the model card depending on the model you use to understand the meaning of the output labels.
For instance for [**cmarkea/bloomz-560m-reranking**](https://huggingface.co/cmarkea/bloomz-560m-reranking), `LABEL1`
near to 1 means that the context in really similar to the query, as [described in the model card](https://huggingface.co/cmarkea/bloomz-560m-reranking#:~:text=context%20in%20contexts%0A%20%20%20%20%5D%0A)-,contexts_reranked,-%3D%20sorted().

#### Guardrail

Use to detect responses that would be toxic for instance : insult, obscene, sexual_explicit, identity_attack...
Our guardrail models are published under  `*-guardrail` [models](https://huggingface.co/collections/cmarkea/odqa-65f56ecd2b3e8e993a9982d6)

```bash
python -m llm_inference --task GUARDRAIL --port 8083 --model cmarkea/bloomz-560m-guardrail
```
Then go to http://localhost:8083/docs.

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
- [Crédit Mutuel Arkéa](https://www.cm-arkea.com/) for supporting this project.

## Contact

For any inquiries or support, open an issue on this repository.
