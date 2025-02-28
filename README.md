# PromptPilot

A powerful tool for generating optimized prompts based on code repository analysis.

## Overview

PromptPilot analyzes code repositories to generate context-aware prompts for AI models. It extracts code structure, semantic relationships, and important context from your codebase, then creates optimized prompts that provide AI models with the most relevant information for the task at hand.

## Features

- **Repository Ingestion**: Process Git repositories, extracting file contents and metadata.
- **Content Analysis**: Generate embeddings and analyze code semantics using OpenAI's text-embedding model.
- **AST Analysis**: Parse and analyze code structure using Tree-sitter for multiple languages.
- **Graph Database Storage**: Store repository data, code relationships, and analysis results in ArangoDB.
- **Enhanced Prompt Generation**: Create optimized prompts using the Gemini API, incorporating repository context.
- **Command-line Interface**: Easy-to-use CLI for processing repositories and generating prompts.

## Supported Languages

- Python
- JavaScript/TypeScript
- Java
- C/C++
- And more (see `core/ast_analyzer.py` for the full list)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- ArangoDB (optional, for database storage)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/promptpilot.git
   cd promptpilot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file and add your API keys and database credentials.

### ArangoDB Setup

For database functionality, you need ArangoDB installed. The easiest way is using Docker:

```
docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD=your_password arangodb/arangodb:latest
```

## Usage

### Process a Repository

```
python promptpilot.py process /path/to/repository
```

### Generate a Prompt

```
python promptpilot.py prompt "Explain how the authentication system works"
```

### List Processed Repositories

```
python promptpilot.py list
```

### Test Database Connection

```
python test_db.py
```

### Process and Store Repository Data

```
python -m utils.db_utils /path/to/repository --full
```

## Architecture

PromptPilot consists of several core modules:

- **ingest.py**: Repository ingestion and file processing
- **analyze.py**: Content analysis and embedding generation
- **ast_analyzer.py**: AST analysis using Tree-sitter
- **enhanced_db.py**: ArangoDB integration for storing repository data
- **enhanced_prompt_generator.py**: Context-aware prompt generation
- **promptpilot.py**: Main CLI interface

## Development

### Project Structure

```
/promptpilot
├── core/                        # Core modules
│   ├── __init__.py
│   ├── ingest.py                # Repository ingestion module
│   ├── analyze.py               # Content analysis and embedding module 
│   ├── ast_analyzer.py          # AST analysis module
│   ├── enhanced_db.py           # Database integration module
│   └── enhanced_prompt_generator.py  # Prompt generation module
├── utils/                       # Utility functions
│   └── db_utils.py              # Database utilities
├── .env.example                 # Example environment variables template
├── .gitignore                   # Git ignore file
├── captainslog.md               # Project log
├── promptpilot.py               # Main CLI interface
├── test_db.py                   # Database test script
└── requirements.txt             # Project dependencies
```

### Running Tests

```
python test_db.py
```

## Limitations

- Large files (>500KB) are skipped during ingestion to maintain reasonable processing times.
- API rate limits may affect prompt generation and embedding creation.
- Tree-sitter parser availability depends on your system configuration.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the project.
