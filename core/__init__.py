"""
Core functionality for PromptPilot.

This package contains the core modules for repository ingestion,
analysis, AST parsing, prompt generation, and database integration.
"""

# Import main classes and functions so they can be imported directly from core
from core.ingest import RepositoryIngestor, ingest_repository
from core.analyze import RepositoryAnalyzer
from core.ast_analyzer import ASTAnalyzer
from core.enhanced_prompt_generator import PromptGenerator
from core.enhanced_db import RepositoryDB, get_repository_db

__all__ = [
    'RepositoryIngestor',
    'ingest_repository',
    'RepositoryAnalyzer',
    'ASTAnalyzer',
    'PromptGenerator',
    'RepositoryDB',
    'get_repository_db',
]
