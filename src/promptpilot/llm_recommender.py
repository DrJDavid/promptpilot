# src/promptpilot/llm_recommender.py
"""Module for recommending the best LLM for specific tasks and repository contexts."""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from promptpilot.task_classifier import TaskCategory


class LLMModel(str, Enum):
    """Enumeration of LLM models."""

    # Claude Models
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_5_HAIKU = "claude-3.5-haiku"
    CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
    CLAUDE_3_7_SONNET = "claude-3.7-sonnet"
    CLAUDE_3_7_SONNET_THINKING = "claude-3.7-sonnet-thinking"
    
    # Cursor Models
    CURSOR_FAST = "cursor-fast"
    CURSOR_SMALL = "cursor-small"
    
    # DeepSeek Models
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_V3 = "deepseek-v3"
    
    # Gemini Models
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_THINKING_EXP = "gemini-2.0-flash-thinking-exp"
    GEMINI_2_0_PRO_EXP = "gemini-2.0-pro-exp"
    GEMINI_EXP_1206 = "gemini-exp-1206"
    
    # GPT Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO_20240409 = "gpt-4-turbo-2024-04-09"
    GPT_4_5_PREVIEW = "gpt-4.5-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    
    # Grok Model
    GROK_2 = "grok-2"
    
    # O1 Models
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"
    
    # O3 Model
    O3_MINI = "o3-mini"


class LLMCapability(str, Enum):
    """Enumeration of LLM capabilities."""

    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    DEBUGGING = "debugging"
    PROBLEM_SOLVING = "problem_solving"
    ARCHITECTURE = "architecture"
    DOCUMENT_GENERATION = "document_generation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DETAILED_ANALYSIS = "detailed_analysis"
    TOKEN_EFFICIENCY = "token_efficiency"
    REAL_TIME_RESPONSE = "real_time_response"
    COMPLEX_REASONING = "complex_reasoning"
    MULTIMODAL_UNDERSTANDING = "multimodal_understanding"


# LLM Model profiles with their capabilities and characteristics
LLM_PROFILES = {
    # Claude Models
    LLMModel.CLAUDE_3_5_SONNET_20241022: {
        "name": "Claude 3.5 Sonnet (Oct 2024)",
        "provider": "Anthropic",
        "max_tokens": 4096,
        "context_window": 200000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.88,
            LLMCapability.CODE_EXPLANATION: 0.92,
            LLMCapability.DEBUGGING: 0.87,
            LLMCapability.PROBLEM_SOLVING: 0.9,
            LLMCapability.ARCHITECTURE: 0.87,
            LLMCapability.DOCUMENT_GENERATION: 0.95,
            LLMCapability.REFACTORING: 0.85,
            LLMCapability.TESTING: 0.83,
            LLMCapability.DETAILED_ANALYSIS: 0.92,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.8,
            LLMCapability.COMPLEX_REASONING: 0.9,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.85,
        },
        "cost_per_1k_tokens": 0.015,
        "knowledge_cutoff": "Oct 2024",
        "strengths": [
            "Very large context window",
            "Excellent documentation capabilities",
            "Strong reasoning and explanation",
            "Good at understanding complex code structures",
            "More recent knowledge cutoff"
        ],
        "weaknesses": [
            "Not as fast as specialized models",
            "Moderate cost",
            "Less multimodal capabilities than GPT-4o"
        ],
    },
    LLMModel.CLAUDE_3_OPUS: {
        "name": "Claude 3 Opus",
        "provider": "Anthropic",
        "max_tokens": 4096,
        "context_window": 200000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.9,
            LLMCapability.CODE_EXPLANATION: 0.95,
            LLMCapability.DEBUGGING: 0.9,
            LLMCapability.PROBLEM_SOLVING: 0.95,
            LLMCapability.ARCHITECTURE: 0.9,
            LLMCapability.DOCUMENT_GENERATION: 0.95,
            LLMCapability.REFACTORING: 0.9,
            LLMCapability.TESTING: 0.85,
            LLMCapability.DETAILED_ANALYSIS: 0.95,
            LLMCapability.TOKEN_EFFICIENCY: 0.8,
            LLMCapability.REAL_TIME_RESPONSE: 0.7,
            LLMCapability.COMPLEX_REASONING: 0.95,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.85,
        },
        "cost_per_1k_tokens": 0.03,
        "knowledge_cutoff": "Aug 2023",
        "strengths": [
            "Extremely large context window",
            "Exceptional at code explanation",
            "Strong reasoning and analysis abilities",
            "Excellent documentation generation",
        ],
        "weaknesses": [
            "Higher cost",
            "Slower response time",
            "Knowledge cutoff date limitations",
            "Limited multimodal abilities compared to GPT-4o"
        ],
    },
    LLMModel.CLAUDE_3_5_HAIKU: {
        "name": "Claude 3.5 Haiku",
        "provider": "Anthropic",
        "max_tokens": 4096,
        "context_window": 200000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.8,
            LLMCapability.CODE_EXPLANATION: 0.85,
            LLMCapability.DEBUGGING: 0.75,
            LLMCapability.PROBLEM_SOLVING: 0.8,
            LLMCapability.ARCHITECTURE: 0.75,
            LLMCapability.DOCUMENT_GENERATION: 0.85,
            LLMCapability.REFACTORING: 0.75,
            LLMCapability.TESTING: 0.75,
            LLMCapability.DETAILED_ANALYSIS: 0.8,
            LLMCapability.TOKEN_EFFICIENCY: 0.9,
            LLMCapability.REAL_TIME_RESPONSE: 0.9,
            LLMCapability.COMPLEX_REASONING: 0.8,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.8,
        },
        "cost_per_1k_tokens": 0.0025,
        "knowledge_cutoff": "Oct 2023",
        "strengths": [
            "Very large context window",
            "Fast response time",
            "Low cost for capabilities",
            "Good token efficiency",
            "Good for routine coding tasks"
        ],
        "weaknesses": [
            "Less sophisticated on complex tasks",
            "Weaker at debugging compared to larger models",
            "Less effective for complex architecture decisions"
        ],
    },
    LLMModel.CLAUDE_3_5_SONNET: {
        "name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "max_tokens": 4096,
        "context_window": 200000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.85,
            LLMCapability.CODE_EXPLANATION: 0.9,
            LLMCapability.DEBUGGING: 0.85,
            LLMCapability.PROBLEM_SOLVING: 0.9,
            LLMCapability.ARCHITECTURE: 0.85,
            LLMCapability.DOCUMENT_GENERATION: 0.95,
            LLMCapability.REFACTORING: 0.85,
            LLMCapability.TESTING: 0.8,
            LLMCapability.DETAILED_ANALYSIS: 0.9,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.8,
            LLMCapability.COMPLEX_REASONING: 0.9,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.85,
        },
        "cost_per_1k_tokens": 0.015,
        "knowledge_cutoff": "Oct 2023",
        "strengths": [
            "Very large context window",
            "Excellent documentation capabilities",
            "Strong reasoning and explanation",
            "Good balance of speed and accuracy",
        ],
        "weaknesses": [
            "Not as strong at testing as some models",
            "Moderate cost",
            "Moderate speed for real-time assistance"
        ],
    },
    LLMModel.CLAUDE_3_7_SONNET: {
        "name": "Claude 3.7 Sonnet",
        "provider": "Anthropic",
        "max_tokens": 4096,
        "context_window": 200000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.93,
            LLMCapability.CODE_EXPLANATION: 0.95,
            LLMCapability.DEBUGGING: 0.92,
            LLMCapability.PROBLEM_SOLVING: 0.93,
            LLMCapability.ARCHITECTURE: 0.9,
            LLMCapability.DOCUMENT_GENERATION: 0.95,
            LLMCapability.REFACTORING: 0.9,
            LLMCapability.TESTING: 0.88,
            LLMCapability.DETAILED_ANALYSIS: 0.95,
            LLMCapability.TOKEN_EFFICIENCY: 0.88,
            LLMCapability.REAL_TIME_RESPONSE: 0.82,
            LLMCapability.COMPLEX_REASONING: 0.95,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.9,
        },
        "cost_per_1k_tokens": 0.02,
        "knowledge_cutoff": "Feb 2025",
        "strengths": [
            "Best-in-class for real-world coding tasks",
            "Superior code explanation capabilities",
            "Excellent for complex debugging scenarios",
            "Very large context window",
            "Strong documentation generation",
            "Most recent knowledge cutoff"
        ],
        "weaknesses": [
            "Higher cost than smaller models",
            "Not as fast as specialized real-time models",
            "Less multimodal capabilities than GPT-4o"
        ],
    },
    LLMModel.CLAUDE_3_7_SONNET_THINKING: {
        "name": "Claude 3.7 Sonnet Thinking",
        "provider": "Anthropic",
        "max_tokens": 4096,
        "context_window": 200000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.92,
            LLMCapability.CODE_EXPLANATION: 0.95,
            LLMCapability.DEBUGGING: 0.93,
            LLMCapability.PROBLEM_SOLVING: 0.95,
            LLMCapability.ARCHITECTURE: 0.92,
            LLMCapability.DOCUMENT_GENERATION: 0.94,
            LLMCapability.REFACTORING: 0.92,
            LLMCapability.TESTING: 0.9,
            LLMCapability.DETAILED_ANALYSIS: 0.96,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.7,
            LLMCapability.COMPLEX_REASONING: 0.98,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.9,
        },
        "cost_per_1k_tokens": 0.025,
        "knowledge_cutoff": "Feb 2025",
        "strengths": [
            "Enhanced reasoning capabilities for complex programming problems",
            "Superior debugging for complicated issues",
            "Excellent at explaining complex code structures",
            "Strong for architectural decisions",
            "Most recent knowledge cutoff"
        ],
        "weaknesses": [
            "Slower response times due to extended reasoning",
            "Higher cost than standard Claude models",
            "Not ideal for simple, routine coding tasks",
        ],
    },
    
    # Cursor Models
    LLMModel.CURSOR_FAST: {
        "name": "Cursor Fast",
        "provider": "Cursor",
        "max_tokens": 4096,
        "context_window": 32000, 
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.85,
            LLMCapability.CODE_EXPLANATION: 0.8,
            LLMCapability.DEBUGGING: 0.82,
            LLMCapability.PROBLEM_SOLVING: 0.8,
            LLMCapability.ARCHITECTURE: 0.75,
            LLMCapability.DOCUMENT_GENERATION: 0.75,
            LLMCapability.REFACTORING: 0.85,
            LLMCapability.TESTING: 0.8,
            LLMCapability.DETAILED_ANALYSIS: 0.75,
            LLMCapability.TOKEN_EFFICIENCY: 0.9,
            LLMCapability.REAL_TIME_RESPONSE: 0.98,
            LLMCapability.COMPLEX_REASONING: 0.75,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.7,
        },
        "cost_per_1k_tokens": 0.01,  # Estimated
        "knowledge_cutoff": "2023",
        "strengths": [
            "Purpose-built for real-time coding in the Cursor IDE",
            "Extremely fast response times",
            "Optimized for coding workflows",
            "Good IDE-specific context awareness"
        ],
        "weaknesses": [
            "Limited utility outside Cursor IDE",
            "Less robust for complex reasoning tasks",
            "Less detailed explanations than larger models",
            "Less effective for architectural decisions"
        ],
    },
    LLMModel.CURSOR_SMALL: {
        "name": "Cursor Small",
        "provider": "Cursor",
        "max_tokens": 2048,
        "context_window": 16000, 
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.75,
            LLMCapability.CODE_EXPLANATION: 0.7,
            LLMCapability.DEBUGGING: 0.7,
            LLMCapability.PROBLEM_SOLVING: 0.65,
            LLMCapability.ARCHITECTURE: 0.6,
            LLMCapability.DOCUMENT_GENERATION: 0.65,
            LLMCapability.REFACTORING: 0.75,
            LLMCapability.TESTING: 0.7,
            LLMCapability.DETAILED_ANALYSIS: 0.6,
            LLMCapability.TOKEN_EFFICIENCY: 0.95,
            LLMCapability.REAL_TIME_RESPONSE: 0.98,
            LLMCapability.COMPLEX_REASONING: 0.6,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.6,
        },
        "cost_per_1k_tokens": 0.005,  # Estimated
        "knowledge_cutoff": "2023",
        "strengths": [
            "Extremely fast response times",
            "Very efficient for simple coding tasks",
            "Low resource usage", 
            "Optimized for coding in Cursor IDE"
        ],
        "weaknesses": [
            "Limited capabilities for complex tasks",
            "Reduced context window compared to larger models",
            "Less robust reasoning capabilities",
            "Less effective for architectural or detailed analysis"
        ],
    },
    
    # DeepSeek Models
    LLMModel.DEEPSEEK_R1: {
        "name": "DeepSeek R1",
        "provider": "DeepSeek",
        "max_tokens": 4096,
        "context_window": 32000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.95,
            LLMCapability.CODE_EXPLANATION: 0.88,
            LLMCapability.DEBUGGING: 0.92,
            LLMCapability.PROBLEM_SOLVING: 0.9,
            LLMCapability.ARCHITECTURE: 0.85,
            LLMCapability.DOCUMENT_GENERATION: 0.8,
            LLMCapability.REFACTORING: 0.92,
            LLMCapability.TESTING: 0.9,
            LLMCapability.DETAILED_ANALYSIS: 0.85,
            LLMCapability.TOKEN_EFFICIENCY: 0.9,
            LLMCapability.REAL_TIME_RESPONSE: 0.82,
            LLMCapability.COMPLEX_REASONING: 0.9,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.7,
        },
        "cost_per_1k_tokens": 0.0,  # Open source
        "knowledge_cutoff": "2024",
        "strengths": [
            "Specifically trained for code (87% code in training data)",
            "Outstanding code generation capabilities",
            "Open source - can be self-hosted",
            "Strong debugging and refactoring abilities",
            "Excellent for algorithmic challenges"
        ],
        "weaknesses": [
            "More limited multimodal capabilities",
            "Moderate context window size",
            "Less strong for documentation generation",
            "Requires significant resources to run locally"
        ],
    },
    LLMModel.DEEPSEEK_V3: {
        "name": "DeepSeek V3",
        "provider": "DeepSeek",
        "max_tokens": 4096,
        "context_window": 32000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.93,
            LLMCapability.CODE_EXPLANATION: 0.87,
            LLMCapability.DEBUGGING: 0.9,
            LLMCapability.PROBLEM_SOLVING: 0.88,
            LLMCapability.ARCHITECTURE: 0.83,
            LLMCapability.DOCUMENT_GENERATION: 0.78,
            LLMCapability.REFACTORING: 0.9,
            LLMCapability.TESTING: 0.88,
            LLMCapability.DETAILED_ANALYSIS: 0.83,
            LLMCapability.TOKEN_EFFICIENCY: 0.88,
            LLMCapability.REAL_TIME_RESPONSE: 0.85,
            LLMCapability.COMPLEX_REASONING: 0.88,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.7,
        },
        "cost_per_1k_tokens": 0.0,  # Open source
        "knowledge_cutoff": "2023",
        "strengths": [
            "Strong code generation capabilities",
            "Open source - can be self-hosted",
            "Good debugging abilities",
            "Better response time than R1",
            "Strong across many programming languages"
        ],
        "weaknesses": [
            "Some limitations with obscure programming environments",
            "Less specialized than DeepSeek R1",
            "Limited multimodal capabilities",
            "Less strong for documentation tasks"
        ],
    },
    
    # Gemini Models
    LLMModel.GEMINI_2_0_FLASH: {
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.82,
            LLMCapability.CODE_EXPLANATION: 0.85,
            LLMCapability.DEBUGGING: 0.8,
            LLMCapability.PROBLEM_SOLVING: 0.83,
            LLMCapability.ARCHITECTURE: 0.78,
            LLMCapability.DOCUMENT_GENERATION: 0.85,
            LLMCapability.REFACTORING: 0.75,
            LLMCapability.TESTING: 0.78,
            LLMCapability.DETAILED_ANALYSIS: 0.8,
            LLMCapability.TOKEN_EFFICIENCY: 0.9,
            LLMCapability.REAL_TIME_RESPONSE: 0.95,
            LLMCapability.COMPLEX_REASONING: 0.78,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.88,
        },
        "cost_per_1k_tokens": 0.0015,
        "knowledge_cutoff": "2023",
        "strengths": [
            "Very fast response times",
            "Multimodal capabilities for understanding code with visuals",
            "Large context window",
            "Integration with Google ecosystem",
            "Good balance of speed and capability"
        ],
        "weaknesses": [
            "Not as strong for complex reasoning tasks",
            "Less specialized for advanced refactoring",
            "Not as capable for architectural decisions"
        ],
    },
    LLMModel.GEMINI_2_0_FLASH_EXP: {
        "name": "Gemini 2.0 Flash Experimental",
        "provider": "Google",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.83,
            LLMCapability.CODE_EXPLANATION: 0.86,
            LLMCapability.DEBUGGING: 0.81,
            LLMCapability.PROBLEM_SOLVING: 0.84,
            LLMCapability.ARCHITECTURE: 0.79,
            LLMCapability.DOCUMENT_GENERATION: 0.86,
            LLMCapability.REFACTORING: 0.76,
            LLMCapability.TESTING: 0.79,
            LLMCapability.DETAILED_ANALYSIS: 0.81,
            LLMCapability.TOKEN_EFFICIENCY: 0.9,
            LLMCapability.REAL_TIME_RESPONSE: 0.94,
            LLMCapability.COMPLEX_REASONING: 0.8,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.89,
        },
        "cost_per_1k_tokens": 0.0015,
        "knowledge_cutoff": "2023",
        "strengths": [
            "Fast response times with experimental features",
            "Enhanced multimodal capabilities",
            "Large context window",
            "Slightly improved reasoning over standard Flash"
        ],
        "weaknesses": [
            "Experimental nature may lead to inconsistencies",
            "Still limited for complex architectural tasks",
            "Less specialized for code than dedicated coding models"
        ],
    },
    LLMModel.GEMINI_2_0_FLASH_THINKING_EXP: {
        "name": "Gemini 2.0 Flash Thinking Experimental",
        "provider": "Google",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.82,
            LLMCapability.CODE_EXPLANATION: 0.87,
            LLMCapability.DEBUGGING: 0.83,
            LLMCapability.PROBLEM_SOLVING: 0.87,
            LLMCapability.ARCHITECTURE: 0.82,
            LLMCapability.DOCUMENT_GENERATION: 0.85,
            LLMCapability.REFACTORING: 0.78,
            LLMCapability.TESTING: 0.8,
            LLMCapability.DETAILED_ANALYSIS: 0.85,
            LLMCapability.TOKEN_EFFICIENCY: 0.88,
            LLMCapability.REAL_TIME_RESPONSE: 0.85,
            LLMCapability.COMPLEX_REASONING: 0.88,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.88,
        },
        "cost_per_1k_tokens": 0.002,
        "knowledge_cutoff": "2023",
        "strengths": [
            "Enhanced reasoning capabilities with 'thinking' approach",
            "Better problem-solving for complex coding tasks",
            "Good explanation abilities",
            "Better architectural understanding than standard Flash",
            "Large context window"
        ],
        "weaknesses": [
            "Slower than standard Flash models",
            "Experimental nature may lead to inconsistencies",
            "Not as specialized for code as dedicated coding models"
        ],
    },
    LLMModel.GEMINI_2_0_PRO_EXP: {
        "name": "Gemini 2.0 Pro Experimental",
        "provider": "Google",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.87,
            LLMCapability.CODE_EXPLANATION: 0.9,
            LLMCapability.DEBUGGING: 0.87,
            LLMCapability.PROBLEM_SOLVING: 0.9,
            LLMCapability.ARCHITECTURE: 0.85,
            LLMCapability.DOCUMENT_GENERATION: 0.9,
            LLMCapability.REFACTORING: 0.83,
            LLMCapability.TESTING: 0.85,
            LLMCapability.DETAILED_ANALYSIS: 0.9,
            LLMCapability.TOKEN_EFFICIENCY: 0.87,
            LLMCapability.REAL_TIME_RESPONSE: 0.83,
            LLMCapability.COMPLEX_REASONING: 0.9,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.92,
        },
        "cost_per_1k_tokens": 0.0035,
        "knowledge_cutoff": "2023",
        "strengths": [
            "Advanced multimodal capabilities for code with visuals",
            "Strong problem-solving and reasoning",
            "Large context window",
            "Good documentation generation",
            "Well-balanced for various coding tasks"
        ],
        "weaknesses": [
            "Experimental features may have some inconsistencies",
            "Not as specialized for code as DeepSeek models",
            "Moderate response times compared to Flash variants"
        ],
    },
    LLMModel.GEMINI_EXP_1206: {
        "name": "Gemini Experimental 1206",
        "provider": "Google",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.85,
            LLMCapability.CODE_EXPLANATION: 0.88,
            LLMCapability.DEBUGGING: 0.85,
            LLMCapability.PROBLEM_SOLVING: 0.87,
            LLMCapability.ARCHITECTURE: 0.83,
            LLMCapability.DOCUMENT_GENERATION: 0.87,
            LLMCapability.REFACTORING: 0.82,
            LLMCapability.TESTING: 0.83,
            LLMCapability.DETAILED_ANALYSIS: 0.87,
            LLMCapability.TOKEN_EFFICIENCY: 0.86,
            LLMCapability.REAL_TIME_RESPONSE: 0.85,
            LLMCapability.COMPLEX_REASONING: 0.88,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.9,
        },
        "cost_per_1k_tokens": 0.003,
        "knowledge_cutoff": "2023",
        "strengths": [
            "Experimental features for advanced code understanding",
            "Good multimodal capabilities",
            "Strong problem-solving abilities",
            "Large context window",
            "Better balance of speed and capability than Pro"
        ],
        "weaknesses": [
            "Experimental nature may lead to inconsistencies",
            "Less information available about specific capabilities",
            "May not be as specialized for code as dedicated coding models"
        ],
    },
    
    # GPT Models
    LLMModel.GPT_3_5_TURBO: {
        "name": "GPT-3.5 Turbo",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 16384,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.75,
            LLMCapability.CODE_EXPLANATION: 0.8,
            LLMCapability.DEBUGGING: 0.7,
            LLMCapability.PROBLEM_SOLVING: 0.75,
            LLMCapability.ARCHITECTURE: 0.65,
            LLMCapability.DOCUMENT_GENERATION: 0.8,
            LLMCapability.REFACTORING: 0.7,
            LLMCapability.TESTING: 0.7,
            LLMCapability.DETAILED_ANALYSIS: 0.7,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.9,
            LLMCapability.COMPLEX_REASONING: 0.65,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.6,
        },
        "cost_per_1k_tokens": 0.0015,
        "knowledge_cutoff": "Oct 2023",
        "strengths": [
            "Very low cost",
            "Fast response time",
            "Good for simpler coding tasks",
            "Decent performance for day-to-day development",
        ],
        "weaknesses": [
            "Struggles with complex code",
            "Less accurate on debugging",
            "Limited architectural understanding",
            "Less reliable for large refactoring",
        ],
    },
    LLMModel.GPT_4: {
        "name": "GPT-4",
        "provider": "OpenAI",
        "max_tokens": 8192,
        "context_window": 8192,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.9,
            LLMCapability.CODE_EXPLANATION: 0.9,
            LLMCapability.DEBUGGING: 0.85,
            LLMCapability.PROBLEM_SOLVING: 0.95,
            LLMCapability.ARCHITECTURE: 0.85,
            LLMCapability.DOCUMENT_GENERATION: 0.9,
            LLMCapability.REFACTORING: 0.9,
            LLMCapability.TESTING: 0.85,
            LLMCapability.DETAILED_ANALYSIS: 0.9,
            LLMCapability.TOKEN_EFFICIENCY: 0.75,
            LLMCapability.REAL_TIME_RESPONSE: 0.7,
            LLMCapability.COMPLEX_REASONING: 0.9,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.8,
        },
        "cost_per_1k_tokens": 0.03,
        "knowledge_cutoff": "Apr 2023",
        "strengths": [
            "Excellent understanding of complex code",
            "Strong problem-solving abilities",
            "High-quality code generation",
            "Good at reasoning through issues",
        ],
        "weaknesses": [
            "Higher cost",
            "Slower response time",
            "Knowledge cutoff date limitations",
            "Smaller context window than newer models",
        ],
    },
    LLMModel.GPT_4_TURBO_20240409: {
        "name": "GPT-4 Turbo (Apr 2024)",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.9,
            LLMCapability.CODE_EXPLANATION: 0.9,
            LLMCapability.DEBUGGING: 0.87,
            LLMCapability.PROBLEM_SOLVING: 0.92,
            LLMCapability.ARCHITECTURE: 0.86,
            LLMCapability.DOCUMENT_GENERATION: 0.9,
            LLMCapability.REFACTORING: 0.87,
            LLMCapability.TESTING: 0.86,
            LLMCapability.DETAILED_ANALYSIS: 0.88,
            LLMCapability.TOKEN_EFFICIENCY: 0.82,
            LLMCapability.REAL_TIME_RESPONSE: 0.8,
            LLMCapability.COMPLEX_REASONING: 0.88,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.87,
        },
        "cost_per_1k_tokens": 0.01,
        "knowledge_cutoff": "Apr 2024",
        "strengths": [
            "Large context window for big codebases",
            "Faster response time than GPT-4",
            "Good balance of quality and cost",
            "Strong across many coding tasks",
            "More recent knowledge cutoff",
        ],
        "weaknesses": [
            "Slightly less accurate than GPT-4 on complex tasks",
            "Higher cost than GPT-3.5",
            "Less optimized for real-time assistance than GPT-4o",
        ],
    },
    LLMModel.GPT_4_5_PREVIEW: {
        "name": "GPT-4.5 Preview",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.93,
            LLMCapability.CODE_EXPLANATION: 0.93,
            LLMCapability.DEBUGGING: 0.9,
            LLMCapability.PROBLEM_SOLVING: 0.95,
            LLMCapability.ARCHITECTURE: 0.89,
            LLMCapability.DOCUMENT_GENERATION: 0.93,
            LLMCapability.REFACTORING: 0.9,
            LLMCapability.TESTING: 0.88,
            LLMCapability.DETAILED_ANALYSIS: 0.92,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.83,
            LLMCapability.COMPLEX_REASONING: 0.95,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.93,
        },
        "cost_per_1k_tokens": 0.015,  # Estimated
        "knowledge_cutoff": "2024",
        "strengths": [
            "Enhanced reasoning capabilities",
            "Improved code generation and explanation",
            "Advanced problem-solving abilities",
            "Large context window",
            "Strong multimodal capabilities",
        ],
        "weaknesses": [
            "Preview status may lead to inconsistencies",
            "Higher cost than GPT-4o",
            "Slower response times than specialized models",
            "Limited availability as a preview model",
        ],
    },
    LLMModel.GPT_4O: {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.92,
            LLMCapability.CODE_EXPLANATION: 0.92,
            LLMCapability.DEBUGGING: 0.88,
            LLMCapability.PROBLEM_SOLVING: 0.92,
            LLMCapability.ARCHITECTURE: 0.87,
            LLMCapability.DOCUMENT_GENERATION: 0.92,
            LLMCapability.REFACTORING: 0.88,
            LLMCapability.TESTING: 0.87,
            LLMCapability.DETAILED_ANALYSIS: 0.9,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.88,
            LLMCapability.COMPLEX_REASONING: 0.9,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.95,
        },
        "cost_per_1k_tokens": 0.005,
        "knowledge_cutoff": "Apr 2023",
        "strengths": [
            "Excellent code optimization suggestions",
            "Fast response times for complex tasks",
            "Strong multimodal capabilities",
            "Better cost-efficiency than GPT-4",
            "Excellent for real-time coding assistance",
        ],
        "weaknesses": [
            "Knowledge cutoff date limitations",
            "More expensive than GPT-3.5 for high-volume use",
        ],
    },
    LLMModel.GPT_4O_MINI: {
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.85,
            LLMCapability.CODE_EXPLANATION: 0.85,
            LLMCapability.DEBUGGING: 0.8,
            LLMCapability.PROBLEM_SOLVING: 0.83,
            LLMCapability.ARCHITECTURE: 0.78,
            LLMCapability.DOCUMENT_GENERATION: 0.85,
            LLMCapability.REFACTORING: 0.8,
            LLMCapability.TESTING: 0.78,
            LLMCapability.DETAILED_ANALYSIS: 0.8,
            LLMCapability.TOKEN_EFFICIENCY: 0.9,
            LLMCapability.REAL_TIME_RESPONSE: 0.92,
            LLMCapability.COMPLEX_REASONING: 0.78,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.85,
        },
        "cost_per_1k_tokens": 0.0015,
        "knowledge_cutoff": "Apr 2023",
        "strengths": [
            "Very fast response times",
            "Good cost efficiency",
            "Large context window",
            "Strong for routine coding tasks",
            "Good multimodal capabilities at lower cost",
        ],
        "weaknesses": [
            "Less capable on complex reasoning tasks",
            "Reduced performance on sophisticated debugging scenarios",
            "Not as strong for architectural decisions",
        ],
    },
    
    # Grok Model
    LLMModel.GROK_2: {
        "name": "Grok-2",
        "provider": "xAI",
        "max_tokens": 4096,
        "context_window": 32000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.88,
            LLMCapability.CODE_EXPLANATION: 0.87,
            LLMCapability.DEBUGGING: 0.85,
            LLMCapability.PROBLEM_SOLVING: 0.88,
            LLMCapability.ARCHITECTURE: 0.82,
            LLMCapability.DOCUMENT_GENERATION: 0.85,
            LLMCapability.REFACTORING: 0.84,
            LLMCapability.TESTING: 0.82,
            LLMCapability.DETAILED_ANALYSIS: 0.85,
            LLMCapability.TOKEN_EFFICIENCY: 0.83,
            LLMCapability.REAL_TIME_RESPONSE: 0.85,
            LLMCapability.COMPLEX_REASONING: 0.88,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.88,
        },
        "cost_per_1k_tokens": 0.0,  # Free with X Premium
        "knowledge_cutoff": "2024",
        "strengths": [
            "State-of-the-art reasoning capabilities",
            "Strong coding capabilities",
            "Access to real-time information through X platform",
            "Multimodal text and vision understanding",
            "Outperforms some leading models on benchmarks",
        ],
        "weaknesses": [
            "Limited availability (requires X Premium subscription)",
            "Less established in the coding community",
            "Potentially controversial outputs due to 'less censored' approach",
            "Less comprehensive testing in production environments",
        ],
    },
    
    # O1 Models
    LLMModel.O1: {
        "name": "O1",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.92,
            LLMCapability.CODE_EXPLANATION: 0.92,
            LLMCapability.DEBUGGING: 0.9,
            LLMCapability.PROBLEM_SOLVING: 0.98,
            LLMCapability.ARCHITECTURE: 0.93,
            LLMCapability.DOCUMENT_GENERATION: 0.9,
            LLMCapability.REFACTORING: 0.9,
            LLMCapability.TESTING: 0.88,
            LLMCapability.DETAILED_ANALYSIS: 0.95,
            LLMCapability.TOKEN_EFFICIENCY: 0.75,
            LLMCapability.REAL_TIME_RESPONSE: 0.65,
            LLMCapability.COMPLEX_REASONING: 0.98,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.88,
        },
        "cost_per_1k_tokens": 0.04,  # Estimated
        "knowledge_cutoff": "2024",
        "strengths": [
            "Superior reasoning capabilities for complex programming problems",
            "Excels at multi-step problem-solving",
            "Outstanding algorithm design and optimization",
            "Deep understanding of complex code structures",
            "Strong architectural decision-making",
        ],
        "weaknesses": [
            "Significantly slower response times",
            "Much higher cost than other models",
            "Overkill for simple coding tasks",
            "Less optimized for real-time assistance",
        ],
    },
    LLMModel.O1_MINI: {
        "name": "O1 Mini",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.9,
            LLMCapability.CODE_EXPLANATION: 0.9,
            LLMCapability.DEBUGGING: 0.88,
            LLMCapability.PROBLEM_SOLVING: 0.95,
            LLMCapability.ARCHITECTURE: 0.9,
            LLMCapability.DOCUMENT_GENERATION: 0.88,
            LLMCapability.REFACTORING: 0.88,
            LLMCapability.TESTING: 0.85,
            LLMCapability.DETAILED_ANALYSIS: 0.92,
            LLMCapability.TOKEN_EFFICIENCY: 0.8,
            LLMCapability.REAL_TIME_RESPONSE: 0.75,
            LLMCapability.COMPLEX_REASONING: 0.95,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.85,
        },
        "cost_per_1k_tokens": 0.02,  # Estimated
        "knowledge_cutoff": "2024",
        "strengths": [
            "Strong reasoning for complex programming tasks",
            "Better speed/performance balance than O1",
            "Excels at algorithm design",
            "Good architectural understanding",
            "Strong for detailed code analysis",
        ],
        "weaknesses": [
            "Still slower than models optimized for speed",
            "Higher cost than general-purpose models",
            "Less suited for simple, routine coding tasks",
        ],
    },
    LLMModel.O1_PREVIEW: {
        "name": "O1 Preview",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.93,
            LLMCapability.CODE_EXPLANATION: 0.93,
            LLMCapability.DEBUGGING: 0.91,
            LLMCapability.PROBLEM_SOLVING: 0.97,
            LLMCapability.ARCHITECTURE: 0.94,
            LLMCapability.DOCUMENT_GENERATION: 0.91,
            LLMCapability.REFACTORING: 0.91,
            LLMCapability.TESTING: 0.89,
            LLMCapability.DETAILED_ANALYSIS: 0.96,
            LLMCapability.TOKEN_EFFICIENCY: 0.75,
            LLMCapability.REAL_TIME_RESPONSE: 0.7,
            LLMCapability.COMPLEX_REASONING: 0.99,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.89,
        },
        "cost_per_1k_tokens": 0.03,  # Estimated
        "knowledge_cutoff": "2024",
        "strengths": [
            "Preview of advanced reasoning capabilities",
            "Exceptionally strong for complex problem-solving",
            "Excellent for sophisticated architectural decisions",
            "Deep code understanding and analysis",
        ],
        "weaknesses": [
            "Preview status may lead to inconsistencies",
            "Slower response times",
            "Higher cost",
            "Limited availability as a preview model",
        ],
    },
    
    # O3 Model
    LLMModel.O3_MINI: {
        "name": "O3 Mini",
        "provider": "OpenAI",
        "max_tokens": 4096,
        "context_window": 128000,
        "capabilities": {
            LLMCapability.CODE_GENERATION: 0.93,
            LLMCapability.CODE_EXPLANATION: 0.9,
            LLMCapability.DEBUGGING: 0.92,
            LLMCapability.PROBLEM_SOLVING: 0.95,
            LLMCapability.ARCHITECTURE: 0.88,
            LLMCapability.DOCUMENT_GENERATION: 0.88,
            LLMCapability.REFACTORING: 0.92,
            LLMCapability.TESTING: 0.9,
            LLMCapability.DETAILED_ANALYSIS: 0.9,
            LLMCapability.TOKEN_EFFICIENCY: 0.85,
            LLMCapability.REAL_TIME_RESPONSE: 0.83,
            LLMCapability.COMPLEX_REASONING: 0.93,
            LLMCapability.MULTIMODAL_UNDERSTANDING: 0.85,
        },
        "cost_per_1k_tokens": 0.015,  # Estimated
        "knowledge_cutoff": "2024",
        "strengths": [
            "Specifically fine-tuned for STEM and programming",
            "Top scores on coding benchmarks like Codeforces Elo and SWE-bench",
            "Faster and more cost-effective than O1 models",
            "Strong reasoning capabilities for complex coding tasks",
            "Good balance between performance and efficiency",
        ],
        "weaknesses": [
            "Newer model with less established track record",
            "Not as fast as models optimized for real-time assistance",
            "More expensive than general-purpose models like GPT-3.5",
        ],
    },
}