"""
Query LLM with retrieved context using structured prompts
Simplified for single work (Tosa Nikki)
"""

import sys
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class Translation(BaseModel):
    """Translation in different languages"""
    zh: Optional[str] = Field(None, description="Chinese translation")
    en: Optional[str] = Field(None, description="English translation")
    ja: Optional[str] = Field(None, description="Japanese translation")


class RetrievedDocument(BaseModel):
    """A single retrieved text segment"""
    text_id: int = Field(..., description="Unique identifier for the text segment")
    original_text: str = Field(..., description="Original Japanese text")
    kana: str = Field(..., description="Kana reading of the text")
    is_poem: Optional[bool] = Field(False, description="Whether this is poetry or prose")
    translations: Translation = Field(..., description="Translations in multiple languages")
    scores: Dict[str, float] = Field(..., description="Relevance scores")
    
    @field_validator('is_poem', mode='before')
    @classmethod
    def validate_is_poem(cls, v):
        """Handle None values for is_poem"""
        if v is None:
            return False
        return v


class SearchResults(BaseModel):
    """Complete search results from the retrieval system"""
    query: str = Field(..., description="User's original query")
    num_results: int = Field(..., description="Number of documents retrieved")
    results: List[RetrievedDocument] = Field(default_factory=list, description="Retrieved documents")


class PromptTemplate(BaseModel):
    """Structured prompt template for LLM queries"""
    system_role: str = Field(..., description="System role definition")
    context_prefix: str = Field(..., description="Introduction to the context")
    document_template: str = Field(..., description="Template for each document")
    context_suffix: str = Field(..., description="Closing of the context section")
    user_query_prefix: str = Field(..., description="Introduction to user query")
    instructions: List[str] = Field(default_factory=list, description="Specific instructions for the LLM")
    output_format: Optional[str] = Field(None, description="Expected output format")


class TosaNikkiPrompt(BaseModel):
    """Specialized prompt for Tosa Nikki (土佐日記) queries"""
    
    @staticmethod
    def get_default_template() -> PromptTemplate:
        """Get the default prompt template for Tosa Nikki"""
        return PromptTemplate(
            system_role="""You are a specialist in classical Japanese literature, with expertise in analyzing texts from the Heian period (794-1185 CE). Your knowledge includes:
- Classical Japanese language (文語/bungo) grammar and vocabulary
- Historical and cultural context of the period
- Literary conventions and poetic forms
- Comparative analysis across translations

You provide accurate, balanced analysis based on textual evidence.""",
            
            context_prefix="""# Retrieved Context

The following passages have been retrieved from the text database based on relevance to the query:
""",
            
            document_template="""
## Passage {index} (ID: {text_id})

**Original Text:**
{original_text}

**Reading:**
{kana}

**Type:** {text_type}

**Chinese Translation:**
{translation_zh}

**English Translation:**
{translation_en}

**Modern Japanese Translation:**
{translation_ja}

**Relevance Score:** {relevance_score}

---
""",
            
            context_suffix="""
# End of Retrieved Context
""",
            
            user_query_prefix="# Query\n\n",
            
            instructions=[
                "Answer based on the retrieved passages above",
                "Cite specific passages when making claims (e.g., 'Passage 1 states...')",
                "Analyze the original text when relevant",
                "Provide historical or cultural context if helpful",
                "Compare translations if there are interesting differences",
                "If the passages don't contain enough information, state what is known and what is unclear"
            ],
            
            output_format=None
        )
    
    @staticmethod
    def format_document(doc: RetrievedDocument, index: int, template: str) -> str:
        """Format a single document using the template"""
        return template.format(
            index=index,
            text_id=doc.text_id,
            original_text=doc.original_text,
            kana=doc.kana,
            text_type="Poetry" if doc.is_poem else "Prose",
            translation_zh=doc.translations.zh or "N/A",
            translation_en=doc.translations.en or "N/A",
            translation_ja=doc.translations.ja or "N/A",
            relevance_score=f"{doc.scores['final']:.3f}"
        )
    
    @staticmethod
    def build_prompt(search_results: SearchResults, template: Optional[PromptTemplate] = None) -> Dict[str, str]:
        """Build complete prompt for LLM from search results"""
        
        if template is None:
            template = TosaNikkiPrompt.get_default_template()
        
        # Build context section
        context_parts = [template.context_prefix]
        
        for idx, doc in enumerate(search_results.results, 1):
            formatted_doc = TosaNikkiPrompt.format_document(doc, idx, template.document_template)
            context_parts.append(formatted_doc)
        
        context_parts.append(template.context_suffix)
        
        # Build instructions section
        instructions_text = "\n# Instructions\n\n"
        for instruction in template.instructions:
            instructions_text += f"- {instruction}\n"
        
        # Build output format section
        output_format_text = ""
        if template.output_format:
            output_format_text = f"\n# Output Format\n\n{template.output_format}\n"
        
        # Combine all parts
        system_message = template.system_role
        
        user_message = (
            "".join(context_parts) +
            instructions_text +
            output_format_text +
            template.user_query_prefix +
            search_results.query
        )
        
        return {
            "system": system_message,
            "user": user_message
        }


def main():
    """Main function to process search results and generate prompt"""
    
    # Read from stdin
    if sys.stdin.isatty():
        print("Usage: python query.py [format] < search_results.json", file=sys.stderr)
        print("   or: echo 'query' | python search_neo4j.py 3 json | python query.py", file=sys.stderr)
        print("\nFormats: json (default), text, markdown", file=sys.stderr)
        sys.exit(1)
    
    # Read JSON from stdin
    try:
        input_data = json.loads(sys.stdin.read())
        search_results = SearchResults(**input_data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to parse search results - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    # Check if we have results
    if search_results.num_results == 0:
        print("Error: No search results found", file=sys.stderr)
        sys.exit(1)
    
    # Build prompt
    prompt = TosaNikkiPrompt.build_prompt(search_results)
    
    # Output format options
    output_format = sys.argv[1] if len(sys.argv) > 1 else 'json'
    
    if output_format == 'json':
        # Output as JSON for API calls
        output = {
            "model": "claude-sonnet-4.5",
            "max_tokens": 4096,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    
    elif output_format == 'text':
        # Output as readable text
        print("=" * 80)
        print("SYSTEM MESSAGE")
        print("=" * 80)
        print(prompt["system"])
        print("\n" + "=" * 80)
        print("USER MESSAGE")
        print("=" * 80)
        print(prompt["user"])
        print("\n" + "=" * 80)
    
    elif output_format == 'markdown':
        # Output as markdown file
        print("# LLM Prompt\n")
        print("## System Role\n")
        print(prompt["system"])
        print("\n## User Query with Context\n")
        print(prompt["user"])
    
    else:
        print(f"Error: Unknown output format '{output_format}'", file=sys.stderr)
        print("Available formats: json, text, markdown", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
