"""
Import Japanese classical text JSON into Neo4j
"""

from neo4j import GraphDatabase
import json
import os
from sentence_transformers import SentenceTransformer

# Database configuration from environment variables (strip whitespace)
NEO4J_URI = os.getenv('NEO4J_URI', '').strip()
NEO4J_USER = os.getenv('NEO4J_USERNAME', '').strip()
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '').strip()

# Validate configuration
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise ValueError("Missing required environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")

print(f"Connecting to: {NEO4J_URI}")
print(f"Username: {NEO4J_USER}")

# Load embedding model (supports Chinese, Japanese, English)
print("\nLoading embedding model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Model loaded")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_indexes():
    """Create indexes if not exists"""
    with driver.session() as session:
        indexes = [
            ("CREATE CONSTRAINT text_id FOR (t:Text) REQUIRE t.id IS UNIQUE", "text_id"),
            ("CREATE CONSTRAINT work_title FOR (w:Work) REQUIRE w.title IS UNIQUE", "work_title"),
            ("CREATE INDEX text_text FOR (t:Text) ON (t.text)", "text_text"),
            ("CREATE INDEX phrase_phrase FOR (p:Phrase) ON (p.phrase)", "phrase_phrase"),
            ("CREATE INDEX word_word FOR (w:Word) ON (w.word)", "word_word"),
        ]
        
        for query, name in indexes:
            try:
                session.run(query)
                print(f"✓ Created index: {name}")
            except Exception:
                print(f"⚠ Index {name} already exists")
        
        # Create vector index
        try:
            session.run("""
                CREATE VECTOR INDEX text_embeddings
                FOR (t:Text) ON (t.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            print("✓ Created vector index: text_embeddings")
        except Exception:
            print("⚠ Vector index text_embeddings already exists")


def get_embedding(text):
    """Generate embedding vector for text"""
    return model.encode(text).tolist()


def get_text_content(paragraph):
    """Get text content from paragraph (handles both 'text' and 'poem' fields)"""
    return paragraph.get('text') or paragraph.get('poem', '')


def import_work_node(session, work_data):
    """Create work (literary work) node"""
    query = """
    MERGE (w:Work {title: $title})
    ON CREATE SET
        w.title_kana = $title_kana,
        w.title_roman = $title_roman,
        w.author = $author,
        w.author_kana = $author_kana,
        w.author_roman = $author_roman
    """
    
    session.run(query,
        title=work_data.get('title', ''),
        title_kana=work_data.get('title_kana', ''),
        title_roman=work_data.get('title_roman', ''),
        author=work_data.get('author', ''),
        author_kana=work_data.get('author_kana', ''),
        author_roman=work_data.get('author_roman', '')
    )
    
    return work_data.get('title', '')


def import_text_node(session, work_title, paragraph):
    """Create main text node with embedding"""
    text_content = get_text_content(paragraph)
    
    if not text_content:
        print(f"  ⚠ Skipping paragraph ID {paragraph.get('id')} - no text/poem content")
        return False
    
    embedding = get_embedding(text_content)
    is_poem = 'poem' in paragraph
    
    query = """
    MATCH (w:Work {title: $work_title})
    CREATE (t:Text {
        id: $id,
        text: $text,
        kana: $kana,
        koutei_yamagen: $koutei_yamagen,
        is_poem: $is_poem,
        embedding: $embedding
    })
    CREATE (w)-[:CONTAINS_TEXT]->(t)
    """
    
    try:
        session.run(query,
            work_title=work_title,
            id=paragraph['id'],
            text=text_content,
            kana=paragraph.get('kana', ''),
            koutei_yamagen=paragraph.get('koutei-yamagen', ''),
            is_poem=is_poem,
            embedding=embedding
        )
        return True
    except Exception as e:
        if 'ConstraintValidationFailed' in str(e):
            print(f"  ⚠ Text ID {paragraph['id']} already exists, skipping")
            return False
        else:
            raise


def import_translations(session, paragraph):
    """Create translation nodes with embeddings"""
    translations = [
        ('ja', paragraph.get('translation-ja'), False),
        ('en', paragraph.get('translation-en'), False),
        ('zh', paragraph.get('translation-zh'), False),
        ('ja', paragraph.get('translation-ja-natural'), True),
        ('en', paragraph.get('translation-en-natural'), True),
        ('zh', paragraph.get('translation-zh-natural'), True)
    ]
    
    query = """
    MATCH (t:Text {id: $text_id})
    CREATE (tr:Translation {
        language: $language,
        text: $translation,
        is_natural: $is_natural,
        embedding: $embedding
    })
    CREATE (t)-[:HAS_TRANSLATION]->(tr)
    """
    
    for lang, text, is_natural in translations:
        if text:
            embedding = get_embedding(text)
            session.run(query, text_id=paragraph['id'], language=lang,
                       translation=text, is_natural=is_natural,
                       embedding=embedding)


def import_phrases_and_words(session, paragraph):
    """Create phrase and word nodes"""
    phrase_query = """
    MATCH (t:Text {id: $text_id})
    CREATE (p:Phrase {
        phrase: $phrase,
        gloss: $gloss,
        gloss_morph: $gloss_morph,
        gloss_zh: $gloss_zh,
        position: $position
    })
    CREATE (t)-[:CONTAINS_PHRASE]->(p)
    """
    
    word_query = """
    MATCH (t:Text {id: $text_id})
    MATCH (p:Phrase {phrase: $phrase_text})
    WHERE (t)-[:CONTAINS_PHRASE]->(p)
    MERGE (w:Word {word: $word})
    ON CREATE SET
        w.gloss = $gloss,
        w.gloss_morph = $gloss_morph,
        w.gloss_zh = $gloss_zh
    CREATE (p)-[:CONTAINS_WORD {position: $position}]->(w)
    """
    
    for phrase_idx, phrase_data in enumerate(paragraph.get('phrase-gloss', [])):
        session.run(phrase_query,
            text_id=paragraph['id'],
            phrase=phrase_data['phrase'],
            gloss=phrase_data.get('gloss', ''),
            gloss_morph=phrase_data.get('gloss-morph', ''),
            gloss_zh=phrase_data.get('gloss-zh', ''),
            position=phrase_idx
        )
        
        for word_idx, word_data in enumerate(phrase_data.get('words', [])):
            session.run(word_query,
                text_id=paragraph['id'],
                phrase_text=phrase_data['phrase'],
                word=word_data['word'],
                gloss=word_data.get('gloss', ''),
                gloss_morph=word_data.get('gloss-morph', ''),
                gloss_zh=word_data.get('gloss-zh', ''),
                position=word_idx
            )


def import_paragraph(work_title, paragraph):
    """Import a single paragraph"""
    with driver.session() as session:
        success = import_text_node(session, work_title, paragraph)
        if not success:
            return
        
        import_translations(session, paragraph)
        import_phrases_and_words(session, paragraph)
    
    print(f"  ✓ Imported paragraph ID: {paragraph['id']}")


def import_work(work_data):
    """Import a work and all its paragraphs"""
    with driver.session() as session:
        work_title = import_work_node(session, work_data)
    
    print(f"✓ Importing work: {work_title}")
    
    for paragraph in work_data.get('paragraph', []):
        import_paragraph(work_title, paragraph)


def import_from_file(json_file):
    """Import from JSON file"""
    print(f"Reading file: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"Number of works: {len(data)}")
        for work_data in data:
            import_work(work_data)
    elif isinstance(data, dict):
        import_work(data)
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python import_to_neo4j.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    print("=== Starting import ===\n")
    
    create_indexes()
    
    try:
        import_from_file(json_file)
    except Exception as e:
        print(f"\n❌ Error during import: {e}")
        import traceback
        traceback.print_exc()
        driver.close()
        sys.exit(1)
    
    driver.close()
    print("\n=== Import complete ===")
