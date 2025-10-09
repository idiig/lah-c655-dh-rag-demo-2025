"""
Search Japanese classical text from Neo4j for RAG
Multi-language vector similarity search
Supports both command line arguments and stdin
"""

from neo4j import GraphDatabase
import os
import sys
from sentence_transformers import SentenceTransformer

# Database configuration from environment variables
NEO4J_URI = os.getenv('NEO4J_URI', '').strip()
NEO4J_USER = os.getenv('NEO4J_USERNAME', '').strip()
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '').strip()

# Validate configuration
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise ValueError("Missing required environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")

# Load embedding model (supports Chinese, Japanese, English)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_embedding(text):
    """Generate embedding vector for text"""
    return model.encode(text).tolist()


def extract_keywords(query_text):
    """Extract potential keywords from query for better matching"""
    words = query_text.lower().split()
    stop_words = {'what', 'did', 'do', 'for', 'us', 'the', 'a', 'an', 'in', 'on', 'at', 
                  'is', 'are', 'was', 'were', 'to', 'of', 'and', 'or', 'but', 'with'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def comprehensive_search(query_text, top_k=5, min_score=0.2):
    """Comprehensive search combining all methods with multi-language vector similarity"""
    query_embedding = get_embedding(query_text)
    keywords = extract_keywords(query_text)
    
    with driver.session() as session:
        result = session.run("""
            WITH $query_embedding AS queryVector, $query_text AS queryText, $keywords AS keywords
            MATCH (w:Work)-[:CONTAINS_TEXT]->(t:Text)
            OPTIONAL MATCH (t)-[:HAS_TRANSLATION]->(tr_zh:Translation {language: 'zh', is_natural: true})
            OPTIONAL MATCH (t)-[:HAS_TRANSLATION]->(tr_en:Translation {language: 'en', is_natural: true})
            OPTIONAL MATCH (t)-[:HAS_TRANSLATION]->(tr_ja:Translation {language: 'ja', is_natural: true})
            
            WITH w, t, tr_zh, tr_en, tr_ja, queryVector, queryText, keywords,
                 vector.similarity.cosine(t.embedding, queryVector) AS vec_score_text,
                 CASE WHEN tr_zh IS NOT NULL THEN vector.similarity.cosine(tr_zh.embedding, queryVector) ELSE 0.0 END AS vec_score_zh,
                 CASE WHEN tr_en IS NOT NULL THEN vector.similarity.cosine(tr_en.embedding, queryVector) ELSE 0.0 END AS vec_score_en,
                 CASE WHEN tr_ja IS NOT NULL THEN vector.similarity.cosine(tr_ja.embedding, queryVector) ELSE 0.0 END AS vec_score_ja
            
            WITH w, t, tr_zh, tr_en, tr_ja, queryText, keywords,
                 [vec_score_text, vec_score_zh, vec_score_en, vec_score_ja] AS vec_scores
            
            WITH w, t, tr_zh, tr_en, tr_ja, queryText, keywords, vec_scores,
                 REDUCE(max = 0.0, score IN vec_scores | CASE WHEN score > max THEN score ELSE max END) AS vec_score
            
            OPTIONAL MATCH (t)-[:CONTAINS_PHRASE]->(p:Phrase)-[:CONTAINS_WORD]->(word:Word)
            
            WITH w, t, tr_zh, tr_en, tr_ja, queryText, keywords, vec_score,
                 collect(DISTINCT {
                     phrase: p.phrase, 
                     word: word.word, 
                     gloss: word.gloss, 
                     gloss_zh: word.gloss_zh,
                     matched: (
                         word.word CONTAINS queryText OR 
                         word.gloss CONTAINS queryText OR 
                         word.gloss_zh CONTAINS queryText OR
                         ANY(kw IN keywords WHERE 
                             toLower(word.word) CONTAINS kw OR 
                             toLower(word.gloss) CONTAINS kw OR 
                             toLower(word.gloss_zh) CONTAINS kw)
                     )
                 }) AS word_info
            
            WITH w, t, tr_zh, tr_en, tr_ja, vec_score, word_info,
                 CASE 
                     WHEN t.text CONTAINS queryText OR t.kana CONTAINS queryText THEN 1.0
                     WHEN tr_zh.text CONTAINS queryText OR tr_en.text CONTAINS queryText OR tr_ja.text CONTAINS queryText THEN 0.8
                     WHEN ANY(kw IN keywords WHERE 
                         toLower(t.text) CONTAINS kw OR 
                         toLower(t.kana) CONTAINS kw OR 
                         toLower(COALESCE(tr_zh.text, '')) CONTAINS kw OR 
                         toLower(COALESCE(tr_en.text, '')) CONTAINS kw OR 
                         toLower(COALESCE(tr_ja.text, '')) CONTAINS kw) THEN 0.6
                     ELSE 0.0
                 END AS keyword_score,
                 REDUCE(s = 0.0, wi IN word_info | s + CASE WHEN wi.matched THEN 0.5 ELSE 0.0 END) AS word_score,
                 [wi IN word_info WHERE wi.word IS NOT NULL] AS filtered_words
            
            WITH w, t, tr_zh, tr_en, tr_ja, vec_score, keyword_score, word_score, filtered_words,
                 (vec_score * 0.5 + keyword_score * 0.3 + word_score * 0.2) AS final_score
            
            WHERE final_score > $min_score
            
            RETURN w.title as work_title, 
                   w.author as work_author,
                   t.id as text_id, 
                   t.text as text, 
                   t.kana as kana, 
                   t.is_poem as is_poem, 
                   tr_zh.text as translation_zh,
                   tr_en.text as translation_en,
                   tr_ja.text as translation_ja,
                   filtered_words, 
                   vec_score, 
                   keyword_score, 
                   word_score, 
                   final_score
            ORDER BY final_score DESC
            LIMIT $top_k
        """, query_embedding=query_embedding, query_text=query_text, 
             keywords=keywords, top_k=top_k, min_score=min_score)
        
        return [dict(record) for record in result]


def format_context_json(search_results, query_text):
    """Format search results as JSON for pipeline processing"""
    import json
    output = {
        "query": query_text,
        "num_results": len(search_results),
        "results": []
    }
    
    for record in search_results:
        result = {
            "work_title": record['work_title'],
            "work_author": record['work_author'],
            "text_id": record['text_id'],
            "original_text": record['text'],
            "kana": record['kana'],
            "is_poem": record['is_poem'],
            "translations": {
                "zh": record.get('translation_zh'),
                "en": record.get('translation_en'),
                "ja": record.get('translation_ja')
            },
            "words": [w for w in record.get('filtered_words', []) if w.get('word')],
            "scores": {
                "vector": record['vec_score'],
                "keyword": record['keyword_score'],
                "word": record['word_score'],
                "final": record['final_score']
            }
        }
        output["results"].append(result)
    
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_context_compact(search_results):
    """Format search results into compact RAG context (for LLM input)"""
    contexts = []
    
    for idx, record in enumerate(search_results, 1):
        ctx = f"[{idx}] {record['work_title']}\n"
        ctx += f"Original: {record['text']}\n"
        
        if record.get('translation_zh'):
            ctx += f"Chinese: {record['translation_zh']}\n"
        if record.get('translation_en'):
            ctx += f"English: {record['translation_en']}\n"
        if record.get('translation_ja'):
            ctx += f"Japanese: {record['translation_ja']}\n"
        
        ctx += f"Relevance: {record['final_score']:.3f}"
        contexts.append(ctx)
    
    return "\n\n".join(contexts)


if __name__ == "__main__":
    # Check if reading from stdin
    if not sys.stdin.isatty():
        # Reading from pipe
        query_text = sys.stdin.read().strip()
        top_k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
        output_format = sys.argv[2] if len(sys.argv) > 2 else 'json'
    else:
        # Reading from command line arguments
        if len(sys.argv) < 2:
            print("Usage: python search_neo4j.py <query> [top_k] [format]", file=sys.stderr)
            print("   or: echo <query> | python search_neo4j.py [top_k] [format]", file=sys.stderr)
            print("\nFormat options: json, compact, text (default: json)", file=sys.stderr)
            sys.exit(1)
        
        query_text = sys.argv[1]
        top_k = 5
        output_format = 'text'
        
        for arg in sys.argv[2:]:
            if arg.isdigit():
                top_k = int(arg)
            elif arg in ['json', 'compact', 'text']:
                output_format = arg
    
    try:
        results = comprehensive_search(query_text, top_k, min_score=0.2)
        
        if not results:
            if output_format == 'json':
                import json
                print(json.dumps({"query": query_text, "num_results": 0, "results": []}, ensure_ascii=False))
            else:
                print("No results found.", file=sys.stderr)
        else:
            if output_format == 'json':
                print(format_context_json(results, query_text))
            elif output_format == 'compact':
                print(format_context_compact(results))
            else:
                # text format (original detailed format)
                print(f"=== Search Results for: '{query_text}' ===\n")
                print(format_context_compact(results))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        driver.close()
