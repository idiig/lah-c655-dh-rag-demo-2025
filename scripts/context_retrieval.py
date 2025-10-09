"""
Search Japanese classical text from Neo4j for RAG
Multi-language vector similarity search with context window
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


def get_context_window(text_ids, window_size=10):
    """Get surrounding context for retrieved text IDs"""
    with driver.session() as session:
        result = session.run("""
            UNWIND $text_ids AS target_id
            MATCH (w:Work)-[:CONTAINS_TEXT]->(t:Text {id: target_id})
            MATCH (w)-[:CONTAINS_TEXT]->(context:Text)
            WHERE context.id >= target_id - $window_size 
              AND context.id <= target_id + $window_size
            OPTIONAL MATCH (context)-[:HAS_TRANSLATION]->(tr_zh:Translation {language: 'zh', is_natural: true})
            OPTIONAL MATCH (context)-[:HAS_TRANSLATION]->(tr_en:Translation {language: 'en', is_natural: true})
            OPTIONAL MATCH (context)-[:HAS_TRANSLATION]->(tr_ja:Translation {language: 'ja', is_natural: true})
            
            RETURN DISTINCT
                   context.id as text_id,
                   context.text as text,
                   context.kana as kana,
                   context.is_poem as is_poem,
                   tr_zh.text as translation_zh,
                   tr_en.text as translation_en,
                   tr_ja.text as translation_ja,
                   target_id as matched_id,
                   CASE WHEN context.id = target_id THEN true ELSE false END as is_matched
            ORDER BY context.id
        """, text_ids=text_ids, window_size=window_size)
        
        return [dict(record) for record in result]


def comprehensive_search(query_text, top_k=5, min_score=0.2, context_window=10):
    """Comprehensive search with context window"""
    query_embedding = get_embedding(query_text)
    keywords = extract_keywords(query_text)
    
    with driver.session() as session:
        # First, find the most relevant passages
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
            
            RETURN t.id as text_id,
                   vec_score, 
                   keyword_score, 
                   word_score, 
                   final_score
            ORDER BY final_score DESC
            LIMIT $top_k
        """, query_embedding=query_embedding, query_text=query_text, 
             keywords=keywords, top_k=top_k, min_score=min_score)
        
        matched_results = [dict(record) for record in result]
    
    if not matched_results:
        return []
    
    # Get context window for all matched IDs
    matched_ids = [r['text_id'] for r in matched_results]
    context_texts = get_context_window(matched_ids, window_size=context_window)
    
    # Organize results in pipeline-compatible format
    final_results = []
    for matched in matched_results:
        matched_id = matched['text_id']
        
        # Get all context texts for this match
        context_window_texts = [ct for ct in context_texts if ct['matched_id'] == matched_id]
        
        # Find the matched passage
        matched_passage = next((ct for ct in context_window_texts if ct['is_matched']), None)
        
        if not matched_passage:
            continue
        
        # Build the result entry with context embedded in text
        context_before = [ct for ct in context_window_texts if ct['text_id'] < matched_id]
        context_after = [ct for ct in context_window_texts if ct['text_id'] > matched_id]
        
        # Construct enriched text with context markers
        full_text_parts = []
        
        # Add context before (last 3)
        if context_before:
            for ct in context_before[-3:]:
                full_text_parts.append(f"[Context ID:{ct['text_id']}] {ct['text']}")
        
        # Add matched passage with marker
        full_text_parts.append(f">>> [MATCHED ID:{matched_id}] {matched_passage['text']} <<<")
        
        # Add context after (first 3)
        if context_after:
            for ct in context_after[:3]:
                full_text_parts.append(f"[Context ID:{ct['text_id']}] {ct['text']}")
        
        full_text = "\n".join(full_text_parts)
        
        # Build enriched translations with context
        translations_zh = []
        translations_en = []
        translations_ja = []
        
        for ct in context_before[-3:]:
            if ct.get('translation_zh'):
                translations_zh.append(f"[ID:{ct['text_id']}] {ct['translation_zh']}")
            if ct.get('translation_en'):
                translations_en.append(f"[ID:{ct['text_id']}] {ct['translation_en']}")
            if ct.get('translation_ja'):
                translations_ja.append(f"[ID:{ct['text_id']}] {ct['translation_ja']}")
        
        translations_zh.append(f">>> [MATCHED ID:{matched_id}] {matched_passage.get('translation_zh', 'N/A')} <<<")
        translations_en.append(f">>> [MATCHED ID:{matched_id}] {matched_passage.get('translation_en', 'N/A')} <<<")
        translations_ja.append(f">>> [MATCHED ID:{matched_id}] {matched_passage.get('translation_ja', 'N/A')} <<<")
        
        for ct in context_after[:3]:
            if ct.get('translation_zh'):
                translations_zh.append(f"[ID:{ct['text_id']}] {ct['translation_zh']}")
            if ct.get('translation_en'):
                translations_en.append(f"[ID:{ct['text_id']}] {ct['translation_en']}")
            if ct.get('translation_ja'):
                translations_ja.append(f"[ID:{ct['text_id']}] {ct['translation_ja']}")
        
        result_entry = {
            "text_id": matched_id,
            "original_text": full_text,
            "kana": matched_passage['kana'],
            "is_poem": matched_passage.get('is_poem', False),
            "translations": {
                "zh": "\n".join(translations_zh) if translations_zh else None,
                "en": "\n".join(translations_en) if translations_en else None,
                "ja": "\n".join(translations_ja) if translations_ja else None
            },
            "scores": {
                "vector": matched['vec_score'],
                "keyword": matched['keyword_score'],
                "word": matched['word_score'],
                "final": matched['final_score']
            }
        }
        
        final_results.append(result_entry)
    
    return final_results


def format_context_json(search_results, query_text):
    """Format search results as JSON for pipeline processing"""
    import json
    output = {
        "query": query_text,
        "num_results": len(search_results),
        "results": search_results
    }
    
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_context_compact(search_results):
    """Format search results into compact RAG context (for LLM input)"""
    contexts = []
    
    for idx, result in enumerate(search_results, 1):
        ctx = f"[{idx}] Passage ID: {result['text_id']}\n"
        ctx += f"\nOriginal:\n{result['original_text']}\n"
        
        if result['translations'].get('zh'):
            ctx += f"\nChinese:\n{result['translations']['zh']}\n"
        if result['translations'].get('en'):
            ctx += f"\nEnglish:\n{result['translations']['en']}\n"
        
        ctx += f"\nRelevance: {result['scores']['final']:.3f}"
        contexts.append(ctx)
    
    return "\n\n" + "="*80 + "\n\n".join(contexts)


if __name__ == "__main__":
    # Check if reading from stdin
    if not sys.stdin.isatty():
        query_text = sys.stdin.read().strip()
        top_k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
        output_format = sys.argv[2] if len(sys.argv) > 2 else 'json'
        context_window = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 10
    else:
        if len(sys.argv) < 2:
            print("Usage: python search_neo4j.py <query> [top_k] [format] [window]", file=sys.stderr)
            print("   or: echo <query> | python search_neo4j.py [top_k] [format] [window]", file=sys.stderr)
            print("\nFormat options: json, compact (default: json)", file=sys.stderr)
            print("Window: context window size (default: 10)", file=sys.stderr)
            sys.exit(1)
        
        query_text = sys.argv[1]
        top_k = 5
        output_format = 'json'
        context_window = 10
        
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg.isdigit() and i == 2:
                top_k = int(arg)
            elif arg in ['json', 'compact'] and i == 3:
                output_format = arg
            elif arg.isdigit() and i == 4:
                context_window = int(arg)
    
    try:
        results = comprehensive_search(query_text, top_k, min_score=0.2, context_window=context_window)
        
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        driver.close()
