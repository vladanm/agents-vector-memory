#!/usr/bin/env python3
"""
Apply comprehensive logging instrumentation to session_memory_store.py
"""

import re
from pathlib import Path

def apply_instrumentation():
    """Apply detailed logging to the session_memory_store.py file"""

    file_path = Path(__file__).parent / "src" / "session_memory_store.py"

    print(f"Reading {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()

    # Backup original
    backup_path = file_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✓ Backup created: {backup_path}")

    # 1. Update embedding_model property with timing and detailed logging
    old_embedding_model = r'''    @property
    def embedding_model\(self\) -> Any:
        """Lazy-load embedding model for semantic search\."""
        if self\._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self\._embedding_model = SentenceTransformer\(
                    'all-MiniLM-L6-v2',
                    device='cpu'
                \)
                logger\.info\("Embedding model loaded successfully"\)
            except Exception as e:
                logger\.warning\(f"Failed to load embedding model: \{e\}"\)
                self\._embedding_model = False
        return self\._embedding_model if self\._embedding_model is not False else None'''

    new_embedding_model = '''    @property
    def embedding_model(self) -> Any:
        """Lazy-load embedding model for semantic search."""
        import time
        if self._embedding_model is None:
            logger.info("[embedding_model] LAZY LOADING - First access detected")
            start_time = time.time()
            try:
                logger.info("[embedding_model] Importing SentenceTransformer...")
                from sentence_transformers import SentenceTransformer
                logger.info("[embedding_model] Creating model instance: all-MiniLM-L6-v2")
                self._embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device='cpu'
                )
                elapsed = time.time() - start_time
                logger.info(f"[embedding_model] ✓ Model loaded successfully in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[embedding_model] ✗ Failed to load model after {elapsed:.2f}s: {e}", exc_info=True)
                self._embedding_model = False
        else:
            if self._embedding_model is False:
                logger.debug("[embedding_model] Model previously failed to load, returning None")
            else:
                logger.debug("[embedding_model] Returning cached model instance")
        return self._embedding_model if self._embedding_model is not False else None'''

    content = re.sub(old_embedding_model, new_embedding_model, content, flags=re.MULTILINE)
    print("✓ Updated embedding_model property with timing logs")

    # 2. Add comprehensive logging to _store_memory_impl
    # Find the method start
    method_start_pattern = r'    def _store_memory_impl\('

    # Find where the try block starts (after docstring)
    try_block_pattern = r'(        """\n        )\n(        try:)'

    # Add overall timing at try block start
    try_block_replacement = r'\1\n        import time\n        logger.info("=" * 80)\n        logger.info(f"[_store_memory_impl] ENTRY")\n        logger.info(f"[_store_memory_impl] memory_type={memory_type}, agent_id={agent_id}, session_id={session_id}")\n        logger.info(f"[_store_memory_impl] session_iter={session_iter}, task_code={task_code}")\n        logger.info(f"[_store_memory_impl] content_length={len(content)}, auto_chunk={auto_chunk}")\n        logger.info(f"[_store_memory_impl] title={title}, description={description}")\n        overall_start = time.time()\n        \n\2'

    content = re.sub(try_block_pattern, try_block_replacement, content, flags=re.MULTILINE)
    print("✓ Added entry logging and timing to _store_memory_impl")

    # 3. Add logging around expensive operations
    # Add chunking logging
    chunking_start_pattern = r'(            if auto_chunk:)\n(                chunk_metadata = )'
    chunking_start_replacement = r'\1\n                logger.info("=" * 60)\n                logger.info(f"[_store_memory_impl] AUTO-CHUNKING ENABLED")\n                logger.info(f"[_store_memory_impl] Content will be chunked before DB insertion")\n                logger.info("=" * 60)\n                chunk_overall_start = time.time()\n\2'
    content = re.sub(chunking_start_pattern, chunking_start_replacement, content, flags=re.MULTILINE)

    # Add chunk_document logging
    chunk_doc_pattern = r'(                # Chunk document \(use placeholder memory_id=0, will update after insert\))\n(                chunks = self\.chunker\.chunk_document)'
    chunk_doc_replacement = r'\1\n                logger.info(f"[_store_memory_impl] Calling chunker.chunk_document()...")\n                chunk_start = time.time()\n\2'
    content = re.sub(chunk_doc_pattern, chunk_doc_replacement, content, flags=re.MULTILINE)

    # Add logging after chunking
    after_chunk_pattern = r'(                chunks = self\.chunker\.chunk_document\(content, 0, chunk_metadata\))\n'
    after_chunk_replacement = r'\1\n                chunk_elapsed = time.time() - chunk_start\n                logger.info(f"[_store_memory_impl] ✓ Chunking complete: {len(chunks)} chunks in {chunk_elapsed:.2f}s")\n'
    content = re.sub(after_chunk_pattern, after_chunk_replacement, content, flags=re.MULTILINE)

    # Add embedding logging
    embed_start_pattern = r'(                if chunks:)\n(                    chunk_texts = )'
    embed_start_replacement = r'\1\n                    logger.info(f"[_store_memory_impl] Generating embeddings for {len(chunks)} chunks")\n                    embed_start = time.time()\n\2'
    content = re.sub(embed_start_pattern, embed_start_replacement, content, flags=re.MULTILINE)

    # Add model access logging
    model_access_pattern = r'(                        # Use the property \(not _embedding_model\) to trigger lazy loading)\n(                        model = self\.embedding_model)'
    model_access_replacement = r'\1\n                        logger.info(f"[_store_memory_impl] Accessing embedding_model property...")\n                        model_access_start = time.time()\n\2\n                        model_access_elapsed = time.time() - model_access_start\n                        logger.info(f"[_store_memory_impl] Model access took {model_access_elapsed:.2f}s")'
    content = re.sub(model_access_pattern, model_access_replacement, content, flags=re.MULTILINE)

    # Add encode logging
    encode_pattern = r'(                        if model is not None:)\n(                            embeddings = model\.encode)'
    encode_replacement = r'\1\n                            logger.info(f"[_store_memory_impl] Calling model.encode() for {len(chunk_texts)} chunks...")\n                            encode_start = time.time()\n\2'
    content = re.sub(encode_pattern, encode_replacement, content, flags=re.MULTILINE)

    # Add post-encode logging
    post_encode_pattern = r'(                            embeddings = model\.encode\(chunk_texts, batch_size=32, show_progress_bar=False\))\n(                            # Convert embeddings)'
    post_encode_replacement = r'\1\n                            encode_elapsed = time.time() - encode_start\n                            logger.info(f"[_store_memory_impl] ✓ Encoding complete: {len(embeddings)} embeddings in {encode_elapsed:.2f}s")\n\2'
    content = re.sub(post_encode_pattern, post_encode_replacement, content, flags=re.MULTILINE)

    # Update final embedding success message
    old_embed_success = r'                            logger\.info\(f"Generated \{len\(embeddings\)\} chunk embeddings"\)'
    new_embed_success = r'                            total_embed_time = time.time() - embed_start\n                            logger.info(f"[_store_memory_impl] ✓ Total embedding process: {total_embed_time:.2f}s")\n                            logger.info("=" * 60)\n                            logger.info(f"[_store_memory_impl] CHUNKING COMPLETE - Total time: {time.time() - chunk_overall_start:.2f}s")\n                            logger.info("=" * 60)'
    content = re.sub(old_embed_success, new_embed_success, content)

    # Add DB connection logging
    db_conn_pattern = r'(            # NOW open connection - all expensive operations are done)\n(            conn = self\._get_connection\(\))'
    db_conn_replacement = r'\1\n            logger.info("=" * 60)\n            logger.info(f"[_store_memory_impl] Opening database connection")\n            logger.info(f"[_store_memory_impl] All expensive operations complete")\n            logger.info("=" * 60)\n            db_start = time.time()\n\2\n            db_connect_elapsed = time.time() - db_start\n            logger.info(f"[_store_memory_impl] ✓ Database connection opened in {db_connect_elapsed:.3f}s")'
    content = re.sub(db_conn_pattern, db_conn_replacement, content, flags=re.MULTILINE)

    # Add success logging at return
    success_return_pattern = r'(                conn\.commit\(\))\n(                conn\.close\(\))\n\n(                return \{)'
    success_return_replacement = r'\1\n                logger.info(f"[_store_memory_impl] ✓ Transaction committed")\n\2\n                logger.info(f"[_store_memory_impl] ✓ Connection closed")\n                total_elapsed = time.time() - overall_start\n                logger.info("=" * 80)\n                logger.info(f"[_store_memory_impl] SUCCESS - memory_id={memory_id}, chunks={chunks_created}, time={total_elapsed:.2f}s")\n                logger.info("=" * 80)\n\n\3'
    content = re.sub(success_return_pattern, success_return_replacement, content, flags=re.MULTILINE)

    print("✓ Added comprehensive instrumentation to _store_memory_impl")

    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Updated {file_path}")
    print(f"\nInstrumentation complete!")
    print(f"Logs will be written to: logs/mcp_server.log")
    print(f"Backup saved to: {backup_path}")

if __name__ == "__main__":
    apply_instrumentation()
