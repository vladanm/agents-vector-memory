#!/usr/bin/env python3
"""Add comprehensive logging to search function."""

import re

# Read the file
with open("src/session_memory_store.py", "r") as f:
    content = f.read()

# Find the fine granularity search section
pattern = r'(        elif granularity == "fine":\s+# Return individual chunks using vector search\s+)(try:)'

replacement = r'''\1logger.info("=" * 80)
            logger.info("FINE GRANULARITY SEARCH STARTING")
            logger.info(f"  memory_type: {memory_type}")
            logger.info(f"  query: {query[:100]}...")
            logger.info(f"  agent_id: {agent_id}")
            logger.info(f"  session_id: {session_id}")
            logger.info(f"  session_iter: {session_iter}")
            logger.info(f"  task_code: {task_code}")
            logger.info(f"  limit: {limit}")
            logger.info(f"  similarity_threshold: {similarity_threshold}")
            logger.info("=" * 80)
            \2'''

content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Add logging after k_value calculation
pattern2 = r'(                k_value = limit \* 10  # Increased for post-filtering\s+)'
replacement2 = r'''\1
                logger.info(f"Using k_value={k_value} for vector search (limit * 10)")
'''

content = re.sub(pattern2, replacement2, content)

# Add logging after query execution
pattern3 = r'(                rows = conn\.execute\(sql, params\)\.fetchall\(\)\s+                conn\.close\(\)\s+)'
replacement3 = r'''\1
                logger.info(f"Vector search returned {len(rows)} raw chunks before filtering")
                if rows:
                    logger.info(f"  First chunk distance: {rows[0][7]}")
                    logger.info(f"  First chunk memory_type: {rows[0][8]}")
                    logger.info(f"  First chunk session_id: {rows[0][10]}")
'''

content = re.sub(pattern3, replacement3, content)

# Add tracking for filtered results
pattern4 = r'(                # Filter results in Python based on metadata\s+                results = \[\]\s+)'
replacement4 = r'''\1                filtered_counts = {
                    "memory_type": 0,
                    "agent_id": 0,
                    "session_id": 0,
                    "session_iter": 0,
                    "task_code": 0,
                    "similarity": 0
                }
'''

content = re.sub(pattern4, replacement4, content)

# Add logging for each filter
pattern5 = r'                    if row_memory_type != memory_type:\s+                        continue'
replacement5 = '''                    if row_memory_type != memory_type:
                        filtered_counts["memory_type"] += 1
                        continue'''

content = content.replace(
    '                    if row_memory_type != memory_type:\n                        continue',
    replacement5
)

content = content.replace(
    '                    if agent_id and row_agent_id != agent_id:\n                        continue',
    '''                    if agent_id and row_agent_id != agent_id:
                        filtered_counts["agent_id"] += 1
                        continue'''
)

content = content.replace(
    '                    if session_id and row_session_id != session_id:\n                        continue',
    '''                    if session_id and row_session_id != session_id:
                        filtered_counts["session_id"] += 1
                        continue'''
)

content = content.replace(
    '                    if session_iter is not None and row_session_iter != session_iter:\n                        continue',
    '''                    if session_iter is not None and row_session_iter != session_iter:
                        filtered_counts["session_iter"] += 1
                        continue'''
)

content = content.replace(
    '                    if task_code and row_task_code != task_code:\n                        continue',
    '''                    if task_code and row_task_code != task_code:
                        filtered_counts["task_code"] += 1
                        continue'''
)

# Add else clause for similarity filtering
pattern6 = r'(\s+if similarity >= similarity_threshold:\s+results\.append\(\{[^}]+\}\))\s+(if len\(results\) >= limit:)'
replacement6 = r'''\1
                    else:
                        filtered_counts["similarity"] += 1

                \2'''

content = re.sub(pattern6, replacement6, content, flags=re.DOTALL)

# Add final logging before return
pattern7 = r'(                    if len\(results\) >= limit:\s+                        break\s+)(                return \{)'
replacement7 = r'''\1
                logger.info("=" * 80)
                logger.info("FILTERING RESULTS:")
                logger.info(f"  Filtered by memory_type: {filtered_counts['memory_type']}")
                logger.info(f"  Filtered by agent_id: {filtered_counts['agent_id']}")
                logger.info(f"  Filtered by session_id: {filtered_counts['session_id']}")
                logger.info(f"  Filtered by session_iter: {filtered_counts['session_iter']}")
                logger.info(f"  Filtered by task_code: {filtered_counts['task_code']}")
                logger.info(f"  Filtered by similarity: {filtered_counts['similarity']}")
                logger.info(f"  FINAL RESULTS: {len(results)} chunks")
                if results:
                    logger.info(f"  Best similarity: {results[0]['similarity']}")
                logger.info("=" * 80)

\2'''

content = re.sub(pattern7, replacement7, content)

# Write back
with open("src/session_memory_store.py", "w") as f:
    f.write(content)

print("Logging added successfully!")
