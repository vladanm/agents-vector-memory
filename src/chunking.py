"""
Document Chunking Module
========================

Hierarchical text chunking using LangChain's MarkdownTextSplitter.
"""

import re
import hashlib
from typing import Any
from dataclasses import dataclass
from .memory_types import ChunkEntry, ContentFormat, get_memory_type_config

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
        ExperimentalMarkdownSyntaxTextSplitter
    )
    LANGCHAIN_AVAILABLE = True
    # DEBUG: Log module load
    try:
        from datetime import datetime
        with open("/tmp/vector_memory_debug.log", "a") as f:
            f.write(f"[CHUNK LOAD] chunking.py loaded at {datetime.now()}, LangChain AVAILABLE (with Experimental)\n")
    except:
        pass
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # DEBUG: Log module load
    try:
        from datetime import datetime
        with open("/tmp/vector_memory_debug.log", "a") as f:
            f.write(f"[CHUNK LOAD] chunking.py loaded at {datetime.now()}, LangChain NOT AVAILABLE\n")
    except:
        pass


@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 450           # Was 800 - reduced to fit within 512 token embedding limit
    chunk_overlap: int = 50         # Was 80 - 11% overlap maintained
    preserve_structure: bool = True
    content_format: str = ContentFormat.MARKDOWN.value
    memory_type: str = "working_memory"
    # Note: min_chunk_size removed per 2-stage best practices
    # Natural boundaries dictate sizes, no artificial minimums


class DocumentChunker:
    """Hierarchical document chunker with LangChain support"""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

        # Initialize tokenizer if available
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

        self._active_chunks = []

    def __del__(self):
        try:
            self.cleanup_chunks()
        except:
            pass

    def cleanup_chunks(self):
        """Cleanup chunk references"""
        if hasattr(self, '_active_chunks'):
            self._active_chunks.clear()

    def chunk_document(self, content: str, parent_id: int, metadata: dict[str, Any] = None) -> list[ChunkEntry]:
        """
        Chunk a document into coherent pieces using LangChain splitters.

        Args:
            content: Document content to chunk
            parent_id: Parent memory entry ID
            metadata: Optional metadata with memory_type hint, title, enable_enrichment

        Returns:
            List of ChunkEntry objects
        """
        if metadata is None:
            metadata = {}

        # Get config based on memory type
        memory_type = metadata.get("memory_type", self.config.memory_type)
        type_config = get_memory_type_config(memory_type)

        chunk_size = type_config.get("chunk_size", self.config.chunk_size)
        chunk_overlap = type_config.get("chunk_overlap", self.config.chunk_overlap)
        preserve_structure = type_config.get("preserve_structure", self.config.preserve_structure)

        # Detect content format
        content_format = self._detect_format(content)

        try:
            with open("/tmp/vector_memory_debug.log", "a") as f:
                f.write(f"[CHUNK DEBUG] LANGCHAIN_AVAILABLE={LANGCHAIN_AVAILABLE}, content_format={content_format}, preserve_structure={preserve_structure}\n")

            if LANGCHAIN_AVAILABLE and content_format == ContentFormat.MARKDOWN.value and preserve_structure:
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write("[CHUNK DEBUG] Using LangChain chunking\n")
                chunks = self._chunk_with_langchain(content, parent_id, chunk_size, chunk_overlap)
            elif content_format == ContentFormat.MARKDOWN.value and preserve_structure:
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write("[CHUNK DEBUG] Using custom markdown hierarchical chunking\n")
                chunks = self._chunk_markdown_hierarchical(content, parent_id, chunk_size, chunk_overlap)
            else:
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write("[CHUNK DEBUG] Using recursive chunking\n")
                chunks = self._chunk_recursive(content, parent_id, chunk_size, chunk_overlap)

            # Apply contextual enrichment if enabled (default for knowledge_base and reports)
            enable_enrichment = metadata.get("enable_enrichment", memory_type in ["knowledge_base", "reports"])
            if enable_enrichment and chunks:
                document_title = metadata.get("title", "Untitled Document")
                self._apply_contextual_enrichment(chunks, document_title, memory_type)

            self._active_chunks.extend(chunks)
            return chunks
        except Exception as e:
            self.cleanup_chunks()
            raise e

    def _detect_format(self, content: str) -> str:
        """Detect content format from patterns"""
        sample = content[:1000].lower()

        if sample.count('#') > 2 and ('##' in sample or '###' in sample):
            return ContentFormat.MARKDOWN.value
        elif sample.startswith('<?xml') or '<html' in sample:
            return ContentFormat.HTML.value
        elif sample.count('{') > 2 and sample.count('}') > 2:
            return ContentFormat.JSON.value
        elif sample.count(':') > 3 and '\n' in sample:
            return ContentFormat.YAML.value
        else:
            return ContentFormat.TEXT.value

    def _chunk_with_langchain(self, content: str, parent_id: int, chunk_size: int, chunk_overlap: int) -> list[ChunkEntry]:
        """
        Chunk markdown using 2-stage approach:
        Stage 1: ExperimentalMarkdownSyntaxTextSplitter (code-aware, preserves indentation)
        Stage 2: RecursiveCharacterTextSplitter (enforces size limits with code-aware separators)

        No artificial minimum size enforcement - natural boundaries dictate chunk sizes.
        """
        with open("/tmp/vector_memory_debug.log", "a") as f:
            f.write(f"[CHUNK] Starting 2-stage code-aware chunking (chunk_size={chunk_size})\n")

        # Define headers to split on (all markdown header levels)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        # Stage 1: Split by markdown headers using experimental splitter (preserves code)
        try:
            markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False  # Keep headers in content for context
            )
            md_header_splits = markdown_splitter.split_text(content)

            with open("/tmp/vector_memory_debug.log", "a") as f:
                f.write(f"[CHUNK] Stage 1: Created {len(md_header_splits)} header-based splits\n")
        except Exception as e:
            # Fallback to standard splitter if experimental fails
            with open("/tmp/vector_memory_debug.log", "a") as f:
                f.write(f"[CHUNK] ExperimentalMarkdownSyntaxTextSplitter failed: {e}, falling back\n")

            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            md_header_splits = markdown_splitter.split_text(content)

        # Stage 2: Apply size constraints with code-aware separators
        # Priority: code blocks > headers > paragraphs > sentences > words
        code_aware_separators = [
            "\n```",    # Code block boundaries (HIGHEST PRIORITY)
            "\n## ",    # Headers
            "\n### ",   # Sub-headers
            "\n\n",     # Paragraphs
            "\n",       # Lines
            ". ",       # Sentences
            " ",        # Words
            ""          # Characters (last resort)
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=code_aware_separators
        )

        # Split the header-based chunks further if needed
        final_splits = text_splitter.split_documents(md_header_splits)

        with open("/tmp/vector_memory_debug.log", "a") as f:
            f.write(f"[CHUNK] Stage 2: Produced {len(final_splits)} final chunks\n")

        # Validate code block integrity
        final_splits = self._validate_code_integrity(final_splits)

        # Convert LangChain documents to ChunkEntry objects with enhanced metadata
        chunks = []
        for idx, doc in enumerate(final_splits):
            # Extract header path from metadata
            header_parts = []
            for i in range(1, 7):
                header_key = f"Header {i}"
                if header_key in doc.metadata:
                    header_parts.append(doc.metadata[header_key])

            header_path = " > ".join(header_parts) if header_parts else ""
            level = len(header_parts) if header_parts else 0

            content_text = doc.page_content
            token_count = self._count_tokens(content_text)

            # Enhance with code-specific metadata
            chunk_metadata = self._enhance_code_metadata(content_text, doc.metadata, token_count, chunk_size)

            chunk_entry = ChunkEntry(
                parent_id=parent_id,
                chunk_index=idx,
                content=content_text.strip(),
                chunk_type=chunk_metadata.get('type', 'section' if level > 0 else 'text'),
                token_count=token_count,
                header_path=header_path,
                level=level,
                content_hash=self._generate_hash(content_text)
            )

            # Store enhanced metadata as dict (will be JSON serialized)
            chunk_entry.metadata = chunk_metadata

            chunks.append(chunk_entry)

        # Link chunks together
        self._link_chunks(chunks)

        with open("/tmp/vector_memory_debug.log", "a") as f:
            f.write(f"[CHUNK] Completed: {len(chunks)} chunks created\n")

            # Log size distribution
            sizes = [c.token_count for c in chunks]
            if sizes:
                f.write(f"[CHUNK] Size range: {min(sizes)}-{max(sizes)} tokens (avg: {sum(sizes)//len(sizes)})\n")

        return chunks

    def _validate_code_integrity(self, splits: list) -> list:
        """
        Ensure code blocks remain intact.

        Validates that code fences are balanced and flags any integrity issues.
        Target: 100% for standard-sized blocks, 95%+ including oversized.
        """
        for i, doc in enumerate(splits):
            content = doc.page_content

            # Check for balanced code fences
            opening_fences = content.count("```")
            if opening_fences % 2 != 0:
                # Unbalanced - code block was split
                doc.metadata['code_integrity_warning'] = True
                doc.metadata['unbalanced_fences'] = True
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write(f"[VALIDATE] WARNING: Chunk {i} has unbalanced code fences ({opening_fences})\n")

        # Calculate integrity score
        code_chunks = [c for c in splits if '```' in c.page_content]
        valid_chunks = [c for c in code_chunks
                        if not c.metadata.get('code_integrity_warning')]

        if code_chunks:
            integrity_score = len(valid_chunks) / len(code_chunks)
            with open("/tmp/vector_memory_debug.log", "a") as f:
                f.write(f"[VALIDATE] Code integrity: {len(valid_chunks)}/{len(code_chunks)} = {integrity_score:.1%}\n")

            # Target: 100% for standard-sized blocks, 95%+ including oversized
            if integrity_score < 0.95:
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write(f"[VALIDATE] WARNING: Code integrity below target: {integrity_score:.1%}\n")

        return splits

    def _apply_contextual_enrichment(self, chunks: list[ChunkEntry], document_title: str, memory_type: str):
        """
        Apply contextual enrichment to all chunks in place.

        Modifies chunk objects to add enriched content and preserve original.
        """
        total_chunks = len(chunks)

        with open("/tmp/vector_memory_debug.log", "a") as f:
            f.write(f"[ENRICH] Applying contextual enrichment to {total_chunks} chunks for '{document_title}'\n")

        for chunk in chunks:
            # Determine granularity based on token count
            if chunk.token_count:
                if chunk.token_count < 400:
                    granularity = "fine"
                elif chunk.token_count <= 1200:
                    granularity = "medium"
                else:
                    granularity = "coarse"
            else:
                granularity = "medium"  # Default

            # Enrich the chunk
            enriched_content, original_content = self._enrich_chunk_with_context(
                chunk_content=chunk.content,
                document_title=document_title,
                header_path=chunk.header_path or "",
                chunk_index=chunk.chunk_index,
                total_chunks=total_chunks,
                memory_type=memory_type,
                granularity_level=granularity
            )

            # Update chunk object
            chunk.original_content = original_content
            chunk.content = enriched_content  # content field gets enriched version for embedding
            chunk.is_contextually_enriched = True
            chunk.granularity_level = granularity  # Set granularity on chunk object

    def _enrich_chunk_with_context(
        self,
        chunk_content: str,
        document_title: str,
        header_path: str,
        chunk_index: int,
        total_chunks: int,
        memory_type: str,
        granularity_level: str
    ) -> tuple[str, str]:
        """
        Create contextual header for chunk embedding while preserving original for display.

        Args:
            chunk_content: Original chunk content
            document_title: Parent document title
            header_path: Section hierarchy path
            chunk_index: Position in document (0-based)
            total_chunks: Total number of chunks
            memory_type: Type of memory (knowledge_base, reports, etc)
            granularity_level: fine/medium/coarse

        Returns:
            tuple: (enriched_content_for_embedding, original_content_for_display)
        """
        # Calculate position ratio
        position_ratio = (chunk_index + 1) / total_chunks if total_chunks > 0 else 0.5
        position_desc = "beginning" if position_ratio < 0.33 else "middle" if position_ratio < 0.67 else "end"

        # Build context header
        context_parts = [
            f"Document: {document_title}",
            f"Section: {header_path or 'Main Content'}",
            f"Position: Chunk {chunk_index + 1} of {total_chunks} ({position_desc}, {granularity_level} granularity)",
            f"Type: {memory_type}",
            "",  # Empty line separator
        ]

        context_header = "\n".join(context_parts)
        enriched_content = context_header + chunk_content

        return enriched_content, chunk_content

    def _enhance_code_metadata(self, content: str, base_metadata: dict[str, Any],
                               token_count: int, chunk_size: int) -> dict[str, Any]:
        """
        Add code-specific metadata for enhanced retrieval.

        Detects code blocks and adds language, type, and integrity flags.
        """
        metadata = base_metadata.copy()

        # Basic metadata
        metadata.update({
            'token_count': token_count,
        })

        # Code-specific metadata
        if '```' in content:
            metadata.update({
                'type': 'code_block',
                'is_code': True,
                'atomic_unit': True,
                'has_code': True,
            })

            # Extract language from fence if present
            lang_match = re.search(r'```(\w+)', content)
            if lang_match:
                metadata['language'] = lang_match.group(1)
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write(f"[METADATA] Detected code language: {lang_match.group(1)}\n")

            # Flag oversized code blocks
            if token_count > chunk_size:
                metadata['exceeds_limit'] = True
                with open("/tmp/vector_memory_debug.log", "a") as f:
                    f.write(f"[METADATA] Code block exceeds chunk_size: {token_count} > {chunk_size}\n")

        # Flag unusually small chunks for monitoring (not an error, just awareness)
        if token_count < 150:
            metadata['below_typical_size'] = True

        return metadata

    def _chunk_markdown_hierarchical(self, content: str, parent_id: int, chunk_size: int, chunk_overlap: int) -> list[ChunkEntry]:
        """Chunk markdown preserving header hierarchy (legacy fallback method)"""
        chunks = []
        sections = self._split_by_headers(content)

        # Note: No longer merging small sections per 2-stage best practices
        # Natural boundaries dictate sizes
        merged_sections = sections

        chunk_index = 0
        for section in merged_sections:
            section_chunks = self._process_section(section, parent_id, chunk_index, chunk_size, chunk_overlap)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        self._link_chunks(chunks)
        return chunks

    def _split_by_headers(self, content: str) -> list[dict[str, Any]]:
        """Split content by markdown headers"""
        sections = []
        lines = content.split('\n')
        current_section = {
            'header': '',
            'level': 0,
            'content': '',
            'start_line': 0,
            'header_path': ''
        }

        header_stack = []

        for line_num, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.*)', line.strip())

            if header_match:
                if current_section['content'].strip():
                    sections.append(current_section)

                header_level = len(header_match.group(1))
                header_text = header_match.group(2)

                header_stack = header_stack[:header_level-1]
                if len(header_stack) < header_level:
                    header_stack.extend([''] * (header_level - len(header_stack)))
                header_stack[header_level-1] = header_text

                current_section = {
                    'header': header_text,
                    'level': header_level,
                    'content': line + '\n',
                    'start_line': line_num,
                    'header_path': ' > '.join(h for h in header_stack[:header_level] if h)
                }
            else:
                current_section['content'] += line + '\n'

        if current_section['content'].strip():
            sections.append(current_section)

        return sections

    def _process_section(self, section: dict[str, Any], parent_id: int, start_index: int, chunk_size: int, chunk_overlap: int) -> list[ChunkEntry]:
        """Process a section, splitting if needed"""
        content = section['content']
        token_count = self._count_tokens(content)

        if token_count <= chunk_size:
            return [ChunkEntry(
                parent_id=parent_id,
                chunk_index=start_index,
                content=content.strip(),
                chunk_type="section" if section['level'] > 0 else "text",
                token_count=token_count,
                header_path=section.get('header_path', ''),
                level=section['level'],
                content_hash=self._generate_hash(content)
            )]
        else:
            return self._split_large_section(section, parent_id, start_index, chunk_size, chunk_overlap)

    def _split_large_section(self, section: dict[str, Any], parent_id: int, start_index: int, chunk_size: int, chunk_overlap: int) -> list[ChunkEntry]:
        """Split large section into smaller chunks"""
        content = section['content']
        chunks = []

        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_index

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > chunk_size and current_chunk:
                chunks.append(ChunkEntry(
                    parent_id=parent_id,
                    chunk_index=chunk_index,
                    content=current_chunk.strip(),
                    chunk_type="text",
                    token_count=current_tokens,
                    header_path=section.get('header_path', ''),
                    level=section['level'],
                    content_hash=self._generate_hash(current_chunk)
                ))

                if chunk_overlap > 0 and current_chunk:
                    overlap_content = self._get_overlap_content(current_chunk, chunk_overlap)
                    current_chunk = overlap_content + '\n\n' + para
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = para
                    current_tokens = para_tokens

                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        if current_chunk.strip():
            chunks.append(ChunkEntry(
                parent_id=parent_id,
                chunk_index=chunk_index,
                content=current_chunk.strip(),
                chunk_type="text",
                token_count=current_tokens,
                header_path=section.get('header_path', ''),
                level=section['level'],
                content_hash=self._generate_hash(current_chunk)
            ))

        return chunks

    def _chunk_recursive(self, content: str, parent_id: int, chunk_size: int, chunk_overlap: int) -> list[ChunkEntry]:
        """Recursive chunking for non-structured content"""
        chunks = []
        sentences = re.split(r'[.!?]+\s+', content)

        current_chunk = ""
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append(ChunkEntry(
                    parent_id=parent_id,
                    chunk_index=chunk_index,
                    content=current_chunk.strip(),
                    chunk_type="text",
                    token_count=current_tokens,
                    content_hash=self._generate_hash(current_chunk)
                ))

                if chunk_overlap > 0:
                    overlap_content = self._get_overlap_content(current_chunk, chunk_overlap)
                    current_chunk = overlap_content + ' ' + sentence
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens

                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens

        if current_chunk.strip():
            chunks.append(ChunkEntry(
                parent_id=parent_id,
                chunk_index=chunk_index,
                content=current_chunk.strip(),
                chunk_type="text",
                token_count=current_tokens,
                content_hash=self._generate_hash(current_chunk)
            ))

        self._link_chunks(chunks)
        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or estimation"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation
            return int(len(text.split()) * 1.3)

    def _get_overlap_content(self, content: str, overlap_tokens: int) -> str:
        """Get last portion of content for overlap"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(content)
            if len(tokens) > overlap_tokens:
                overlap_token_ids = tokens[-overlap_tokens:]
                return self.tokenizer.decode(overlap_token_ids)
            else:
                return content
        else:
            words = content.split()
            overlap_words = int(overlap_tokens / 1.3)
            if len(words) > overlap_words:
                return ' '.join(words[-overlap_words:])
            else:
                return content

    def _link_chunks(self, chunks: list[ChunkEntry]) -> None:
        """Link chunks with prev/next references"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i-1].id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].id

    def _generate_hash(self, content: str) -> str:
        """Generate SHA256 hash for content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def reconstruct_document(self, chunks: list[ChunkEntry]) -> str:
        """Reconstruct document from chunks"""
        if not chunks:
            return ""

        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index)
        content_parts = [chunk.content for chunk in sorted_chunks]
        return '\n\n'.join(content_parts)
