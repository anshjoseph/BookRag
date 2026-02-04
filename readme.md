# Book RAG with Disjoint Connections

Complete Python implementation using **Nomic embeddings via LM Studio** and **ChromaDB** for vector storage.

## Features

- **Nomic Embeddings**: Uses `text-embedding-nomic-embed-text-v1.5` via LM Studio API
- **ChromaDB**: Persistent vector storage with cosine similarity search
- **Atomic Unit Splitting**: Breaks documents into smallest meaningful units
- **Semantic Sectioning**: Groups units into coherent sections
- **Disjoint Connection Discovery**: LLM finds meaningful relationships between distant sections
- **Intelligent Query Expansion**: Retrieves both similar and conceptually related content
- **Offline Processing**: All connections built during indexing for fast queries

## Installation

```bash
pip install PyPDF2 chromadb openai numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- **LM Studio** running locally with:
  - `text-embedding-nomic-embed-text-v1.5@f16` loaded for embeddings
  - `llama-3.2-3b-instruct` loaded for reasoning/queries
  - API server enabled (default: http://localhost:1234)

## LM Studio Setup

1. Download and install LM Studio
2. Download these models:
   - `nomic-ai/nomic-embed-text-v1.5-GGUF` (use f16 quantization)
   - `bartowski/Llama-3.2-3B-Instruct-GGUF`
3. Start local server in LM Studio (Developer tab → Local Server → Start)
4. Load the embedding model first, then the LLM model

## Quick Start

### 1. Build RAG System from PDF

```bash
python book_rag_system.py --pdf your_book.pdf --build
```

This will:
- Extract and normalize text from PDF
- Create atomic units and sections
- Generate Nomic embeddings via LM Studio
- Store sections in ChromaDB
- Build disjoint connections using Llama
- Save the system to `rag_system.pkl`

### 2. Query the System

```bash
python book_rag_system.py --query "Why does X happen?"
```

### 3. Custom Configuration

```bash
python book_rag_system.py --pdf book.pdf --build --lm-studio-url http://localhost:5000/v1 --chroma-dir ./my_chroma
```

## Usage as Library

```python
from book_rag_system import BookRAGSystem

# Initialize with custom settings
rag = BookRAGSystem(
    lm_studio_url="http://localhost:1234/v1",
    chroma_persist_dir="./chroma_db",
    similarity_threshold=0.7,
    max_section_tokens=512
)

# Build from PDF
rag.build_from_pdf("my_book.pdf", "my_rag.pkl")

# Query
answer = rag.query("What are the main themes?")
print(answer)

# Load existing system
rag.load("my_rag.pkl")
answer = rag.query("How does chapter 1 relate to chapter 10?")
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lm_studio_url` | http://localhost:1234/v1 | LM Studio API endpoint |
| `chroma_persist_dir` | ./chroma_db | ChromaDB storage directory |
| `similarity_threshold` | 0.7 | Threshold for grouping units into sections |
| `max_section_tokens` | 512 | Maximum tokens per section |
| `low_similarity_threshold` | 0.3 | Threshold for disjoint candidates |
| `max_candidates_per_section` | 10 | Max disjoint candidates per section |
| `max_connections_per_section` | 5 | Max disjoint connections per section |
| `min_confidence` | 0.6 | Minimum LLM confidence for connections |

## How It Works

### Phase 1-2: Document Processing
1. Extract text from PDF
2. Normalize and clean text
3. Split into atomic units (paragraphs/sentence groups)
4. Generate Nomic embeddings via LM Studio

### Phase 3: Section Creation
5. Group nearby atomic units into sections based on:
   - Embedding similarity (cosine)
   - Sequential position
   - Token limits
6. Store sections in ChromaDB with embeddings

### Phase 4-5: Disjoint Discovery
7. Compute pairwise similarity between all sections
8. Select low-similarity pairs with shared keywords

### Phase 6-7: Connection Generation
9. Use Llama 3.2 to analyze each candidate pair
10. Extract relation type, common ground, and combined meaning
11. Build bidirectional graph of connections

### Phase 8-9: Query Processing
12. Embed query with Nomic
13. Retrieve top-K similar sections from ChromaDB
14. Expand context via disjoint connections
15. Assemble enriched context with reasoning bridges
16. Generate answer using Llama 3.2

## Architecture

```
PDF Input
    ↓
[Phase 1] Text Extraction & Normalization
    ↓
[Phase 2] Atomic Unit Splitting → Nomic Embeddings
    ↓
[Phase 3] Section Grouping → ChromaDB Storage
    ↓
[Phase 4] Similarity Matrix Computation
    ↓
[Phase 5] Disjoint Candidate Selection
    ↓
[Phase 6] LLM Connection Generation (Llama 3.2)
    ↓
[Phase 7] Graph Construction
    ↓
[Save] rag_system.pkl + ChromaDB

Query → Nomic Embed → ChromaDB Search → Context + Disjoint → Llama 3.2 → Answer
```

## Relation Types

The system identifies these connection types:

- **cause-effect**: Causal relationships
- **definition-application**: Concepts and their usage
- **question-answer**: Questions posed early, answered later
- **assumption-outcome**: Assumptions leading to results
- **concept-example**: Abstract concepts with concrete examples
- **contrast**: Opposing viewpoints or approaches

## Example Queries

```bash
# Causal reasoning
python book_rag_system.py --query "Why does the protagonist make this decision?"

# Cross-chapter analysis
python book_rag_system.py --query "How does the introduction relate to the conclusion?"

# Conceptual connections
python book_rag_system.py --query "What assumptions lead to this outcome?"
```

## File Structure

```
book_rag_system.py          # Main implementation
requirements.txt            # Dependencies
chroma_db/                  # ChromaDB storage (created automatically)
rag_system.pkl              # Saved system (after build)
README.md                   # This file
```

## Advanced Usage

### Debugging Connections

```python
rag.load("rag_system.pkl")

# Inspect all connections
for conn in rag.disjoint_connections:
    print(f"{conn.source_id} → {conn.target_id}")
    print(f"Type: {conn.relation_type}")
    print(f"Why: {conn.common_ground}")
    print(f"Insight: {conn.combined_meaning}")
    print(f"Confidence: {conn.confidence:.2f}")
    print("-" * 60)
```

### Query ChromaDB Directly

```python
# Check what's in ChromaDB
results = rag.sections_collection.peek(10)
print(results)

# Get collection stats
print(rag.sections_collection.count())
```

### Batch Processing

```python
import glob

for pdf_file in glob.glob("books/*.pdf"):
    rag = BookRAGSystem(chroma_persist_dir=f"./chroma_{pdf_file}")
    rag.build_from_pdf(pdf_file, pdf_file.replace('.pdf', '.pkl'))
```

## Troubleshooting

**LM Studio connection error**:
- Ensure LM Studio is running with local server enabled
- Check the URL (default: http://localhost:1234/v1)
- Verify both Nomic and Llama models are loaded
- Try the "Test Connection" button in LM Studio

**Embedding errors**:
- Make sure `text-embedding-nomic-embed-text-v1.5@f16` is loaded
- Check LM Studio console for error messages
- Verify the model is loaded in "Embedding" mode, not "Chat"

**ChromaDB errors**:
- Delete `./chroma_db` folder and rebuild if corrupted
- Check write permissions in the directory
- Use a different `chroma_persist_dir` if needed

**Out of memory**:
- Reduce `max_section_tokens`
- Reduce `max_candidates_per_section`
- Process document in chunks
- Use quantized models in LM Studio

**Poor connections**:
- Increase `min_confidence` (e.g., 0.7 or 0.8)
- Adjust `low_similarity_threshold`
- Check if Llama 3.2 is properly loaded
- Review LLM temperature and max_tokens

## Performance Notes

- **Embedding speed**: ~10 units/second with Nomic f16
- **Connection generation**: ~2-5 seconds per candidate pair
- **Query time**: <1 second (after connections are built)
- **Memory**: ~2-4GB for average book (200 pages)

## Differences from Standard RAG

Standard RAG only retrieves similar content. This system:

1. **Finds distant connections**: Low-similarity but conceptually related sections
2. **Explains relationships**: Common ground and emergent insights
3. **Better causal reasoning**: Can trace assumptions to outcomes across chapters
4. **Reduced hallucination**: Explicit reasoning bridges ground the LLM

## License

MIT

## Credits

- **Nomic AI** for nomic-embed-text-v1.5
- **Meta** for Llama 3.2
- **ChromaDB** for vector storage
- **LM Studio** for local inference