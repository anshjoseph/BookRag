"""
Book RAG with Disjoint Connections - FIXED VERSION
Using Nomic embeddings via LM Studio and ChromaDB for vector storage

FIXES:
- Removed memory system (causing issues)
- Fixed ChromaDB not loading sections on query
- Better error handling for empty results
- Improved disjoint connection generation

Usage:
    python book_rag_system_fixed.py --pdf book.pdf --build
    python book_rag_system_fixed.py --query "Why does X happen?"
"""

import os
import re
import json
import pickle
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

import PyPDF2
import chromadb
from openai import OpenAI


@dataclass
class AtomicUnit:
    """Smallest meaningful text unit"""
    id: str
    text: str
    position: int
    embedding: Optional[List[float]] = None


@dataclass
class Section:
    """Coherent group of atomic units"""
    section_id: str
    text: str
    embedding: Optional[List[float]] = None
    position_range: Tuple[int, int] = (0, 0)
    atomic_unit_ids: List[str] = None
    
    def __post_init__(self):
        if self.atomic_unit_ids is None:
            self.atomic_unit_ids = []


@dataclass
class DisjointConnection:
    """Reasoning edge between sections"""
    source_id: str
    target_id: str
    relation_type: str
    common_ground: str
    combined_meaning: str
    confidence: float


class NomicEmbedder:
    """Wrapper for Nomic embeddings via LM Studio"""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = "text-embedding-nomic-embed-text-v1.5"
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding text: {e}")
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embed([text])[0]


class BookRAGSystem:
    """Main RAG system with disjoint connections"""
    
    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1",
        lm_studio_api_key: str = "lm-studio",
        chroma_persist_dir: str = "./chroma_db",
        similarity_threshold: float = 0.7,
        max_section_tokens: int = 512,
        low_similarity_threshold: float = 0.4,
        max_candidates_per_section: int = 15,
        max_connections_per_section: int = 5,
        min_confidence: float = 0.5,
        min_shared_keywords: int = 1
    ):
        """Initialize the RAG system"""
        print("Initializing Book RAG System...")
        
        print("Initializing Nomic embedder via LM Studio...")
        self.embedder = NomicEmbedder(lm_studio_url, lm_studio_api_key)
        
        self.llm_client = OpenAI(
            base_url=lm_studio_url,
            api_key=lm_studio_api_key
        )
        
        print(f"Initializing ChromaDB at {chroma_persist_dir}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.chroma_persist_dir = chroma_persist_dir
        
        # Try to get existing collection
        try:
            self.sections_collection = self.chroma_client.get_collection("sections")
            print(f"Loaded existing collection with {self.sections_collection.count()} sections")
        except:
            print("No existing collection found")
            self.sections_collection = None
        
        self.similarity_threshold = similarity_threshold
        self.max_section_tokens = max_section_tokens
        self.low_similarity_threshold = low_similarity_threshold
        self.max_candidates_per_section = max_candidates_per_section
        self.max_connections_per_section = max_connections_per_section
        self.min_confidence = min_confidence
        self.min_shared_keywords = min_shared_keywords
        
        self.atomic_units: List[AtomicUnit] = []
        self.sections: List[Section] = []
        self.disjoint_connections: List[DisjointConnection] = []
        self.disjoint_graph: Dict[str, List[DisjointConnection]] = defaultdict(list)
        
    def load_pdf(self, pdf_path: str) -> str:
        """Load and extract text from PDF"""
        print(f"Loading PDF: {pdf_path}")
        
        text_parts = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
        
        full_text = "\n".join(text_parts)
        print(f"Extracted {len(full_text)} characters from {len(pdf_reader.pages)} pages")
        return full_text
    
    def normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text"""
        print("Normalizing text...")
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def split_into_atomic_units(self, text: str) -> List[AtomicUnit]:
        """Split text into atomic units"""
        print("Splitting into atomic units...")
        
        paragraphs = re.split(r'\n\n+|\.\s*\n', text)
        
        atomic_units = []
        position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            
            if len(para) > 1000:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_group = []
                current_length = 0
                
                for sent in sentences:
                    if current_length + len(sent) > 500 and current_group:
                        group_text = ' '.join(current_group)
                        unit = AtomicUnit(
                            id=f"unit_{position}",
                            text=group_text,
                            position=position
                        )
                        atomic_units.append(unit)
                        position += 1
                        current_group = [sent]
                        current_length = len(sent)
                    else:
                        current_group.append(sent)
                        current_length += len(sent)
                
                if current_group:
                    group_text = ' '.join(current_group)
                    unit = AtomicUnit(
                        id=f"unit_{position}",
                        text=group_text,
                        position=position
                    )
                    atomic_units.append(unit)
                    position += 1
            else:
                unit = AtomicUnit(
                    id=f"unit_{position}",
                    text=para,
                    position=position
                )
                atomic_units.append(unit)
                position += 1
        
        print(f"Created {len(atomic_units)} atomic units")
        return atomic_units
    
    def embed_atomic_units(self, units: List[AtomicUnit]):
        """Generate embeddings for atomic units"""
        print("Generating embeddings for atomic units...")
        
        texts = [unit.text for unit in units]
        
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"Embedding units {i+1} to {min(i+batch_size, len(texts))} of {len(texts)}")
            embeddings = self.embedder.embed(batch)
            
            for j, embedding in enumerate(embeddings):
                units[i+j].embedding = embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def create_sections(self, units: List[AtomicUnit]) -> List[Section]:
        """Group atomic units into sections"""
        print("Creating sections from atomic units...")
        
        if not units:
            return []
        
        sections = []
        section_id = 0
        
        current_section_units = [units[0]]
        current_section_text = units[0].text
        current_section_tokens = len(units[0].text.split())
        
        for i in range(1, len(units)):
            unit = units[i]
            unit_tokens = len(unit.text.split())
            
            if current_section_tokens + unit_tokens <= self.max_section_tokens:
                current_embeddings = [u.embedding for u in current_section_units]
                current_embedding = np.mean(current_embeddings, axis=0).tolist()
                
                similarity = self.cosine_similarity(current_embedding, unit.embedding)
                
                if similarity >= self.similarity_threshold:
                    current_section_units.append(unit)
                    current_section_text += " " + unit.text
                    current_section_tokens += unit_tokens
                    continue
            
            section_embedding = np.mean([u.embedding for u in current_section_units], axis=0).tolist()
            
            section = Section(
                section_id=f"section_{section_id}",
                text=current_section_text,
                embedding=section_embedding,
                position_range=(current_section_units[0].position, current_section_units[-1].position),
                atomic_unit_ids=[u.id for u in current_section_units]
            )
            sections.append(section)
            section_id += 1
            
            current_section_units = [unit]
            current_section_text = unit.text
            current_section_tokens = unit_tokens
        
        if current_section_units:
            section_embedding = np.mean([u.embedding for u in current_section_units], axis=0).tolist()
            
            section = Section(
                section_id=f"section_{section_id}",
                text=current_section_text,
                embedding=section_embedding,
                position_range=(current_section_units[0].position, current_section_units[-1].position),
                atomic_unit_ids=[u.id for u in current_section_units]
            )
            sections.append(section)
        
        print(f"Created {len(sections)} sections")
        return sections
    
    def add_sections_to_chroma(self, sections: List[Section]):
        """Add sections to ChromaDB"""
        print("Adding sections to ChromaDB...")
        
        # Delete old collection
        try:
            self.chroma_client.delete_collection("sections")
            print("Deleted old collection")
        except:
            pass
        
        # Create new collection
        self.sections_collection = self.chroma_client.create_collection(
            name="sections",
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = [s.section_id for s in sections]
        documents = [s.text for s in sections]
        embeddings = [s.embedding for s in sections]
        metadatas = [
            {
                "position_start": s.position_range[0],
                "position_end": s.position_range[1]
            }
            for s in sections
        ]
        
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.sections_collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        
        print(f"Added {len(sections)} sections to ChromaDB")
    
    def compute_similarity_matrix(self, sections: List[Section]) -> np.ndarray:
        """Compute pairwise similarity"""
        print("Computing pairwise similarity matrix...")
        
        embeddings = np.array([s.embedding for s in sections])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords"""
        words = []
        
        words += re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        words += re.findall(r'\b[a-z]{4,}\b', text.lower())
        words += re.findall(r'\b\d+\b', text)
        words += re.findall(r'\b[a-z]+-[a-z]+\b', text.lower())
        
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 
                      'could', 'should', 'their', 'there', 'these', 'those', 'when', 'what',
                      'which', 'where', 'while', 'about', 'after', 'before', 'into', 'such',
                      'also', 'just', 'only', 'some', 'than', 'them', 'then', 'very'}
        words = [w for w in words if w.lower() not in stop_words]
        
        return set(words[:100])
    
    def select_disjoint_candidates(
        self,
        sections: List[Section],
        similarity_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Select candidate pairs"""
        print("Selecting disjoint candidates...")
        
        candidates = []
        n_sections = len(sections)
        
        for i in range(n_sections):
            section_i = sections[i]
            keywords_i = self.extract_keywords(section_i.text)
            
            low_sim_indices = np.where(
                (similarity_matrix[i] < self.low_similarity_threshold) &
                (similarity_matrix[i] > 0)
            )[0]
            
            print(f"Section {i}: Found {len(low_sim_indices)} low-similarity sections, {len(keywords_i)} keywords")
            
            section_candidates = []
            
            for j in low_sim_indices:
                if i >= j:
                    continue
                
                section_j = sections[j]
                keywords_j = self.extract_keywords(section_j.text)
                shared_keywords = keywords_i & keywords_j
                
                if len(shared_keywords) >= self.min_shared_keywords:
                    position_distance = abs(section_i.position_range[0] - section_j.position_range[0])
                    score = len(shared_keywords) * 100 - position_distance * 0.1
                    section_candidates.append((i, j, score, shared_keywords))
            
            section_candidates.sort(key=lambda x: x[2], reverse=True)
            
            for i_idx, j_idx, score, shared in section_candidates[:self.max_candidates_per_section]:
                candidates.append((i_idx, j_idx))
                print(f"  Candidate: section_{i_idx} <-> section_{j_idx} (shared: {list(shared)[:5]})")
        
        print(f"Selected {len(candidates)} candidate pairs")
        return candidates
    
    def generate_disjoint_connection(
        self,
        section_a: Section,
        section_b: Section
    ) -> Optional[DisjointConnection]:
        """Use LLM to determine relationship"""
        
        prompt = f"""Analyze if these two text sections have a meaningful conceptual relationship.

Section A (position {section_a.position_range}):
{section_a.text[:1000]}

Section B (position {section_b.position_range}):
{section_b.text[:1000]}

Determine if they are related through:
- cause-effect: One explains why the other happens
- definition-application: One defines a concept used in the other
- question-answer: One poses a question answered by the other
- assumption-outcome: One states assumptions leading to outcomes in the other
- concept-example: One introduces a concept illustrated by the other
- contrast: They present opposing viewpoints on the same topic
- elaboration: One expands on ideas mentioned briefly in the other

Respond ONLY with valid JSON:
{{
    "related": true or false,
    "relation_type": "one of the types above or none",
    "common_ground": "brief explanation of the connection",
    "combined_meaning": "what insight emerges from both together",
    "confidence": 0.0 to 1.0
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.2-3b-instruct",
                messages=[
                    {"role": "system", "content": "You analyze text relationships. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                print(f"  ✗ No JSON found")
                return None
            
            result = json.loads(json_match.group())
            
            if not result.get('related', False):
                print(f"  ✗ Not related")
                return None
            
            confidence = result.get('confidence', 0)
            if confidence < self.min_confidence:
                print(f"  ✗ Low confidence: {confidence:.2f}")
                return None
            
            return DisjointConnection(
                source_id=section_a.section_id,
                target_id=section_b.section_id,
                relation_type=result.get('relation_type', 'unknown'),
                common_ground=result.get('common_ground', ''),
                combined_meaning=result.get('combined_meaning', ''),
                confidence=confidence
            )
            
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def build_disjoint_connections(
        self,
        sections: List[Section],
        candidates: List[Tuple[int, int]]
    ) -> List[DisjointConnection]:
        """Generate disjoint connections"""
        print(f"Building disjoint connections from {len(candidates)} candidates...")
        
        connections = []
        connections_per_section = defaultdict(int)
        
        for i, (idx_a, idx_b) in enumerate(candidates):
            section_a = sections[idx_a]
            section_b = sections[idx_b]
            
            if connections_per_section[section_a.section_id] >= self.max_connections_per_section:
                continue
            if connections_per_section[section_b.section_id] >= self.max_connections_per_section:
                continue
            
            print(f"\nProcessing {i+1}/{len(candidates)}: {section_a.section_id} <-> {section_b.section_id}")
            
            connection = self.generate_disjoint_connection(section_a, section_b)
            
            if connection:
                connections.append(connection)
                connections_per_section[section_a.section_id] += 1
                connections_per_section[section_b.section_id] += 1
                print(f"  ✓ Created: {connection.relation_type} (confidence: {connection.confidence:.2f})")
        
        print(f"\n{'='*60}")
        print(f"Created {len(connections)} disjoint connections")
        print(f"{'='*60}")
        return connections
    
    def build_disjoint_graph(self, connections: List[DisjointConnection]):
        """Build graph structure"""
        print("Building disjoint graph...")
        
        self.disjoint_graph.clear()
        
        for conn in connections:
            self.disjoint_graph[conn.source_id].append(conn)
            reverse_conn = DisjointConnection(
                source_id=conn.target_id,
                target_id=conn.source_id,
                relation_type=conn.relation_type,
                common_ground=conn.common_ground,
                combined_meaning=conn.combined_meaning,
                confidence=conn.confidence
            )
            self.disjoint_graph[conn.target_id].append(reverse_conn)
    
    def query(self, query_text: str, top_k: int = 5, expand_disjoint: bool = True) -> str:
        """Query the RAG system"""
        print(f"\nQuery: {query_text}")
        
        query_embedding = self.embedder.embed_single(query_text)
        
        try:
            results = self.sections_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return "Error: Could not retrieve sections. Please rebuild the system."
        
        retrieved_section_ids = results['ids'][0]
        print(f"Retrieved {len(retrieved_section_ids)} primary sections")
        
        if len(retrieved_section_ids) == 0:
            return "No relevant sections found. Please rebuild the system with --build."
        
        retrieved_sections = [s for s in self.sections if s.section_id in retrieved_section_ids]
        
        context_parts = []
        added_section_ids = set()
        
        for i, section in enumerate(retrieved_sections):
            context_parts.append(f"[Primary Section {i+1}]")
            context_parts.append(section.text)
            context_parts.append("")
            added_section_ids.add(section.section_id)
            
            if expand_disjoint and section.section_id in self.disjoint_graph:
                connections = self.disjoint_graph[section.section_id]
                connections = sorted(connections, key=lambda x: x.confidence, reverse=True)
                
                for conn in connections[:2]:
                    if conn.target_id not in added_section_ids:
                        target_section = next((s for s in self.sections if s.section_id == conn.target_id), None)
                        
                        if target_section:
                            context_parts.append(f"[Connected Section via {conn.relation_type}]")
                            context_parts.append(f"Connection: {conn.common_ground}")
                            context_parts.append(f"Insight: {conn.combined_meaning}")
                            context_parts.append(target_section.text)
                            context_parts.append("")
                            added_section_ids.add(conn.target_id)
        
        context = "\n".join(context_parts)


        print(f"CONTEXT : {context}")
        
        print("Generating answer...")
        answer_prompt = f"""
        
Context:
{context}


Question: {query_text}

Answer:"""
        system = """
You are answering questions based on an insurance policy document.

The document contains:
- Policy rules and definitions
- Sample illustrations with names, ages, years, and scenarios

IMPORTANT RULES:
- Use sample illustrations ONLY to understand policy rules.
- NEVER reuse names, ages, years, or story details unless the user explicitly asks for an example.
- If the question is hypothetical, summarize ONLY the policy rules.
- Do NOT infer option numbers or timelines from examples.
"""

        try:
            groq_client = OpenAI(
                api_key="gsk_...", 
                base_url="https://api.groq.com/openai/v1"
            )
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def build_from_pdf(self, pdf_path: str, save_path: str = "rag_system.pkl"):
        """Complete pipeline"""
        print("=" * 60)
        print("BUILDING RAG SYSTEM FROM PDF")
        print("=" * 60)
        
        raw_text = self.load_pdf(pdf_path)
        normalized_text = self.normalize_text(raw_text)
        
        self.atomic_units = self.split_into_atomic_units(normalized_text)
        self.embed_atomic_units(self.atomic_units)
        
        self.sections = self.create_sections(self.atomic_units)
        self.add_sections_to_chroma(self.sections)
        
        similarity_matrix = self.compute_similarity_matrix(self.sections)
        candidates = self.select_disjoint_candidates(self.sections, similarity_matrix)
        self.disjoint_connections = self.build_disjoint_connections(self.sections, candidates)
        self.build_disjoint_graph(self.disjoint_connections)
        
        self.save(save_path)
        
        print("=" * 60)
        print("BUILD COMPLETE")
        print(f"Atomic units: {len(self.atomic_units)}")
        print(f"Sections: {len(self.sections)}")
        print(f"Disjoint connections: {len(self.disjoint_connections)}")
        print("=" * 60)
    
    def save(self, path: str):
        """Save the system"""
        print(f"Saving system to {path}...")
        
        data = {
            'atomic_units': self.atomic_units,
            'sections': self.sections,
            'disjoint_connections': self.disjoint_connections,
            'disjoint_graph': dict(self.disjoint_graph),
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'max_section_tokens': self.max_section_tokens,
                'low_similarity_threshold': self.low_similarity_threshold
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print("System saved successfully")
    
    def load(self, path: str):
        """Load saved system"""
        print(f"Loading system from {path}...")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.atomic_units = data['atomic_units']
        self.sections = data['sections']
        self.disjoint_connections = data['disjoint_connections']
        self.disjoint_graph = defaultdict(list, data['disjoint_graph'])
        
        # CRITICAL: Reload sections into ChromaDB
        if self.sections_collection is None or self.sections_collection.count() == 0:
            print("ChromaDB empty - reloading sections...")
            self.add_sections_to_chroma(self.sections)
        
        print("System loaded successfully")
        print(f"Loaded {len(self.sections)} sections, {len(self.disjoint_connections)} connections")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Book RAG with Disjoint Connections")
    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--build', action='store_true', help='Build system')
    parser.add_argument('--load', type=str, default='rag_system.pkl', help='Saved system path')
    parser.add_argument('--query', type=str, help='Query the system')
    parser.add_argument('--lm-studio-url', type=str, default='http://localhost:1234/v1')
    parser.add_argument('--chroma-dir', type=str, default='./chroma_db')
    
    args = parser.parse_args()
    
    rag = BookRAGSystem(
        lm_studio_url=args.lm_studio_url,
        chroma_persist_dir=args.chroma_dir
    )
    
    if args.build:
        if not args.pdf:
            print("Error: --pdf required for --build")
            return
        rag.build_from_pdf(args.pdf, args.load)
    
    elif args.query:
        rag.load(args.load)
        answer = rag.query(args.query)
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
    
    else:
        print("Use --build or --query")
        print("Example: python book_rag_system_fixed.py --pdf book.pdf --build")
        print("Example: python book_rag_system_fixed.py --query 'Your question?'")


if __name__ == "__main__":
    main()