"""
Project Indexer for Donna Email Research.

This module handles Phase 1 of project-based email search:
- Search all emails for a project name
- Fetch all unique threads
- Extract ALL content from PDFs (text or Gemini Vision for drawings)
- Build comprehensive ThreadDocs ready for vectorization

Usage:
    indexer = ProjectIndexer("88 SuperMarket", "user@email.com", "gmail")
    thread_docs = indexer.index_project()
"""

import json
import os
import re
import io
import time
import base64
import requests
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

# Local imports
from tools.gmail_tools import get_gmail_tools, GmailTools
from llm_brain import get_llm_brain


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AttachmentDoc:
    """Extracted attachment data."""
    attachment_id: str
    filename: str
    content_type: str
    extraction_method: str  # "text" or "vision"
    content: str  # Full extracted text or Gemini description


@dataclass 
class MessageDoc:
    """Single message within a thread."""
    sequence: int
    message_id: str
    timestamp: str
    from_email: str
    from_name: str
    to: List[str]
    cc: List[str]
    subject: str
    body: str  # Cleaned body (no signatures)
    attachments: List[AttachmentDoc]


@dataclass
class ThreadDoc:
    """Complete thread document ready for vectorization."""
    thread_id: str
    project_id: str
    subject: str
    date_first: str
    date_last: str
    participants: List[str]
    message_count: int
    pdf_count: int
    messages: List[MessageDoc]


@dataclass
class ProjectIndex:
    """Metadata about an indexed project."""
    project_id: str
    project_name: str
    user_email: str
    indexed_at: str
    last_email_timestamp: str
    thread_count: int
    message_count: int
    pdf_count: int
    thread_docs: List[ThreadDoc]


@dataclass
class AttachmentIndexItem:
    """Single item in the attachments index."""
    attachment_id: str
    filename: str
    content_type: str
    extraction_method: str
    sender_email: str
    sender_name: str
    receiver_emails: List[str]
    email_subject: str
    email_timestamp: str
    message_id: str
    thread_id: str
    ai_analysis: str  # The comprehensive extraction/explanation


@dataclass
class AttachmentIndex:
    """Index of all attachments processed for a project."""
    project_id: str
    project_name: str
    indexed_at: str
    total_attachments: int
    attachments: List[AttachmentIndexItem]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _normalize_project_id(project_name: str) -> str:
    """Convert project name to a safe ID."""
    return re.sub(r'[^a-z0-9]+', '_', project_name.lower()).strip('_')


def _clean_body(raw: str) -> str:
    """
    Clean an email body:
    - Convert HTML to text
    - Remove quoted replies
    - Strip signatures
    """
    if not raw:
        return ""
    
    # HTML to text
    def _html_to_text(s: str) -> str:
        from bs4 import BeautifulSoup
        s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
        s = re.sub(r'</p>', '\n', s, flags=re.IGNORECASE)
        s = re.sub(r'</div>', '\n', s, flags=re.IGNORECASE)
        s = re.sub(r'</li>', '\n', s, flags=re.IGNORECASE)
        try:
            soup = BeautifulSoup(s, 'html.parser')
            return soup.get_text('\n')
        except:
            return re.sub(r'<[^>]+>', '', s)
    
    # Remove quoted content (replies)
    def _extract_new_content(text: str) -> str:
        lines = text.split('\n')
        out = []
        for ln in lines:
            # Skip common reply indicators
            if ln.strip().startswith('>'):
                continue
            if re.match(r'^On .+ wrote:$', ln.strip()):
                break
            if re.match(r'^From:', ln.strip()) and 'Sent:' in text[text.find(ln):text.find(ln)+200]:
                break
            out.append(ln)
        return '\n'.join(out)
    
    # Strip signature
    def _strip_signature(text: str) -> str:
        lines = text.split('\n')
        valedictions = ['thanks', 'thank you', 'regards', 'kind regards', 'best', 
                       'best regards', 'sincerely', 'cheers', 'warm regards']
        
        # Look for signature in last 12 lines
        for i in range(len(lines) - 1, max(0, len(lines) - 12) - 1, -1):
            s = lines[i].strip().lower().rstrip(',')
            if any(s == v or s.startswith(v + ',') for v in valedictions):
                # Check if remaining lines look like signature (short, contact-ish)
                tail = lines[i:]
                if len(tail) <= 8:
                    return '\n'.join(lines[:i]).strip()
        return text.strip()
    
    text = _html_to_text(raw)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = _extract_new_content(text)
    text = _strip_signature(text)
    
    return text.strip()


def _extract_pdf_all_pages(pdf_bytes: bytes, max_chars: int = 500000) -> str:
    """
    Extract text from ALL pages of a PDF using pypdf.
    Returns empty string if no text found (likely a drawing/scan).
    """
    if not pdf_bytes:
        return ""
    
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(io.BytesIO(pdf_bytes))
        all_text = []
        
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                all_text.append(page_text)
        
        combined = '\n\n--- PAGE BREAK ---\n\n'.join(all_text)
        
        # Normalize whitespace
        combined = combined.replace('\r\n', '\n').replace('\r', '\n')
        combined = re.sub(r'[ \t]+\n', '\n', combined)
        combined = re.sub(r'\n{3,}', '\n\n', combined).strip()
        
        # Cap at max chars
        if len(combined) > max_chars:
            combined = combined[:max_chars] + '\n\n...[TRUNCATED]...'
        
        return combined
    
    except Exception as e:
        print(f"⚠️ PDF text extraction failed: {e}")
        return ""


def _build_comprehensive_extraction_prompt(filename: str) -> str:
    """
    Build a prompt that extracts and explains EVERYTHING from PDFs like a very smart human.
    Understands drawings, relationships, context, and meaning - not just text extraction.
    """
    return f"""You are an expert analyst examining this PDF document: "{filename}"

**Approach this like a VERY SMART HUMAN who understands everything:**
- Read and understand every word, number, and detail
- Look at drawings and understand what they mean, not just what they show
- Understand relationships, connections, and context
- Explain everything thoroughly - as if teaching someone who needs to understand every detail

**Your Mission: Extract and Explain EVERYTHING - leave NOTHING out.**

**1. TEXT CONTENT - Extract and Explain Everything:**
   - Extract every word of text exactly as written (verbatim)
   - Explain the meaning and context of every section
   - Include all headers, footers, watermarks, marginal notes
   - Extract all tables completely - every cell, every row, every column
   - Extract all forms - every field, every label, every value
   - Extract all lists, bullet points, and numbered items completely
   - Explain what each section means and why it matters

**2. DRAWINGS AND VISUAL ELEMENTS - Understand and Explain Deeply:**
   - Look at every drawing, diagram, floor plan, schematic, or illustration
   - Don't just describe what you see - EXPLAIN what it means
   - Extract ALL text, labels, annotations, dimensions, and callouts from drawings
   - Understand spatial relationships - what connects to what, what is adjacent, what is above/below
   - Understand scale and measurements - explain what dimensions mean in context
   - Identify all symbols, legends, and notations - explain what each represents
   - Understand the purpose and function of each drawing element
   - Explain relationships between different parts of the drawing
   - If it's a floor plan - explain the layout, room relationships, flow, dimensions
   - If it's a schematic - explain the connections, flow, and how it works
   - If it's a detail drawing - explain what it shows, dimensions, specifications
   - Extract every dimension, measurement, and annotation - explain what each means
   - Describe all visual elements: colors, patterns, line types, shading
   - Explain the overall purpose and meaning of the drawing

**3. DATA AND NUMBERS - Extract Every Detail:**
   - Extract EVERY number exactly as written (no rounding, no approximation)
   - Include all measurements with units (inches, feet, meters, etc.)
   - Include all dates and timestamps (creation, revision, issue, expiration, due dates)
   - Include all financial figures with currency symbols
   - Include all quantities, counts, and amounts
   - Explain what each number means in context
   - Include all calculations, formulas, or computed values if visible

**4. IDENTIFIERS AND REFERENCES - Extract All:**
   - Every part number, serial number, model number, item number
   - Every contract number, PO number, invoice number, order number
   - Every drawing number, revision number, sheet number
   - Every reference number, code, identifier, and classification
   - Every file number, case number, project number, job number
   - Explain what each identifier refers to

**5. NAMES AND ENTITIES - Extract All:**
   - Every person's name, title, role, department
   - Every company name, organization, division
   - Every address (street, city, state, zip, country)
   - Every contact detail (phone, email, website)
   - Every product name, material name, component name
   - Explain relationships between entities

**6. SPECIFICATIONS AND REQUIREMENTS - Extract All:**
   - Every specification, standard, requirement, and condition
   - Every term, clause, and provision
   - Every standard, code, regulation, or guideline referenced
   - Every tolerance, allowance, and limitation
   - Explain what each means and why it matters

**7. CONTEXT AND RELATIONSHIPS - Understand and Explain:**
   - Understand the document's purpose and context
   - Explain relationships between different parts
   - Explain how elements connect or relate to each other
   - Understand the overall structure and organization
   - Explain the significance and importance of details

**CRITICAL RULES:**
- Process EVERY page completely - do not skip anything
- Extract EVERYTHING - every word, every number, every detail
- Explain EVERYTHING - don't just list, help the reader understand
- Understand drawings like a smart human - extract meaning, not just appearance
- NO summarization - be thorough and complete
- NO omission - if it's in the document, include it
- Preserve exact wording for important terms, specifications, legal language
- Organize clearly but include every detail
- Think like an expert analyst - understand context, relationships, and implications

**Output Structure:**
Organize your comprehensive extraction clearly:

1. **Document Overview**
   - Document type, purpose, and context
   - All metadata (dates, reference numbers, revision info)
   - Key identifying information

2. **Complete Text Content**
   - Full text from all pages (with page references)
   - Explanation of each section's meaning and purpose
   - All tables, forms, and structured data completely extracted

3. **Complete Visual Analysis (Drawings, Diagrams, etc.)**
   - For each drawing/diagram:
     * What type of drawing it is and its purpose
     * Complete description of what it shows
     * Deep explanation of what it means (like a smart human explaining it)
     * All text, labels, annotations, dimensions extracted
     * Explanation of spatial relationships, connections, and layout
     * All measurements and dimensions with explanations
     * All symbols, legends, and notations explained
     * Overall interpretation and meaning

4. **Complete Data Extraction**
   - All numbers, measurements, quantities (with explanations)
   - All dates and timestamps
   - All financial figures
   - All identifiers and reference numbers

5. **Complete Entity Information**
   - All parties, names, addresses, contact details
   - All products, materials, components
   - Relationships and connections explained

6. **Complete Specifications and Requirements**
   - All terms, conditions, specifications
   - All standards, codes, regulations
   - All requirements and constraints

Remember: Approach this like a VERY SMART HUMAN who sees everything, understands everything, and explains everything thoroughly. Extract and explain every detail without omitting anything."""


def _upload_pdf_to_gemini(pdf_bytes: bytes, filename: str, api_key: str) -> Optional[str]:
    """
    Upload PDF file to Gemini File API using REST API.
    Returns file_uri if successful, None otherwise.
    """
    upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(pdf_bytes)),
        "X-Goog-Upload-Header-Content-Type": "application/pdf",
        "Content-Type": "application/json"
    }
    meta = {"file": {"display_name": filename}}
    
    try:
        # Start upload session
        resp = requests.post(upload_url, headers=headers, json=meta)
        resp.raise_for_status()
        session_uri = resp.headers.get("X-Goog-Upload-URL")
        
        if not session_uri:
            return None

        # Upload actual bytes
        upload_headers = {
            "Content-Length": str(len(pdf_bytes)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        
        resp = requests.post(session_uri, headers=upload_headers, data=pdf_bytes)
        resp.raise_for_status()
        
        result = resp.json()
        return result.get("file", {}).get("uri")
        
    except Exception as e:
        print(f"⚠️ PDF upload error for {filename}: {e}")
        return None


def _wait_for_gemini_file_active(file_uri: str, api_key: str, max_wait_seconds: int = 30) -> bool:
    """
    Poll the file endpoint until state is ACTIVE.
    Returns True when active, False if failed or timed out.
    """
    file_name = file_uri.split("/files/")[-1]
    url = f"https://generativelanguage.googleapis.com/v1beta/files/{file_name}?key={api_key}"
    
    max_retries = max_wait_seconds
    for _ in range(max_retries):
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            state = resp.json().get("state")
            
            if state == "ACTIVE":
                return True
            elif state == "FAILED":
                print(f"⚠️ File processing failed: {file_uri}")
                return False
            
            # Still PROCESSING
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error checking file state: {e}")
            return False
            
    print(f"⚠️ Timed out waiting for file to be active: {file_uri}")
    return False


def _describe_pdf_with_vision(pdf_bytes: bytes, filename: str) -> str:
    """
    Use Gemini File API to process PDF directly with comprehensive extraction.
    Extracts and explains everything like a very smart human would.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return f"[No Gemini API key - cannot process PDF: {filename}]"
    
    # Upload PDF to Gemini
    file_uri = _upload_pdf_to_gemini(pdf_bytes, filename, api_key)
    if not file_uri:
        return f"[Failed to upload PDF: {filename}]"
    
    # Wait for processing
    if not _wait_for_gemini_file_active(file_uri, api_key):
        return f"[File processing failed or timed out: {filename}]"
    
    # Build comprehensive extraction prompt
    prompt = _build_comprehensive_extraction_prompt(filename)
    
    # Generate content using REST API
    model = "gemini-3-flash-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": "application/pdf", "file_uri": file_uri}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2
        }
    }
    
    # IMPROVED RETRY LOGIC FOR RATE LIMITS
    max_retries = 8  # Increased from 3
    base_delay = 4   # Increased from 2
    
    import random
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload)
            
            # Handle rate limits (429)
            if resp.status_code == 429:
                if attempt < max_retries:
                    # Exponential backoff + Jitter
                    # 4s, 8s, 16s, 32s... plus random 0-2s
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 2)
                    print(f"⚠️ Rate limit (429) for '{filename}'. Retrying in {sleep_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                else:
                    return f"[Rate limit exceeded for {filename}]"
            
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response text
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                print(f"⚠️ Unexpected response structure for {filename}: {data}")
                return f"[Unexpected response structure for {filename}]"
    
        except Exception as e:
            if attempt < max_retries:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 2)
                print(f"⚠️ Error processing {filename}, retrying in {sleep_time:.1f}s: {e}")
                time.sleep(sleep_time)
            else:
                print(f"⚠️ Gemini processing failed for {filename}: {e}")
                return f"[Vision processing failed for {filename}: {str(e)}]"
    
    return f"[Failed to process PDF after {max_retries} retries: {filename}]"


# ============================================================
# PROJECT INDEXER CLASS
# ============================================================

class ProjectIndexer:
    """
    Indexes all emails for a project into comprehensive ThreadDocs.
    
    Phase 1: Data preparation for vectorization.
    """
    
    def __init__(
        self,
        project_name: str,
        user_email: str,
        provider: str = "gmail",
        credentials: Optional[Dict] = None,
        max_threads: Optional[int] = None,
        max_workers: int = 10
    ):
        """
        Initialize the project indexer.
        
        Args:
            project_name: Name of the project to index (e.g., "88 SuperMarket")
            user_email: User's email address
            provider: Email provider ("gmail" or "outlook")
            credentials: Optional credentials for web mode
            max_threads: Maximum number of threads to index (None = unlimited)
            max_workers: Number of parallel workers for downloads
        """
        self.project_name = project_name
        self.project_id = _normalize_project_id(project_name)
        self.user_email = user_email
        self.provider = provider.lower()
        self.credentials = credentials
        self.max_threads = max_threads
        self.max_workers = max_workers
        
        # Initialize email tools
        if self.provider == "gmail":
            self.email_tools: GmailTools = get_gmail_tools(credentials=credentials)
        else:
            raise NotImplementedError("Only Gmail is supported currently")
        
        # Stats
        self.stats = {
            "threads_found": 0,
            "threads_processed": 0,
            "messages_processed": 0,
            "pdfs_found": 0,
            "pdfs_text_extracted": 0,
            "pdfs_vision_processed": 0,
            "errors": []
        }
    
    def index_project(self, progress_callback=None) -> Tuple[ProjectIndex, AttachmentIndex]:
        """
        Main entry point: Index all emails for the project.
        
        Args:
            progress_callback: Optional function(message, percent) for progress updates
        
        Returns:
            Tuple[ProjectIndex, AttachmentIndex]
        """
        start_time = time.time()
        
        def _progress(msg: str, pct: int = 0):
            elapsed = time.time() - start_time
            print(f"\n📊 [{pct:3d}%] [{elapsed:6.1f}s] {msg}")
            if progress_callback:
                progress_callback(msg, pct)
        
        print("\n" + "="*60)
        print(f"🚀 PROJECT INDEXER - Starting")
        print("="*60)
        print(f"   Project:     {self.project_name}")
        print(f"   Project ID:  {self.project_id}")
        print(f"   User:        {self.user_email}")
        print(f"   Provider:    {self.provider}")
        print(f"   Max threads: {self.max_threads or 'unlimited'}")
        print(f"   Workers:     {self.max_workers}")
        print("="*60)
        
        _progress(f"Starting index for project: {self.project_name}", 0)
        
        # Step 1: Search for all emails with project name
        _progress("Searching emails...", 5)
        thread_ids = self._search_and_get_thread_ids()
        self.stats["threads_found"] = len(thread_ids)
        _progress(f"Found {len(thread_ids)} unique threads", 15)
        
        if not thread_ids:
            empty_idx = self._build_empty_index()
            empty_att_idx = AttachmentIndex(
                project_id=self.project_id,
                project_name=self.project_name,
                indexed_at=datetime.utcnow().isoformat() + "Z",
                total_attachments=0,
                attachments=[]
            )
            return empty_idx, empty_att_idx
        
        # Step 2: Fetch all threads in parallel
        _progress("Fetching thread contents...", 20)
        raw_threads = self._fetch_all_threads(thread_ids, _progress)
        _progress(f"Fetched {len(raw_threads)} threads", 40)
        
        # Step 3: Discover all PDF attachments
        _progress("Discovering PDF attachments...", 45)
        pdf_tasks = self._discover_pdf_attachments(raw_threads)
        self.stats["pdfs_found"] = len(pdf_tasks)
        _progress(f"Found {len(pdf_tasks)} PDFs to process", 50)
        
        # Step 4: Download and extract all PDFs in parallel
        _progress("Processing PDFs (this may take a while)...", 55)
        pdf_results = self._process_all_pdfs(pdf_tasks, _progress)
        _progress(f"Processed {len(pdf_results)} PDFs", 80)
        
        # Step 5: Build ThreadDocs
        _progress("Building ThreadDocs...", 85)
        thread_docs = self._build_thread_docs(raw_threads, pdf_results)
        self.stats["threads_processed"] = len(thread_docs)
        _progress(f"Built {len(thread_docs)} ThreadDocs", 90)

        # Step 6: Build Attachment Index
        _progress("Building Attachments Index...", 95)
        attachment_index = self._build_attachment_index(raw_threads, pdf_results)
        _progress(f"Built Attachments Index with {len(attachment_index.attachments)} items", 98)
        
        # Step 7: Build final project index
        last_timestamp = self._get_latest_timestamp(raw_threads)
        
        elapsed = time.time() - start_time
        _progress(f"Indexing complete in {elapsed:.1f}s", 100)
        
        # Print stats
        print(f"\n{'='*50}")
        print(f"📈 INDEXING STATS FOR: {self.project_name}")
        print(f"{'='*50}")
        print(f"Threads found:        {self.stats['threads_found']}")
        print(f"Threads processed:    {self.stats['threads_processed']}")
        print(f"Messages processed:   {self.stats['messages_processed']}")
        print(f"PDFs found:           {self.stats['pdfs_found']}")
        print(f"PDFs (text):          {self.stats['pdfs_text_extracted']}")
        print(f"PDFs (vision):        {self.stats['pdfs_vision_processed']}")
        print(f"Errors:               {len(self.stats['errors'])}")
        print(f"Total time:           {elapsed:.1f}s")
        print(f"{'='*50}\n")
        
        project_index = ProjectIndex(
            project_id=self.project_id,
            project_name=self.project_name,
            user_email=self.user_email,
            indexed_at=datetime.utcnow().isoformat() + "Z",
            last_email_timestamp=last_timestamp,
            thread_count=len(thread_docs),
            message_count=self.stats["messages_processed"],
            pdf_count=self.stats["pdfs_found"],
            thread_docs=thread_docs
        )
        
        return project_index, attachment_index
    
    def _search_and_get_thread_ids(self) -> List[str]:
        """Search for project emails and extract unique thread IDs."""
        # Strip any existing quotes, then wrap in quotes for exact phrase search
        # Note: Gmail search is case-insensitive by default, so "88 SuperMarket",
        # "88 supermarket", and "88 SUPERMARKET" will all match the same emails
        project_name_clean = self.project_name.strip().strip('"').strip("'")
        search_query = f'"{project_name_clean}"'
        print(f"   🔍 Searching Gmail for: {search_query}")
        all_thread_ids = set()
        page_cursor = None
        page_num = 0
        
        while True:
            page_num += 1
            print(f"      📄 Fetching search results page {page_num}...")
            
            result = self.email_tools.search_messages(
                query=search_query,
                top=100,  # Max per page
                page_cursor=page_cursor
            )
            
            if result.get("error"):
                print(f"      ❌ Search error: {result['error']}")
                self.stats["errors"].append(f"Search error: {result['error']}")
                break
            
            # Gmail returns 'emails', handle both formats
            messages = result.get("messages") or result.get("emails") or []
            before_count = len(all_thread_ids)
            
            for msg in messages:
                thread_id = msg.get("thread_id") or msg.get("threadId") or msg.get("conversationId")
                if thread_id:
                    all_thread_ids.add(thread_id)
            
            new_threads = len(all_thread_ids) - before_count
            print(f"      ✓ Page {page_num}: Found {len(messages)} emails, {new_threads} new threads (total: {len(all_thread_ids)})")
            
            # Check for more pages
            page_cursor = result.get("next_page_cursor") or result.get("page_cursor")
            has_more = result.get("has_more", False)
            if not page_cursor or not has_more:
                break
            if self.max_threads and len(all_thread_ids) >= self.max_threads:
                print(f"      ⚠️ Reached max threads limit ({self.max_threads})")
                break
        
        print(f"   ✅ Search complete: {len(all_thread_ids)} unique threads found")
        return list(all_thread_ids)[:self.max_threads] if self.max_threads else list(all_thread_ids)
    
    def _fetch_all_threads(self, thread_ids: List[str], progress_cb) -> List[Dict]:
        """Fetch all threads in parallel."""
        print(f"   📥 Fetching {len(thread_ids)} threads (using {self.max_workers} parallel workers)...")
        raw_threads = []
        errors = 0
        
        def fetch_one(tid: str) -> Optional[Dict]:
            try:
                return self.email_tools.get_conversation_threads(tid)
            except Exception as e:
                self.stats["errors"].append(f"Thread fetch error {tid}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_one, tid): tid for tid in thread_ids}
            done_count = 0
            
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                if result and not result.get("error"):
                    raw_threads.append(result)
                    # Log every 5 threads
                    if done_count % 5 == 0 or done_count == len(thread_ids):
                        print(f"      ✓ {done_count}/{len(thread_ids)} threads fetched ({len(raw_threads)} successful)")
                else:
                    errors += 1
                
                if done_count % 10 == 0:
                    pct = 20 + int((done_count / len(thread_ids)) * 20)
                    progress_cb(f"Fetched {done_count}/{len(thread_ids)} threads...", pct)
        
        print(f"   ✅ Thread fetch complete: {len(raw_threads)} successful, {errors} errors")
        return raw_threads
    
    def _discover_pdf_attachments(self, raw_threads: List[Dict]) -> List[Tuple[str, str, str, str]]:
        """
        Find all PDF attachments across all threads.
        
        Returns:
            List of (thread_id, message_id, attachment_id, filename)
        """
        print(f"   🔎 Scanning {len(raw_threads)} threads for PDF attachments...")
        pdf_tasks = []
        messages_with_attachments = 0
        total_attachments = 0
        
        for thread_idx, thread in enumerate(raw_threads, start=1):
            thread_id = thread.get("conversation_id") or thread.get("thread_id", "")
            messages = thread.get("messages") or []
            
            for msg in messages:
                msg_id = msg.get("id") or msg.get("message_id", "")
                if not msg_id:
                    continue
                
                # Check for attachments
                has_attachments = msg.get("has_attachments") or msg.get("hasAttachments")
                if not has_attachments:
                    continue
                
                messages_with_attachments += 1
                
                # List attachments
                try:
                    att_result = self.email_tools.list_attachments(msg_id)
                    attachments = att_result.get("attachments") or []
                    total_attachments += len(attachments)
                    
                    for att in attachments:
                        content_type = (att.get("contentType") or att.get("content_type") or "").lower()
                        filename = att.get("name") or att.get("filename") or ""
                        att_id = att.get("id") or att.get("attachment_id") or ""
                        
                        # Only PDFs
                        if "pdf" in content_type or filename.lower().endswith(".pdf"):
                            pdf_tasks.append((thread_id, msg_id, att_id, filename))
                
                except Exception as e:
                    self.stats["errors"].append(f"List attachments error {msg_id}: {e}")
            
            # Log progress every 10 threads
            if thread_idx % 10 == 0 or thread_idx == len(raw_threads):
                print(f"      ✓ Scanned {thread_idx}/{len(raw_threads)} threads, found {len(pdf_tasks)} PDFs so far")
        
        print(f"   ✅ Attachment scan complete:")
        print(f"      - Messages with attachments: {messages_with_attachments}")
        print(f"      - Total attachments found: {total_attachments}")
        print(f"      - PDFs to process: {len(pdf_tasks)}")
        
        return pdf_tasks
    
    def _process_all_pdfs(
        self,
        pdf_tasks: List[Tuple[str, str, str, str]],
        progress_cb
    ) -> Dict[Tuple[str, str], AttachmentDoc]:
        """
        Download and process all PDFs with hash-based deduplication.
        Downloads each PDF once, hashes it, and only processes unique content.
        
        Returns:
            Dict mapping (message_id, attachment_id) -> AttachmentDoc
        """
        if not pdf_tasks:
            print("   ⚠️ No PDFs to process")
            return {}
        
        # ============================================================
        # PHASE 1: Download all PDFs and compute hashes (fast, no API calls)
        # ============================================================
        print(f"   📥 Downloading {len(pdf_tasks)} PDFs and computing hashes...")
        
        # Store: task_index -> (pdf_bytes, content_hash) or None if error
        downloaded = {}
        # Store: content_hash -> first task_index (to identify which task is the "original")
        hash_to_first_task = {}
        
        for i, task in enumerate(pdf_tasks):
            thread_id, msg_id, att_id, filename = task
            
            try:
                dl_result = self.email_tools.download_attachment(msg_id, att_id)
                if dl_result.get("error"):
                    downloaded[i] = None
                    continue
                
                pdf_bytes = dl_result.get("content_bytes") or dl_result.get("content")
                if isinstance(pdf_bytes, str):
                    try:
                        pdf_bytes = base64.urlsafe_b64decode(pdf_bytes)
                    except Exception:
                        pdf_bytes = pdf_bytes.encode('utf-8')
                
                if not pdf_bytes:
                    downloaded[i] = None
                    continue
                
                content_hash = hashlib.sha256(pdf_bytes).hexdigest()
                downloaded[i] = (pdf_bytes, content_hash)
                
                # Track first occurrence of each hash
                if content_hash not in hash_to_first_task:
                    hash_to_first_task[content_hash] = i
                
            except Exception as e:
                self.stats["errors"].append(f"PDF download error {filename}: {e}")
                downloaded[i] = None
            
            if (i + 1) % 20 == 0 or i + 1 == len(pdf_tasks):
                print(f"      ✓ Downloaded {i + 1}/{len(pdf_tasks)}")
        
        # Calculate unique count
        unique_count = len(hash_to_first_task)
        successful_downloads = len([d for d in downloaded.values() if d is not None])
        duplicate_count = successful_downloads - unique_count
        error_download_count = len([d for d in downloaded.values() if d is None])
        
        print(f"   ✅ Download complete:")
        print(f"      - Total PDFs: {len(pdf_tasks)}")
        print(f"      - Unique PDFs: {unique_count}")
        print(f"      - Duplicates: {duplicate_count}")
        print(f"      - Download errors: {error_download_count}")
        
        if unique_count == 0:
            print("   ⚠️ No unique PDFs to process")
            return {}
        
        # ============================================================
        # PHASE 2: Process only unique PDFs (slow, API calls with rate limiting)
        # ============================================================
        print(f"   🤖 Processing {unique_count} unique PDFs with Gemini Vision...")
        
        results = {}
        processed_hashes = {}  # content_hash -> AttachmentDoc
        vision_count = 0
        error_count = 0
        start_time = time.time()
        
        # Process only the first occurrence of each unique hash
        unique_tasks = [(i, pdf_tasks[i]) for i in hash_to_first_task.values()]
        
        for processed_idx, (task_idx, task) in enumerate(unique_tasks, 1):
            thread_id, msg_id, att_id, filename = task
            pdf_bytes, content_hash = downloaded[task_idx]
            
            try:
                description = _describe_pdf_with_vision(pdf_bytes, filename)
                doc = AttachmentDoc(
                    attachment_id=att_id,
                    filename=filename,
                    content_type="application/pdf",
                    extraction_method="vision",
                    content=description
                )
                
                processed_hashes[content_hash] = doc
                results[(msg_id, att_id)] = doc
                vision_count += 1
                self.stats["pdfs_vision_processed"] += 1
                print(f"      👁 [{processed_idx}/{unique_count}] VISION: {filename[:50]}...")
                
            except Exception as e:
                self.stats["errors"].append(f"PDF process error {filename}: {e}")
                error_count += 1
                print(f"      ❌ [{processed_idx}/{unique_count}] ERROR: {filename[:50]}...")
            
            if processed_idx % 5 == 0:
                pct = 55 + int((processed_idx / unique_count) * 25)
                progress_cb(f"Processed {processed_idx}/{unique_count} unique PDFs...", pct)
            
            # Rate limit only between actual API calls
            if processed_idx < len(unique_tasks):
                time.sleep(5)
        
        # ============================================================
        # PHASE 3: Copy results to duplicates (instant, no API calls)
        # ============================================================
        duplicates_copied = 0
        if duplicate_count > 0:
            print(f"   ♻️  Copying results to {duplicate_count} duplicate PDFs...")
            
            for i, task in enumerate(pdf_tasks):
                thread_id, msg_id, att_id, filename = task
                key = (msg_id, att_id)
                
                if key in results:
                    continue  # Already processed
                
                if downloaded[i] is None:
                    continue  # Download failed
                
                pdf_bytes, content_hash = downloaded[i]
                
                if content_hash in processed_hashes:
                    # Copy from processed original
                    original_doc = processed_hashes[content_hash]
                    duplicate_doc = AttachmentDoc(
                        attachment_id=att_id,
                        filename=filename,
                        content_type=original_doc.content_type,
                        extraction_method=original_doc.extraction_method,
                        content=original_doc.content
                    )
                    results[key] = duplicate_doc
                    duplicates_copied += 1
        
        elapsed = time.time() - start_time
        print(f"   ✅ PDF processing complete in {elapsed:.1f}s:")
        print(f"      - Unique processed: {vision_count}")
        print(f"      - Duplicates copied: {duplicates_copied}")
        print(f"      - Errors: {error_count}")
        
        return results
    
    def _build_thread_docs(
        self,
        raw_threads: List[Dict],
        pdf_results: Dict[Tuple[str, str], AttachmentDoc]
    ) -> List[ThreadDoc]:
        """Build ThreadDoc objects from raw threads and processed PDFs."""
        print(f"   🔨 Building ThreadDocs from {len(raw_threads)} threads and {len(pdf_results)} PDFs...")
        thread_docs = []
        total_messages = 0
        total_pdfs_attached = 0
        
        for thread_idx, thread in enumerate(raw_threads, start=1):
            thread_id = thread.get("conversation_id") or thread.get("thread_id", "")
            messages = thread.get("messages") or []
            
            if not messages:
                continue
            
            # Build message docs
            message_docs = []
            participants = set()
            pdf_count = 0
            
            for seq, msg in enumerate(messages, start=1):
                msg_id = msg.get("id") or msg.get("message_id", "")
                from_email = msg.get("from_email") or msg.get("from") or ""
                from_name = msg.get("from_name") or ""
                
                # Track participants
                if from_email:
                    participants.add(from_email)
                
                # Parse to/cc
                to_list = []
                cc_list = []
                to_raw = msg.get("to") or ""
                cc_raw = msg.get("cc") or ""
                if isinstance(to_raw, str):
                    to_list = [e.strip() for e in to_raw.split(",") if e.strip()]
                elif isinstance(to_raw, list):
                    to_list = to_raw
                if isinstance(cc_raw, str):
                    cc_list = [e.strip() for e in cc_raw.split(",") if e.strip()]
                elif isinstance(cc_raw, list):
                    cc_list = cc_raw
                
                # Clean body
                body_raw = msg.get("body") or msg.get("body_html") or msg.get("bodyPreview") or ""
                body_clean = _clean_body(body_raw)
                
                # Get attachments for this message from pdf_results
                # The raw threads don't include attachment details, so we look up by message_id
                attachments = []
                for (result_msg_id, result_att_id), att_doc in pdf_results.items():
                    if result_msg_id == msg_id:
                        attachments.append(att_doc)
                        pdf_count += 1
                
                message_docs.append(MessageDoc(
                    sequence=seq,
                    message_id=msg_id,
                    timestamp=msg.get("receivedDateTime") or msg.get("received") or msg.get("date") or "",
                    from_email=from_email,
                    from_name=from_name,
                    to=to_list,
                    cc=cc_list,
                    subject=msg.get("subject") or "",
                    body=body_clean,
                    attachments=attachments
                ))
                
                self.stats["messages_processed"] += 1
            
            if not message_docs:
                continue
            
            # Get subject from first message with subject
            subject = ""
            for md in message_docs:
                if md.subject:
                    subject = md.subject
                    break
            
            # Get date range
            timestamps = [md.timestamp for md in message_docs if md.timestamp]
            date_first = min(timestamps) if timestamps else ""
            date_last = max(timestamps) if timestamps else ""
            
            thread_docs.append(ThreadDoc(
                thread_id=thread_id,
                project_id=self.project_id,
                subject=subject,
                date_first=date_first,
                date_last=date_last,
                participants=sorted(participants),
                message_count=len(message_docs),
                pdf_count=pdf_count,
                messages=message_docs
            ))
            
            total_messages += len(message_docs)
            total_pdfs_attached += pdf_count
            
            # Log progress every 10 threads
            if thread_idx % 10 == 0 or thread_idx == len(raw_threads):
                print(f"      ✓ Built {thread_idx}/{len(raw_threads)} ThreadDocs")
        
        print(f"   ✅ ThreadDoc building complete:")
        print(f"      - ThreadDocs created: {len(thread_docs)}")
        print(f"      - Total messages: {total_messages}")
        print(f"      - PDFs attached: {total_pdfs_attached}")
        
        return thread_docs
    
    def _build_attachment_index(
        self,
        raw_threads: List[Dict],
        pdf_results: Dict[Tuple[str, str], AttachmentDoc]
    ) -> AttachmentIndex:
        """Build the standalone attachment index."""
        attachments = []
        
        # Create lookups for message details
        msg_details = {}
        for thread in raw_threads:
            thread_id = thread.get("conversation_id") or thread.get("thread_id", "")
            for msg in thread.get("messages") or []:
                msg_id = msg.get("id") or msg.get("message_id", "")
                if msg_id:
                    to_raw = msg.get("to") or []
                    if isinstance(to_raw, str):
                        to_list = [e.strip() for e in to_raw.split(",") if e.strip()]
                    else:
                        to_list = to_raw

                    msg_details[msg_id] = {
                        "thread_id": thread_id,
                        "subject": msg.get("subject") or "",
                        "timestamp": msg.get("receivedDateTime") or msg.get("received") or msg.get("date") or "",
                        "from_email": msg.get("from_email") or msg.get("from") or "",
                        "from_name": msg.get("from_name") or "",
                        "to_list": to_list
                    }

        # Build index items from pdf_results
        for (msg_id, att_id), doc in pdf_results.items():
            details = msg_details.get(msg_id, {})
            
            attachments.append(AttachmentIndexItem(
                attachment_id=att_id,
                filename=doc.filename,
                content_type=doc.content_type,
                extraction_method=doc.extraction_method,
                sender_email=details.get("from_email", ""),
                sender_name=details.get("from_name", ""),
                receiver_emails=details.get("to_list", []),
                email_subject=details.get("subject", ""),
                email_timestamp=details.get("timestamp", ""),
                message_id=msg_id,
                thread_id=details.get("thread_id", ""),
                ai_analysis=doc.content
            ))
            
        return AttachmentIndex(
            project_id=self.project_id,
            project_name=self.project_name,
            indexed_at=datetime.utcnow().isoformat() + "Z",
            total_attachments=len(attachments),
            attachments=attachments
        )
    
    def _get_latest_timestamp(self, raw_threads: List[Dict]) -> str:
        """Get the latest email timestamp for incremental updates."""
        latest = ""
        for thread in raw_threads:
            for msg in thread.get("messages") or []:
                ts = msg.get("receivedDateTime") or msg.get("received") or msg.get("date") or ""
                if ts > latest:
                    latest = ts
        return latest
    
    def _build_empty_index(self) -> ProjectIndex:
        """Build an empty index when no threads found."""
        return ProjectIndex(
            project_id=self.project_id,
            project_name=self.project_name,
            user_email=self.user_email,
            indexed_at=datetime.utcnow().isoformat() + "Z",
            last_email_timestamp="",
            thread_count=0,
            message_count=0,
            pdf_count=0,
            thread_docs=[]
        )
    
    def save_index(self, index: ProjectIndex, output_dir: str = "project_indexes") -> str:
        """
        Save the project index to disk.
        
        Args:
            index: The ProjectIndex to save
            output_dir: Directory to save in
        
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{index.project_id}_threads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert dataclasses to dicts for JSON serialization
        def _to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(_to_dict(index), f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved THREADS index to: {filepath}")
        return filepath

    def save_attachments_index(self, index: AttachmentIndex, output_dir: str = "project_indexes") -> str:
        """
        Save the attachments index to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{index.project_id}_attachments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        def _to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(_to_dict(index), f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saved ATTACHMENTS index to: {filepath}")
        return filepath
    
    def save_as_pdf(self, index: ProjectIndex, output_dir: str = "project_indexes") -> str:
        """
        Save the project index as a readable PDF report.
        
        Args:
            index: The ProjectIndex to save
            output_dir: Directory to save in
        
        Returns:
            Path to saved PDF file
        """
        print(f"\n📝 Generating THREADS PDF report...")
        print(f"   - Threads: {index.thread_count}")
        print(f"   - Messages: {index.message_count}")
        print(f"   - PDFs: {index.pdf_count}")
        
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{index.project_id}_threads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        )
        
        thread_style = ParagraphStyle(
            'Thread',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            textColor=colors.darkgreen,
            backColor=colors.lightgrey
        )
        
        message_style = ParagraphStyle(
            'Message',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            spaceBefore=10,
            leftIndent=10,
            textColor=colors.black
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=5,
            leftIndent=20,
            textColor=colors.darkgrey
        )
        
        attachment_style = ParagraphStyle(
            'Attachment',
            parent=styles['Normal'],
            fontSize=8,
            spaceAfter=3,
            leftIndent=30,
            backColor=colors.beige,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=5
        )
        
        meta_style = ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            leftIndent=20
        )
        
        # Build content
        content = []
        
        # Title
        content.append(Paragraph(f"📁 PROJECT INDEX: {index.project_name}", title_style))
        content.append(Spacer(1, 10))
        
        # Summary table
        summary_data = [
            ["Project ID", index.project_id],
            ["User Email", index.user_email],
            ["Indexed At", index.indexed_at],
            ["Last Email", index.last_email_timestamp or "N/A"],
            ["Threads", str(index.thread_count)],
            ["Messages", str(index.message_count)],
            ["PDFs Extracted", str(index.pdf_count)]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        content.append(summary_table)
        content.append(Spacer(1, 20))
        
        # Threads
        print(f"   Building PDF content for {len(index.thread_docs)} threads...")
        for thread_idx, thread in enumerate(index.thread_docs, start=1):
            if thread_idx % 10 == 0 or thread_idx == len(index.thread_docs):
                print(f"      ✓ Processing thread {thread_idx}/{len(index.thread_docs)}")
            
            # Thread header
            content.append(Paragraph(
                f"📧 THREAD {thread_idx}: {self._escape_html(thread.subject[:80])}{'...' if len(thread.subject) > 80 else ''}",
                thread_style
            ))
            
            # Thread meta
            participants_str = ", ".join(thread.participants[:5])
            if len(thread.participants) > 5:
                participants_str += f" (+{len(thread.participants) - 5} more)"
            
            content.append(Paragraph(
                f"<b>Participants:</b> {self._escape_html(participants_str)}<br/>"
                f"<b>Messages:</b> {thread.message_count} | <b>PDFs:</b> {thread.pdf_count}<br/>"
                f"<b>Date Range:</b> {thread.date_first} → {thread.date_last}",
                meta_style
            ))
            content.append(Spacer(1, 10))
            
            # Messages in thread
            for msg in thread.messages:
                # Message header
                content.append(Paragraph(
                    f"<b>[Message {msg.sequence}]</b> {msg.timestamp}<br/>"
                    f"<b>From:</b> {self._escape_html(msg.from_name or msg.from_email)}<br/>"
                    f"<b>To:</b> {self._escape_html(', '.join(msg.to[:3]))}{'...' if len(msg.to) > 3 else ''}",
                    message_style
                ))
                
                # Message body (Full content)
                if msg.body:
                    content.append(Paragraph(self._escape_html(msg.body), body_style))
                
                # Attachments
                for att in msg.attachments:
                    content.append(Spacer(1, 5))
                    content.append(Paragraph(
                        f"📎 <b>ATTACHMENT:</b> {self._escape_html(att.filename)}<br/>"
                        f"<i>Extraction: {att.extraction_method}</i>",
                        message_style
                    ))
                    
                    # Attachment content (Full content)
                    if att.content:
                        content.append(Paragraph(
                            f"<font size=7>{self._escape_html(att.content)}</font>",
                            attachment_style
                        ))
                
                content.append(Spacer(1, 10))
            
            # Page break between threads (every 3 threads)
            if thread_idx % 3 == 0 and thread_idx < len(index.thread_docs):
                content.append(PageBreak())
        
        # Build PDF
        doc.build(content)
        
        print(f"📄 Saved THREADS PDF report to: {filepath}")
        return filepath

    def save_attachments_pdf(self, index: AttachmentIndex, output_dir: str = "project_indexes") -> str:
        """
        Save the attachments index as a standalone PDF 'Book of Attachments'.
        """
        print(f"\n📝 Generating ATTACHMENTS PDF report...")
        print(f"   - Total Attachments: {index.total_attachments}")
        
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{index.project_id}_attachments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'Title', parent=styles['Heading1'], fontSize=24, 
            alignment=1, textColor=colors.darkblue, spaceAfter=20
        )
        
        header_style = ParagraphStyle(
            'Header', parent=styles['Heading2'], fontSize=16, 
            textColor=colors.darkred, spaceBefore=15, spaceAfter=10
        )
        
        meta_style = ParagraphStyle(
            'Meta', parent=styles['Normal'], fontSize=10, 
            textColor=colors.black, spaceAfter=5, leftIndent=10
        )
        
        content_style = ParagraphStyle(
            'Content', parent=styles['Normal'], fontSize=9, 
            textColor=colors.darkgrey, spaceBefore=5, leftIndent=20
        )
        
        story = []
        story.append(Paragraph(f"📎 ATTACHMENTS INDEX: {index.project_name}", title_style))
        story.append(Spacer(1, 20))
        
        for idx, att in enumerate(index.attachments, 1):
            story.append(Paragraph(f"{idx}. {self._escape_html(att.filename)}", header_style))
            
            # Metadata
            meta_text = (
                f"<b>Sender:</b> {self._escape_html(att.sender_name)} ({self._escape_html(att.sender_email)})<br/>"
                f"<b>Receiver:</b> {self._escape_html(', '.join(att.receiver_emails[:3]))}<br/>"
                f"<b>Date:</b> {att.email_timestamp}<br/>"
                f"<b>Subject:</b> {self._escape_html(att.email_subject)}"
            )
            story.append(Paragraph(meta_text, meta_style))
            story.append(Spacer(1, 10))
            
            # AI Analysis (Full content)
            story.append(Paragraph("<b>🤖 AI Analysis:</b>", meta_style))
            story.append(Paragraph(self._escape_html(att.ai_analysis), content_style))
            story.append(PageBreak())
            
        doc.build(story)
        print(f"📄 Saved ATTACHMENTS PDF report to: {filepath}")
        return filepath
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for ReportLab."""
        if not text:
            return ""
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('\n', '<br/>')
            .replace('\t', '    ')
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def index_project(
    project_name: str,
    user_email: str,
    provider: str = "gmail",
    credentials: Optional[Dict] = None,
    save: bool = True,
    save_pdf: bool = False
) -> Tuple[ProjectIndex, AttachmentIndex]:
    """
    Convenience function to index a project.
    
    Args:
        project_name: Name of the project (e.g., "88 SuperMarket")
        user_email: User's email address
        provider: Email provider ("gmail")
        credentials: Optional credentials for web mode
        save: Whether to save the index to disk (JSON)
        save_pdf: Whether to save a PDF report for verification
    
    Returns:
        Tuple[ProjectIndex, AttachmentIndex]
    """
    indexer = ProjectIndexer(
        project_name=project_name,
        user_email=user_email,
        provider=provider,
        credentials=credentials
    )
    
    index, att_index = indexer.index_project()
    
    if save:
        indexer.save_index(index)
        indexer.save_attachments_index(att_index)
    
    if save_pdf:
        indexer.save_as_pdf(index)
        indexer.save_attachments_pdf(att_index)
    
    return index, att_index


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python project_indexer.py <project_name> [user_email] [--pdf]")
        print("Example: python project_indexer.py '88 SuperMarket' harv@buildsmartr.com --pdf")
        print("\nOptions:")
        print("  --pdf    Also generate a PDF report for verification")
        sys.exit(1)
    
    # Parse args
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    
    project = args[0] if args else ""
    email = args[1] if len(args) > 1 else "user@example.com"
    generate_pdf = "--pdf" in flags
    
    if not project:
        print("Error: Project name is required")
        sys.exit(1)
    
    print(f"\n🚀 Indexing project: {project}")
    print(f"   User: {email}")
    print(f"   PDF Report: {'Yes' if generate_pdf else 'No'}\n")
    
    result, att_result = index_project(project, email, save=True, save_pdf=generate_pdf)
    
    print(f"\n✅ Done! Indexed {result.thread_count} threads with {result.pdf_count} PDFs")
    print(f"✅ Attachments index created with {att_result.total_attachments} items")

