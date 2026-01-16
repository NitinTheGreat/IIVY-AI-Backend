"""
PDF tools for Donna SUPERHUMAN.

Two-layer approach:
1. Skimmer - Fast, lightweight summarization for triage (2-3 sentences)
2. Deep Reader - Detailed analysis for specific questions (comprehensive)
"""
import requests
import json
import base64
import time
from typing import Dict, Any, Optional
from config import GOOGLE_API_KEY, PDF_SKIMMER_MODEL, PDF_SKIMMER_TEMPERATURE, PDF_READER_MODEL, PDF_READER_TEMPERATURE


class PDFSkimmer:
    """
    Two-tier PDF analysis tool using Gemini REST API (Thread-Safe).
    
    Bypasses the google.generativeai Python client to avoid gRPC crashes 
    in multi-threaded environments. Uses standard requests for full parallelism.
    """
    
    def __init__(self):
        """Initialize configuration."""
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required.")
        
        self.api_key = GOOGLE_API_KEY
        self.skimmer_model = PDF_SKIMMER_MODEL
        self.skimmer_temp = PDF_SKIMMER_TEMPERATURE
        self.reader_model = PDF_READER_MODEL
        self.reader_temp = PDF_READER_TEMPERATURE
        self.base_url = "https://generativelanguage.googleapis.com"

    def skim_pdf(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """Skim a PDF using Local pypdf (fast) -> Gemini REST API (fallback)."""
        try:
            # 1. Try Local Extraction First (Lightning Fast)
            local_text = self._local_extract_text(pdf_bytes)
            if local_text and len(local_text.strip()) > 100:
                # OPTIMIZATION: Truncate to first 12k chars (~3-4 pages) to reduce LLM latency
                # Invoices/Contracts almost always have key info on Page 1.
                truncated_text = local_text[:12000]
                if len(local_text) > 12000:
                    truncated_text += "\n...[Text Truncated for Speed]..."
                
                print(f"Local PDF Text Extracted ({len(local_text)} chars) -> Truncated to {len(truncated_text)}")
                
                # Wrap in a summary-like structure
                return {
                    "summary": f"**Content Extracted Locally:**\n{truncated_text}",
                    "success": True,
                    "size_mb": len(pdf_bytes) / (1024 * 1024),
                    "method": "local_pypdf"
                }

            # 2. Fallback to Gemini (Slower, Vision-capable)
            print("Local extraction failed/empty. Falling back to Gemini Vision.")
            
            # Upload File via REST
            file_uri = self._upload_file_rest(pdf_bytes, filename)
            if not file_uri:
                return {"error": "Failed to upload file to Gemini", "success": False}

            # Wait for Processing (CRITICAL)
            if not self._wait_for_file_active(file_uri):
                 return {"error": "File processing failed or timed out", "success": False}

            # Generate Content via REST
            prompt = self._build_skim_prompt(filename)
            summary = self._generate_content_rest(
                self.skimmer_model, 
                prompt, 
                file_uri,
                self.skimmer_temp
            )

            if not summary:
                return {"error": "Empty response from Gemini", "success": False}

            return {
                "summary": summary,
                "success": True,
                "size_mb": len(pdf_bytes) / (1024 * 1024),
                "method": "gemini_api"
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def read_pdf(self, pdf_bytes: bytes, filename: str = "document.pdf", question: Optional[str] = None) -> Dict[str, Any]:
        """Deep read a PDF using Local pypdf (fast) -> Gemini REST API (fallback)."""
        try:
            # 1. Try Local Extraction First
            local_text = self._local_extract_text(pdf_bytes)
            if local_text and len(local_text.strip()) > 100:
                print(f"Local PDF Text Extracted ({len(local_text)} chars)")
                return {
                    "analysis": f"**Full Text Extracted Locally:**\n{local_text}",
                    "key_details": {}, # TODO: could implement local regex extraction here
                    "success": True,
                    "size_mb": len(pdf_bytes) / (1024 * 1024),
                    "model_used": "local_pypdf",
                    "method": "local_pypdf"
                }

            # 2. Fallback to Gemini
            print("Local extraction failed/empty. Falling back to Gemini Vision.")
            
            # Upload File via REST
            file_uri = self._upload_file_rest(pdf_bytes, filename)
            if not file_uri:
                return {"error": "Failed to upload file to Gemini", "success": False}

            # Wait for Processing (CRITICAL)
            if not self._wait_for_file_active(file_uri):
                 return {"error": "File processing failed or timed out", "success": False}

            # Generate Content via REST
            prompt = self._build_read_prompt(filename, question)
            analysis = self._generate_content_rest(
                self.reader_model, 
                prompt, 
                file_uri, 
                self.reader_temp
            )

            if not analysis:
                return {"error": "Empty response from Gemini", "success": False}

            # Extract Details locally
            key_details = self._extract_key_details(analysis)

            return {
                "analysis": analysis,
                "key_details": key_details,
                "success": True,
                "size_mb": len(pdf_bytes) / (1024 * 1024),
                "model_used": self.reader_model,
                "method": "gemini_api"
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def _local_extract_text(self, pdf_bytes: bytes) -> Optional[str]:
        """Attempt to extract text locally using pypdf."""
        import io
        from pypdf import PdfReader
        
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text if text.strip() else None
        except Exception as e:
            print(f"Local PDF Extraction Error: {e}")
            return None

    def _upload_file_rest(self, pdf_bytes: bytes, filename: str) -> Optional[str]:
        """Upload file using the Media Upload REST API."""
        # 1. Initial Resumable Upload Request
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
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

            # 2. Upload Actual Bytes
            upload_headers = {
                "Content-Length": str(len(pdf_bytes)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize"
            }
            
            # Send bytes
            resp = requests.post(session_uri, headers=upload_headers, data=pdf_bytes)
            resp.raise_for_status()
            
            # Parse result
            result = resp.json()
            return result.get("file", {}).get("uri")
            
        except Exception as e:
            print(f"REST Upload Error: {e}")
            return None

    def _wait_for_file_active(self, file_uri: str) -> bool:
        """Poll the file endpoint until state is ACTIVE."""
        # Extract file name from URI: https://.../files/NAME
        file_name = file_uri.split("/files/")[-1]
        url = f"https://generativelanguage.googleapis.com/v1beta/files/{file_name}?key={self.api_key}"
        
        max_retries = 10
        for _ in range(max_retries):
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                state = resp.json().get("state")
                
                if state == "ACTIVE":
                    return True
                elif state == "FAILED":
                    print(f"File processing failed: {file_uri}")
                    return False
                
                # Still PROCESSING
                time.sleep(1)
            except Exception as e:
                print(f"Error checking file state: {e}")
                return False
                
        print(f"Timed out waiting for file to be active: {file_uri}")
        return False

    def _generate_content_rest(self, model: str, prompt: str, file_uri: str, temperature: float) -> Optional[str]:
        """Generate content using the REST API with retry for rate limits."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"file_data": {"mime_type": "application/pdf", "file_uri": file_uri}}
                ]
            }],
            "generationConfig": {
                "temperature": temperature
            }
        }
        
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(url, json=payload)
                
                # Check for Rate Limit (429)
                if resp.status_code == 429:
                    if attempt < max_retries:
                        # Exponential backoff: 2s, 4s, 8s...
                        sleep_time = base_delay * (2 ** attempt)
                        # Add jitter to prevent thundering herd
                        import random
                        sleep_time += random.uniform(0, 1)
                        
                        print(f"REST Gen Rate Limit (429). Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        print("REST Gen Failed: Max retries exceeded for 429.")
                        return None

                resp.raise_for_status()
                
                data = resp.json()
                # Safely extract text
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    print(f"REST Gen: Unexpected structure or blocked: {data}")
                    return None
                    
            except Exception as e:
                print(f"REST Gen Error: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Response: {e.response.text}")
                return None
        
        return None
    
    def _build_skim_prompt(self, filename: str) -> str:
        """Build simple skim prompt."""
        return f"""Analyze this PDF attachment ("{filename}").
Provide a concise summary (2-3 sentences) of what this document is about.
Identify key entities (companies, people) and dates if visible.
If it looks like an invoice, quote, or contract, state that clearly."""

    def _build_read_prompt(self, filename: str, question: Optional[str] = None) -> str:
        """
        Build the deep reading prompt for Gemini Pro.
        """
        base_prompt = f"""You are performing a detailed analysis of this PDF attachment ("{filename}").

**Your task:**
1. Identify the document type precisely
2. Extract ALL key information comprehensively:
   - Parties involved (full names, addresses, contact info)
   - Dates (all mentioned dates - effective dates, due dates, expiration, etc.)
   - Amounts (all financial figures with currency)
   - Terms and conditions (key clauses, obligations, restrictions)
   - Deliverables or items (what's being provided/exchanged)
   - Any deadlines, milestones, or action items
   - Important references (contract numbers, PO numbers, invoice numbers, etc.)

**Rules:**
- Analyze the ENTIRE document (all pages) - this is a deep read, not a skim
- Be thorough and detailed - extract ALL relevant information
- Organize information logically (by category)
- Use clear formatting (bullet points, sections)
- Include specific numbers, dates, and names (don't summarize)
- Highlight any unusual clauses, risks, or important notices

"""
        
        if question:
            base_prompt += f"""**Specific Question to Answer:**
{question}

Make sure to address this question directly in your analysis, in addition to providing comprehensive document details.

"""
        
        base_prompt += """**Output Format:**
Structure your analysis clearly with sections like:
- Document Type
- Parties Involved
- Financial Details
- Key Dates
- Terms & Conditions
- Deliverables/Items
- Action Items/Deadlines
- Important Notes

Begin your analysis now:"""
        
        return base_prompt
    
    def _extract_key_details(self, analysis: str) -> Dict[str, Any]:
        """
        Extract structured key details from the analysis text.
        """
        details = {}
        
        # Extract amounts (basic pattern matching)
        import re
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', analysis)
        if amounts:
            details["amounts"] = amounts
        
        # Extract dates (basic pattern matching)
        dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', analysis)
        if dates:
            details["dates"] = dates
        
        return details


# Singleton instance
_pdf_skimmer_instance = None

def get_pdf_skimmer() -> PDFSkimmer:
    """Get or create the PDF skimmer singleton."""
    global _pdf_skimmer_instance
    if _pdf_skimmer_instance is None:
        _pdf_skimmer_instance = PDFSkimmer()
    return _pdf_skimmer_instance
