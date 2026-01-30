"""
Azure Function: generate_quotes

HTTP Trigger for generating vendor quotes using LLM.
Parses vendor pricing rules and calculates quotes for a project.
"""
import logging
import sys
import os
import json
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

logger = logging.getLogger(__name__)

# LLM Configuration
QUOTE_MODEL = "gemini-2.0-flash"  # Fast model for quote generation
QUOTE_TEMPERATURE = 0.1  # Low temp for consistent calculations


def get_gemini_client():
    """Get the Gemini client."""
    from google import genai
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    return genai.Client(api_key=api_key)


def build_quote_prompt(
    segment: str,
    segment_name: str,
    project_sqft: int,
    city: str,
    region: str,
    country: str,
    options: Dict,
    vendors: List[Dict]
) -> str:
    """Build the LLM prompt for quote generation."""
    
    # Format vendor information
    vendor_sections = []
    for i, vendor in enumerate(vendors, 1):
        vendor_section = f"""
---
Vendor #{i}: {vendor.get('company_name', 'Unknown')}
Email: {vendor.get('user_email', 'N/A')}
Pricing Rules: "{vendor.get('pricing_rules', 'No pricing rules provided')}"
Lead Time: {vendor.get('lead_time', 'Not specified')}
Notes: {vendor.get('notes', 'None')}
---"""
        vendor_sections.append(vendor_section)
    
    vendors_text = "\n".join(vendor_sections)
    
    # Format options
    options_text = ", ".join(f"{k}: {v}" for k, v in options.items()) if options else "Standard"
    
    prompt = f"""You are a construction cost estimator. Parse vendor pricing rules and calculate accurate quotes.

PROJECT DETAILS:
- Segment: {segment_name}
- Project Size: {project_sqft:,} square feet
- Location: {city or 'Unknown city'}, {region}, {country}
- Options: {options_text}

VENDORS TO QUOTE:
{vendors_text}

INSTRUCTIONS:
For EACH vendor, carefully analyze their pricing rules and calculate:

1. base_rate_per_sf: Extract the base rate per square foot from their rules
2. adjustments: Apply ANY modifiers that match the project:
   - Location adjustments (city, region matches)
   - Option adjustments (premium, standard, etc.)
   - Size-based adjustments
3. final_rate_per_sf: base_rate + sum of adjustment deltas
4. total: final_rate_per_sf × project_sqft
   - If minimum charge is stated and total < minimum, use minimum
5. lead_time: Extract from vendor info
6. notes: List any important conditions, exclusions, or notes

CALCULATION RULES:
- Be precise with math
- If a location modifier applies (e.g., "Richmond +$2/sf"), apply it
- If no matching modifiers, don't add any adjustments
- Respect minimum charges
- If pricing rules are unclear, make reasonable estimates and note uncertainty

OUTPUT FORMAT - Return ONLY valid JSON array:
[
  {{
    "company_name": "Vendor Name",
    "user_email": "vendor@email.com",
    "base_rate_per_sf": 14.00,
    "adjustments": [
      {{"reason": "Richmond surcharge", "delta_per_sf": 2.00}}
    ],
    "final_rate_per_sf": 16.00,
    "total": 128000,
    "lead_time": "3-4 weeks",
    "notes": ["Excludes tax", "Minimum $5,000 met"]
  }}
]

Generate quotes for all {len(vendors)} vendors:"""

    return prompt


def parse_llm_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into structured vendor quotes."""
    
    # Clean up response - find JSON array
    text = response_text.strip()
    
    # Try to extract JSON array from response
    start_idx = text.find('[')
    end_idx = text.rfind(']') + 1
    
    if start_idx == -1 or end_idx == 0:
        logger.error(f"No JSON array found in response: {text[:200]}")
        return []
    
    json_text = text[start_idx:end_idx]
    
    try:
        quotes = json.loads(json_text)
        
        # Validate and clean each quote
        validated_quotes = []
        for quote in quotes:
            validated = {
                "company_name": quote.get("company_name", "Unknown"),
                "user_email": quote.get("user_email", ""),
                "base_rate_per_sf": float(quote.get("base_rate_per_sf", 0)),
                "adjustments": quote.get("adjustments", []),
                "final_rate_per_sf": float(quote.get("final_rate_per_sf", 0)),
                "total": float(quote.get("total", 0)),
                "lead_time": quote.get("lead_time", "Not specified"),
                "notes": quote.get("notes", [])
            }
            
            # Ensure notes is a list
            if isinstance(validated["notes"], str):
                validated["notes"] = [validated["notes"]]
            
            validated_quotes.append(validated)
        
        return validated_quotes
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}\nResponse: {json_text[:500]}")
        return []


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    POST /api/generate_quotes
    
    Generate vendor quotes using LLM.
    
    Request body:
    {
        "segment": "windows_exterior_doors",
        "segment_name": "Window & exterior door supplier/installer",
        "project_sqft": 8000,
        "city": "Richmond",
        "region": "BC",
        "country": "CA",
        "options": {"finish": "standard"},
        "vendors": [
            {
                "company_name": "ABC Windows",
                "user_email": "abc@windows.com",
                "pricing_rules": "Base $14/sf. Richmond +$2. Min $5k.",
                "lead_time": "3-4 weeks",
                "notes": "Excludes tax"
            }
        ]
    }
    
    Returns:
    {
        "vendor_quotes": [
            {
                "company_name": "ABC Windows",
                "user_email": "abc@windows.com",
                "base_rate_per_sf": 14,
                "adjustments": [{"reason": "Richmond surcharge", "delta_per_sf": 2}],
                "final_rate_per_sf": 16,
                "total": 128000,
                "lead_time": "3-4 weeks",
                "notes": ["Excludes tax", "Min $5k met"]
            }
        ]
    }
    """
    logger.info("generate_quotes function called")
    
    try:
        # Parse request
        try:
            body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract parameters
        segment = body.get("segment")
        segment_name = body.get("segment_name", segment)
        project_sqft = body.get("project_sqft")
        city = body.get("city")
        region = body.get("region")
        country = body.get("country", "CA")
        options = body.get("options", {})
        vendors = body.get("vendors", [])
        
        # Validate required fields
        if not segment:
            return func.HttpResponse(
                json.dumps({"error": "segment is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not project_sqft or project_sqft <= 0:
            return func.HttpResponse(
                json.dumps({"error": "Valid project_sqft is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not vendors:
            return func.HttpResponse(
                json.dumps({"vendor_quotes": [], "message": "No vendors provided"}),
                status_code=200,
                mimetype="application/json"
            )
        
        # Build prompt
        prompt = build_quote_prompt(
            segment=segment,
            segment_name=segment_name,
            project_sqft=project_sqft,
            city=city,
            region=region,
            country=country,
            options=options,
            vendors=vendors
        )
        
        logger.info(f"Generating quotes for {len(vendors)} vendors, {project_sqft} sqft")
        
        # Call LLM
        try:
            client = get_gemini_client()
            
            response = client.models.generate_content(
                model=QUOTE_MODEL,
                contents=prompt,
                config={
                    "temperature": QUOTE_TEMPERATURE,
                    "max_output_tokens": 2000
                }
            )
            
            response_text = response.text
            logger.info(f"LLM response length: {len(response_text)}")
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"LLM call failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )
        
        # Parse response
        vendor_quotes = parse_llm_response(response_text)
        
        logger.info(f"Generated {len(vendor_quotes)} quotes successfully")
        
        return func.HttpResponse(
            json.dumps({
                "vendor_quotes": vendor_quotes,
                "vendors_processed": len(vendors),
                "quotes_generated": len(vendor_quotes)
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in generate_quotes: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
