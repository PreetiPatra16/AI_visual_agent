import os
import traceback
import base64
from groq import Groq
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# INIT CLIENTS
# -------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# -------------------------------
# FORMAT OUTPUT
# -------------------------------
def format_llm_output(text: str) -> str:
    if not text:
        return "No analysis generated."

    text = text.replace("**", "").strip()

    sections = {
        "Overall Assessment": "",
        "Key Observations": "",
        "Impact Analysis": "",
        "Final Verdict": ""
    }

    current = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        for key in sections:
            if key.lower() in line.lower():
                current = key
                break
        else:
            if current:
                sections[current] += line + "\n"

    result = ""
    for key, value in sections.items():
        if value.strip():
            result += f"{key}:\n{value.strip()}\n\n"

    return result.strip() or text


# -------------------------------
# MAIN LLM FUNCTION
# -------------------------------
def generate_llm_report(result, summary_text, detailed_report, baseline_bytes, current_bytes):

    # -------------------------------
    # PURE VISUAL PROMPT
    # -------------------------------
    prompt = """
You are a senior QA engineer performing visual regression testing.

You are given TWO UI screenshots:
1. Baseline (expected UI)
2. Current (updated UI)

Your job is to compare them and explain ONLY what changed.

STRICT RULES:
- DO NOT describe images separately
- DO NOT use structured data
- IGNORE pixel-level noise

Focus ONLY on meaningful UI differences:
- new sections added
- removed elements
- new labels / buttons / fields
- layout expansion or compression
- visible UI content changes

Think like a human comparing two screens side-by-side.

OUTPUT FORMAT:

Overall Assessment:
Key Observations:
Impact Analysis:
Final Verdict:
"""

    # -------------------------------
    # ENCODE IMAGES 
    # -------------------------------
    baseline_b64 = base64.b64encode(baseline_bytes).decode("utf-8")
    current_b64 = base64.b64encode(current_bytes).decode("utf-8")

    baseline_image = f"data:image/png;base64,{baseline_b64}"
    current_image = f"data:image/png;base64,{current_b64}"

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": baseline_image}},
        {"type": "image_url", "image_url": {"url": current_image}},
    ]

    # -------------------------------
    # GROQ CALL
    # -------------------------------
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=1200
        )

        output = response.choices[0].message.content

        if output:
            return format_llm_output(output)

    except Exception:
        print("\nGROQ FAILED:")
        traceback.print_exc()

    # -------------------------------
    # GEMINI FALLBACK
    # -------------------------------
    try:
        response = gemini_model.generate_content(prompt)
        return format_llm_output(response.text)

    except Exception:
        print("\nGEMINI FAILED:")
        traceback.print_exc()

        return "LLM report generation failed."