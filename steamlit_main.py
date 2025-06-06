import streamlit as st
import json
import re
from openai import OpenAI
from typing import List, Dict, Any
import weaviate
from weaviate.classes.init import Auth, Timeout, AdditionalConfig

# Page configuration
st.set_page_config(
    page_title="Indian Income Tax Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'reasoning_content' not in st.session_state:
    st.session_state['reasoning_content'] = ""
if 'answer_content' not in st.session_state:
    st.session_state['answer_content'] = ""
if 'extracted_sections' not in st.session_state:
    st.session_state['extracted_sections'] = []
if 'combined_sections' not in st.session_state:
    st.session_state['combined_sections'] = []
if 'section_contents' not in st.session_state:
    st.session_state['section_contents'] = {}
if 'legal_analysis' not in st.session_state:
    st.session_state['legal_analysis'] = ""

# Hardcoded configuration values
DEEPSEEK_API_KEY = "sk-7440c63fd9684aff921335cea5f08a84"  # Replace with your actual API key
DEEPSEEK_MODEL_NAME = "deepseek-reasoner"
WEAVIATE_HOST = "https://18ggj3djrq2x8oco7fqxa.c0.asia-southeast1.gcp.weaviate.cloud"  # Replace with your Weaviate host
WEAVIATE_API_KEY = "2QBDWwNyIulMQUZ1wp4CLZx0aoLWGiIxHSF1"  # Replace with your actual API key
WEAVIATE_CLASS_NAME = "Income_Tax_Sections_Key_Value"  # Your class name in Weaviate

# UI Elements
st.title("Indian Income Tax Assistant")
st.subheader(
    "An AI assistant that extracts relevant section numbers from the Income-tax Act, 1961 and provides legal analysis")

# Main query section
st.header("Tax Query")
user_query = st.text_area("Enter your income tax related query:", height=150)

# Add debug toggle
show_debug = st.checkbox("Show debug options", value=False)


# Function to call the DeepSeek API for reasoning and section extraction
def query_deepseek_api_with_reasoning(query, api_key, model):
    try:
        # Initialize OpenAI client with DeepSeek base URL
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Format the prompt
        formatted_prompt = f"""You are an expert legal assistant specialized in Indian tax law, especially the Income-tax Act, 1961.

Your task is to extract ALL relevant **section numbers** (e.g., 45, 50, 56(2), 64(1), etc.) from the Act that could apply to the given query for sloving the user query. 

STRICT RULES:
- Return ONLY Income-tax Act section numbers in a **flat JSON list**.
- Include any clauses that may reasonably apply (e.g., gift, transfer, exemptions, capital gains, depreciation, clubbing, etc.).
- Do NOT add explanations or extra text.
- Omit section labels like "Section" or "Sec." – just give the number (e.g., "56(2)", not "Section 56(2)").
- Limit to one level of subsection, e.g., use 56(2) dont give  56(2)(i) just one level down.
- Think like a cautious tax advisor — err on the side of including all possibly relevant provisions.

Respond only in this format: ["45", "56(2)", "64(1)", "49", "50", "54", "112"]

Query: {query}"""

        # Set up messages
        messages = [{"role": "user", "content": formatted_prompt}]

        # Make streaming API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        # Initialize containers for reasoning and answer content
        reasoning_content = ""
        answer_content = ""

        # Process streaming response
        reasoning_container = st.empty()
        for chunk in response:
            # Extract reasoning content
            if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[
                0].delta.reasoning_content is not None:
                reasoning_content += chunk.choices[0].delta.reasoning_content
                reasoning_container.text(reasoning_content)

            # Extract answer content
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                answer_content += chunk.choices[0].delta.content

        return reasoning_content, answer_content

    except Exception as e:
        return f"Error: {str(e)}", ""


# Function to query DeepSeek to analyze sections and answer the query
def query_deepseek_for_analysis(query, sections_data, api_key, model):
    try:
        # Initialize OpenAI client with DeepSeek base URL
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Format the sections data
        formatted_sections = ""
        for section_number, content in sections_data.items():
            formatted_sections += f"Section {section_number}:\n{content.get('text', 'No content available')}\n\n"

        # Format the prompt with query and sections
        formatted_prompt = f"""You are an expert legal assistant specialized in Indian tax law, especially the Income-tax Act, 1961.

I will provide you with a tax-related query and the text of relevant sections from the Income Tax Act. Please analyze these sections and provide a clear, thorough answer to the query.

QUERY:
{query}

RELEVANT SECTIONS FROM INCOME TAX ACT:
{formatted_sections}

Please provide a detailed answer that:
1. Addresses all aspects of the query
2. Cites the specific sections of the Income Tax Act that apply
3. Explains how these sections apply to the query
4. Provides a clear conclusion
5. Notes any limitations or additional information that might be needed

Your answer should be written in a professional but accessible tone, suitable for someone who is not a tax expert."""

        # Set up messages
        messages = [{"role": "user", "content": formatted_prompt}]

        # Make streaming API call
        analysis_container = st.empty()
        analysis_content = ""

        # Display a message while waiting
        analysis_container.markdown("*Analyzing sections and preparing answer...*")

        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        # Process streaming response
        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                analysis_content += chunk.choices[0].delta.content
                analysis_container.markdown(analysis_content)

        return analysis_content

    except Exception as e:
        return f"Error analyzing sections: {str(e)}"


# Function to extract section numbers from the answer content
def extract_sections(content):
    try:
        # Look for any JSON-like arrays in the content
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)

        if json_match:
            section_list_str = json_match.group(0)
            try:
                section_list = json.loads(section_list_str)
                return section_list
            except json.JSONDecodeError:
                pass

        return []
    except Exception as e:
        st.error(f"Error extracting sections: {str(e)}")
        return []


# Extract sections from reasoning content
def extract_sections_from_reasoning(reasoning_content):
    if not reasoning_content:
        return []

    # Patterns to match section references in text
    patterns = [
        r'[Ss]ection (\d+(?:\(\d+\))?[A-Za-z]?)',  # Section 32, Section 56(2), Section 10A
        r'[Ss]ec\. (\d+(?:\(\d+\))?[A-Za-z]?)',  # Sec. 32, Sec. 56(2)
        r'[Ss]ec (\d+(?:\(\d+\))?[A-Za-z]?)',  # Sec 32, Sec 56(2)
        r'[Ss]\. (\d+(?:\(\d+\))?[A-Za-z]?)',  # S. 32, S. 56(2)
        r'\b[Ss](\d+(?:\(\d+\))?[A-Za-z]?)\b',  # S32, S56(2)
        r'(?<![a-zA-Z0-9])(\d+(?:\(\d+\))?[A-Za-z]?)(?= of the Act| of the Income-tax Act| of IT Act)'
        # 32 of the Act, 56(2) of the Income-tax Act
    ]

    extracted_sections = []

    # Process reasoning content line by line
    for line in reasoning_content.split('\n'):
        for pattern in patterns:
            matches = re.findall(pattern, line)
            extracted_sections.extend(matches)

    # Clean up the extracted sections and remove duplicates
    cleaned_sections = []
    for section in extracted_sections:
        # Remove any unwanted characters
        cleaned_section = re.sub(r'^[Ss]', '', section)  # Remove leading S or s
        if cleaned_section not in cleaned_sections:
            cleaned_sections.append(cleaned_section)

    return sorted(cleaned_sections)


# Combine sections from model's final answer and reasoning content
def combine_sections(model_sections, reasoning_sections):
    # Create a set to automatically handle duplicates
    combined_set = set()

    # Add all sections from both sources
    if model_sections:
        combined_set.update(model_sections)
    if reasoning_sections:
        combined_set.update(reasoning_sections)

    # Convert back to a list and sort
    combined_list = sorted(list(combined_set))

    return combined_list


# Initialize Weaviate client
def init_weaviate_client():
    try:
        # Initialize Weaviate client with hardcoded values
        # Using the exact same configuration as in the example
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_HOST,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=60)  # Increase timeout to 60 seconds
            ),
            skip_init_checks=True  # Skip initial health checks
        )

        return client
    except Exception as e:
        st.error(f"Error connecting to Weaviate: {str(e)}")
        return None


# Retrieve section content from Weaviate database using BM25
def get_section_content(client, section_numbers):
    if not client or not section_numbers:
        return {}

    section_contents = {}

    try:
        # Get the collection - exactly as in the example
        tax_sections = client.collections.get(WEAVIATE_CLASS_NAME)

        for section in section_numbers:
            try:
                # Use BM25 search with the exact same parameters as in the example
                response = tax_sections.query.bm25(
                    query=section,  # The keyword-based query to search for
                    query_properties=["section_number"],  # The properties to search in
                    limit=5  # The maximum number of results to return
                )

                # Filter results to find exact matches - exactly as in the example
                exact_matches = []
                for obj in response.objects:
                    if obj.properties.get("section_number") == section:
                        exact_matches.append(obj)

                # If there's an exact match, use it
                if exact_matches:
                    obj = exact_matches[0]
                    section_contents[section] = {
                        "section_number": obj.properties.get("section_number", ""),
                        "text": obj.properties.get("text", "")
                    }
                # If no exact match but results were found, use the first result
                elif response.objects:
                    obj = response.objects[0]
                    section_contents[section] = {
                        "section_number": obj.properties.get("section_number", ""),
                        "text": obj.properties.get("text", "")
                    }
                else:
                    # If section not found, add a placeholder
                    section_contents[section] = {
                        "section_number": section,
                        "text": "No content available for this section."
                    }

            except Exception as e:
                st.error(f"Error searching for section {section}: {str(e)}")
                # If there's an error, add a placeholder
                section_contents[section] = {
                    "section_number": section,
                    "text": f"Error retrieving content: {str(e)}"
                }

        return section_contents

    except Exception as e:
        st.error(f"Error retrieving section content: {str(e)}")
        return {}


# Display section content in an organized way
def display_section_contents(section_contents):
    if not section_contents:
        st.warning("No section content available to display.")
        return

    st.header("📚 Relevant Tax Sections")

    # Create tabs for each section
    tabs = st.tabs([f"Section {s}" for s in section_contents.keys()])

    # Fill each tab with content
    for i, (section, content) in enumerate(section_contents.items()):
        with tabs[i]:
            st.subheader(f"Section {section}")

            # Display content in an expandable container
            with st.expander("View Full Section Content", expanded=True):
                st.markdown(content.get("text", "Content not available"))

            # Add a divider
            st.divider()

            # Add a note about interpretation
            st.caption(
                "Note: This is the legal text of the section. For interpretation or specific application to your case, consult a tax professional.")


# Button to submit the query
if st.button("Find Relevant Sections and Analyze", key="find_sections_button"):
    if user_query:
        # Run the API query with streaming
        with st.spinner("Processing query..."):
            # Create a placeholder for the reasoning output
            st.subheader("Model's Reasoning Process")
            reasoning_placeholder = st.empty()

            # Get reasoning and answer content
            reasoning_content, answer_content = query_deepseek_api_with_reasoning(
                user_query, DEEPSEEK_API_KEY, DEEPSEEK_MODEL_NAME
            )

            # Store in session state
            st.session_state['reasoning_content'] = reasoning_content
            st.session_state['answer_content'] = answer_content

            # Display the reasoning process
            reasoning_placeholder.text(reasoning_content)

            # Extract sections from reasoning content
            extracted_sections = extract_sections_from_reasoning(reasoning_content)
            st.session_state['extracted_sections'] = extracted_sections

            # Display raw response for debugging if enabled
            if show_debug:
                with st.expander("Raw Answer Content"):
                    st.text(answer_content)

            # Extract final section numbers from model answer
            model_sections = extract_sections(answer_content)

            # Combine sections from both sources
            combined_sections = combine_sections(model_sections, extracted_sections)
            st.session_state['combined_sections'] = combined_sections

            # Display all sets of sections
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Model's Final Sections")
                if model_sections:
                    st.json(model_sections)
                else:
                    st.warning("No sections found in model's final answer")

            with col2:
                st.subheader("Sections From Reasoning")
                if extracted_sections:
                    st.json(extracted_sections)
                else:
                    st.warning("No sections extracted from reasoning content")

            with col3:
                st.subheader("Combined Sections")
                if combined_sections:
                    st.json(combined_sections)
                else:
                    st.warning("No sections found from either source")

            # Fetch and display section contents from Weaviate
            if combined_sections:
                with st.spinner("Retrieving section contents from database..."):
                    # Initialize Weaviate client
                    client = init_weaviate_client()

                    if client:
                        try:
                            # Get section contents
                            section_contents = get_section_content(client, combined_sections)

                            # Store in session state
                            st.session_state['section_contents'] = section_contents

                            # Display section contents
                            display_section_contents(section_contents)

                            # Use sections as knowledge to answer the query
                            st.header("🧠 Legal Analysis and Answer")
                            st.info("Using the retrieved sections to analyze and answer your query...")

                            # Get legal analysis
                            legal_analysis = query_deepseek_for_analysis(
                                user_query,
                                section_contents,
                                DEEPSEEK_API_KEY,
                                DEEPSEEK_MODEL_NAME
                            )

                            # Store analysis in session state
                            st.session_state['legal_analysis'] = legal_analysis

                        except Exception as e:
                            st.error(f"Error during section content retrieval and display: {str(e)}")
                        finally:
                            # Always close the Weaviate client
                            client.close()
                    else:
                        st.error("Could not connect to Weaviate database. Please check your connection settings.")
    else:
        st.warning("Please enter a tax query.")

# Display previously retrieved section contents and analysis if available
if st.session_state['section_contents'] and not st.button:
    display_section_contents(st.session_state['section_contents'])

    # Display legal analysis if available
    if 'legal_analysis' in st.session_state and st.session_state['legal_analysis']:
        st.header("🧠 Legal Analysis and Answer")
        st.markdown(st.session_state['legal_analysis'])