import sys
import time
import argparse
import logging

import config
import llm_setup
import profile_extraction
import data_processing
import query_engine

# Set up module-level logger
logger = logging.getLogger(__name__)

def chatbot_interface(index):
    """
    Provides a simple chatbot interface for user interaction.
    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
    """
    print("\nYou can now ask more in-depth questions about this person. Type 'exit', 'quit', or 'bye' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        print("Bot is typing...", end='')
        sys.stdout.flush()
        time.sleep(1)  # Simulate typing delay
        print('\r', end='')
        
        response = query_engine.answer_user_query(index, user_query)
        print(f"Bot: {response.response.strip()}\n")

def process_linkedin(linkedin_url, api_key=None, mock=True):
    try:
        logger.info(f"Processing LinkedIn URL: {linkedin_url} (mock mode: {mock})")

        profile_data = profile_extraction.extract_linkedin_profile(linkedin_url, api_key, mock=mock)
        if not profile_data:
            logger.error("Failed to extract LinkedIn profile data.")
            return

        nodes = data_processing.split_profile_data(profile_data)
        logger.info(f"Split profile into {len(nodes)} nodes.")

        vectordb_index = data_processing.create_vector_database(nodes)
        logger.info("Vector database index created.")

        if not data_processing.verify_embeddings(vectordb_index):
            logger.warning("Some nodes may be missing embeddings.")
        else:
            logger.info("Embedding verification passed.")

        initial_facts = query_engine.generate_initial_facts(vectordb_index)
        print("\nHere are 3 interesting facts about this person:")
        print(initial_facts)

        chatbot_interface(vectordb_index)

    except Exception:
        logger.exception("Fatal error during profile processing.")

def main():
    """Main function to run the Icebreaker Bot."""
    parser = argparse.ArgumentParser(description='Icebreaker Bot - LinkedIn Profile Analyzer')
    parser.add_argument('--url', type=str, help='LinkedIn profile URL')
    parser.add_argument('--api-key', type=str, help='ProxyCurl API key')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of API')
    parser.add_argument('--model', type=str, help='LLM model to use (e.g., "ibm/granite-3-2-8b-instruct")')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logging to output to the terminal (stderr)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    logger.info("Initializing Icebreaker Bot...")
    
    # Use command line arguments or prompt user for input
    linkedin_url = args.url or input("Enter LinkedIn profile URL (or press Enter to use mock data): ")
    use_mock = args.mock or not linkedin_url
    
    # Configure LlamaIndex Settings — must happen before any index or query engine is created.
    # args.model is None if not provided, which makes configure() use config.QUERY_MODEL.
    llm_setup.configure(model_name=args.model)

    api_key = args.api_key or config.PROXYCURL_API_KEY

    if not use_mock and not api_key:
        api_key = input("Enter ProxyCurl API key: ")

    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

    process_linkedin(linkedin_url, api_key, mock=use_mock)

if __name__ == "__main__":
    main()

