"""
Command-line interface for the Verbatim RAG system.
"""

import os
import argparse
import json

from verbatim_rag import (
    VerbatimIndex,
    VerbatimRAG,
    DocumentLoader,
    TextSplitter,
    TemplateManager,
)


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY=your_api_key_here")
        return

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Verbatim RAG Command-Line Interface")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "--input", "-i", required=True, nargs="+", help="Paths to input documents"
    )
    index_parser.add_argument(
        "--output", "-o", required=True, help="Path to save the index"
    )
    index_parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Size of document chunks"
    )
    index_parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Overlap between chunks"
    )

    # Template commands
    template_parser = subparsers.add_parser("template", help="Template management")
    template_subparsers = template_parser.add_subparsers(
        dest="template_command", help="Template command to execute"
    )

    # Create template command
    create_template_parser = template_subparsers.add_parser(
        "create", help="Create templates for questions"
    )
    create_template_parser.add_argument(
        "--questions",
        "-q",
        required=True,
        nargs="+",
        help="Questions to create templates for",
    )
    create_template_parser.add_argument(
        "--output", "-o", required=True, help="Path to save the templates"
    )
    create_template_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )

    # Match template command
    match_template_parser = template_subparsers.add_parser(
        "match", help="Match a question to existing templates"
    )
    match_template_parser.add_argument(
        "--question", "-q", required=True, help="Question to match"
    )
    match_template_parser.add_argument(
        "--templates", "-t", required=True, help="Path to templates file"
    )
    match_template_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold (0-1)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the Verbatim RAG system")
    query_parser.add_argument("--index", "-i", required=True, help="Path to the index")
    query_parser.add_argument("--question", "-q", required=True, help="Question to ask")
    query_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )
    query_parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )
    query_parser.add_argument(
        "--template", "-t", help="Simple template with [CONTENT] placeholder"
    )
    query_parser.add_argument(
        "--templates-file", help="Path to templates file for template matching"
    )
    query_parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for template matching",
    )
    query_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    query_parser.add_argument(
        "--output", "-o", help="Path to save the response as JSON"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start an interactive session"
    )
    interactive_parser.add_argument(
        "--index", "-i", required=True, help="Path to the index"
    )
    interactive_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )
    interactive_parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )
    interactive_parser.add_argument(
        "--template", "-t", help="Simple template with [CONTENT] placeholder"
    )
    interactive_parser.add_argument(
        "--templates-file", help="Path to templates file for template matching"
    )
    interactive_parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for template matching",
    )
    interactive_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    if args.command == "index":
        # Load documents
        print(f"Loading documents from: {args.input}")
        documents = []
        for input_path in args.input:
            if os.path.isdir(input_path):
                documents.extend(DocumentLoader.load_directory(input_path))
            else:
                documents.extend(DocumentLoader.load_file(input_path))
        print(f"Loaded {len(documents)} documents")

        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = TextSplitter(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        chunked_documents = text_splitter.split_documents(documents)
        print(f"Created {len(chunked_documents)} document chunks")

        # Create and populate the index
        print("Creating and populating the index...")
        index = VerbatimIndex()
        index.add_documents(chunked_documents)

        # Save the index
        print(f"Saving index to: {args.output}")
        index.save(args.output)
        print("Indexing complete")

    elif args.command == "template":
        if args.template_command == "create":
            # Create templates for questions
            print(f"Creating templates for {len(args.questions)} questions...")
            template_manager = TemplateManager(model=args.model)

            templates = template_manager.create_templates_batch(args.questions)

            # Print the generated templates
            for question, template in templates.items():
                print(f"\nQuestion: {question}")
                print(f"Template: {template}")

            # Save the templates
            template_manager.save_templates(args.output)
            print(f"Templates saved to: {args.output}")

        elif args.template_command == "match":
            # Match a question to existing templates
            print(f"Matching question to templates in: {args.templates}")
            template_manager = TemplateManager()
            template_manager.load_templates(args.templates)

            matched_template, score = template_manager.match_template(
                args.question, args.threshold
            )

            print(f"Question: {args.question}")
            print(f"Match Score: {score:.2f}")

            if matched_template:
                print(f"Matched Template: {matched_template}")
            else:
                print("No matching template found")

        else:
            template_parser.print_help()

    elif args.command == "query":
        # Load the index
        print(f"Loading index from: {args.index}")
        index = VerbatimIndex.load(args.index)

        # Initialize template manager if templates file is provided
        template_manager = None
        if args.templates_file:
            print(f"Loading templates from: {args.templates_file}")
            template_manager = TemplateManager()
            template_manager.load_templates(args.templates_file)

        # Initialize VerbatimRAG
        verbatim_rag = VerbatimRAG(
            index,
            model=args.model,
            k=args.k,
            simple_template=args.template,
            template_manager=template_manager,
            template_match_threshold=args.match_threshold
            if args.templates_file
            else 0.0,
        )

        # Process the query
        print(f"Question: {args.question}")
        response, details = verbatim_rag.query(args.question)

        if args.verbose:
            print("\nTemplate:")
            print(details["template"])

            print("\nRelevant Spans:")
            for i, span in enumerate(details["relevant_spans"], 1):
                print(f"{i}. {span}")

        print("\nResponse:")
        print(response)

        # Save the response if requested
        if args.output:
            output_data = {
                "question": args.question,
                "response": response,
                "details": details,
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"Response saved to: {args.output}")

        # Save updated templates if using template manager
        if template_manager and args.templates_file:
            template_manager.save_templates(args.templates_file)
            print(f"Updated templates saved to: {args.templates_file}")

    elif args.command == "interactive":
        # Load the index
        print(f"Loading index from: {args.index}")
        index = VerbatimIndex.load(args.index)

        # Initialize template manager if templates file is provided
        template_manager = None
        if args.templates_file:
            print(f"Loading templates from: {args.templates_file}")
            template_manager = TemplateManager()
            template_manager.load_templates(args.templates_file)

        # Initialize VerbatimRAG
        verbatim_rag = VerbatimRAG(
            index,
            model=args.model,
            k=args.k,
            simple_template=args.template,
            template_manager=template_manager,
            template_match_threshold=args.match_threshold
            if args.templates_file
            else 0.0,
        )

        print("Verbatim RAG Interactive Mode")
        print("Type 'exit' or 'quit' to end the session")

        while True:
            # Get user input
            question = input("\nQuestion: ")

            # Check if the user wants to exit
            if question.lower() in ["exit", "quit"]:
                break

            # Process the query
            response, details = verbatim_rag.query(question)

            if args.verbose:
                print("\nTemplate:")
                print(details["template"])

                print("\nRelevant Spans:")
                for i, span in enumerate(details["relevant_spans"], 1):
                    print(f"{i}. {span}")

            print("\nResponse:")
            print(response)

        # Save updated templates if using template manager
        if template_manager and args.templates_file:
            template_manager.save_templates(args.templates_file)
            print(f"Updated templates saved to: {args.templates_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
