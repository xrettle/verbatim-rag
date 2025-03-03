"""
Command-line interface for the Verbatim RAG system.
"""

import argparse
import json
import os

from verbatim_rag import (
    DocumentLoader,
    QueryRequest,
    TemplateManager,
    VerbatimIndex,
    VerbatimRAG,
)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY=your_api_key_here")
        return

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

    # Template command
    template_parser = subparsers.add_parser("template", help="Template management")
    template_parser.add_argument(
        "--questions",
        "-q",
        required=True,
        nargs="+",
        help="Questions to create templates for",
    )
    template_parser.add_argument(
        "--output", "-o", required=True, help="Path to save templates"
    )
    template_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )

    query_parser = subparsers.add_parser("query", help="Query the Verbatim RAG system")
    query_parser.add_argument("--index", "-i", required=True, help="Path to the index")
    query_parser.add_argument("--question", "-q", required=True, help="Question to ask")
    query_parser.add_argument(
        "--num-docs", "-n", type=int, default=5, help="Number of documents to retrieve"
    )
    query_parser.add_argument("--templates", "-t", help="Path to templates file")
    query_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )
    query_parser.add_argument(
        "--output", "-o", help="Path to save the response as JSON"
    )

    args = parser.parse_args()

    if args.command == "index":
        print(f"Loading documents from: {args.input}")
        documents = []
        for input_path in args.input:
            if os.path.isdir(input_path):
                documents.extend(DocumentLoader.load_directory(input_path))
            else:
                documents.extend(DocumentLoader.load_file(input_path))
        print(f"Loaded {len(documents)} documents")

        print("Creating and populating the index...")
        index = VerbatimIndex()
        index.add_documents(documents)

        print(f"Saving index to: {args.output}")
        index.save(args.output)
        print("Indexing complete")

    elif args.command == "template":
        print(f"Creating templates for {len(args.questions)} questions...")
        template_manager = TemplateManager(model=args.model)

        templates = template_manager.create_templates_batch(args.questions)
        template_manager.save_templates(args.output)

        print("\nGenerated Templates:")
        for question, template in templates.items():
            print(f"\nQuestion: {question}")
            print(f"Template: {template}")
        print(f"\nTemplates saved to: {args.output}")

    elif args.command == "query":
        print(f"Loading index from: {args.index}")
        index = VerbatimIndex.load(args.index)

        template_manager = None
        if args.templates:
            print(f"Loading templates from: {args.templates}")
            template_manager = TemplateManager()
            template_manager.load_templates(args.templates)

        rag = VerbatimRAG(
            index=index,
            model=args.model,
            template_manager=template_manager,
        )

        request = QueryRequest(question=args.question, num_docs=args.num_docs)
        response = rag.query(request.question)

        print("\nQuestion:", response.question)
        print("\nAnswer:", response.answer)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(response.model_dump(), f, indent=2)
            print(f"\nResponse saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
