import argparse
import logging

from film.service import film_list_service
from film.service import film_get_service
from film.service import film_insert_service
from film.service import film_update_service
from film.service import film_delete_service
from film.service import film_search_service
from film.service import film_rag_service

# Global variable start
logger = logging.getLogger(__name__)
# Global variable ends

# Private functions start
def _validate_list(values):
    """Validate each value in the list."""
    
    validated_values = []
    max = 100000
    for value in values:
        try:
            # Convert the value to an integer
            number = int(value)
            # Check if the number is within a specific range (e.g., 0 to 100000)
            if 0 <= number <= max:
                validated_values.append(number)
            else:
                logger.error(f"Value '{value}' is out of range (0-{max}).")
        except ValueError:
            logger.error(f"Value '{value}' is not a valid integer.")
    
    return validated_values
# Private functions end

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Manage a vector db with CRUD operations")

    # Define command-line arguments
    parser.add_argument("--operation", choices=["list", "get", "update", "delete", "insert", "search"], required=True, help="Operation to perform")
    parser.add_argument("--ids", type=str, help="List of comma separated id's to list, only necessary for 'list' operation")
    parser.add_argument("--id", type=str, help="ID or index for the operation")
    parser.add_argument("--payload", type=str, help="Payload for the operation")
    parser.add_argument("--prompt", type=str, help="Prompt for search operation")
    parser.add_argument("--type", choices=["semantic", "rag"], help="Type of search, default to 'semantic'")

    # Parse arguments
    args = parser.parse_args()

    # Execute command based on parsed arguments
    if args.operation == "list":
        if args.ids:
            # Split the comma-separated list into a Python list
            values = [value.strip() for value in args.ids.split(",")]
            # Validate the list
            validated_values = _validate_list(values)
            film_list_service.list_items(validated_values)
        else:
            parser.error("The 'list' operation requires an --ids argument")

    elif args.operation == "get":
        if args.id is None:
            parser.error("The 'get' operation requires an --id argument")
        film_get_service.get_item(args.id)

    elif args.operation == "update":
        if args.id is None or args.payload is None:
            parser.error("The 'update' operation requires both --id and --payload arguments")
        film_update_service.update_item(args.id, args.payload)

    elif args.operation == "delete":
        if args.id is None:
            parser.error("The 'delete' operation requires an --id argument")
        film_delete_service.delete_item(args.id)

    elif args.operation == "insert":
        if args.id is None or args.payload is None:
            parser.error("The 'insert' operation requires both --id and --payload arguments")
        film_insert_service.insert_item(args.id, args.payload)

    elif args.operation == "search":
        if args.prompt is None:
            parser.error("The 'search' operation requires --prompt arguments")

        if args.type is None:
            results = film_search_service.film_search(args.prompt)
            logger.debug(f"Found [{len(results)}] results {results}")
        else:
            result = film_rag_service.film_rag(args.prompt)
            logger.debug(f"Response : \n{result}")
            

if __name__ == "__main__":
    main()


