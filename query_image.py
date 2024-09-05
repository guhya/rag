import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from PIL import Image
import base64
from io import BytesIO

PROMPT_TEMPLATE = """
You are a programmer and coding assistant.

{question}

Describe the answer with 5 bullet points or more.
If there is text in the picture, spell it.
Translate the answer into Korean Language.
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    
    args = parser.parse_args()
    query_text = args.query_text
    image_path = args.image_path
    
    query_rag(query_text, image_path)


def query_rag(query_text: str, image_path: str):
    print(f"Query Text: {query_text}")
    print(f"Image Path: {image_path}")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text)
    
    # 0.0 temperature means LLM will respond with exact answer to exact prompt
    model = Ollama(model="llava:7b",temperature = 0.0)
    image_b64 = load_image(image_path)
    response_text = model.invoke(prompt, images=[image_b64])

    formatted_response = f"\nResponse:\n{response_text.strip()}"
    print(formatted_response)
    return response_text

def convert_to_base64(pil_image: Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_image(image_path: str):
    pil_image = Image.open(image_path)
    image_b64 = convert_to_base64(pil_image)
    print("Loaded image successfully!")
    return image_b64


if __name__ == "__main__":
    main()
