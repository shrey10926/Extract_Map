# aws sso login --profile shrey_bedrock
from PIL import Image
from pathlib import Path
import boto3, json, yaml, io


try:
    with open(r"bedrock_config.yaml", "r") as f:
        APP_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("config.yaml not found! Please ensure it is in the same directory.")



session = boto3.Session(profile_name="shrey_bedrock")

client = session.client(
    "bedrock-runtime",
    region_name="us-east-1"
)



try:
    with open(r"Prompt\prompt.yaml", "r") as f:
        prompt = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("prompt.yaml not found! Please ensure it is in the same directory.")

sys_prompt = prompt["system_prompt"]


with open(r"Response_Schema\response_schema_updated.json", 'r') as file:
    response_schema = json.load(file)




def preprocess_page(image_path: str, max_edge=1568, webp_quality=85) -> bytes:
    # Open image and keep it in RGB to preserve color-coded hierarchy
    img = Image.open(image_path).convert("RGB")
    
    # Check BOTH width and height to prevent server-side downscaling
    width, height = img.size
    if max(width, height) > max_edge:
        if width > height:
            new_width = max_edge
            new_height = int(height * (max_edge / width))
        else:
            new_height = max_edge
            new_width = int(width * (max_edge / height))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save as WebP: Delivers the low latency of JPEG with the sharp text edges of PNG
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
    
    return buffer.getvalue()


x = preprocess_page(r"saved_images\01029293\01029293_page_1.png")


# Specify the directory path
directory_path = Path(r"saved_images\01029292")
# Loop through all items in the directory
page_list = []
for item in directory_path.iterdir():
    if item.is_file():
        print(f"File: {item.name} | Full Path: {item}")
        page_list.append(str(item))


# --- read all images and build the content blocks ---
content_blocks = []
for path in page_list:
    with open(path, "rb") as f:
        image_bytes = f.read()
    content_blocks.append({
        "image": {
            "format": "png",          # change to "jpeg" or "gif" if needed
            "source": {
                "bytes": image_bytes
            }
        }
    })

# --- call the API (everything else unchanged) ---
response = client.converse(
    modelId=APP_CONFIG['api']['model_name'],
    system=[
        {
            "text": sys_prompt          # keep your system prompt
        }
    ],
    messages=[
        {
            "role": "user",
            "content": content_blocks    # ← multiple images in one message
        }
    ],
    outputConfig={                       # ← same outputConfig, with your JSON schema
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "name": "invoice_extraction",
                    "schema": json.dumps(response_schema)
                }
            }
        }
    },
    inferenceConfig={
        "temperature": APP_CONFIG['api']['temperature'],
        "maxTokens": APP_CONFIG['api']['max_tokens']
    }
)







# png_path = r"saved_images\1008_1234\1008_1234_page_1.png"

# # Read image
# with open(png_path, "rb") as f:
#     image_bytes = f.read()

# response = client.converse(
#     modelId = APP_CONFIG['api']['model_name'],
#     system = [
#         {
#             "text" : sys_prompt
#         }
#     ],
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "image": {
#                         "format": "webp",
#                         "source": {
#                             "bytes": x#image_bytes
#                         }
#                     }
#                 }
#             ]
#         }
#     ],
#     outputConfig={
#         "textFormat": {
#             "type": "json_schema",
#             "structure": {
#                 "jsonSchema": {
#                     "name": "invoice_extraction",
#                     "schema": json.dumps(response_schema)
#                 }
#             }
#         }
#     },
#     inferenceConfig={
#     "temperature": APP_CONFIG['api']['temperature'],
#     # "topP": 0.1,
#     "maxTokens": APP_CONFIG['api']['max_tokens']
# }
# )


try:
    response_text = response["output"]["message"]["content"][0]["text"].strip()
    response_text = response_text.removeprefix("```json")
    response_text = response_text.removeprefix("```")
    response_text = response_text.removesuffix("```")
except Exception as e:
    print("Error extracting text from response:", e)
    raise

try:
    parsed = json.loads(response_text)
except json.JSONDecodeError:
    print("Error parsing JSON:", response_text)
    raise

# print(type(parsed))
with open("result.json", "w") as file:
    json.dump(parsed, file, indent=4)