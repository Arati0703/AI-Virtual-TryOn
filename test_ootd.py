
from pathlib import Path
from gradio_client import Client, handle_file
import shutil
import os

# ---------------------------------
# 1. Project file paths
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent

PERSON_IMAGE = BASE_DIR / "personimg" / "images.jpg"
GARMENT_IMAGE = BASE_DIR / "outfits" / "lower-body" / "blue_jeans.jpg"
OUTPUT_DIR = BASE_DIR / "generated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# 2. Check input files
# ---------------------------------
print("Checking input images...")

if not PERSON_IMAGE.is_file():
    raise FileNotFoundError(
        f"Person image not found:\n{PERSON_IMAGE}"
    )

if not GARMENT_IMAGE.is_file():
    raise FileNotFoundError(
        f"Garment image not found:\n{GARMENT_IMAGE}"
    )

print("Person image found:", PERSON_IMAGE)
print("Garment image found:", GARMENT_IMAGE)

# ---------------------------------
# 3. Connect to OOTDiffusion
# ---------------------------------
print("\nConnecting to OOTDiffusion...")
client = Client("levihsu/OOTDiffusion")

print("Connected successfully.")

# ---------------------------------
# 4. Generate virtual try-on
# ---------------------------------
print("\nGenerating lower-body try-on...")
print("Please wait. The remote model may take some time.")

result = client.predict(
    vton_img=handle_file(str(PERSON_IMAGE)),
    garm_img=handle_file(str(GARMENT_IMAGE)),
    category="Lower-body",
    n_samples=1,
    n_steps=20,
    image_scale=2.0,
    seed=-1,
    api_name="/process_dc",
)

print("\nGeneration request completed.")
print("Returned result:")
print(repr(result))

# ---------------------------------
# 5. Find and save returned image
# ---------------------------------
def save_image_result(item, output_dir):
    """
    Save a returned local image path or copy an existing
    local image into the generated folder.
    """
    if isinstance(item, (str, os.PathLike)):
        source = Path(item)

        if source.is_file():
            destination = output_dir / "ootdiffusion_lower_body.png"

            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

            return destination

    if isinstance(item, dict):
        # Some Gradio endpoints return dictionaries.
        for key in ("image", "path", "name"):
            if key in item:
                saved = save_image_result(item[key], output_dir)
                if saved:
                    return saved

    if isinstance(item, (list, tuple)):
        for value in item:
            saved = save_image_result(value, output_dir)
            if saved:
                return saved

    return None


saved_image = save_image_result(result, OUTPUT_DIR)

if saved_image:
    print("\nSuccess! Image saved at:")
    print(saved_image)
else:
    print(
        "\nThe API returned a result, but no accessible local "
        "image file was found."
    )
    print(
        "Check the result printed above. If it contains a "
        "remote /tmp/gradio/... path, that path may exist only "
        "on the server and cannot be copied directly on Windows."
    )
