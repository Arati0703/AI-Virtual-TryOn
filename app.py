
import os
import tempfile
import traceback
from pathlib import Path
from shutil import copy2

import gradio as gr
from PIL import Image
from gradio_client import Client, handle_file


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTFITS_DIR = BASE_DIR / "outfits"
OUTPUT_DIR = BASE_DIR / "generated"

IDM_SPACE_ID = "yisol/IDM-VTON"
OOT_SPACE_ID = "levihsu/OOTDiffusion"

CATEGORIES = {
    "Upper-body": "upper-body",
    "Lower-body": "lower-body",
    "Dress": "dress",
}

GARMENT_DESCRIPTIONS = {
    "Upper-body": "upper-body clothing",
    "Lower-body": "lower-body clothing",
    "Dress": "a dress",
}

IDM_CLIENT = None
OOT_CLIENT = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# OUTFIT COLLECTION
# =========================================================

def get_outfits(category):
    folder = OUTFITS_DIR / CATEGORIES.get(category, "upper-body")

    if not folder.is_dir():
        return []

    items = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            try:
                with Image.open(path) as image:
                    items.append(
                        (
                            image.convert("RGB").copy(),
                            path.stem.replace("_", " ").title(),
                        )
                    )
            except Exception as exc:
                print(f"Cannot load {path}: {exc}", flush=True)

    return items


def load_catalog(category):
    items = get_outfits(category)

    if items:
        message = f"Collection loaded: {len(items)} outfit(s)."
    else:
        folder = OUTFITS_DIR / CATEGORIES[category]
        message = f"No outfit images found. Add images to:\n{folder}"

    return (
        items,
        None,
        message,
        "Select an outfit card or upload your own garment.",
    )


def select_outfit(evt: gr.SelectData, category):
    items = get_outfits(category)

    try:
        index = evt.index
        if isinstance(index, (tuple, list)):
            index = index[0]

        index = int(index)

        if index < 0 or index >= len(items):
            return None, "Please select a valid outfit."

        folder = OUTFITS_DIR / CATEGORIES[category]
        extensions = {".jpg", ".jpeg", ".png", ".webp"}

        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

        path = paths[index]
        name = path.stem.replace("_", " ").title()

        return str(path), f"Selected outfit: **{name}**"

    except Exception as exc:
        print(f"Outfit selection error: {exc}", flush=True)
        return None, "Could not select this outfit. Please try again."


def change_garment_method(method):
    ready = method == "Ready-made collection"

    return (
        gr.update(visible=ready),
        gr.update(visible=not ready),
        None,
        None,
        "Choose your garment to continue.",
    )


# =========================================================
# RESULT IMAGE EXTRACTION
# =========================================================

def find_image_path(result):
    """
    Supports common IDM-VTON and OOTDiffusion responses,
    including Gallery results such as:
    [{'image': 'C:/.../image.webp', 'caption': None}]
    """

    candidates = result if isinstance(result, (list, tuple)) else [result]

    for item in candidates:
        if isinstance(item, (str, os.PathLike)):
            path = os.fspath(item)

            if os.path.isfile(path):
                return path

        elif isinstance(item, dict):
            for key in ("path", "image", "name"):
                candidate = item.get(key)

                if isinstance(candidate, dict):
                    candidate = (
                        candidate.get("path")
                        or candidate.get("name")
                    )

                if isinstance(candidate, (str, os.PathLike)):
                    path = os.fspath(candidate)

                    if os.path.isfile(path):
                        return path

    return None


# =========================================================
# GENERATE VIRTUAL TRY-ON
# =========================================================

def generate_tryon(
    person_image,
    garment_method,
    selected_path,
    manual_image,
    category,
    description,
    steps,
    seed,
    progress=gr.Progress(),
):
    global IDM_CLIENT, OOT_CLIENT

    temp_files = []

    try:
        if person_image is None:
            yield None, "Please upload your person photo first."
            return

        # Get the selected garment.
        if garment_method == "Ready-made collection":
            if not selected_path or not os.path.isfile(selected_path):
                yield None, "Please select an outfit card first."
                return

            garment_description = GARMENT_DESCRIPTIONS[category]
            garment_source = selected_path

        else:
            if manual_image is None:
                yield None, "Please upload a garment image."
                return

            garment_description = (
                description.strip()
                if description and description.strip()
                else GARMENT_DESCRIPTIONS[category]
            )

            garment_temp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            )
            garment_temp.close()
            temp_files.append(garment_temp.name)

            manual_image.convert("RGB").save(garment_temp.name)
            garment_source = garment_temp.name

        # Save a temporary copy of the person photo.
        person_temp = tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        )
        person_temp.close()
        temp_files.append(person_temp.name)

        person_image.convert("RGB").save(person_temp.name)

        # Copy catalog garment to a temporary PNG.
        if garment_method == "Ready-made collection":
            garment_temp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            )
            garment_temp.close()
            temp_files.append(garment_temp.name)

            with Image.open(garment_source) as image:
                image.convert("RGB").save(garment_temp.name)

            garment_source = garment_temp.name

        # -------------------------------------------------
        # ROUTE MODEL BY CATEGORY
        # -------------------------------------------------

        if category == "Lower-body":
            model_name = "OOTDiffusion"

            yield None, "Connecting to OOTDiffusion. Please wait…"
            progress(0.15, desc="Connecting to OOTDiffusion")

            if OOT_CLIENT is None:
                OOT_CLIENT = Client(OOT_SPACE_ID)

            print("Sending images to OOTDiffusion...", flush=True)
            progress(0.30, desc="Preparing images")

            result = OOT_CLIENT.predict(
                vton_img=handle_file(person_temp.name),
                garm_img=handle_file(garment_source),
                category="Lower-body",
                n_samples=1,
                n_steps=int(steps),
                image_scale=2.0,
                seed=int(seed),
                api_name="/process_dc",
            )

        else:
            model_name = "IDM-VTON"

            yield None, "Connecting to IDM-VTON. Please wait…"
            progress(0.15, desc="Connecting to IDM-VTON")

            if IDM_CLIENT is None:
                IDM_CLIENT = Client(IDM_SPACE_ID)

            print("Sending images to IDM-VTON...", flush=True)
            progress(0.30, desc="Preparing images")

            result = IDM_CLIENT.predict(
                dict={
                    "background": handle_file(person_temp.name),
                    "layers": [],
                    "composite": None,
                },
                garm_img=handle_file(garment_source),
                garment_des=garment_description,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=int(steps),
                seed=int(seed),
                api_name="/tryon",
            )

        print(f"{model_name} response:", repr(result), flush=True)
        progress(0.90, desc="Receiving generated image")

        result_path = find_image_path(result)

        if result_path is None:
            yield None, (
                f"{model_name} responded, but its image file could not "
                "be located. Check the VS Code terminal."
            )
            return

        # Save the result in the project.
        output_path = OUTPUT_DIR / "latest_tryon.png"
        copy2(result_path, output_path)

        # Confirm that the output is a valid image.
        with Image.open(output_path) as image:
            image.verify()

        progress(1.0, desc="Complete")
        print(f"Saved image: {output_path}", flush=True)

        yield str(output_path), (
            f"Your virtual try-on is ready! Model used: {model_name}."
        )

    except Exception as exc:
        traceback.print_exc()

        yield None, (
            f"Generation failed: {exc}\n\n"
            "Check your internet connection and the VS Code terminal. "
            "The hosted model may be busy, unavailable, or have changed."
        )

    finally:
        for path in temp_files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass


# =========================================================
# DESIGN
# =========================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

:root {
    --ivory: #F8F4EE;
    --paper: #FFFDFA;
    --espresso: #30231F;
    --rose: #B87578;
    --rose-dark: #92585C;
    --gold: #BD9A62;
    --muted: #887C74;
}

body, .gradio-container {
    background: var(--ivory) !important;
    color: var(--espresso) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

h1, h2, h3 {
    color: var(--espresso) !important;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(42px, 6vw, 68px) !important;
    letter-spacing: -1.5px !important;
}

h2 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 500 !important;
}

.hero {
    padding: 42px 20px 30px;
    text-align: center;
    border-bottom: 1px solid #E7DCD1;
    margin-bottom: 28px;
}

.eyebrow, .section-kicker {
    color: var(--rose);
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 11px;
    font-weight: 700;
}

.hero-subtitle {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-style: italic;
    color: var(--muted);
}

.hero-copy {
    max-width: 570px;
    margin: 10px auto 0;
    line-height: 1.8;
    color: var(--muted);
    font-size: 14px;
}

.panel {
    background: var(--paper);
    border: 1px solid #E9DED4;
    border-radius: 18px;
    padding: 22px;
}

button.primary {
    background: var(--rose) !important;
    border: 1px solid var(--rose) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 48px;
}

button.primary:hover {
    background: var(--rose-dark) !important;
    border-color: var(--rose-dark) !important;
}

.status-box {
    background: #F6EDE9;
    border: 1px solid #E8D4D0;
    border-radius: 10px;
    padding: 12px;
    white-space: pre-wrap;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding: 28px 0 16px;
    border-top: 1px solid #E7DCD1;
    margin-top: 28px;
}

footer {
    display: none !important;
}
"""


# =========================================================
# GRADIO APP
# =========================================================

with gr.Blocks(title="AI TRY-ON | Virtual Fitting Studio") as demo:

    gr.HTML("""
    <div class="hero">
        <div class="eyebrow">THE VIRTUAL FITTING STUDIO</div>
        <h1>AI TRY-ON</h1>
        <div class="hero-subtitle">Discover your next look.</div>
        <div class="hero-copy">
            Choose a garment from our collection or upload your own.
            See how it looks with AI.
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.HTML("""
            <div class="section-kicker">01 — YOUR PHOTO</div>
            <h2>Start with you</h2>
            <p>Upload a clear, front-facing full-body photo.</p>
            """)

            person_image = gr.Image(
                label="Your photo",
                type="pil",
                image_mode="RGB",
                height=320,
            )

            gr.HTML("""
            <div class="section-kicker">02 — YOUR GARMENT</div>
            <h2>Find your look</h2>
            """)

            garment_method = gr.Radio(
                choices=[
                    "Ready-made collection",
                    "Upload my own garment",
                ],
                value="Ready-made collection",
                label="Garment source",
            )

            category = gr.Dropdown(
                choices=["Upper-body", "Lower-body", "Dress"],
                value="Upper-body",
                label="Garment category",
            )

            with gr.Group(visible=True) as ready_made_panel:
                outfit_gallery = gr.Gallery(
                    label="Choose an outfit",
                    columns=3,
                    rows=2,
                    height=340,
                    object_fit="contain",
                    allow_preview=False,
                )

            with gr.Group(visible=False) as manual_panel:
                manual_image = gr.Image(
                    label="Upload your garment",
                    type="pil",
                    image_mode="RGB",
                    height=260,
                )

                description = gr.Textbox(
                    label="Garment description (optional)",
                    placeholder="Example: a white cotton shirt",
                    lines=2,
                )

            selected_path = gr.State(None)

            selection_status = gr.Markdown(
                "Select an outfit card or upload your own garment."
            )

            steps = gr.Slider(
                minimum=10,
                maximum=40,
                value=30,
                step=5,
                label="Generation steps",
            )

            seed = gr.Number(
                value=42,
                precision=0,
                label="Seed",
            )

            generate_button = gr.Button(
                "Generate my look",
                variant="primary",
            )

            status = gr.Markdown(
                "Ready when you are.",
                elem_classes=["status-box"],
            )

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.HTML("""
            <div class="section-kicker">YOUR VIRTUAL FITTING</div>
            <h2>The new perspective</h2>
            <p style="color:#887C74">
                Your generated photo will appear here.
            </p>
            """)

            result_image = gr.Image(
                label="Final try-on photo",
                type="filepath",
                height=620,
                interactive=False,
            )

    gr.HTML("""
    <div class="footer">
        AI TRY-ON · A virtual fitting experience
    </div>
    """)

    # Switch between collection and manual upload.
    garment_method.change(
        fn=change_garment_method,
        inputs=[garment_method],
        outputs=[
            ready_made_panel,
            manual_panel,
            selected_path,
            manual_image,
            selection_status,
        ],
    )

    # Reload outfit collection when category changes.
    category.change(
        fn=load_catalog,
        inputs=[category],
        outputs=[
            outfit_gallery,
            selected_path,
            status,
            selection_status,
        ],
    )

    # Select a ready-made garment.
    outfit_gallery.select(
        fn=select_outfit,
        inputs=[category],
        outputs=[selected_path, selection_status],
    )

    # Load initial collection.
    demo.load(
        fn=load_catalog,
        inputs=[category],
        outputs=[
            outfit_gallery,
            selected_path,
            status,
            selection_status,
        ],
    )

    # Generate try-on image.
    generate_button.click(
        fn=generate_tryon,
        inputs=[
            person_image,
            garment_method,
            selected_path,
            manual_image,
            category,
            description,
            steps,
            seed,
        ],
        outputs=[result_image, status],
        show_progress="full",
    )


# =========================================================
# START APP
# =========================================================

if __name__ == "__main__":
    print("Starting AI TRY-ON...", flush=True)
    print("Open http://127.0.0.1:7860 in your browser.", flush=True)

    demo.queue().launch(
        theme=gr.themes.Base(
            primary_hue="rose",
            secondary_hue="amber",
            neutral_hue="stone",
        ),
        css=CUSTOM_CSS,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
