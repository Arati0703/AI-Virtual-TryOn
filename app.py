
import gradio as gr


def virtual_try_on(person_image, garment_image):
    if person_image is None:
        return None

    if garment_image is None:
        return None

    return person_image


with gr.Blocks(title="AI Virtual Try-On Fashion System") as demo:
    gr.Markdown("# AI-Based Virtual Try-On Fashion System")
    gr.Markdown(
        "Upload a photo of yourself and a garment image "
        "to explore our virtual try-on interface."
    )

    with gr.Row():
        person_image = gr.Image(
            label="Upload Your Photo",
            type="pil"
        )

        garment_image = gr.Image(
            label="Upload Garment Image",
            type="pil"
        )

    try_on_button = gr.Button("Try On", variant="primary")

    result_image = gr.Image(label="Result")

    try_on_button.click(
        fn=virtual_try_on,
        inputs=[person_image, garment_image],
        outputs=result_image
    )


if __name__ == "__main__":
    demo.launch()
