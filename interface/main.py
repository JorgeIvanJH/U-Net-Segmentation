import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("🖱️ Click on the Image to Get Pixel Coordinates")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB mode

    # Display image (optional)
    st.image(image, caption="Click on the image below")

    # Create canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Fill color
        stroke_width=1,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas",
    )

    # Get clicked point
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            last_click = objects[-1]
            x = int(last_click["left"])
            y = int(last_click["top"])
            st.write(f"📍 Last clicked pixel: **({x}, {y})**")
