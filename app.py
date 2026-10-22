
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from pathlib import Path
import io

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Medical Image Processing AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Image Processing for Efficient Disease Diagnosis")
st.write(
    "Upload a medical image to explore image processing "
    "techniques and AI-based image-type classification."
)

# ==================================================
# 2. OUTPUT FOLDER
# ==================================================

OUTPUT_DIR = Path("processed_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 3. UPLOAD IMAGE
# ==================================================

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

# ==================================================
# 4. LOAD PRETRAINED AI MODEL
# ==================================================

@st.cache_resource
def load_ai_model():
    model_name = "openai/clip-vit-base-patch32"

    return pipeline(
        task="zero-shot-image-classification",
        model=model_name,
        device=-1
    )

# ==================================================
# 5. PROCESS UPLOADED IMAGE
# ==================================================

if uploaded_file is not None:

    try:
        # Read uploaded image
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(pil_image)

        # OpenCV uses BGR format
        image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        st.success("Image uploaded successfully!")

        # Display original image
        st.subheader("1. Original Medical Image")
        st.image(pil_image, use_container_width=True)

        # Image information
        st.subheader("2. Image Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Width", image.shape[1])

        with col2:
            st.metric("Height", image.shape[0])

        with col3:
            st.metric("Dimensions", image.ndim)

        with col4:
            st.metric("Data Type", str(image.dtype))

        # ==================================================
        # 6. GRAYSCALE CONVERSION
        # ==================================================

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # 7. NOISE REDUCTION
        # ==================================================

        denoised_image = cv2.GaussianBlur(
            gray_image,
            (5, 5),
            0
        )

        # ==================================================
        # 8. CONTRAST ENHANCEMENT
        # ==================================================

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced_image = clahe.apply(
            denoised_image
        )

        # ==================================================
        # 9. DISPLAY PROCESSED IMAGES
        # ==================================================

        st.subheader("3. Image Processing Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                gray_image,
                caption="Grayscale Image",
                clamp=True,
                use_container_width=True
            )

        with col2:
            st.image(
                denoised_image,
                caption="Noise Reduced (Gaussian Blur)",
                clamp=True,
                use_container_width=True
            )

        with col3:
            st.image(
                enhanced_image,
                caption="Contrast Enhanced (CLAHE)",
                clamp=True,
                use_container_width=True
            )

        # ==================================================
        # 10. SAVE PROCESSED IMAGES
        # ==================================================

        cv2.imwrite(
            str(OUTPUT_DIR / "grayscale.png"),
            gray_image
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "denoised.png"),
            denoised_image
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "enhanced.png"),
            enhanced_image
        )

        # Download buttons
        st.subheader("4. Download Processed Images")

        col1, col2, col3 = st.columns(3)

        with col1:
            success, encoded = cv2.imencode(
                ".png", gray_image
            )
            if success:
                st.download_button(
                    "Download Grayscale",
                    data=encoded.tobytes(),
                    file_name="grayscale.png",
                    mime="image/png"
                )

        with col2:
            success, encoded = cv2.imencode(
                ".png", denoised_image
            )
            if success:
                st.download_button(
                    "Download Denoised",
                    data=encoded.tobytes(),
                    file_name="denoised.png",
                    mime="image/png"
                )

        with col3:
            success, encoded = cv2.imencode(
                ".png", enhanced_image
            )
            if success:
                st.download_button(
                    "Download Enhanced",
                    data=encoded.tobytes(),
                    file_name="enhanced.png",
                    mime="image/png"
                )

        # ==================================================
        # 11. AI IMAGE-TYPE CLASSIFICATION
        # ==================================================

        st.subheader("5. AI Image-Type Classification")
        st.write(
            "The Hugging Face CLIP model compares the image "
            "with the image categories listed below."
        )

        candidate_labels = [
            "a chest X-ray radiograph",
            "a CT scan",
            "an MRI scan",
            "an ultrasound image",
            "a skin photograph",
            "a microscopic tissue image"
        ]

        if st.button("Run AI Classification", type="primary"):

            try:
                with st.spinner(
                    "Loading the AI model and classifying the image..."
                ):
                    classifier = load_ai_model()

                    predictions = classifier(
                        pil_image,
                        candidate_labels=candidate_labels
                    )

                st.success("AI classification completed!")

                best_prediction = predictions[0]["label"]

                st.write("**Top-ranked image category:**")
                st.info(best_prediction)

                st.write("**Category similarity scores:**")

                for result in predictions:
                    score = result["score"] * 100

                    st.write(
                        f'{result["label"]}: {score:.2f}%'
                    )

                    st.progress(
                        min(max(float(result["score"]), 0.0), 1.0)
                    )

                # Save AI results
                result_path = OUTPUT_DIR / "ai_results.txt"

                with open(
                    result_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    file.write(
                        "AI IMAGE-TYPE CLASSIFICATION RESULTS\n"
                    )
                    file.write("=" * 40 + "\n\n")

                    file.write(
                        f"Top-ranked category: {best_prediction}\n\n"
                    )

                    for result in predictions:
                        file.write(
                            f'{result["label"]}: '
                            f'{result["score"] * 100:.2f}%\n'
                        )

                    file.write(
                        "\nThese are image-category similarity scores, "
                        "not disease probabilities or diagnoses.\n"
                    )

                st.download_button(
                    "Download AI Results",
                    data=result_path.read_text(encoding="utf-8"),
                    file_name="ai_results.txt",
                    mime="text/plain"
                )

            except Exception as error:
                st.error(
                    "Could not run the AI model. Check your internet "
                    "connection and installed packages."
                )
                st.exception(error)

        # ==================================================
        # 12. SAVE COMBINED RESULTS FIGURE
        # ==================================================

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].imshow(image_array)
        axes[0, 0].set_title("Original Image")

        axes[0, 1].imshow(
            gray_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[0, 1].set_title("Grayscale Image")

        axes[1, 0].imshow(
            denoised_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[1, 0].set_title("Noise Reduced")

        axes[1, 1].imshow(
            enhanced_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[1, 1].set_title("Contrast Enhanced")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()

        figure_buffer = io.BytesIO()

        fig.savefig(
            figure_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        figure_bytes = figure_buffer.getvalue()

        with open(
            OUTPUT_DIR / "processing_results.png",
            "wb"
        ) as file:
            file.write(figure_bytes)

        st.subheader("6. Combined Processing Figure")

        st.image(
            figure_bytes,
            caption="Original and processed images",
            use_container_width=True
        )

        st.download_button(
            "Download Combined Figure",
            data=figure_bytes,
            file_name="processing_results.png",
            mime="image/png"
        )

        # ==================================================
        # 13. SAFETY NOTE
        # ==================================================

        st.warning(
            "Medical safety note: This is an educational demonstration. "
            "CLIP classifies image categories and does not diagnose "
            "diseases. Similarity scores are not disease probabilities. "
            "Do not use this application to make medical decisions."
        )

    except Exception as error:
        st.error("The uploaded image could not be processed.")
        st.exception(error)

else:
    st.info(
        "Please upload an image using the uploader above to begin."
    )
