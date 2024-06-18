from dotenv import load_dotenv
load_dotenv() # load environment variables from .env file
import pathlib
import google.generativeai as genai
import streamlit as st
import os
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Generation Configurations
gen_config = {
    "max_output_tokens": 10000,
    "temperature": 1,
    "top_p": 1.0,
    "top_k": 64,
    "response_mime_type": "text/plain",
}

# Model Configurations
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=gen_config,
    safety_settings=safety_settings,
)


# Framework selection (e.g., Tailwind, Bootstrap, etc.)
framework = "Regular CSS use flex grid etc"

# Chat History
chat_session = model.start_chat(history = [])

# Function to send a message to the model
def send_message_to_model(message, image_path):
    image_input = {
        'mime_type': 'image/jpeg',
        'data': pathlib.Path(image_path).read_bytes()
    }
    response = chat_session.send_message([message, image_input])
    return response.text


def main():
    st.title("Gemini 1.5 Pro Application")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # Save the image temporarily
            temp_image_path = pathlib.Path("temp_image.jpg")
            image.save(temp_image_path, format="JPEG")

            if st.button("Code UI"):
                st.write("Reviewing the image for UI code generation...")
                prompt = "Please provide a detailed description of this user interface (UI), including the various elements it contains. When referring to a specific UI element, enclose its name and bounding box coordinates in square brackets using the following format: [object name (y_min, x_min, y_max, x_max)]. Additionally, please specify the color of each element using standard color names or hexadecimal color codes. Your description should be thorough and accurate, capturing all relevant details of the UI's layout, design, and functionality."
                description = send_message_to_model(prompt, temp_image_path)
                st.write(description)

                # Refine the description
                st.write("Working on the refined description...")
                refine_prompt = f"""Carefully compare the UI elements described in the provided text with the corresponding image. Identify any missing UI elements that were not mentioned in the description or any inaccuracies in the described details (such as dimensions, positions, or colors). For each UI element, provide the following information:

                1. The element name and bounding box coordinates in the format: [object name (y_min, x_min, y_max, x_max)]
                2. The accurate color of the element using a standard color name or hexadecimal color code.

                Based on your comparison, provide a refined and accurate description of the UI elements, ensuring that all elements present in the image are accounted for and that their details (dimensions, positions, colors) are correctly represented.

                Here is the initial description:

                {description}

                Please use this initial description as a starting point and make necessary corrections, additions, or modifications to ensure a precise representation of the UI based on the provided image."""
                refined_description = send_message_to_model(refine_prompt, temp_image_path)
                st.write(refined_description)

                # Generate HTML
                st.write("Generating HTML...")
                html_prompt = f"""Create an HTML file based on the following UI description, using the UI elements described in the previous response. Embed {framework} CSS styles within the HTML file to style the elements. Ensure that the colors used for each element match the specified colors in the original UI description. The UI should be responsive and designed with a mobile-first approach, aiming to closely resemble the original UI.
                Do not include any explanations or comments within the code. Avoid using backticks (```) or any other formatting for the HTML code. Simply return the HTML code with inline CSS styles.
                Here is the refined description:
                {refined_description}
                This improved prompt provides the following enhancements:
                It clarifies that the CSS styles should be embedded within the HTML file using inline styles.
                It emphasizes the importance of matching the colors specified in the original UI description.
                It specifies that the UI should be responsive and follow a mobile-first design approach.
                It reiterates the requirement to avoid using backticks or any other formatting for the HTML code, ensuring that only the HTML code with inline CSS styles is returned.

                By incorporating these improvements, the prompt becomes more clear, concise, and easier to understand, reducing the likelihood of misinterpretation or confusion."""
                initial_html = send_message_to_model(html_prompt, temp_image_path)
                st.code(initial_html, language='html')

                # Refine HTML
                st.write("Refining HTML...")
                refine_html_prompt = f"Validate the following HTML code based on the UI description and image and provide a refined version of the HTML code with {framework} CSS that improves accuracy, responsiveness, and adherence to the original design. ONLY return the refined HTML code with inline CSS. Avoid using ```html. and ``` at the end. Here is the initial HTML: {initial_html}"
                refined_html = send_message_to_model(refine_html_prompt, temp_image_path)
                st.code(refined_html, language='html')

                # Save the refined HTML to a file
                with open("index.html", "w") as file:
                    file.write(refined_html)
                st.success("HTML file 'index.html' has been created.")

                # Provide download link for HTML
                st.download_button(label="Download HTML", data=refined_html, file_name="index.html", mime="text/html")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
