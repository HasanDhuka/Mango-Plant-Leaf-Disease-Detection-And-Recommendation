import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import cv2
import pandas as pd
import json
import unittest
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, green, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import datetime

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="Mango Plant Leaf Disease Detection And Recommendation",
    page_icon="🥭",
    layout="wide"
)

# Register DejaVu Sans fonts
pdfmetrics.registerFont(TTFont('DejaVuSans', 'ttf/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'ttf/DejaVuSans-Bold.ttf'))

# Custom CSS for improved visibility
st.markdown("""
<style>
.stApp {
    background-color: #f4f8f2;  /* Light sage green background */
}
.stMarkdown {
    color: #1e4620;  /* Dark forest green text */
}
.stButton>button {
    background-color: #4a7c59;  /* Medium forest green buttons */
    color: white;
    border-radius: 4px;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #5c8c69;  /* Slightly lighter green on hover */
}
.stTextInput>div>div>input {
    background-color: white;
    color: #2d502f;  /* Dark green text in inputs */
    border: 1px solid #4a7c59;
}
.stSelectbox>div>div {
    background-color: white;
    color: #2d502f;
    border: 1px solid #4a7c59;
}
h1, h2, h3 {
    color: #2d502f !important;  /* Dark green headers */
}
.stDataFrame {
    border: 1px solid #4a7c59;
}
</style>
""", unsafe_allow_html=True)

# Disease information database
DISEASE_INFO = {
    'Anthracnose': {
        'scientific_name': 'Colletotrichum gloeosporioides',
        'symptoms': [
            'Dark brown to black spots on leaves',
            'Circular or angular lesions',
            'Spots may have yellow halos',
            'Leaves may wither and fall prematurely'
        ],
        'treatments': [
            'Apply copper-based fungicides every 15-30 days during the growing season',
            'Prune infected branches and destroy fallen leaves',
            'Improve air circulation by proper spacing and pruning',
            'Avoid overhead irrigation to reduce leaf wetness'
        ],
        'preventive_measures': [
            'Maintain proper tree spacing',
            'Regular pruning to improve air circulation',
            'Collect and destroy fallen leaves',
            'Time irrigation to allow leaves to dry before evening'
        ],
        'organic_remedies': [
            'Neem oil spray (15ml/L of water)',
            'Garlic extract spray',
            'Cow urine spray (1:10 dilution)',
            'Trichoderma-based biological control'
        ]
    },
    'Cutting Weevil': {
        'scientific_name': 'Hypomeces squamosus',
        'symptoms': [
            'Irregular holes in leaves',
            'Notched leaf margins',
            'Skeletonized leaves',
            'Presence of adult weevils'
        ],
        'treatments': [
            'Spray carbaryl (0.1%) or malathion (0.1%)',
            'Use sticky bands around tree trunk',
            'Apply neem-based insecticides',
            'Remove affected leaves and destroy'
        ],
        'preventive_measures': [
            'Regular monitoring of trees',
            'Maintain orchard hygiene',
            'Remove alternative host plants',
            'Use light traps for adult insects'
        ],
        'organic_remedies': [
            'Neem oil spray (2%)',
            'Beauveria bassiana application',
            'Metarhizium anisopliae spray',
            'Garlic-chili extract spray'
        ]
    },
    'Die Back': {
        'scientific_name': 'Lasiodiplodia theobromae',
        'symptoms': [
            'Progressive death of twigs from tip downward',
            'Dark discoloration of bark',
            'Internal wood browning',
            'Leaf wilting and drying'
        ],
        'treatments': [
            'Prune affected branches 15-20 cm below infection',
            'Apply carbendazim (0.1%)',
            'Use copper oxychloride paste on cuts',
            'Maintain proper irrigation'
        ],
        'preventive_measures': [
            'Avoid tree stress',
            'Proper nutrition management',
            'Regular pruning of dead wood',
            'Protect from sunscald'
        ],
        'organic_remedies': [
            'Trichoderma viride application',
            'Bordeaux paste on wounds',
            'Cow dung slurry coating',
            'Neem oil treatment'
        ]
    },
    'Gall Midge': {
        'scientific_name': 'Procontarinia matteiana',
        'symptoms': [
            'Small wart-like galls on leaves',
            'Yellowing around galls',
            'Premature leaf fall',
            'Reduced photosynthesis'
        ],
        'treatments': [
            'Spray dimethoate (0.06%)',
            'Apply systemic insecticides',
            'Remove and destroy affected leaves',
            'Time spraying with pest emergence'
        ],
        'preventive_measures': [
            'Monitor pest population',
            'Maintain orchard sanitation',
            'Proper pruning and training',
            'Encourage natural predators'
        ],
        'organic_remedies': [
            'Neem seed kernel extract (5%)',
            'Verticillium lecanii application',
            'Yellow sticky traps',
            'Karanj oil spray'
        ]
    },
    'Powdery Mildew': {
        'scientific_name': 'Oidium mangiferae',
        'symptoms': [
            'White powdery coating on leaves',
            'Curling and distortion of leaves',
            'Premature flower/fruit drop',
            'Stunted growth of new shoots'
        ],
        'treatments': [
            'Spray wettable sulfur (0.2%)',
            'Apply systemic fungicides',
            'Remove infected plant parts',
            'Improve air circulation'
        ],
        'preventive_measures': [
            'Proper tree spacing',
            'Regular pruning',
            'Avoid excessive nitrogen',
            'Monitor humidity levels'
        ],
        'organic_remedies': [
            'Milk spray (1:10 dilution)',
            'Potassium bicarbonate solution',
            'Neem oil spray',
            'Sulfur dust application'
        ]
    },
    'Sooty Mould': {
        'scientific_name': 'Capnodium mangiferae',
        'symptoms': [
            'Black powdery coating on leaves',
            'Reduced photosynthesis',
            'Associated with honeydew secretion',
            'Weakening of tree vigor'
        ],
        'treatments': [
            'Control sap-sucking insects',
            'Spray starch solution (5%)',
            'Apply insecticide-fungicide combination',
            'Fish oil resin soap spray'
        ],
        'preventive_measures': [
            'Monitor for insect infestations',
            'Maintain tree vigor',
            'Proper spacing and pruning',
            'Regular orchard monitoring'
        ],
        'organic_remedies': [
            'Soap solution spray',
            'Neem oil application',
            'Vermiwash spray',
            'Cow urine spray'
        ]
    }
}

def translate_text(text, to_lang='hi'):
    try:
        return GoogleTranslator(source='auto', target=to_lang).translate(str(text))
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def get_display_text(text, is_hindi=False):
    return translate_text(text) if is_hindi else text

def language_toggle():
    st.sidebar.header("Language Settings / भाषा सेटिंग्स")
    is_hindi = st.sidebar.toggle("English / हिंदी", value=False)
    return is_hindi

def generate_pdf_report(disease_name, confidence, info, image_path=None, is_hindi=False):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import black
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        from PIL import Image
        
        # Prepare output stream
        output = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(output, pagesize=letter, 
                                rightMargin=72, leftMargin=72, 
                                topMargin=72, bottomMargin=18)
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles with DejaVu Sans
        title_style = ParagraphStyle(
            'Title', 
            parent=styles['Title'], 
            fontName='DejaVuSans-Bold', 
            fontSize=16, 
            textColor=black
        )
        
        heading_style = ParagraphStyle(
            'Heading', 
            parent=styles['Heading3'], 
            fontName='DejaVuSans-Bold', 
            fontSize=14, 
            textColor=black
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='DejaVuSans',
            fontSize=12,
            textColor=black
        )
        
        # Prepare story for PDF
        story = []
        
        # Title (always in English)
        story.append(Paragraph(f"Mango Leaf Disease Report: {disease_name}", title_style))
        story.append(Spacer(1, 12))
        
        # Confidence
        story.append(Paragraph(f"Detection Confidence: {confidence:.2f}%", heading_style))
        story.append(Spacer(1, 8))

        # Scientific Name
        if 'scientific_name' in info:
            story.append(Paragraph("<b>Scientific Name:</b> " + info['scientific_name'], body_style))
            story.append(Spacer(1, 8))

        # Symptoms
        if 'symptoms' in info:
            story.append(Paragraph("<b>Symptoms:</b>", heading_style))
            for sym in info['symptoms']:
                story.append(Paragraph(f"• {sym}", body_style))
            story.append(Spacer(1, 8))

        # Treatments
        if 'treatments' in info:
            story.append(Paragraph("<b>Treatments:</b>", heading_style))
            for treat in info['treatments']:
                story.append(Paragraph(f"• {treat}", body_style))
            story.append(Spacer(1, 8))

        # Preventive Measures
        if 'preventive_measures' in info:
            story.append(Paragraph("<b>Preventive Measures:</b>", heading_style))
            for pm in info['preventive_measures']:
                story.append(Paragraph(f"• {pm}", body_style))
            story.append(Spacer(1, 8))

        # Organic Remedies
        if 'organic_remedies' in info:
            story.append(Paragraph("<b>Organic Remedies:</b>", heading_style))
            for remedy in info['organic_remedies']:
                story.append(Paragraph(f"• {remedy}", body_style))
            story.append(Spacer(1, 12))

        # Add image if provided
        if image_path:
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
                aspect = img_height / img_width
                img_width = 4 * inch  # Set a standard width
                img_height = img_width * aspect
                img_obj = RLImage(image_path, width=img_width, height=img_height)
                story.append(img_obj)
            except Exception as e:
                print(f"Could not add image: {e}")
        story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        return output.getvalue()
    
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def load_model():
    return tf.keras.models.load_model('models/best_model.keras')

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    img = image.resize((224, 224))
    img_array = np.array(img)
    # Convert image from RGBA to RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prediction(model, image):
    """Get prediction for the uploaded image"""
    class_names = ['Anthracnose', 'Cutting Weevil', 'Die Back', 
                  'Gall Midge', 'Powdery Mildew', 'Sooty Mould']
    
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence

def predict(image):
    model = load_model()
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class = ['Anthracnose', 'Cutting Weevil', 'Die Back', 
                  'Gall Midge', 'Powdery Mildew', 'Sooty Mould'][np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence

def display_disease_info(disease, info, is_hindi=False):
    st.subheader(get_display_text(disease, is_hindi))
    
    # Scientific Name
    st.write("**" + get_display_text("Scientific Name:", is_hindi) + "**", info['scientific_name'])
    
    # Create tabs for different information sections
    tabs = st.tabs([get_display_text("Symptoms", is_hindi), get_display_text("Treatments", is_hindi), get_display_text("Preventive Measures", is_hindi), get_display_text("Organic Remedies", is_hindi)])
    
    with tabs[0]:
        st.write("### " + get_display_text("Symptoms", is_hindi))
        for symptom in info['symptoms']:
            st.write("• " + get_display_text(symptom, is_hindi))
    
    with tabs[1]:
        st.write("### " + get_display_text("Treatments", is_hindi))
        for treatment in info['treatments']:
            st.write("• " + get_display_text(treatment, is_hindi))
    
    with tabs[2]:
        st.write("### " + get_display_text("Preventive Measures", is_hindi))
        for measure in info['preventive_measures']:
            st.write("• " + get_display_text(measure, is_hindi))
    
    with tabs[3]:
        st.write("### " + get_display_text("Organic Remedies", is_hindi))
        for remedy in info['organic_remedies']:
            st.write("• " + get_display_text(remedy, is_hindi))

def main():
    # Language toggle
    is_hindi = language_toggle()
    
    st.title(get_display_text("Mango Plant Leaf Disease Detection And Recommendation", is_hindi))
    
    # File uploader with container width
    uploaded_file = st.file_uploader(
        get_display_text("Upload a mango leaf image", is_hindi),
        type=['jpg', 'jpeg', 'png'],
        help=get_display_text("Please upload a clear image of a mango leaf", is_hindi)
    )
    
    if uploaded_file is not None:
        # Display image with container width
        image = Image.open(uploaded_file)
        st.image(image, caption=get_display_text('Uploaded Image', is_hindi))
        
        # Prediction and display
        disease, confidence = predict(image)
        
        if disease:
            # Result display
            st.success(get_display_text(f"Detected Disease: {disease} (Confidence: {confidence:.2f}%)", is_hindi))
            
            # Disease information
            display_disease_info(disease, DISEASE_INFO[disease], is_hindi)
            
            # PDF Report Generation
            pdf_bytes = generate_pdf_report(disease, confidence, DISEASE_INFO[disease], uploaded_file)
            
            # Download button
            st.download_button(
                label=get_display_text("Download Report as PDF", is_hindi),
                data=pdf_bytes,
                file_name=f"mango_disease_report_{disease.lower().replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
    
    # System Features Section
    with st.expander(get_display_text("System Features and Roadmap", is_hindi)):
        st.markdown("""
        ### """ + get_display_text("Current Features", is_hindi) + """
        - 🌿 """ + get_display_text("AI-Powered Disease Detection", is_hindi) + """
        - 📋 """ + get_display_text("Detailed Disease Information", is_hindi) + """
        - 📄 """ + get_display_text("PDF Report Generation", is_hindi) + """
        - 🌱 """ + get_display_text("Treatment Recommendations", is_hindi) + """

        ### """ + get_display_text("Upcoming Features", is_hindi) + """
        - 🚀 """ + get_display_text("Mobile App Integration", is_hindi) + """
        - 🌦️ """ + get_display_text("Weather-based Recommendations", is_hindi) + """
        - 👥 """ + get_display_text("Farmer Community Network", is_hindi) + """
        - 📊 """ + get_display_text("Advanced Disease Severity Analysis", is_hindi) + """
        """)
if __name__ == '__main__':
    main()