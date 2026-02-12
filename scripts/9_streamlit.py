import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Fashion AI Recommender",
    page_icon="🛍️",
    layout="wide"
)

# Title
st.title("🎯 AI Fashion Recommendation System")
st.markdown("Upload product image and review to get AI-powered recommendation!")

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Product Image")
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        # Resize image for better display
        max_size = (300, 300)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        st.image(image, caption="Uploaded Image", width=300)  # Fixed width

with col2:
    st.subheader("📝 Product Review")
    review_text = st.text_area(
        "Enter product review:",
        placeholder="Example: This shirt is amazing! Perfect fit and great quality...",
        height=150
    )

# Predict button
if st.button("🎯 Get Recommendation", type="primary"):
    if uploaded_image is not None and review_text:
        try:
            # Prepare data for API
            files = {"image": uploaded_image.getvalue()}
            data = {"review_text": review_text}
            
            # Show loading
            with st.spinner("🤖 AI is analyzing your product..."):
                # Call FastAPI
                response = requests.post(
                    "http://localhost:8000/predict",
                    files=files,
                    data=data
                )
                
            if response.status_code == 200:
                result = response.json()
                
                if result["success"]:
                    # CHECK: Agar confidence low hai to unknown product
                    vision_confidence = result["vision_confidence"]
                    
                    if vision_confidence < 0.6:  # 60% threshold
                        st.error("❌ **Unknown Product** - This doesn't appear to be a shirt or shoe. Please upload fashion product images only.")
                        
                        # Show low confidence results
                        st.warning(f"Detected as: {result['vision_prediction']} (Low confidence: {vision_confidence:.1%})")
                        
                    else:
                        # Display results (only if confidence is high)
                        st.success("✅ Analysis Complete!")
                        
                        # Results in columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            confidence_color = "🟢" if vision_confidence > 0.8 else "🟡"
                            st.metric(
                                label=f"{confidence_color} Product Type",
                                value=result["vision_prediction"],
                                delta=f"{vision_confidence:.1%} confidence"
                            )
                        
                        with col2:
                            st.metric(
                                label="😊 Sentiment",
                                value=result["sentiment"],
                                delta=f"{result['sentiment_confidence']:.1%} confidence"
                            )
                        
                        with col3:
                            # Color based on score
                            score = result["recommendation_score"]
                            if score > 0.7:
                                color = "🟢"
                            elif score > 0.4:
                                color = "🟡" 
                            else:
                                color = "🔴"
                                
                            st.metric(
                                label=f"{color} Recommendation Score",
                                value=f"{score:.2f}",
                                delta=result["recommendation_status"]
                            )
                        
                        # Progress bar for recommendation
                        st.progress(score)
                        
                        # Final verdict
                        st.subheader("🎯 Final Verdict")
                        if score > 0.7:
                            st.success(f"**{result['recommendation_status']}** - This product has high potential!")
                        elif score > 0.4:
                            st.warning(f"**{result['recommendation_status']}** - Consider other options too.")
                        else:
                            st.error(f"**{result['recommendation_status']}** - Not recommended for purchase.")
                        
                else:
                    st.error(f"Error: {result['error']}")
            else:
                st.error("Server error. Please try again.")
                
        except Exception as e:
            st.error(f"Connection error: Make sure the API server is running!")
            
    else:
        st.warning("⚠️ Please upload an image and enter review text")

# Instructions
st.markdown("---")
st.subheader("📖 How to Use:")
st.markdown("""
1. **Upload** a product image (Shirt or Shoe)
2. **Enter** a product review (Positive/Negative/Neutral)  
3. **Click** 'Get Recommendation' button
4. **View** AI-powered analysis and recommendation
""")

st.info("💡 **Tip:** For best results, upload clear images of shirts or shoes only!")