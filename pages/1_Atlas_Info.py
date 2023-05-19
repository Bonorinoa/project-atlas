import streamlit as st

# TODO: Fill in the information about Atlas

# this function should only provide information about Atlas (Home page)
def main():
    
    st.title("Atlas Intelligence: your path to well-being")
    
    st.subheader("About Atlas")
    st.write("Atlas Intelligence leverages the latest in Artificial Intelligence, Wellness, and Behavioral science to offer users access to a holistic wellness coach in their pocket.\n " \
            + "After users answer a short questionnaire based on the PERMA+4 wellness model, Atlas generates a comprehensive wellness report, identifies users’ top three areas for wellness improvement, helps users set goals aimed at improving these areas, and suggests some actions to help users get started.\n" \
            + "Users can then select a goal and interact with Atlas’ in-built AI wellness coach and digital nudging protocols to define a concrete path forward and ensure success throughout their wellness journeys.")
    
    st.subheader("Atlas Mission")
    st.write("To close the wellness service gap and help people all around the globe become the best version of themselves.")
    
    st.subheader("Atlas Vision")
    st.write("We hope to offer low cost, personalized wellness coaching that changes and adapts to best suit users’ needs throughout the lifespan and around the world.\n " \
           + "We envision a world where everyone has access to cutting edge wellness coaching and self-improvement techniques at their fingertips, a world where cost is no longer a prohibitive factor in people realizing their highest potential and living the best lives they possibly can, and a world in which the help we need is always at the push of a button.")

if __name__ == "__main__":
    main()