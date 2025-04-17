# ⚖️ **LegalRe: AI-Powered Legal Assistant**

[![GitHub stars](https://img.shields.io/github/stars/lawglance/lawglance?style=social)](https://github.com/lawglance/lawglance/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/lawglance/lawglance?style=social)](https://github.com/lawglance/lawglance/forks)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yrS2Kp-kprYWot_sEu7JeWMIRAei_vov?usp=sharing)
[![Loom](https://img.shields.io/badge/Loom-Tutorial-8A2BE2?logo=loom)](https://www.loom.com/share/dcc6b14c653c4618829f46a9aa2ab68c?sid=00d0d3c1-9d4b-4cf7-8684-cdee76718bd5)
[![LangChain](https://img.shields.io/badge/LangChain-Open%20Source-5e9cff?logo=langchain&logoColor=white)](https://python.langchain.com/docs/introduction/)
[![Crew AI](https://img.shields.io/badge/Crew%20AI-Multi--Agent%20Workflows-00bda?style=flat-square)](https://www.crewai.com/)

### _Bridging the Gap Between People and Legal Access_ 🌍

🌐 **Website:** [LawGlance](https://lawglance.com/)

**LegalRe** is a free, open-source, people-centric initiative 💡 designed to make legal guidance accessible to everyone, based on provided documents. Using **AI-powered Retriever-Augmented Generation (RAG)**, **LegalRe** delivers quick, accurate legal support tailored to your needs by searching through your document collection.

> 🛡️ **Mission:** “Justice should be accessible to everyone. LegalRe aims to provide tools for better access to information within specific legal documents.”

This project is developed with support from mentors and experts at [Data Science Academy](https://datascience.one/) and [Curvelogics](https://www.curvelogics.com/). 💼

---

## 📚 **Legal Coverage**

**LegalRe** works with the PDF documents you provide in the `pdf_data` directory. The accuracy and scope of its responses depend entirely on the content of those documents.

---

## 🎥 **Video Tutorial**

## 💻 **Developer Quick Start Guide**

Ready to get started? Follow these simple steps to set up **LegalRe** on your machine:

1. **Clone the Repository** 🌀

   ```bash
   git clone https://github.com/lawglance/lawglance.git
   cd legalre
   ```

2. **Install uv** 📂

   First, let's install uv and set up our Python project and environment

   MacOS/Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Windows:

   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.

3. **Install Dependencies** 📦

   ```bash
   uv sync
   ```

4. **Set Your Groq API Key** 🔑

   Open `.env` and add your Groq API key:

   ```bash
   GROQ_API_KEY=your-api-key-here
   ```

5. **Add Your Documents** 📄
   Create a folder named `pdf_data` in the root of the project directory.
   Place all your legal PDF documents inside this `pdf_data` folder.

6. **Generate Embeddings** ✨
   Run the embedding script to process your PDFs and create the vector database:

   ```bash
   cd src
   python pdf_emb.py
   cd ..
   ```

7. **Run the Application** 🚀

   ```bash
   uv run streamlit run app.py
   ```

8. **Access the App** 🌐  
   Open your browser and visit:
   ```bash
   http://127.0.0.1:8501
   ```

---

## 🔧 **Tools & Technologies**

| 💡 **Technology**         | 🔍 **Description**                           |
| ------------------------- | -------------------------------------------- |
| **LangChain**             | Framework for building language applications |
| **ChromaDB**              | Vector database for RAG implementation       |
| **Groq API**              | Powering fast large language model inference |
| **Sentence Transformers** | Generating text embeddings                   |
| **Streamlit**             | Python web framework for the UI              |
| **PyPDF2**                | Extracting text from PDF files               |

---

## 🌟 **Future Roadmap**

Exciting developments are planned for **LegalRe**! Here's what's coming next:

1.  **🤝 Smarter Together: Introducing Our Multi-Agentic Framework 🤖**

    - Imagine a team of specialized AI agents working seamlessly in the background to provide you with the most comprehensive and efficient legal insights. Our new multi-agent framework makes this a reality, boosting platform performance like never before!

2.  **🗣️ Your Voice is the Key: Introducing Voice Interaction 🎙️**

    - Navigate and access legal information effortlessly with our new voice command feature. Simply speak your queries and let LegalRe do the rest – making legal research more intuitive and accessible.

3.  **🌍 Bridging Language Barriers: Multi-Lingual Legal Assistance 🌐**

    - We're committed to serving a global audience. LegalRe will soon offer legal assistance in multiple languages, breaking down communication barriers and making our platform truly inclusive.

4.  **🎯 Precision & Personalization: Advanced Search & Tailored Assistance 🔍**

    - Say goodbye to endless scrolling! Our enhanced search engine will pinpoint the exact legal information you need with lightning speed. Plus, enjoy personalized suggestions and assistance crafted just for you.

5.  **✍️ Draft with Confidence: Introducing Legal Document Generation 📄**

    - Need a contract or agreement? Our upcoming legal document generation feature will empower you to create essential legal documents using customizable templates and intuitive user input.

6.  **🗓️ Stay Organized, Stay Ahead: Introducing Case Management 📁**
    - Effortlessly manage your legal matters with our new case management feature. Track crucial deadlines, appointments, and important events all in one centralized location, keeping you in control.

---

## 🤝 **Contribute**

We are always looking for contributors! Whether you want to help with development, report issues, or request features, we welcome you to fork the repo and submit a pull request. Every contribution helps to make **LegalRe** better for everyone! 🚀

---

**LegalRe** is more than just an AI tool—it's a framework for building custom RAG applications for specific document sets. Together, let's make information more accessible! ✨
