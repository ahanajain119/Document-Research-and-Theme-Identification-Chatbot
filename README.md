# Document Research & Theme Identification Chatbot

A powerful chatbot system designed to analyze documents and identify themes within them. This system combines document processing capabilities with advanced natural language processing to provide meaningful insights from uploaded documents.

## Features

- Document upload and processing (PDF, TXT, Images)
- OCR for scanned documents
- Theme identification and analysis
- Interactive chat interface
- Accurate citations and references
- Vector-based document search

## Project Structure

```
chatbot_theme_identifier/
|-- backend/
|   |-- app/
|   |   |-- api/        # API endpoints
|   |   |-- core/       # Core functionality
|   |   |-- models/     # Data models
|   |   |-- services/   # Business logic
|   |   |-- main.py     # Application entry point
|   |-- config.py       # Configuration settings
|   |-- data/          # Data storage
|   |   |-- uploads/   # Uploaded documents
|   |   |-- processed/ # Processed documents
|   |   |-- faiss_index/ # Vector store
|   |-- requirements.txt # Python dependencies
|-- frontend/
|   |-- app.py         # Streamlit application
|   |-- requirements.txt # Frontend dependencies
|-- docs/              # Documentation
|-- tests/             # Test cases
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd chatbot_theme_identifier
```

2. Set up the backend:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env and add your OpenAI API key
```

3. Set up the frontend:
```bash
cd ../frontend
pip install -r requirements.txt
```

4. Run the application:
```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend
streamlit run app.py
```

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DEBUG`: Set to True for development
- `ENVIRONMENT`: Set to 'development' or 'production'
- `MAX_UPLOAD_SIZE`: Maximum file upload size in bytes
- `MAX_DOCUMENTS`: Maximum number of documents to process

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for the interactive API documentation.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

