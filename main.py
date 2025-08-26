import os
import tempfile
import logging
from pathlib import Path
from git import Repo
from langchain_community.chat_models import ChatOllama  # 👈 using community wrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("repo_analysis.log"),
        logging.StreamHandler()
    ]
)

# ✅ Choose Ollama model here (must be pulled or running)
MODEL_NAME = "phi"  # or llama3, mistral, phi, etc.

# Initialize the Ollama model
llm = ChatOllama(model=MODEL_NAME)

def clone_repo(repo_url):
    temp_dir = tempfile.mkdtemp()
    logging.info(f"Cloning repository: {repo_url}")
    Repo.clone_from(repo_url, temp_dir)
    logging.info(f"Repository cloned to: {temp_dir}")
    return temp_dir

def get_code_files(path):
    extensions = [".py", ".js", ".ts", ".java", ".cpp", ".go", ".rb", ".cs"]
    files = [f for f in Path(path).rglob("*") if f.suffix in extensions]
    logging.info(f"Found {len(files)} code files")
    return files

def analyze_code(file_path):
    logging.info(f"Analyzing file: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        prompt = f"""
You are an expert software engineer and code reviewer. 

Please analyze the JavaScript code below. Your response must include:
1. A brief summary of what this code does.
2. A list of any issues, bugs, or areas of improvement.
3. Suggested code improvements or best practices, with updated code if needed.

Only respond with the analysis — do not ask questions or thank the user.

File name: {file_path.name}

```javascript
{code}
"""
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        logging.error(f"Error analyzing {file_path}: {e}")
        return f"Error analyzing {file_path.name}: {e}"

def main(repo_url):
    repo_path = clone_repo(repo_url)
    code_files = get_code_files(repo_path)

    for file in code_files:
        result = analyze_code(file)
        logging.info(f"Finished analysis of {file.name}")
        print(f"\n📄 {file.relative_to(repo_path)}\n{result}\n" + "="*80)

if __name__ == "__main__":
    repo_url = input("Enter a public GitHub repository URL: ")
    main(repo_url)
