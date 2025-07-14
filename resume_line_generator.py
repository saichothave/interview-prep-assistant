from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    story: str
    instructions: str = ""


class ChatRequest(BaseModel):
    story: str
    bullet: str
    questions: str
    user_query: str

    
def run_ollama_prompt(input_text, additional_instructions="", model="mistral"):
    prompt = f"""
       You are a professional resume writing assistant. Convert the following user experience story into a high-impact resume bullet using the what-how-effect structure.

        General Expectations:

        Begin with a strong action verb.

        Clearly convey: What was done + How it was done + What impact or metric was achieved.

        Mention relevant tools or technologies.

        Absolutely avoid personal pronouns like "I" or "my".

        Formatting Rules:

        If the user provides additional instructions, you must follow them strictly.

        If no additional instruction is given:

        Output exactly one sentence between 80 and 95 characters (including spaces).

        Do not use more or fewer characters than this range.

        Do not add explanations, line breaks, or extra formatting.

        Avoid orphan lines: do not end with 2-3 weak words on a separate line. Optimize the sentence so it fills a full line effectively.

        Additional Instructions (optional): {additional_instructions}

        Input Story: {input_text}

        Output (resume bullet only, no explanation):

    """
    command = ['ollama', 'run', model]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=prompt.encode())
    return stdout.decode()

def run_ollama_questions_prompt(resume_line, model="mistral"):
    prompt = f"""
        You are a technical interviewer. Based on the following resume bullet point, generate 3 to 5 interview questions that assess technical depth, decision-making, tools used, and measurable impact.

        Guidelines:
        - Do not explain anything.
        - Only list the questions.
        - Make them relevant and concise.

        Resume Line:
        \"{resume_line}\"

        Output:
    """
    command = ['ollama', 'run', model]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=prompt.encode())
    return stdout.decode()

def run_ollama_chat_prompt(story, bullet, questions, user_query, model="mistral"):
    prompt = f"""
        You are a career assistant helping a user prepare for interviews.

        Context:
        - Original story: {story}
        - Resume bullet point: {bullet}
        - Suggested interview questions: {questions}

        User's question: {user_query}

        Answer:
        - Be relevant to the resume bullet and job preparation.
        - Keep it clear, practical, and helpful.
    """
    command = ['ollama', 'run', model]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=prompt.encode())
    return stdout.decode()


@app.post("/generate")
async def generate_output(req: StoryRequest):
    bullet = run_ollama_prompt(req.story, req.instructions)
    questions = run_ollama_questions_prompt(bullet)
    return {"bullet": bullet.strip(), "questions": questions.strip()}


@app.post("/chat")
async def chat_response(req: ChatRequest):
    reply = run_ollama_chat_prompt(req.story, req.bullet, req.questions, req.user_query)
    return {"reply": reply.strip()}
