from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Import necessary components for chaining with dictionaries
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class StoryRequest(BaseModel):
    story: str
    instructions: str = ""

class ChatRequest(BaseModel):
    story: str
    bullet: str
    questions: str
    user_query: str

# Prompts
bullet_prompt = PromptTemplate(
    input_variables=["story", "instructions"],
    template="""
            You are a resume bullet point generator.

            Convert the following paragraph into 2–3 single-line bullet points.

            Rules:
            - Each bullet must start with a **strong action verb** or impact phrase
            - Follow What → How → Effect structure
            - Each bullet must be a **single line only**, fitting a standard resume without wrapping
            - Start and end each bullet on the same line (no orphan words)
            - Bullet 1 = objective/context
            - Bullet 2 = what you built + outcome
            - Bullet 3 = effect or value added (if applicable)
            - Output must always be 2 or 3 bullets, no explanations

            User Instructions: {instructions}

            Input Paragraph:
            {story}

            Output (formatted bullet points only):
        """
    )

# Corrected chain: prompt | llm | parser
bullet_chain = bullet_prompt | llm | StrOutputParser()

highlight_prompt = PromptTemplate(
    input_variables=["bullet"],
    template="""
        You are a resume enhancement expert. Rewrite the following resume bullet point by highlighting **important words**, **technologies**, or **action verbs** using Markdown-style bold (e.e.g., **Python**, **optimized**, **95%**).

        Resume Bullet: {bullet}

        Output (with important keywords in bold):
    """
    )

# Corrected chain: prompt | llm | parser
highlight_chain = highlight_prompt | llm | StrOutputParser()


refine_prompt = PromptTemplate(
    input_variables=["raw_bullets"],
    template = """
        You are a resume formatting assistant.

            Your task is to refine the given resume bullets using the following rules:

            - Begin each bullet with a strong **action verb**
            - Follow the **What - How - Effect** structure
            - Each bullet must be a **single line only**, with **no line breaks or wrapping**
            - Use as much space as possible, but **do not exceed 110 characters or 20 words**
            - Avoid short or vague lines — make each line information-dense and valuable
            - Output **exactly 2 or 3 refined bullet points**
            - Return only the final list of bullets — no explanation or extra text

            Raw Bullets:
            {raw_bullets}

            Refined Output:
      """
)

# Corrected chain: prompt | llm | parser
refine_chain = refine_prompt | llm | StrOutputParser()

questions_prompt = PromptTemplate(
    input_variables=["bullet"],
    template="""
        You are a technical interviewer. Based on the following resume bullet point, generate 3 to 5 interview questions that assess technical depth, decision-making, tools used, and measurable impact.

        Guidelines:
        - Do not explain anything.
        - Only list the questions.
        - Make them short, relevant and concise.

        Resume Line:
        "{bullet}"

        Output:
    """
)

# Corrected chain: prompt | llm | parser
questions_chain = questions_prompt | llm | StrOutputParser()

chat_prompt = PromptTemplate(
    input_variables=["story", "bullet", "questions", "user_query"],
    template="""
        You are a career assistant helping a user prepare for interviews.

        Context:
        - Original story: {story}
        - Resume bullet point: {bullet}
        - Suggested interview questions: {questions}

        User's question: {user_query}

        Answer:
        - Be relevant to the resume bullet and job preparation.
        - Keep it clear, practical, and helpful.
        - Clearly convey: What was done + How it was done + What impact was achieved (if applicable).
        """
)
# Corrected chain: prompt | llm | parser
chat_chain = chat_prompt | llm | StrOutputParser()

# Routes

@app.post("/generate")
async def generate_output(req: StoryRequest):
    # First generate raw bullets from story and instructions
    # Input dictionary should directly match prompt's input_variables
    raw_bullet = bullet_chain.invoke({
        "story": req.story,
        "instructions": req.instructions
    })
    print("Raw Bullet:", raw_bullet)

    # Refine the raw bullets using validation/refinement LLM
    # Input dictionary should directly match prompt's input_variables
    bullet = refine_chain.invoke({
        "raw_bullets": raw_bullet
    })
    print("Refined Bullet:", bullet)

    # Highlight keywords in the refined bullet
    # Input dictionary should directly match prompt's input_variables
    highlighted_bullet = highlight_chain.invoke({
        "bullet": bullet
    })
    print("Highlighted Bullet:", highlighted_bullet)

    # (Optional) Generate questions if needed later
    questions = ""  # Placeholder
    # questions = questions_chain.invoke({ "bullet": bullet })

    return {
        "bullet": bullet,
        "highlighted_bullet": highlighted_bullet,
        "questions": questions
    }

@app.post("/chat")
async def chat_response(req: ChatRequest):
    # Input dictionary should directly match prompt's input_variables
    reply = chat_chain.invoke(
        {
            "story": req.story,
            "bullet": req.bullet,
            "questions": req.questions,
            "user_query": req.user_query
        }
    )
    return {"reply": reply}