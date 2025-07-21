from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF

# Request Models
class StoryRequest(BaseModel):
    story: str
    instructions: str = ""

class QuestionRequest(BaseModel):
    story: str
    bullet: str

class ChatRequest(BaseModel):
    story: str
    bullet: str
    questions: str
    user_query: str


# LLM Service
class LLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self._init_chains()

    def _build_chain(self, prompt: PromptTemplate):
        return prompt | self.llm | StrOutputParser()

    def _init_chains(self):
        self.bullet_chain = self._build_chain(PromptTemplate(
            input_variables=["story", "instructions"],
            template="""..."""
        ))

        self.highlight_chain = self._build_chain(PromptTemplate(
            input_variables=["bullet"],
            template="""..."""
        ))

        self.refine_chain = self._build_chain(PromptTemplate(
            input_variables=["raw_bullets"],
            template="""..."""
        ))

        self.questions_chain = self._build_chain(PromptTemplate(
            input_variables=["bullet", "story"],
            template="""..."""
        ))

        self.chat_chain = self._build_chain(PromptTemplate(
            input_variables=["story", "bullet", "questions", "user_query"],
            template="""..."""
        ))

        self.resume_parse_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text"],
            template="""
                You are a resume parsing assistant. Extract the following:
                - Sections and its content from the resume
                - Use markdown headers like ### Section
                - Use bullets for list items

                Resume Text:
                {resume_text}

                Parsed Output:
            """
        ))

        self.resume_analysis_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text", "job_title", "job_description"],
            template="""
                You are a resume analysis assistant.

                Analyze how well the following resume matches the job description. Score based on:
                - Overall match
                - Keyword alignment (matched/missing)
                - Skills analysis (present/suggested)
                - ATS compatibility (0-100)
                - Readability (0-100)
                - Format structure and length
                - Impact of phrasing
                - Suggestions for improvement

                Return JSON only. Structure:
                {{
                    "overallScore": <int>,
                    "overallMatch": <int>,
                    "readabilityScore": <int>,
                    "atsCompatibilityScore": <int>,
                    "keywordAnalysis": {{
                        "relevantKeywords": [<list>],
                        "missingKeywords": [<list>]
                    }},
                    "impactAnalysis": {{
                        "impactfulPhrases": [<list>],
                        "weakPhrases": [<list>]
                    }},
                    "skillsGapAnalysis": {{
                        "presentSkills": [<list>],
                        "suggestedSkills": [<list>]
                    }},
                    "formatAnalysis": {{
                        "structure": "<string>",
                        "length": "<string>"
                    }},
                    "overallImprovementSuggestions": [<list>],
                    "generalRecommendations": [<list>],
                    "industrySpecificFeedback": "<string>"
                }}

                Resume Text:
                {resume_text}

                Job Title:
                {job_title}

                Job Description:
                {job_description}
            """
        ))


        self.keyword_match_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text", "job_description"],
            template="""
            Compare the resume against the job description. Identify and return:
            - relevantKeywords (matched from JD)
            - missingKeywords (important ones not in resume)

            Return JSON:
            {{
                "relevantKeywords": [<list>],
                "missingKeywords": [<list>]
            }}

            Resume:
            {resume_text}

            Job Description:
            {job_description}
            """
        ))

        self.skills_gap_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text", "job_description"],
            template="""
            From the resume and job description, analyze skills.

            Return JSON:
            {{
                "presentSkills": [<list>],
                "suggestedSkills": [<list>]
            }}

            Resume:
            {resume_text}

            Job Description:
            {job_description}
            """
        ))

        self.ats_format_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text"],
            template="""
            Evaluate resume formatting and readability for ATS systems.

            Return JSON:
            {{
                "atsCompatibilityScore": <int>,  # 0-100
                "readabilityScore": <int>,       # 0-100
                "formatAnalysis": {{
                    "structure": "<string>",
                    "length": "<string>"
                }}
            }}

            Resume:
            {resume_text}
            """
        ))

        self.impact_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text"],
            template="""
            Analyze the language impact in resume. Identify:

            Return JSON:
            {{
                "impactfulPhrases": [<list>],
                "weakPhrases": [<list>]
            }}

            Resume:
            {resume_text}
            """
        ))

        self.summary_suggestions_chain = self._build_chain(PromptTemplate(
            input_variables=["resume_text", "job_title", "job_description"],
            template="""
            Analyze overall resume strength for the job title and suggest improvements.

            Return JSON:
            {{
                "overallScore": <int>,         # 0-100
                "overallMatch": <int>,         # 0-100
                "overallImprovementSuggestions": [<list>],
                "generalRecommendations": [<list>],
                "industrySpecificFeedback": "<string>"
            }}

            Resume:
            {resume_text}

            Job Title:
            {job_title}

            Job Description:
            {job_description}
            """
        ))


    def generate_bullet(self, story: str, instructions: str = "") -> str:
        return self.bullet_chain.invoke({"story": story, "instructions": instructions})

    def refine_bullet(self, raw_bullets: str) -> str:
        return self.refine_chain.invoke({"raw_bullets": raw_bullets})

    def highlight_bullet(self, bullet: str) -> str:
        return self.highlight_chain.invoke({"bullet": bullet})

    def generate_questions(self, bullet: str, context: str) -> str:
        return self.questions_chain.invoke({"bullet": bullet, "story": context})

    def chat(self, story: str, bullet: str, questions: str, user_query: str) -> str:
        return self.chat_chain.invoke({
            "story": story,
            "bullet": bullet,
            "questions": questions,
            "user_query": user_query
        })

    def parse_resume_text(self, resume_text: str) -> str:
        return self.resume_parse_chain.invoke({"resume_text": resume_text})

    def analyze_resume(self, resume_text: str, job_title: str, job_description: str) -> dict:
        import json, re

        def try_parse(raw):
            try:
                return json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
            except:
                return {}

        try:
            keyword_data = try_parse(self.keyword_match_chain.invoke({
                "resume_text": resume_text, "job_description": job_description
            }))

            skill_data = try_parse(self.skills_gap_chain.invoke({
                "resume_text": resume_text, "job_description": job_description
            }))

            ats_data = try_parse(self.ats_format_chain.invoke({
                "resume_text": resume_text
            }))

            impact_data = try_parse(self.impact_chain.invoke({
                "resume_text": resume_text
            }))

            summary_data = try_parse(self.summary_suggestions_chain.invoke({
                "resume_text": resume_text,
                "job_title": job_title,
                "job_description": job_description
            }))

            # Combine all results into one response
            return {
                "overallScore": summary_data.get("overallScore", 0),
                "overallMatch": summary_data.get("overallMatch", 0),
                "readabilityScore": ats_data.get("readabilityScore", 0),
                "atsCompatibilityScore": ats_data.get("atsCompatibilityScore", 0),
                "keywordAnalysis": {
                    "relevantKeywords": keyword_data.get("relevantKeywords", []),
                    "missingKeywords": keyword_data.get("missingKeywords", [])
                },
                "impactAnalysis": {
                    "impactfulPhrases": impact_data.get("impactfulPhrases", []),
                    "weakPhrases": impact_data.get("weakPhrases", [])
                },
                "skillsGapAnalysis": {
                    "presentSkills": skill_data.get("presentSkills", []),
                    "suggestedSkills": skill_data.get("suggestedSkills", [])
                },
                "formatAnalysis": ats_data.get("formatAnalysis", {
                    "structure": "", "length": ""
                }),
                "overallImprovementSuggestions": summary_data.get("overallImprovementSuggestions", []),
                "generalRecommendations": summary_data.get("generalRecommendations", []),
                "industrySpecificFeedback": summary_data.get("industrySpecificFeedback", "")
            }

        except Exception as e:
            print("Failed full analysis:", e)
            return {
                "overallScore": 0,
                "overallMatch": 0,
                "readabilityScore": 0,
                "atsCompatibilityScore": 0,
                "keywordAnalysis": {
                    "relevantKeywords": [],
                    "missingKeywords": []
                },
                "impactAnalysis": {
                    "impactfulPhrases": [],
                    "weakPhrases": []
                },
                "skillsGapAnalysis": {
                    "presentSkills": [],
                    "suggestedSkills": []
                },
                "formatAnalysis": {
                    "structure": "",
                    "length": ""
                },
                "overallImprovementSuggestions": [],
                "generalRecommendations": [],
                "industrySpecificFeedback": ""
            }




# Resume Scorer
class ResumeScorer:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def parse_resume(self, text: str):
        parsed = self.llm_service.parse_resume_text(text)
        return parsed


# Controller
class ResumeController:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_resume_content(self, req: StoryRequest):
        raw_bullet = self.llm_service.generate_bullet(req.story, req.instructions)
        bullet = self.llm_service.refine_bullet(raw_bullet)
        highlighted_bullet = self.llm_service.highlight_bullet(bullet)
        return {
            "bullet": bullet,
            "highlighted_bullet": highlighted_bullet,
            "questions": ""
        }

    def generate_questions(self, req: QuestionRequest):
        questions = self.llm_service.generate_questions(req.bullet, req.story)
        return {"questions": questions}

    def chat_response(self, req: ChatRequest):
        reply = self.llm_service.chat(
            req.story, req.bullet, req.questions, req.user_query
        )
        return {"reply": reply}


# FastAPI App
def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    llm_service = LLMService()
    controller = ResumeController(llm_service)
    scorer = ResumeScorer(llm_service)

    @app.post("/generate")
    async def generate_output(req: StoryRequest):
        return controller.generate_resume_content(req)

    @app.post("/generate_questions")
    async def generate_questions(req: QuestionRequest):
        return controller.generate_questions(req)

    @app.post("/chat")
    async def chat_response(req: ChatRequest):
        return controller.chat_response(req)

    @app.post("/parse_resume")
    async def parse_resume(
        resume: UploadFile = File(...),
        job_title: str = Form(...),
        job_description: str = Form(...)
    ):
        if resume.content_type != "application/pdf":
            return {"error": "Only PDF files are supported."}
        try:
            contents = await resume.read()
            doc = fitz.open(stream=contents, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            parsed = scorer.parse_resume(text)
            scores = scorer.llm_service.analyze_resume(parsed, job_title, job_description)
            return {
                "analysis": scores
            }
        except Exception as e:
            return {"error": f"Failed to parse resume: {str(e)}"}

    return app


app = create_app()
