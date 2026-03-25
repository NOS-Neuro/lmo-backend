from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator


class CompetitorIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    website: HttpUrl

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Competitor name too short")
        return v


class QuestionIn(BaseModel):
    prompt_name: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=10, max_length=1000)


class ScanRequest(BaseModel):
    businessName: str = Field(..., min_length=2, max_length=200)
    industry: Optional[str] = Field(default=None, max_length=100)
    website: HttpUrl
    contactEmail: EmailStr
    requestContact: bool = False
    captchaToken: str = Field(..., min_length=10)
    models: List[str] = Field(default=[], max_length=5)
    competitors: List[CompetitorIn] = Field(default=[], max_length=10)
    questions: List[QuestionIn] = Field(default=[], max_length=20)

    @field_validator("businessName")
    @classmethod
    def validate_business_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("Business name too short")
        return v


class QAPair(BaseModel):
    question: str
    answer: str
    prompt_name: Optional[str] = None


class ScanResponse(BaseModel):
    scan_id: Optional[str] = None
    created_at: Optional[str] = None
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int
    package_recommendation: str
    package_explanation: str
    strategy_summary: str
    findings: List[str]
    qa_pairs: Optional[List[QAPair]] = None
    email_sent: Optional[bool] = None
    entity_status: Optional[str] = None
    entity_confidence: Optional[int] = None
    warnings: Optional[List[str]] = None
    disclaimer: str = (
        "This scan is evidence-based when run in Real Scan mode. "
        "Fallback mode is an honest AI-assisted estimate."
    )


class ScanStatusResponse(BaseModel):
    scan_id: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    failure_message: Optional[str] = None
    result: Optional[ScanResponse] = None
