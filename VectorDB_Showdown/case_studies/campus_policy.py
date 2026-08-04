"""
Case Study 2 — University Engineering Campus Handbook
===========================================================
One long unstructured document (no metadata tags) that must be CHUNKED
before indexing. This case study is used to demonstrate SCALE & SPEED:

  • Both FAISS and Chroma have to embed and index dozens of chunks.
  • FAISS (in-memory ANN index, backed by the `faiss` C++ library) is
    typically the faster of the two to build and to query at this scale.
  • Chroma carries extra overhead (SQLite-backed metadata store, client
    layer) but gives you a persistent, queryable database out of the box.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HANDBOOK_TEXT = """
ENGINEERING CAMPUS HANDBOOK — ACADEMIC YEAR POLICIES

SECTION 1: ATTENDANCE REQUIREMENTS
All students must maintain a minimum of 75% attendance in each registered course to be
eligible to sit for the end-semester examination. Attendance is calculated per subject,
not as an overall average across subjects. Students falling between 65% and 75% attendance
may apply for condonation through the Head of Department, subject to valid medical or
family emergency documentation submitted within 7 days of the absence. Students below 65%
attendance are debarred from the examination for that subject and must repeat the course
in a subsequent semester. Attendance for lab sessions is tracked separately from theory
lectures and requires a separate 75% threshold, since lab sessions carry practical
examination weight.

SECTION 2: EXAMINATION RULES
End-semester examinations are conducted over a 15-day window at the end of each semester.
Students must carry their institute ID card to every examination; entry without an ID card
requires a temporary pass issued by the examination cell at least one hour before the exam.
Unfair means during an examination — including possession of unauthorized material, use of
smart devices, or copying from another student's answer sheet — results in immediate
cancellation of that examination and a disciplinary hearing before the Academic Integrity
Committee. First-time unfair means violations typically result in a zero for that subject;
repeat violations can lead to suspension for one full semester.

SECTION 3: LAB HOURS AND SAFETY
Engineering labs are open Monday through Saturday, 9:00 AM to 6:00 PM, with extended hours
until 9:00 PM during project submission weeks (announced each semester by the department).
Closed-toe shoes and lab coats are mandatory in all chemical, electronics, and workshop labs;
students without proper safety attire will not be permitted entry. Each lab session must have
a minimum of one Teaching Assistant present for every 20 students. Damage to lab equipment due
to negligence is billed to the student at replacement cost, assessed by the lab-in-charge.

SECTION 4: CAPSTONE PROJECT GUIDELINES
Final-year capstone projects are conducted in teams of 3 to 5 students and span two
semesters. Project topics must be approved by a faculty guide by the end of the third week
of the first semester; late approvals require a written justification to the department
project committee. Mid-term evaluation occurs at the end of the first semester and
contributes 30% of the total capstone grade; the second semester's final demonstration,
report, and viva contribute the remaining 70%. Plagiarism in the capstone report, verified
through similarity-checking software, is treated as an academic integrity violation and can
result in the team failing the entire capstone course.

SECTION 5: SCHOLARSHIPS AND FINANCIAL AID
Merit scholarships covering up to 50% of tuition are awarded to students in the top 5% of
their department by cumulative GPA at the end of each academic year, renewed annually
subject to maintaining a minimum 8.5 CGPA. Need-based financial aid applications open at
the start of every academic year and require submission of family income certificates,
processed by the Student Financial Aid Office within 30 working days. Students receiving
any form of institute scholarship must maintain at least 85% attendance across all subjects,
a stricter threshold than the general 75% requirement, or risk scholarship suspension for
the following semester.

SECTION 6: HOSTEL AND CAMPUS CONDUCT
Hostel curfew is set at 10:30 PM on weekdays and 12:00 AM on weekends; students returning
after curfew must sign the late-entry register at the security desk and may be asked to
provide a reason to the hostel warden. Ragging in any form — physical, verbal, or online —
is strictly prohibited under national anti-ragging law and campus policy; confirmed
incidents result in immediate suspension pending a formal inquiry by the Anti-Ragging
Committee, and in severe cases, expulsion. Visitors are permitted in hostel common areas
only between 4:00 PM and 8:00 PM and must register at the gate with a valid photo ID.

SECTION 7: GRADING AND CREDIT SYSTEM
The institute follows a 10-point CGPA grading scale, with individual subject grades ranging
from O (Outstanding, 90-100%) down to F (Fail, below 40%). Each subject carries a defined
credit weight — typically 3 to 4 credits for theory courses and 1 to 2 credits for
laboratory courses — and a minimum of 160 total credits is required to graduate with a
Bachelor's degree in Engineering. Students may retake a maximum of two subjects per semester
under the improvement scheme to raise a previously passed grade, provided they have not
already used their two-semester improvement quota for that subject.

SECTION 8: INTERNSHIP AND PLACEMENT POLICY
Summer internships between the third and fourth year are mandatory for all engineering
branches and must be of a minimum 6-week duration to count toward the internship credit
requirement. Placement eligibility requires a minimum CGPA of 6.0 with no active backlog
subjects at the time of the placement drive. Students who accept a job offer through the
campus placement process are not permitted to sit for further placement interviews in the
same academic year, in accordance with the one-offer policy agreed upon with recruiting
companies.
""".strip()


def load_documents() -> list[Document]:
    """Chunks the handbook text with the same splitter settings used across
    this repo's other RAG demos (chunk_size=300, chunk_overlap=50) so the
    lecture can compare chunk counts directly against the other RAG scripts."""
    # Splitter tries each separator in order (paragraph, then line, then sentence,
    # then word, then hard-cut) — keeps chunks on natural boundaries where possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
    )
    # Single Document wrapping the whole handbook — the splitter is what fans it out
    # into the many chunks that actually get embedded/indexed.
    doc = Document(page_content=HANDBOOK_TEXT, metadata={"source": "campus_policy_handbook"})
    return splitter.split_documents([doc])
