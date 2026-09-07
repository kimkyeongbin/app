import json
import os
import re
from typing import Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


YNU_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
GPT_MODEL = "claude-sonnet-5"

_client: Any = None


FORBIDDEN_QUESTION_PATTERNS = [
    "성함",
    "이름",
    "배우자",
    "고향",
    "요일",
    "날짜",
    "생년월일",
    "나이",
    "몇 년",
    "몇 월",
    "며칠",
    "치매",
    "검사",
    "진단",
    "정답",
    "맞히",
    "틀렸",
    "기억력",
]


NEGATIVE_ASSUMPTION_PATTERNS = [
    "걱정",
    "불안",
    "우울",
    "외롭",
    "힘들",
    "무섭",
    "신경 쓰",
]


OVERLY_CLINICAL_EMPATHY_PATTERNS = [
    "상담",
    "치료",
    "진료",
    "우울증",
    "불안장애",
    "극복",
    "힘내",
    "괜찮아질",
]


QUESTION_STOPWORDS = {
    "요즘",
    "오늘",
    "최근에",
    "혹시",
    "조금",
    "다시",
    "있으세요",
    "기억나세요",
    "떠오르세요",
    "어떠셨어요",
    "무엇",
    "뭐가",
    "어떤",
    "하나",
    "이번에는",
    "이번에",
    "이어서",
    "그럼",
    "그러면",
    "그러셨군요",
    "그렇군요",
    "그랬군요",
    "알겠습니다",
    "좋으셨겠어요",
    "좋았겠어요",
    "아하",
    "음",
}


EVERYDAY_TOPIC_EXAMPLES = [
    "보고 싶은 사람",
    "오늘 먹은 음식",
    "다녀온 장소",
    "집에서 한 일",
    "본 방송이나 들은 노래",
    "산책이나 이동",
    "어릴 적 기억",
    "최근 떠오른 물건",
    "동네 풍경",
    "계절이나 날씨",
    "시장이나 마트",
    "명절이나 가족 행사",
    "사진이나 오래된 물건",
    "취미나 손으로 하던 일",
]


POSITIVE_CUE_PATTERNS = [
    "좋",
    "재밌",
    "즐거",
    "맛있",
    "상쾌",
    "개운",
    "편안",
    "기쁘",
    "반가",
]


IMPROVEMENT_CUE_PATTERNS = [
    "나아",
    "지금은 괜찮",
    "이제 괜찮",
    "괜찮아졌",
    "호전",
]


NEGATIVE_CUE_PATTERNS = [
    "아프",
    "아파",
    "아팠",
    "다쳤",
    "넘어",
    "심심",
    "외롭",
    "힘들",
    "불편",
    "속상",
    "무서",
    "걱정",
    "피곤",
    "기운이 없",
    "기운 없",
    "재미없",
    "재미 없",
    "재미가 없",
    "맛없",
    "맛이 없",
    "입맛이 없",
    "입맛 없",
    "밥맛이 없",
    "밥맛 없",
    "식욕이 없",
    "식욕 없",
    "안 좋",
    "좋지 않",
    "나쁘",
    "나빴",
    "아쉽",
    "아쉬",
]


NEGATED_NEGATIVE_CUE_PATTERNS = [
    "별로 걱정되지 않",
    "걱정 안",
    "안 아파",
    "안 아프",
    "안 힘든",
    "안 힘들",
    "안 나쁘",
    "안 불편",
    "안 외롭",
    "안 피곤",
    "안 무섭",
    "안 속상",
    "안 슬프",
    "안 우울",
    "힘들지 않",
    "나쁘지 않",
    "불편하지 않",
    "걱정되지 않",
    "걱정하지 않",
    "외롭지 않",
    "피곤하지 않",
    "무섭지 않",
    "속상하지 않",
    "슬프지 않",
    "우울하지 않",
]


NEGATED_NEGATIVE_CUE_PATTERN = re.compile(
    r"(?:"
    r"(?:아프|힘들|나쁘|외롭|무섭|슬프)(?:지|진|지는)\s*않|"
    r"(?:불편하|피곤하|속상하|우울하|걱정되)(?:지|진|지는)\s*않|"
    r"안\s*(?:아파|아프|힘들|나빠|나쁘|불편|외로|피곤|무서|속상|슬프|우울)|"
    r"걱정(?:은|이)?\s*안"
    r")"
)


TOPIC_SIMILARITY_GROUPS = [
    ("방송", "프로그램", "텔레비전", "티비", "tv", "TV", "노래", "가수", "트롯", "미스터트롯", "임영웅"),
    ("집", "집안", "집밖", "집 밖", "집에", "집에서", "방", "거실"),
    ("동네", "풍경", "바깥", "밖에", "나가", "다녀온", "초록", "산책", "공원", "길"),
    ("음식", "식사", "밥", "아침", "점심", "저녁", "김치볶음밥", "피자"),
    ("가족", "아들", "딸", "손주", "배우자", "자식", "연락"),
    ("병원", "약", "진료", "의사", "간호사", "아프", "다쳤", "무릎"),
    ("날씨", "바람", "비", "눈", "햇빛", "더워", "추워", "쌀쌀"),
]


QUESTION_FOCUS_PATTERNS = {
    "PERSON": ("누구", "누가", "사람", "그분", "함께", "같이", "혼자"),
    "PLACE": ("어디", "어느 곳", "장소"),
    "TIME": ("언제", "몇 시", "시간", "아침", "점심", "저녁"),
    "REASON": ("왜", "이유"),
    "FEELING": ("기분", "마음", "어떠셨어"),
    "CATEGORY": (
        "어떤 방송",
        "무슨 방송",
        "어떤 프로그램",
        "무슨 프로그램",
        "어떤 음식",
        "무슨 음식",
        "어떤 노래",
        "무슨 노래",
    ),
    "ACTION": ("무엇을 하", "뭘 하", "하신 일", "하는 일", "하시던 일", "자주 하는"),
    "CONTENT": ("내용", "부분", "장면", "모습", "변화", "어떤 색", "눈에 띄"),
    "FOOD_TARGET": ("음식", "드신 것", "먹은 것"),
    "DETAIL": ("무엇", "뭐", "어떤", "장면", "모습"),
}


GENERIC_ANCHORED_FOLLOWUP_WORDS = {
    "그때",
    "그",
    "이야기",
    "방금",
    "조금",
    "누구",
    "누가",
    "같이",
    "함께",
    "혼자",
    "계셨어요",
    "있었나요",
    "있으셨어요",
    "어디",
    "언제",
    "언제쯤",
    "시간",
    "장면",
    "모습",
    "기분",
    "마음",
    "주변",
    "생각",
    "떠올리면",
    "기억나는",
    "가장",
    "먼저",
    "말씀",
    "일",
    "점",
    "것",
    "드셨어요",
    "가셨어요",
    "보셨어요",
    "하셨어요",
    "그분",
    "사람",
    "최근",
    "최근에",
    "나누셨어요",
    "드셨나요",
    "먹었나요",
    "다녀오셨어요",
    "보셨나요",
    "들으셨어요",
    "하셨나요",
    "어땠나요",
    "어떠셨나요",
    "내용",
    "부분",
    "곳",
    "장소",
    "음식",
    "방송",
    "프로그램",
    "노래",
    "음악",
}


PERSON_GROUNDING_WORDS = (
    "아들",
    "딸",
    "손주",
    "동생",
    "형",
    "누나",
    "언니",
    "오빠",
    "엄마",
    "아빠",
    "어머니",
    "아버지",
    "남편",
    "아내",
    "조카",
    "사촌",
    "친구",
    "가족",
)


GROUNDING_PRONOUN_GROUPS = [
    (PERSON_GROUNDING_WORDS, ("그분", "그 사람")),
    (("먹", "음식", "식사", "밥", "반찬", "국", "찌개", "김치", "피자"), ("그 음식", "드신 것")),
    (("집", "병원", "마트", "시장", "공원", "동네", "장소", "다녀", "갔"), ("그곳", "그 장소")),
    (("방송", "프로그램", "텔레비전", "티비", "드라마", "뉴스"), ("그 방송", "그 프로그램", "그 장면")),
    (("노래", "가수", "들었"), ("그 노래", "그 음악")),
]


def _get_client() -> Any:
    global _client

    if OpenAI is None:
        raise ImportError("openai package is not installed.")

    if _client is None:
        api_key = os.getenv("YNU_API_KEY")

        if not api_key:
            raise EnvironmentError("YNU_API_KEY is not set.")

        _client = OpenAI(
            api_key=api_key,
            base_url=YNU_BASE_URL,
        )

    return _client


def clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _get_final_correction_segment(text: str) -> str:
    text = str(text or "")
    correction_patterns = (
        r"다시\s*생각해\s*보니",
        r"정정(?:할게요?|하면)",
        r"(?:^|[,.;!?]\s*|\s+)(?:아니에요|아니요|아니)(?![가-힣])\s*[,，]?\s*",
    )
    last_end = -1

    for pattern in correction_patterns:
        for match in re.finditer(pattern, text):
            last_end = max(last_end, match.end())

    corrected_text = text if last_end < 0 else text[last_end:].strip()
    alternative_matches = list(
        re.finditer(r"(?:아니라|아니고|말고)\s*", corrected_text)
    )

    if alternative_matches:
        alternative_text = corrected_text[alternative_matches[-1].end():].strip()

        if alternative_text:
            corrected_text = alternative_text

    return corrected_text or text


def _normalize_similarity_word(word: str) -> str:
    word = clean_text(word)

    for suffix in (
        "\uc5d0\uc11c\ub294",
        "\uc5d0\uac8c\ub294",
        "\uc73c\ub85c\ub294",
        "\uc5d0\ub294",
        "\uc5d0\uc11c",
        "\uc73c\ub85c",
        "\uc5d0\uac8c",
        "\ud55c\ud14c",
        "\uc774\ub098",
        "\ub791",
        "\uc640",
        "\uacfc",
        "\ub85c",
        "\uc740",
        "\ub294",
        "\uc774",
        "\uac00",
        "\uc744",
        "\ub97c",
        "\uc5d0",
        "\uc758",
        "\ub3c4",
        "\ub9cc",
    ):
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            return word[: -len(suffix)]

    return word


def normalize_question_for_similarity(text: str) -> str:
    text = clean_text(text).replace("?", "")
    text = re.sub(r"[^\w\s가-힣]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = [_normalize_similarity_word(word) for word in text.strip().split()]
    return " ".join(word for word in words if word)


def _has_empathy_prefix(question: str) -> bool:
    question = clean_text(question)
    prefixes = (
        "\uc88b\uc73c\uc168\uaca0\uc5b4\uc694",
        "\uc7ac\ubbf8\uc788\uc73c\uc168\uaca0\uc5b4\uc694",
        "\uadf8\ub7ec\uc168\uad70\uc694",
        "\uadf8\ub807\uad70\uc694",
        "\uadf8\ub7ac\uad70\uc694",
        "\uc54c\uaca0\uc2b5\ub2c8\ub2e4",
        "\uc88b\uc558\uaca0\uc5b4\uc694",
        "\uc544\ud558",
        "\uc74c",
        "\uc544\uc774\uace0",
        "\ub2e4\ud589\uc774\ub124\uc694",
    )
    return question.startswith(prefixes)


def _shares_topic_group(text: str, previous_text: str) -> bool:
    text = clean_text(text).lower()
    previous_text = clean_text(previous_text).lower()

    if not text or not previous_text:
        return False

    for group in TOPIC_SIMILARITY_GROUPS:
        normalized_group = [term.lower() for term in group]
        if any(term in text for term in normalized_group) and any(
            term in previous_text for term in normalized_group
        ):
            return True

    return False


def _meaningful_words(text: str) -> set[str]:
    normalized = normalize_question_for_similarity(text)
    return {
        word
        for word in normalized.split()
        if len(word) >= 2 and word not in QUESTION_STOPWORDS
    }


def _question_focuses(text: str) -> set[str]:
    text = clean_text(text)

    for focus, patterns in QUESTION_FOCUS_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return {focus}

    return set()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return False


def _has_pronoun_grounding(question: str, history_text: str) -> bool:
    for history_patterns, question_patterns in GROUNDING_PRONOUN_GROUPS:
        if any(pattern in history_text for pattern in history_patterns) and any(
            pattern in question
            for pattern in question_patterns
        ):
            return True

    return False


def _latest_answer_has_unfinished_action(text: str) -> bool:
    text = clean_text(_get_final_correction_segment(text))

    if re.search(r"[가-힣]+고\s*싶", text):
        return True

    if any(
        phrase in text
        for phrase in (
            "먹고 싶",
            "먹고싶",
            "보고 싶",
            "보고싶",
            "하고 싶",
            "하고싶",
            "내일",
            "모레",
            "예정",
            "갈 거",
            "갈거",
            "할 거",
            "할거",
            "먹을 거",
            "먹을거",
            "볼 거",
            "볼거",
            "만날 거",
            "만날거",
            "갈게",
            "다녀올게",
            "할게",
            "먹을게",
            "볼게",
            "만날게",
            "쉴게",
            "살게",
            "올게",
            "가려고",
            "다녀오려고",
            "하려고",
            "먹으려고",
            "보려고",
            "만나려고",
            "쉬려고",
            "자려고",
            "사려고",
            "오려고",
            "갈래",
            "할래",
            "먹을래",
            "볼래",
            "만날래",
            "쉴래",
            "살래",
            "올래",
        )
    ):
        return True

    return bool(
        re.search(
            r"(?:^|\s)(?:안|못)\s*"
            r"(?:먹|먹었|마시|마셨|가|갔|다녀|보|봤|만나|만났|사|샀|"
            r"하|했|오|왔|나가|나갔|듣|들었|쉬|쉬었|자|잤|읽|읽었|"
            r"쓰|썼|타|탔|통화|전화|연락)[가-힣]*",
            text,
        )
        or re.search(r"(?:^|\s)[가-힣]+지\s*않[가-힣]*", text)
    )


def _question_assumes_completed_action(question: str) -> bool:
    return any(
        phrase in clean_text(question)
        for phrase in (
            "드셨어",
            "먹었",
            "가셨어",
            "다녀오셨어",
            "보셨어",
            "들으셨어",
            "하셨어",
            "사셨어",
            "만나셨어",
            "통화하셨어",
            "연락하셨어",
            "계셨어",
            "있었",
            "쉬고 나서는",
            "쉬셨어",
            "주무셨어",
            "낮잠을 주무",
            "낮잠 잤",
        )
    )


def is_question_grounded_in_history(
    question: str,
    conversation_history: List[str],
) -> bool:
    question = clean_text(question)
    history = [
        clean_text(_get_final_correction_segment(text))
        for text in conversation_history
        if clean_text(_get_final_correction_segment(text))
    ]

    if not question or not history:
        return True

    joined_history = " ".join(history)
    latest_text = history[-1]

    if (
        _latest_answer_has_unfinished_action(latest_text)
        and _question_assumes_completed_action(question)
    ):
        return False

    asks_about_completed_contact = any(
        phrase in question
        for phrase in ("이야기를 나누셨어", "통화하셨어", "연락하셨어")
    )
    has_contact_history = any(
        phrase in joined_history
        for phrase in (
            "통화",
            "전화",
            "대화",
            "이야기했",
            "연락",
            "말했",
            "만났",
            "다녀갔",
            "찾아왔",
            "들렀",
        )
    )

    if asks_about_completed_contact and not has_contact_history:
        return False

    question_words = _meaningful_words(question)
    history_words = _meaningful_words(joined_history)

    has_direct_grounding = bool(question_words & history_words)
    has_pronoun_grounding = _has_pronoun_grounding(question, joined_history)

    if has_direct_grounding or has_pronoun_grounding:
        unsupported_words = (
            question_words
            - history_words
            - GENERIC_ANCHORED_FOLLOWUP_WORDS
        )
        return not unsupported_words

    anchored_followup_phrases = [
        "그때",
        "그 이야기",
        "방금",
        "조금 더",
    ]

    if not any(phrase in question for phrase in anchored_followup_phrases):
        return False

    unsupported_words = question_words - GENERIC_ANCHORED_FOLLOWUP_WORDS
    return not unsupported_words


def _format_history(conversation_history: List[str]) -> str:
    lines = []

    for index, text in enumerate(conversation_history, start=1):
        cleaned = clean_text(text)

        if cleaned:
            lines.append(f"{index}. {cleaned}")

    return "\n".join(lines) if lines else "없음"


def _format_topic_examples() -> str:
    return ", ".join(EVERYDAY_TOPIC_EXAMPLES)


def _extract_json_object(text: str) -> dict:
    text = clean_text(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def is_safe_followup_question(question: str) -> bool:
    question = clean_text(question)

    if not question:
        return False

    if len(question) < 8 or len(question) > 95:
        return False

    if question.count("?") > 1:
        return False

    if not question.endswith("?"):
        return False

    if any(pattern in question for pattern in FORBIDDEN_QUESTION_PATTERNS):
        return False

    if any(pattern in question for pattern in NEGATIVE_ASSUMPTION_PATTERNS):
        return False

    if any(pattern in question for pattern in OVERLY_CLINICAL_EMPATHY_PATTERNS):
        return False

    return True


def is_weak_free_talk_answer(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return True

    exact_weak_answers = {
        "응",
        "네",
        "아니",
        "몰라",
        "없어",
        "글쎄",
    }

    if text in exact_weak_answers:
        return True

    weak_phrases = [
        "모르겠",
        "없어요",
        "딱히",
        "그냥",
        "기억 안",
        "기억이 안",
        "생각 안",
        "생각이 안",
        "라니까",
        "말했",
        "했잖",
    ]

    return any(phrase in text for phrase in weak_phrases)


def build_fallback_with_empathy(
    question: str,
    conversation_history: List[str],
) -> str:
    latest_text = clean_text(
        _get_final_correction_segment(
            conversation_history[-1] if conversation_history else ""
        )
    )
    question = clean_text(question)

    if not latest_text or not question:
        return question

    negative_prefixes = ("그러셨군요", "그랬군요", "아이고")
    positive_prefixes = ("좋으셨겠어요", "좋았겠어요", "재미있으셨겠어요")

    negative_evidence_text = latest_text

    for pattern in NEGATED_NEGATIVE_CUE_PATTERNS:
        negative_evidence_text = negative_evidence_text.replace(pattern, "")

    negative_evidence_text = NEGATED_NEGATIVE_CUE_PATTERN.sub(
        "",
        negative_evidence_text,
    )

    has_negative_cue = any(
        pattern in negative_evidence_text
        for pattern in NEGATIVE_CUE_PATTERNS
    )

    if has_negative_cue:
        if question.startswith(positive_prefixes):
            _prefix, separator, remainder = question.partition(".")
            question = remainder.strip() if separator and remainder.strip() else question

        if _has_empathy_prefix(question):
            return question

        return f"그러셨군요. {question}"

    if any(pattern in latest_text for pattern in IMPROVEMENT_CUE_PATTERNS):
        if question.startswith(negative_prefixes):
            _prefix, separator, remainder = question.partition(".")
            question = remainder.strip() if separator and remainder.strip() else question

        if _has_empathy_prefix(question):
            return question

        return f"다행이네요. {question}"

    positive_evidence_text = latest_text
    for pattern in ("좋겠", "좋을 것 같", "좋을것 같"):
        positive_evidence_text = positive_evidence_text.replace(pattern, "")

    if any(pattern in positive_evidence_text for pattern in POSITIVE_CUE_PATTERNS):
        if question.startswith(negative_prefixes):
            _prefix, separator, remainder = question.partition(".")
            question = remainder.strip() if separator and remainder.strip() else question

        if _has_empathy_prefix(question):
            return question

        return f"좋으셨겠어요. {question}"

    if _has_empathy_prefix(question):
        return question

    return question


def is_similar_to_previous_question(
    question: str,
    previous_questions: List[str],
) -> bool:
    question = normalize_question_for_similarity(question)
    question_words = {
        word
        for word in question.split()
        if len(word) >= 2 and word not in QUESTION_STOPWORDS
    }

    if not question_words:
        return False

    for previous_question in previous_questions:
        previous = normalize_question_for_similarity(previous_question)
        previous_words = {
            word
            for word in previous.split()
            if len(word) >= 2 and word not in QUESTION_STOPWORDS
        }

        if not previous_words:
            continue

        question_focuses = _question_focuses(question)
        previous_focuses = _question_focuses(previous)

        if question_focuses and previous_focuses and question_focuses.isdisjoint(previous_focuses):
            continue

        overlap = len(question_words & previous_words)
        smaller_size = min(len(question_words), len(previous_words))

        if overlap >= 2:
            return True

        if smaller_size > 0 and overlap / smaller_size >= 0.45:
            return True

        if (
            _shares_topic_group(question, previous)
            and overlap >= 1
            and smaller_size > 0
            and overlap / smaller_size >= 0.3
        ):
            return True

    return False


def _build_followup_prompt(
    conversation_history: List[str],
    stage: str,
    previous_questions: List[str],
    rejection_note: str = "",
) -> str:
    return f"""
당신은 고령자와 자연스럽게 대화하는 한국어 말동무 AI입니다.

대화 목표:
- 검사처럼 느껴지지 않게 편안하게 대화를 이어갑니다.
- 자연스러운 대화 중 회상 질문으로 쓸 수 있는 구체적인 단서를 모읍니다.
- 유효한 기억 단서가 충분히 쌓일 때까지 한 주제를 너무 오래 캐묻지 않고 자연스럽게 구체화합니다.
- 정해진 질문지를 반복하지 말고, 사용자의 말에서 핵심 단서를 잡아 이어 묻습니다.
- 새 주제를 열 때는 매번 비슷한 사람/음식 질문만 반복하지 말고 다양한 일상 주제를 사용합니다.
- 사용자의 말이 긍정적이면 짧게 좋은 반응을 하고, 부정적이면 짧게 받아준 뒤 자연스럽게 이어 묻습니다.

열 수 있는 일상 주제 예시:
{_format_topic_examples()}

현재 단계:
{stage}

단계 의미:
- OPEN: 새로운 자유대화 주제를 엽니다. 너무 넓지 않게 사람, 음식, 장소, 집에서 한 일, 방송/노래 같은 일상 주제 중 하나를 자연스럽게 묻습니다.
- DEEPEN: 사용자가 방금 말한 내용을 자연스럽게 한 단계 더 이어갑니다. 이미 답한 내용은 다시 묻지 않습니다.
- ANCHOR: 현재 이야기에서 정답이 명확한 기억 사건이 부족할 때만 장면, 장소, 사람, 시간 중 하나를 구체적으로 묻습니다.

최근 자유대화 답변:
{_format_history(conversation_history)}

이미 물어본 질문:
{_format_history(previous_questions)}
{rejection_note}
반드시 지킬 규칙:
1. 질문은 한국어 1문장만 만듭니다.
2. 질문은 반드시 물음표로 끝납니다.
3. 이름, 배우자, 고향, 날짜, 요일, 나이, 생년월일은 묻지 않습니다.
4. 치매, 검사, 진단, 정답, 오답, 기억력 같은 표현은 쓰지 않습니다.
5. 사용자의 말을 부정적으로 넘겨짚지 않습니다.
6. 너무 넓은 질문은 피합니다.
7. 이미 물어본 질문과 같은 질문은 만들지 않습니다.
8. 이미 물어본 질문과 의도가 비슷한 질문도 만들지 않습니다.
9. 사용자가 답한 내용을 정답처럼 확인하지 말고, 편하게 이어 묻습니다.
10. 사용자의 답변 안에 있는 핵심 단어를 그대로 복사만 하지 말고, 그 단서에서 자연스럽게 한 단계만 구체화합니다.
11. 감정 반응은 최대 1문장, 25자 안팎으로 짧게 합니다.
12. 상담, 치료, 진단처럼 들리는 위로는 하지 않습니다.
13. 전체 출력 질문은 "짧은 반응 + 질문" 형태여도 되지만, 합쳐서 1~2문장 이내로 유지합니다.

좋은 예:
- 좋으셨겠어요. 그때 어떤 장면이 제일 기억나세요?
- 아이고, 불편하셨겠어요. 그때는 어디에 계셨어요?
- 그러셨군요. 그 이야기를 떠올리면 어떤 모습이 먼저 생각나세요?
- 그때 어디에서 있었던 일인지 기억나세요?
- 그분과 함께했던 장면 중에 먼저 떠오르는 게 있으세요?
- 그 음식을 누구와 같이 드셨던 기억이 있으세요?

나쁜 예:
- 오늘 하루 중 기억나는 순간이 있으세요?
- 혹시 치매 검사를 받아보신 적 있으세요?
- 아까 말씀하신 정답이 맞나요?
- 많이 힘드셨겠지만 앞으로는 괜찮아질 거예요.
- 그건 상담을 받아보시는 게 좋겠어요.

출력 형식:
{{"nextQuestion":"질문","shouldChangeTopic":false,"reason":"짧은 이유"}}
""".strip()


_REJECTION_NOTE_TEMPLATES = {
    "repeated_question": "방금 만든 질문 '{question}'은 이미 물어본 질문과 완전히 같아서 사용할 수 없습니다. 같은 질문을 반복하지 말고 다른 표현으로 다시 만드세요.",
    "similar_question": "방금 만든 질문 '{question}'은 이미 물어본 질문과 의도가 너무 비슷해서 사용할 수 없습니다. 겹치지 않는 다른 각도로 다시 만드세요.",
    "ungrounded_question": "방금 만든 질문 '{question}'은 최근 자유대화 답변에 나오지 않은 내용을 묻고 있어 사용할 수 없습니다. 사용자가 실제로 말한 내용에 더 밀착해서 다시 만드세요.",
    "unsafe_question": "방금 만든 질문 '{question}'은 형식이나 금지 표현 규칙을 어겨서 사용할 수 없습니다. 규칙을 다시 확인하고 짧은 물음표 문장 하나로 다시 만드세요.",
}


def _call_followup_llm(prompt: str) -> dict:
    response = _get_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "한국어 말동무 AI입니다. 반드시 JSON만 출력합니다.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.35,
    )

    content = response.choices[0].message.content
    parsed = _extract_json_object(content)

    return {
        "question": clean_text(parsed.get("nextQuestion", "")),
        "shouldChangeTopic": _coerce_bool(parsed.get("shouldChangeTopic", False)),
        "reason": clean_text(parsed.get("reason", "")),
    }


def _validate_followup_question(
    question: str,
    stage: str,
    conversation_history: List[str],
    previous_questions: List[str],
    should_change_topic: bool,
) -> tuple[str | None, str]:
    if question in {clean_text(item) for item in previous_questions}:
        return None, "repeated_question"

    if is_similar_to_previous_question(question, previous_questions):
        return None, "similar_question"

    question = build_fallback_with_empathy(question, conversation_history)

    if (
        stage in {"DEEPEN", "ANCHOR"}
        and not should_change_topic
        and not is_question_grounded_in_history(question, conversation_history)
    ):
        return None, "ungrounded_question"

    if not is_safe_followup_question(question):
        return None, "unsafe_question"

    return question, ""


def generate_safe_followup_question(
    conversation_history: List[str],
    stage: str,
    fallback_question: str,
    previous_questions: List[str] | None = None,
) -> dict:
    stage = str(stage or "").upper()
    fallback_question = build_fallback_with_empathy(
        fallback_question,
        conversation_history,
    )
    previous_questions = previous_questions or []

    if stage not in {"OPEN", "DEEPEN", "ANCHOR"}:
        return {
            "nextQuestion": fallback_question,
            "shouldChangeTopic": False,
            "reason": "unsupported_stage",
        }

    is_weak_answer = is_weak_free_talk_answer(
        conversation_history[-1] if conversation_history else ""
    )
    rejection_note = ""

    for _attempt in range(2):
        prompt = _build_followup_prompt(
            conversation_history=conversation_history,
            stage=stage,
            previous_questions=previous_questions,
            rejection_note=rejection_note,
        )

        try:
            generated = _call_followup_llm(prompt)
        except Exception:
            return {
                "nextQuestion": fallback_question,
                "shouldChangeTopic": is_weak_answer,
                "reason": "llm_failed",
            }

        should_change_topic = generated["shouldChangeTopic"] or is_weak_answer

        validated_question, rejection_reason = _validate_followup_question(
            question=generated["question"],
            stage=stage,
            conversation_history=conversation_history,
            previous_questions=previous_questions,
            should_change_topic=should_change_topic,
        )

        if validated_question:
            return {
                "nextQuestion": validated_question,
                "shouldChangeTopic": should_change_topic,
                "reason": generated["reason"],
            }

        rejection_note = "\n" + _REJECTION_NOTE_TEMPLATES[rejection_reason].format(
            question=generated["question"] or "(빈 질문)"
        ) + "\n"

    return {
        "nextQuestion": fallback_question,
        "shouldChangeTopic": is_weak_answer,
        "reason": rejection_reason,
    }
