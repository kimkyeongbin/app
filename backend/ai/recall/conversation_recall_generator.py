import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

import requests
from kiwipiepy import Kiwi
from openai import OpenAI

from recall.memory_candidate_service import validate_recall_candidate_contract
from recall.memory_event import MemoryAnswerType, MemoryEvent, infer_answer_type
from recall.recall_score_calculator import calculate_similarity_score

MEMORY_POINT_SEMANTIC_SIMILARITY_THRESHOLD = 70.0


logger = logging.getLogger(__name__)
YNU_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
GPT_MODEL = "claude-sonnet-5"
BASE_URL = "http://localhost:8080"
MIN_RECALL_MEMORY_CANDIDATES = 2
RECENT_RECALL_HISTORY_LIMIT = 12

_client: OpenAI | None = None


FORBIDDEN_RECALL_KEYWORDS = [
    "성함",
    "이름",
    "생년월일",
    "생년 월일",
    "배우자",
    "고향",
    "요일",
    "날짜",
    "몇 년도",
    "몇년도",
    "몇 월",
    "몇월",
    "며칠",
    "오늘 날짜",
    "오늘은 무슨 요일",
]


WEAK_MEMORY_PHRASES = [
    "응",
    "네",
    "아니",
    "몰라",
    "모르겠",
    "기억 안",
    "기억이 안",
    "그냥",
    "없어",
    "없다",
    "없었",
    "없어요",
    "없다니까",
]


MEMORY_DETAIL_HINTS = [
    "먹",
    "마시",
    "마셨",
    "마셨어",
    "잤",
    "낮잠",
    "쉬었",
    "읽었",
    "썼",
    "줬",
    "갔",
    "다녀오",
    "왔",
    "만났",
    "봤",
    "보았",
    "했",
    "샀",
    "사왔",
    "넣었",
    "전화",
    "통화",
    "이야기",
    "산책",
    "운동",
    "청소",
    "빨래",
    "설거지",
    "요리",
    "목욕",
    "정리",
    "병원",
    "아팠",
    "마트",
    "시장",
    "공원",
    "집",
    "티비",
    "텔레비전",
    "뉴스",
    "노래",
    "드라마",
    "사과",
    "우유",
    "생선",
    "냉장고",
    "약",
    "허리",
    "몸",
    "불편",
    "손주",
    "아들",
    "딸",
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
    "점심",
    "저녁",
    "아침",
    "음식",
]

MEMORY_ACTION_HINTS = [
    "먹었",
    "마셨",
    "마셨어",
    "잤",
    "쉬었",
    "읽었",
    "썼",
    "줬",
    "갔",
    "갔다",
    "다녀왔",
    "봤",
    "보고",
    "보았",
    "들었",
    "샀",
    "넣었",
    "만났",
    "통화",
    "전화",
    "이야기",
    "산책",
    "운동",
    "청소",
    "빨래",
    "설거지",
    "요리",
    "목욕",
    "정리",
    "있었",
    "앉았",
    "누워",
]


MEMORY_ACTION_CONCEPTS = {
    "EAT": ("먹", "식사", "드시", "드셨"),
    "DRINK": ("마시",),
    "GO": ("갔", "다녀오", "왔", "가서"),
    "SEE": ("봤", "보았", "보고", "시청"),
    "HEAR": ("들었", "들어", "노래"),
    "BUY": ("샀", "사왔", "장 봤", "장봤"),
    "MEET": ("만났", "만나"),
    "CONTACT": ("통화", "전화", "연락"),
    "ACTIVITY": ("산책", "운동", "청소", "빨래", "설거지", "요리", "목욕", "정리"),
    "REST": ("쉬었", "휴식", "낮잠", "누워"),
    "GIVE": ("줬", "주었", "드렸"),
    "PLACE": ("뒀", "두었", "놓았", "넣었"),
    "WRITE": ("썼", "적었", "작성"),
    "READ": ("읽었", "읽어"),
    "PLAY": ("뒀", "쳤", "놀이", "게임"),
}


GENERAL_PAST_ACTION_PATTERN = re.compile(
    r"(?:았|었|였|했|왔|갔|봤|샀|줬|뒀|쳤|탔|썼|읽었|만들었)"
    r"(?:어|어요|습니다|다)?(?:[.!?]|$)"
)


GENERIC_OBJECT_ACTION_PATTERN = re.compile(
    r"(?:"
    r"[가-힣a-zA-Z0-9]+(?:을|를|에|에서|에게|한테)\s+[가-힣]{2,}"
    r"|[가-힣]{2,}\s+[가-힣]{2,}"
    r")[.!?]?$"
)


STATE_PREDICATE_PREFIXES = (
    "있",
    "없",
    "좋",
    "나쁘",
    "아프",
    "아팠",
    "피곤",
    "힘들",
    "예쁘",
    "예뻤",
    "괜찮",
    "덥",
    "더웠",
    "춥",
    "추웠",
    "슬프",
    "슬펐",
    "외롭",
    "외로웠",
    "불편",
    "어렵",
    "어려웠",
    "맛있",
    "맛없",
    "재미있",
    "재미없다",
    "즐겁",
    "기쁘",
    "반갑",
    "무섭",
    "속상하",
    "우울",
    "걱정",
)


COMPARISON_STOPWORDS = {
    "오늘",
    "어제",
    "아까",
    "조금",
    "전에",
    "오전",
    "오후",
    "아침",
    "점심",
    "저녁",
    "기억",
    "기억나는",
    "이야기",
    "일",
    "것",
}


SHORT_MEMORY_CONTENT_WORDS = {
    "딸",
    "형",
    "약",
    "국",
    "배",
    "집",
    "밥",
    "물",
    "차",
    "책",
    "꽃",
    "옷",
    "산",
    "눈",
    "손",
    "발",
    "팔",
    "방",
    "길",
    "비",
}


NON_PARTICLE_COMPOUND_WORDS = {
    "약과",
    "국가",
}


KEYWORD_POLITE_SUFFIXES = (
    "이었어요",
    "이었죠",
    "이었지",
    "이었어",
    "였어요",
    "였죠",
    "였지",
    "였어",
    "이죠",
    "이지",
    "이야",
    "이라고요",
    "이에요",
    "이라고",
    "라고요",
    "입니다",
    "예요",
    "라고",
    "이요",
    "요",
)


KEYWORD_PARTICLE_SUFFIXES = (
    "에서는",
    "에게는",
    "한테는",
    "께서는",
    "으로는",
    "이랑",
    "에서",
    "에게",
    "한테",
    "께서",
    "까지",
    "부터",
    "으로",
    "하고",
    "이나",
    "랑",
    "로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "와",
    "과",
    "께",
    "도",
    "만",
)


MEMORY_CONCRETE_HINTS = [
    "김치",
    "볶음밥",
    "피자",
    "라면",
    "국",
    "밥",
    "사과",
    "우유",
    "물",
    "차",
    "생선",
    "책",
    "꽃",
    "옷",
    "낮잠",
    "약",
    "허리",
    "몸",
    "티비",
    "텔레비전",
    "뉴스",
    "노래",
    "드라마",
    "병원",
    "마트",
    "시장",
    "공원",
    "집",
    "냉장고",
    "소파",
    "산책",
    "운동",
    "청소",
    "빨래",
    "설거지",
    "요리",
    "목욕",
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
]


PERSON_MEMORY_HINTS = (
    "아들", "딸", "손주", "동생", "형", "누나", "언니", "오빠",
    "엄마", "아빠", "어머니", "아버지", "남편", "아내", "조카",
    "사촌", "친구", "가족", "배우자",
)


PLACE_MEMORY_HINTS = (
    "병원", "마트", "시장", "공원", "집", "경로당", "복지관",
    "식당", "카페", "약국", "은행", "교회", "성당", "절",
)


TIME_MEMORY_HINTS = (
    "오늘", "어제", "아침", "점심", "저녁", "오전", "오후",
    "새벽", "밤", "낮", "시", "분",
)


FUTURE_OR_WISH_PHRASES = [
    "고 싶",
    "고싶",
    "먹고 싶",
    "먹고싶",
    "보고 싶",
    "보고싶",
    "하고 싶",
    "하고싶",
    "내일",
    "모레",
    "다음 주",
    "다음주",
    "갈 거",
    "갈거",
    "할 거",
    "할거",
    "먹을 거",
    "먹을거",
    "만날 거",
    "만날거",
    "예정",
    "가려고",
    "하려고",
    "먹으려고",
    "다녀오려고",
    "보려고",
    "만나려고",
    "쉬려고",
    "사려고",
    "오려고",
    "갈게",
    "다녀올게",
    "할게",
    "먹을게",
    "볼게",
    "만날게",
    "쉴게",
    "살게",
    "올게",
    "갈래",
    "할래",
    "먹을래",
    "볼래",
    "만날래",
    "쉴래",
    "살래",
    "올래",
    "으면 좋",
    "면 좋겠",
]

INCOMPLETE_OR_NEGATED_ACTION_PHRASES = [
    "안 먹",
    "못 먹",
    "깜빡했",
    "잊어버렸",
    "잊었",
    "뻔했",
    "아직 안",
    "아직은 안",
    "연락은 못",
    "연락을 못",
    "안 나갔",
    "못 나갔",
]


UNCERTAIN_MEMORY_PHRASES = [
    "것 같",
    "듯해",
    "듯했",
    "아마",
    "수도 있",
    "지도 몰라",
    "을걸",
    "ㄹ걸",
]


STATE_ONLY_MEMORY_PHRASES = (
    "피곤",
    "어려웠",
    "힘들",
    "속상하",
    "슬펐",
    "우울",
    "외롭",
    "불안",
    "걱정",
    "괜찮았",
    "좋았",
    "나빴",
    "더웠",
    "추웠",
)


UNCERTAIN_ACTION_PATTERN = re.compile(
    r"(?:았|었|였|했|왔|갔|봤|샀|먹었|마셨|만났|다녀왔)는지\s*"
    r"(?:잘\s*)?(?:모르|기억이?\s*안)"
)


UNCOMPLETED_ACTION_PATTERN = re.compile(
    r"(?:"
    r"(?:을|ㄹ)까\s*(?:하고\s*)?생각|"
    r"기로\s*했|"
    r"려다(?:가)?|"
    r"(?:았|었|했)어야\s*했|"
    r"해야\s*했|"
    r"[가-힣]+야지"
    r")"
)


FUTURE_ENDING_PATTERN = re.compile(
    r"[가-힣]+\s+거(?:야|예요|에요|다)(?:[.!?]|$)"
)


HEARSAY_PATTERN = re.compile(
    r"(?:"
    r"다고\s*(?:했|말했)|"
    r"[가-힣]+(?:았대|었대|했대)|"
    r"(?:왔대|갔대|봤대|샀대|탔대)"
    r")"
)


NEGATED_ACTION_PATTERN = re.compile(
    r"(?:"
    r"(?:^|\s)(?:안|못)\s*"
    r"(?:먹|먹었|마시|마셨|가|갔|다녀|보|봤|만나|만났|사|샀|"
    r"하|했|오|왔|나가|나갔|들|쉬|자|잤|읽|읽었|쓰|썼|타|탔|통화|전화|"
    r"연락|운동|산책|청소|요리|아프|아팠)[가-힣]*"
    r"|(?:^|\s)[가-힣]+지\s*않[가-힣]*"
    r")"
)


def _get_client() -> OpenAI:
    global _client

    if _client is None:
        api_key = os.getenv("YNU_API_KEY")

        if not api_key:
            raise EnvironmentError("환경변수 YNU_API_KEY가 설정되지 않았습니다.")

        _client = OpenAI(
            api_key=api_key,
            base_url=YNU_BASE_URL,
        )

    return _client


def clean_text(text: str) -> str:
    text = str(text).strip()
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


def _comparison_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", clean_text(text).lower())


def _character_bigrams(text: str) -> set[str]:
    normalized = _comparison_text(text)

    if len(normalized) < 2:
        return {normalized} if normalized else set()

    return {
        normalized[index:index + 2]
        for index in range(len(normalized) - 1)
    }


def _content_words(text: str) -> set[str]:
    words = set()

    for raw_word in re.findall(r"[0-9a-zA-Z가-힣]+", clean_text(text).lower()):
        word = raw_word

        for suffix in (() if word in NON_PARTICLE_COMPOUND_WORDS else (
            "에서는",
            "이랑",
            "하고",
            "에게",
            "한테",
            "에서",
            "으로",
            "이나",
            "와",
            "과",
            "랑",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "도",
        )):
            candidate = word[:-len(suffix)] if word.endswith(suffix) else ""

            if (
                candidate
                and (len(candidate) >= 2 or candidate in SHORT_MEMORY_CONTENT_WORDS)
            ):
                word = candidate
                break

        if (
            (len(word) >= 2 or word in SHORT_MEMORY_CONTENT_WORDS)
            and word not in COMPARISON_STOPWORDS
        ):
            if any(
                pattern in word
                for patterns in MEMORY_ACTION_CONCEPTS.values()
                for pattern in patterns
            ):
                continue

            words.add(word)

    return words


def _normalize_keyword_word(word: str) -> str:
    word = _comparison_text(word)

    for suffix in KEYWORD_POLITE_SUFFIXES:
        if len(word) > len(suffix) and word.endswith(suffix):
            word = word[:-len(suffix)]
            break

    if len(word) > 2 and word.endswith("야"):
        word = word[:-1]

    if word in NON_PARTICLE_COMPOUND_WORDS:
        return word

    for suffix in KEYWORD_PARTICLE_SUFFIXES:
        candidate = word[:-len(suffix)] if word.endswith(suffix) else ""

        if (
            candidate
            and (len(candidate) >= 2 or candidate in SHORT_MEMORY_CONTENT_WORDS)
        ):
            word = candidate
            break

    return word


_kiwi = Kiwi()

# voice_reply_handler와 동일한 원칙: 키워드 리스트 항목은 활용 조각이
# 아니라 사전형 어간으로 관리하고, 매칭은 부분 문자열이 아니라
# 형태소(토큰) 단위로만 한다.
_NOUN_TAGS = {"NNG", "NNP", "NNB", "NR", "NP"}
_EXACT_MATCH_TAGS = _NOUN_TAGS | {"MAG", "XR", "SL", "SH", "SN", "IC"}
# 동사/형용사는 접두사로 비교한다. "듣다/눕다/어렵다/덥다/춥다" 같은
# 불규칙 활용 어간은 kiwi가 "VV-I"/"VA-I"처럼 -I가 붙은 태그를 쓴다.
_PREDICATE_TAG_PREFIXES = ("VV", "VA", "VX", "XSA", "XSV")
_tokenize_cache: dict[str, tuple] = {}


def _is_match_tag(tag: str) -> bool:
    return tag in _EXACT_MATCH_TAGS or tag.startswith(_PREDICATE_TAG_PREFIXES)


def _tokenize_cached(text: str) -> tuple:
    text = str(text or "")
    cached = _tokenize_cache.get(text)

    if cached is not None:
        return cached

    tokens = tuple(_kiwi.tokenize(text))

    if len(_tokenize_cache) > 2000:
        _tokenize_cache.clear()

    _tokenize_cache[text] = tokens
    return tokens


def _match_token_forms(text: str) -> list[str]:
    return [token.form for token in _tokenize_cached(text) if _is_match_tag(token.tag)]


def _full_token_forms(text: str) -> list[str]:
    return [token.form for token in _tokenize_cached(text)]


def _is_contiguous_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False

    span = len(needle)

    return any(
        haystack[start:start + span] == needle
        for start in range(len(haystack) - span + 1)
    )


# 조사(격조사/보조사/접속조사)는 의미를 가른다("집에만" vs "집을").
# 어미(EP/EF/EC 등 활용형 꼬리)는 그냥 시제/문체 차이라 무시해도 된다.
_PARTICLE_TAG_PREFIXES = ("JK", "JX", "JC")


def _keyword_has_meaningful_particle(word: str) -> bool:
    return any(
        token.tag.startswith(_PARTICLE_TAG_PREFIXES)
        for token in _tokenize_cached(word)
    )


def _word_occurs_as_token(word: str, text: str, strict: bool = False) -> bool:
    """word(키워드 하나 또는 구)가 text 안에 실제 단어(형태소)로 등장하는지 확인한다.

    조사가 붙은 활용("형이", "형을")은 형태소 분석으로 정확히 잡아내고,
    "형섭"처럼 다른 단어 속에 우연히 낀 글자는 매칭되지 않는다.

    strict=True는 어미(시제/의향 등)까지 그대로 지켜야 하는 키워드용이다.
    예: FUTURE_OR_WISH_PHRASES의 "먹으려고"는 "먹었어"(과거)와는 달라야
    하므로, 어간만 남기는 느슨한 매칭을 쓰면 안 된다.
    """
    # "약과"(과자)를 "약"+"과"(조사)로, "국가"를 "국"+"가"(조사)로 잘못
    # 쪼개는 것처럼 형태소 분석기 자체가 헷갈리는 소수의 복합어는
    # 별도로 관리한다(NON_PARTICLE_COMPOUND_WORDS).
    for compound in NON_PARTICLE_COMPOUND_WORDS:
        if compound != word and compound.startswith(word) and compound in text:
            return False

    keyword_content_forms = _match_token_forms(word)

    if not keyword_content_forms:
        return False

    if (
        not strict
        and len(keyword_content_forms) == 1
        and not _keyword_has_meaningful_particle(word)
    ):
        # 단일 형태소 키워드는 조사/어미와 무관하게 그 형태소가
        # 문장 안에 등장하는지만 본다(활용형에 안정적).
        return keyword_content_forms[0] in _match_token_forms(text)

    # 조사·어미가 의미를 가르는 키워드는 그걸 빼면 의미가 사라지거나
    # ("집에만"→"집") 다른 뜻이 되므로("먹으려고"→"먹", 과거와 충돌)
    # 어미·조사를 포함한 전체 토큰 나열이 연속으로 등장하는지 확인한다.
    return _is_contiguous_subsequence(_full_token_forms(word), _full_token_forms(text))


def _single_syllable_keyword_occurs(answer_keyword: str, text: str) -> bool:
    return _word_occurs_as_token(answer_keyword, text)


def _contains_any(text: str, keywords, strict: bool = False) -> bool:
    return any(_word_occurs_as_token(keyword, text, strict=strict) for keyword in keywords)


def _keyword_occurs_in_text(answer_keyword: str, text: str) -> bool:
    compact_keyword = _comparison_text(answer_keyword)

    if len(compact_keyword) == 1:
        return _single_syllable_keyword_occurs(compact_keyword, text)

    keyword_words = [
        _normalize_keyword_word(word)
        for word in clean_text(answer_keyword).split()
        if _normalize_keyword_word(word)
    ]
    text_words = [
        _normalize_keyword_word(word)
        for word in clean_text(text).split()
        if _normalize_keyword_word(word)
    ]

    if not keyword_words or not text_words:
        return False

    target = "".join(keyword_words)

    for start_index in range(len(text_words)):
        combined = ""

        for end_index in range(start_index, len(text_words)):
            combined += text_words[end_index]

            if combined == target:
                return True

            if len(combined) >= len(target):
                break

    return False


def _memory_hint_occurs(hint: str, text: str) -> bool:
    return _word_occurs_as_token(hint, text)


def _memory_detail_hint_occurs(hint: str, text: str) -> bool:
    if hint in MEMORY_CONCRETE_HINTS:
        return _memory_hint_occurs(hint, text)

    return _word_occurs_as_token(hint, text)


def _has_completed_generic_object_action(text: str) -> bool:
    if not GENERIC_OBJECT_ACTION_PATTERN.search(text):
        return False

    words = re.findall(r"[가-힣]+", clean_text(text))

    if not words:
        return False

    subject = _normalize_keyword_word(words[-2]) if len(words) >= 2 else ""

    if subject in COMPARISON_STOPWORDS | {
        "방금",
        "그거",
        "이거",
        "저거",
        "뭔가",
        "무언가",
        "기분",
        "날씨",
        "상태",
        "형편",
        "마음",
        "정말",
        "너무",
        "많이",
    }:
        return False

    predicate = words[-1]

    if predicate.startswith(STATE_PREDICATE_PREFIXES):
        return False

    decomposed = unicodedata.normalize("NFD", predicate)
    return "\u11bb" in decomposed


def _has_state_only_predicate(text: str) -> bool:
    words = re.findall(r"[가-힣]+", clean_text(text))
    return bool(words and words[-1].startswith(STATE_PREDICATE_PREFIXES))


def _has_completed_generic_food_action(text: str) -> bool:
    match = re.search(
        r"(?:^|\s)([가-힣]{1,12})(?:을|를|은|는|도|만)?\s*"
        r"(?:먹었|마셨어|마셨)[가-힣]*",
        clean_text(text),
    )

    if not match:
        return False

    food_word = match.group(1)
    return food_word not in {"그거", "이거", "저거", "뭔가", "뭐", "무언가"}


def _memory_signature(text: str) -> tuple[set[str], set[str]]:
    text = clean_text(text)
    actions = {
        concept
        for concept, patterns in MEMORY_ACTION_CONCEPTS.items()
        if _contains_any(text, patterns)
    }
    concrete = {
        hint
        for hint in MEMORY_CONCRETE_HINTS
        if _memory_hint_occurs(hint, text)
    }
    return actions, concrete


def extract_memory_action_concepts(text: str) -> set[str]:
    actions, _concrete = _memory_signature(text)
    return actions


def extract_recall_question_action_concepts(question: str) -> set[str]:
    """Extract the recalled event action without treating an intro as evidence."""
    question_units = [
        unit.strip()
        for unit in re.split(r"[,.!?]+\s*", clean_text(question))
        if unit.strip()
    ]

    for unit in reversed(question_units):
        actions = extract_memory_action_concepts(unit)
        if actions:
            return actions

    return set()


def is_similar_memory_point(memory_point: str, previous_memory_point: str) -> bool:
    current = _comparison_text(memory_point)
    previous = _comparison_text(previous_memory_point)

    if not current or not previous:
        return False

    if min(len(current), len(previous)) >= 5 and (
        current in previous or previous in current
    ):
        return True

    current_actions, current_concrete = _memory_signature(memory_point)
    previous_actions, previous_concrete = _memory_signature(previous_memory_point)

    if (
        current_actions & previous_actions
        and current_concrete
        and previous_concrete
        and (
            current_concrete <= previous_concrete
            or previous_concrete <= current_concrete
        )
    ):
        return True

    # 짧은 답변은 실질 단어가 1~2개뿐이라 글자 단위 비교(단어 겹침·bigram)로는
    # "제육볶음을 먹었다" vs "제육볶음 먹었어" 같은 조사·어미 차이만 있는
    # 패러프레이즈조차 다른 내용으로 오판하기 쉽다. 이미 recall_score_calculator에서
    # 검증된 한국어 문장 임베딩 모델로 의미 유사도를 재사용해 판단한다.
    return calculate_similarity_score(
        memory_point,
        previous_memory_point,
    ) >= MEMORY_POINT_SEMANTIC_SIMILARITY_THRESHOLD


def memory_point_match_score(memory_point: str, candidate_text: str) -> float:
    current = _comparison_text(memory_point)
    candidate = _comparison_text(candidate_text)

    if not current or not candidate:
        return 0.0

    score = 0.0

    if current == candidate:
        score += 1.0
    elif min(len(current), len(candidate)) >= 5 and (
        current in candidate or candidate in current
    ):
        score += 0.6

    current_bigrams = _character_bigrams(memory_point)
    candidate_bigrams = _character_bigrams(candidate_text)
    bigram_union = current_bigrams | candidate_bigrams

    if bigram_union:
        score += len(current_bigrams & candidate_bigrams) / len(bigram_union)

    current_actions, current_concrete = _memory_signature(memory_point)
    candidate_actions, candidate_concrete = _memory_signature(candidate_text)

    if current_actions & candidate_actions:
        score += 0.25

    concrete_union = current_concrete | candidate_concrete

    if concrete_union:
        score += 0.4 * (
            len(current_concrete & candidate_concrete) / len(concrete_union)
        )

    current_words = _content_words(memory_point)
    candidate_words = _content_words(candidate_text)
    word_union = current_words | candidate_words

    if word_union:
        score += 0.4 * len(current_words & candidate_words) / len(word_union)

    return score


def is_memory_point_grounded(memory_point: str, source_text: str) -> bool:
    original_source_text = source_text
    source_text = _get_final_correction_segment(source_text)
    memory_words = _content_words(memory_point)
    source_words = _content_words(source_text)

    if memory_words - source_words:
        return False

    memory_actions, _memory_concrete = _memory_signature(memory_point)
    source_actions, _source_concrete = _memory_signature(source_text)

    if memory_actions and not memory_actions <= source_actions:
        return False

    explicit_time_hints = (
        "오늘",
        "어제",
        "아침",
        "점심",
        "저녁",
        "오전",
        "오후",
        "새벽",
        "밤에",
        "낮에",
    )

    if any(
        _word_occurs_as_token(hint, memory_point)
        and not _word_occurs_as_token(hint, original_source_text)
        for hint in explicit_time_hints
    ):
        return False

    memory_numbers = set(re.findall(r"\d+", memory_point))
    source_numbers = set(re.findall(r"\d+", original_source_text))

    return memory_numbers <= source_numbers


def is_similar_recall_question(question: str, previous_question: str) -> bool:
    current = _comparison_text(question)
    previous = _comparison_text(previous_question)

    if not current or not previous:
        return False

    if current == previous:
        return True

    current_bigrams = _character_bigrams(question)
    previous_bigrams = _character_bigrams(previous_question)
    union = current_bigrams | previous_bigrams

    if not union:
        return False

    return len(current_bigrams & previous_bigrams) / len(union) >= 0.6


def question_reveals_memory_answer(answer_keyword: str, question: str) -> bool:
    return _keyword_occurs_in_text(answer_keyword, question)


def is_answer_keyword_grounded(
    answer_keyword: str,
    memory_point: str,
    source_text: str,
) -> bool:
    corrected_source_text = _get_final_correction_segment(source_text)
    return (
        _keyword_occurs_in_text(answer_keyword, memory_point)
        and _keyword_occurs_in_text(answer_keyword, corrected_source_text)
    )


def is_valid_answer_keyword_format(answer_keyword: str) -> bool:
    answer_keyword = clean_text(answer_keyword)

    if not answer_keyword or len(answer_keyword) > 30:
        return False

    if any(separator in answer_keyword for separator in (",", "/", "·", "그리고")):
        return False

    return len(answer_keyword.split()) <= 4


def is_answer_keyword_compatible_with_question(
    answer_keyword: str,
    question: str,
    source_text: str,
) -> bool:
    answer_keyword = clean_text(answer_keyword)
    question = clean_text(question)
    source_text = clean_text(source_text)
    keyword_concrete = {
        hint
        for hint in MEMORY_CONCRETE_HINTS
        if _memory_hint_occurs(hint, answer_keyword)
    }

    answer_type = infer_answer_type(question)
    asks_person = answer_type == MemoryAnswerType.PERSON

    if asks_person:
        if any(_memory_hint_occurs(hint, answer_keyword) for hint in PERSON_MEMORY_HINTS):
            return True

        if keyword_concrete:
            return False

        source_actions, _source_concrete = _memory_signature(source_text)
        escaped_keyword = re.escape(answer_keyword)
        has_person_relation = bool(
            re.search(
                rf"{escaped_keyword}(?:이|가|은|는)?(?:와|과|랑|하고|에게|한테)",
                source_text,
            )
        )
        return has_person_relation and bool(source_actions & {"MEET", "CONTACT"})

    asks_place = answer_type == MemoryAnswerType.PLACE

    if asks_place:
        if any(_memory_hint_occurs(hint, answer_keyword) for hint in PLACE_MEMORY_HINTS):
            return True

        if keyword_concrete:
            return False

        escaped_keyword = re.escape(answer_keyword)
        return bool(
            re.search(
                rf"{escaped_keyword}(?:에|에서|으로|까지|부터)",
                source_text,
            )
        )

    asks_time = answer_type == MemoryAnswerType.TIME

    if asks_time:
        return bool(re.search(r"\d", answer_keyword)) or any(
            _memory_hint_occurs(hint, answer_keyword)
            for hint in TIME_MEMORY_HINTS
        )

    asks_food = answer_type == MemoryAnswerType.FOOD

    if asks_food:
        if any(_memory_hint_occurs(hint, answer_keyword) for hint in PERSON_MEMORY_HINTS):
            return False

        if any(_memory_hint_occurs(hint, answer_keyword) for hint in PLACE_MEMORY_HINTS):
            return False

        if bool(re.search(r"\d", answer_keyword)) or any(
            _memory_hint_occurs(hint, answer_keyword)
            for hint in TIME_MEMORY_HINTS
        ):
            return False

        source_actions, _source_concrete = _memory_signature(source_text)
        return bool(source_actions & {"EAT", "DRINK"})

    return True


def is_forbidden_recall_content(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return True

    return _contains_any(text, FORBIDDEN_RECALL_KEYWORDS)


def is_valid_recall_question_format(question: str) -> bool:
    question = clean_text(question)

    if len(question) < 8 or len(question) > 120:
        return False

    if question.count("?") != 1:
        return False

    return question.endswith("?")


def is_valid_conversation_text(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return False

    if len(text) < 5:
        return False

    meaningless_words = {
        "음",
        "어",
        "아",
        "네",
        "응",
        "예",
        "몰라",
        "모르겠어",
        "글쎄",
    }

    if text in meaningless_words:
        return False

    if len(set(text)) <= 2:
        return False

    return True


def evaluate_memory_candidate(text: str) -> Dict[str, object]:
    text = clean_text(text)

    reasons = []
    score = 0

    if not is_valid_conversation_text(text):
        return {
            "isValid": False,
            "score": 0,
            "reasons": ["too_short_or_meaningless"],
        }

    if is_forbidden_recall_content(text):
        return {
            "isValid": False,
            "score": 0,
            "reasons": ["forbidden_fixed_question_content"],
        }

    if _contains_any(text, WEAK_MEMORY_PHRASES, strict=True):
        reasons.append("weak_or_uncertain_expression")
        score -= 20

    if len(text) >= 8:
        score += 20
        reasons.append("enough_length")

    if len(text) >= 14:
        score += 15
        reasons.append("specific_length")

    if GENERAL_PAST_ACTION_PATTERN.search(text):
        score += 20
        reasons.append("has_past_tense")

    detail_hits = [
        hint
        for hint in MEMORY_DETAIL_HINTS
        if _memory_detail_hint_occurs(hint, text)
    ]

    if detail_hits:
        score += min(45, len(detail_hits) * 15)
        reasons.append("has_daily_detail")

    if re.search(r"\d", text):
        score += 10
        reasons.append("has_number_detail")

    score = max(0, min(score, 100))

    return {
        "isValid": score >= 30,
        "score": score,
        "reasons": reasons,
    }


def _has_confirmed_action_after_wish(text: str) -> bool:
    wish_end_positions = [
        index + len(phrase)
        for phrase in ("고 싶", "고싶")
        if (index := text.rfind(phrase)) >= 0
    ]

    if not wish_end_positions:
        return False

    suffix = text[max(wish_end_positions):].strip()

    if any(hint in suffix for hint in MEMORY_ACTION_HINTS):
        return True

    return len(suffix) >= 4 and bool(GENERAL_PAST_ACTION_PATTERN.search(suffix))


def _get_confirmed_action_after_negation(text: str) -> str:
    text = clean_text(_get_final_correction_segment(text))
    matches = list(NEGATED_ACTION_PATTERN.finditer(text))

    if not matches:
        return ""

    suffix = text[matches[-1].end():].strip(" ,.;!?")
    suffix = re.sub(
        r"^(?:지만|고|대신|그래도|그러나|그런데)\s*",
        "",
        suffix,
    )

    if any(hint in suffix for hint in MEMORY_ACTION_HINTS):
        return suffix

    if len(suffix) >= 4 and GENERAL_PAST_ACTION_PATTERN.search(suffix):
        return suffix

    return ""


def score_recall_memory_candidate(text: str) -> Dict[str, object]:
    text = clean_text(_get_final_correction_segment(text))
    text = _get_confirmed_action_after_negation(text) or text
    evaluation = evaluate_memory_candidate(text)

    if text.endswith("?"):
        return {
            **evaluation,
            "isValid": False,
            "recallScore": 0,
            "reasons": [*evaluation.get("reasons", []), "interrogative_utterance"],
        }

    if re.search(r"(?:감사합니다|감사해요|고맙습니다|고마워요)[.!]*$", text):
        return {
            **evaluation,
            "isValid": False,
            "recallScore": 0,
            "reasons": [*evaluation.get("reasons", []), "social_utterance"],
        }

    hard_failure_reasons = {
        "too_short_or_meaningless",
        "forbidden_fixed_question_content",
    }

    if not evaluation["isValid"] and hard_failure_reasons.intersection(
        evaluation["reasons"]
    ):
        return {
            **evaluation,
            "recallScore": 0,
        }

    recall_score = int(evaluation["score"])
    reasons = list(evaluation["reasons"])

    action_hits = [hint for hint in MEMORY_ACTION_HINTS if _word_occurs_as_token(hint, text)]
    action_concepts, _concrete_concepts = _memory_signature(text)
    has_generic_object_action = _has_completed_generic_object_action(text)
    has_generic_food_action = _has_completed_generic_food_action(text)
    concrete_hits = [
        hint
        for hint in MEMORY_CONCRETE_HINTS
        if _memory_hint_occurs(hint, text)
    ]

    if action_hits:
        recall_score += min(30, len(action_hits) * 10)
        reasons.append("has_past_action")

    if (
        "has_past_tense" in reasons
        and not action_hits
    ):
        recall_score += 15
        reasons.append("has_general_past_action")

    if has_generic_object_action:
        if not action_hits:
            recall_score += 40
        elif not concrete_hits:
            recall_score += 20

        reasons.append("has_generic_object_action")

    if has_generic_food_action and not concrete_hits:
        recall_score += 15
        reasons.append("has_generic_food_action")

    if concrete_hits:
        recall_score += min(30, len(concrete_hits) * 10)
        reasons.append("has_concrete_object")

    future_or_wish_hits = [
        phrase
        for phrase in FUTURE_OR_WISH_PHRASES
        if _word_occurs_as_token(phrase, text, strict=True)
    ]
    has_only_wish_hits = bool(future_or_wish_hits) and all(
        "싶" in phrase
        for phrase in future_or_wish_hits
    )
    has_future_or_wish = (
        bool(future_or_wish_hits)
        or bool(FUTURE_ENDING_PATTERN.search(text))
    ) and not (
        has_only_wish_hits
        and _has_confirmed_action_after_wish(text)
    )
    has_incomplete_or_negated_action = (
        _contains_any(text, INCOMPLETE_OR_NEGATED_ACTION_PHRASES, strict=True)
        or bool(NEGATED_ACTION_PATTERN.search(text))
        or bool(UNCOMPLETED_ACTION_PATTERN.search(text))
    )
    has_uncertain_memory = (
        _contains_any(text, UNCERTAIN_MEMORY_PHRASES, strict=True)
        or bool(UNCERTAIN_ACTION_PATTERN.search(text))
        or bool(HEARSAY_PATTERN.search(text))
    )
    has_state_without_action = (
        _has_state_only_predicate(text)
        or (
            _contains_any(text, STATE_ONLY_MEMORY_PHRASES, strict=True)
            and not action_hits
            and not action_concepts
            and not has_generic_object_action
        )
    )

    if has_future_or_wish:
        recall_score -= 70
        reasons.append("future_or_wish_expression")

    if has_incomplete_or_negated_action:
        recall_score -= 70
        reasons.append("incomplete_or_negated_action")

    if has_uncertain_memory:
        recall_score -= 70
        reasons.append("uncertain_memory")

    if has_state_without_action:
        recall_score -= 70
        reasons.append("state_without_action")

    if _contains_any(text, WEAK_MEMORY_PHRASES, strict=True):
        recall_score -= 35
        reasons.append("low_info_expression")

    recall_score = max(0, min(recall_score, 100))

    return {
        **evaluation,
        "isValid": (
            recall_score >= 40
            and not has_future_or_wish
            and not has_incomplete_or_negated_action
            and not has_uncertain_memory
            and not has_state_without_action
        ),
        "recallScore": recall_score,
        "reasons": reasons,
    }


def select_recall_memory_candidates(
    conversation_history: List[str],
    max_candidates: int = 3,
) -> List[str]:
    scored_candidates = []

    for index, text in enumerate(conversation_history):
        cleaned = clean_text(text)
        candidate_text = _get_confirmed_action_after_negation(cleaned) or cleaned
        evaluation = score_recall_memory_candidate(candidate_text)

        if evaluation["isValid"]:
            scored_candidates.append(
                {
                    "index": index,
                    "text": candidate_text,
                    "score": evaluation["recallScore"],
                    "reasons": evaluation["reasons"],
                }
            )

    scored_candidates = [
        item
        for item in scored_candidates
        if (
            item["score"] >= 50
            or "has_concrete_object" in item["reasons"]
            or "has_generic_object_action" in item["reasons"]
        )
    ]

    unique_candidates = []

    for item in scored_candidates:
        duplicate_index = next(
            (
                index
                for index, selected in enumerate(unique_candidates)
                if is_similar_memory_point(item["text"], selected["text"])
            ),
            None,
        )

        if duplicate_index is None:
            unique_candidates.append(item)
            continue

        selected = unique_candidates[duplicate_index]

        if (item["score"], len(item["text"])) > (
            selected["score"],
            len(selected["text"]),
        ):
            unique_candidates[duplicate_index] = item

    unique_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = sorted(unique_candidates[:max_candidates], key=lambda item: item["index"])

    return [item["text"] for item in selected]


def filter_valid_conversation_history(
    conversation_history: List[str],
) -> List[str]:
    valid_history = []

    for text in conversation_history:
        cleaned = clean_text(text)

        evaluation = evaluate_memory_candidate(cleaned)

        if evaluation["isValid"]:
            valid_history.append(cleaned)

    return valid_history


def build_conversation_text(conversation_history: List[str]) -> str:
    lines = []

    for index, text in enumerate(conversation_history, start=1):
        text = clean_text(text)

        if text:
            lines.append(f"{index}. {text}")

    return "\n".join(lines)


def build_previous_questions_text(previous_questions: Optional[List[str]]) -> str:
    if not previous_questions:
        return "없음"

    lines = []

    for index, question in enumerate(previous_questions, start=1):
        question = clean_text(question)

        if question:
            lines.append(f"{index}. {question}")

    return "\n".join(lines) if lines else "없음"


def build_used_memory_points_text(
    used_memory_points: Optional[List[str]],
) -> str:
    if not used_memory_points:
        return "없음"

    lines = []

    for index, memory_point in enumerate(used_memory_points, start=1):
        memory_point = clean_text(memory_point)

        if memory_point:
            lines.append(f"{index}. {memory_point}")

    return "\n".join(lines) if lines else "없음"


def normalize_structured_memory_candidates(
    memory_candidates: Optional[List[Dict]],
    conversation_history: List[str],
) -> List[Dict]:
    history_by_text = {
        clean_text(text): str(text or "").strip()
        for text in conversation_history
        if clean_text(text)
    }
    normalized = []
    seen = set()

    for candidate in memory_candidates or []:
        if not isinstance(candidate, dict):
            continue

        try:
            validate_recall_candidate_contract(candidate)
        except (TypeError, ValueError):
            continue

        source_text = clean_text(candidate.get("sourceText"))
        answer_value = clean_text(candidate.get("answerValue"))

        try:
            source_record_id = int(candidate.get("sourceRecordId") or 0)
            answer_type = MemoryAnswerType(
                str(candidate.get("answerType") or "").strip().upper()
            )
        except (TypeError, ValueError):
            continue

        event_id = str(candidate.get("eventId") or "").strip()
        identity = (event_id, source_record_id, answer_type.value, answer_value)

        if (
            not event_id
            or source_record_id <= 0
            or not source_text
            or source_text not in history_by_text
            or not answer_value
            or answer_type == MemoryAnswerType.UNKNOWN
            or identity in seen
        ):
            continue

        normalized_candidate = {
            "eventId": event_id,
            "sourceRecordId": source_record_id,
            "sourceText": history_by_text[source_text],
            "answerType": answer_type.value,
            "answerValue": answer_value,
        }
        for context_field in (
            "topic",
            "sourceRecordIds",
            "evidence",
            "recallClue",
            "recallClues",
            "answerValues",
            "qualityScore",
        ):
            if context_field in candidate:
                normalized_candidate[context_field] = candidate[context_field]

        normalized.append(normalized_candidate)
        seen.add(identity)

    return normalized


def build_structured_memory_candidates_text(
    memory_candidates: List[Dict],
) -> str:
    if not memory_candidates:
        return "없음"

    blocks = []

    for index, candidate in enumerate(memory_candidates, start=1):
        lines = [
            f"후보 {index}",
            f"eventId: {candidate['eventId']}",
            f"sourceRecordId: {candidate['sourceRecordId']}",
            f"sourceText: {candidate['sourceText']}",
            f"answerType: {candidate['answerType']}",
            f"answerValue: {candidate['answerValue']}",
        ]
        evidence = candidate.get("evidence")

        if candidate.get("topic") and isinstance(evidence, list):
            lines.append(f"eventTopic: {candidate['topic']}")
            lines.append("eventEvidence:")
            lines.extend(
                f"- [{item['sourceRecordId']}] {item['sourceText']}"
                for item in evidence
            )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def select_structured_recall_target(
    memory_candidates: List[Dict],
) -> Optional[Dict]:
    """Select the single engine-ranked clue that the LLM may verbalize."""
    return memory_candidates[0] if memory_candidates else None


def parse_recall_generation_content(content: str) -> Dict[str, str]:
    parsed = {
        "memoryPoint": "",
        "answerKeyword": "",
        "question": "",
    }

    for raw_line in str(content or "").splitlines():
        line = raw_line.strip().lstrip("-*•> ")
        line = line.replace("**", "").replace("`", "").strip()
        label, separator, value = line.partition(":")

        if not separator:
            continue

        normalized_label = label.strip().lower()
        value = value.strip()

        if normalized_label == "memorypoint":
            parsed["memoryPoint"] = value
        elif normalized_label == "answerkeyword":
            parsed["answerKeyword"] = value
        elif normalized_label == "question":
            parsed["question"] = value

    return parsed


def fetch_existing_recall_questions(
    user_id: int,
    base_url: str = BASE_URL,
) -> Dict[str, List[str]]:
    response = requests.get(
        f"{base_url}/api/recall/questions/{user_id}",
        timeout=10,
    )
    response.raise_for_status()

    questions = response.json()

    conversation_questions = []

    for item in questions:
        question_text = str(item.get("questionText", "")).strip()
        expected_answer = str(item.get("expectedAnswer", "")).strip()
        category = str(item.get("category", "")).strip()

        if category == "INITIAL_FIXED":
            continue

        conversation_questions.append(
            {
                "questionId": int(item.get("questionId") or 0),
                "questionText": question_text,
                "expectedAnswer": expected_answer,
            }
        )

    conversation_questions.sort(key=lambda item: item["questionId"])
    recent_questions = conversation_questions[-RECENT_RECALL_HISTORY_LIMIT:]

    return {
        "previousQuestions": [
            item["questionText"]
            for item in recent_questions
            if item["questionText"]
        ],
        "usedMemoryPoints": [
            item["expectedAnswer"]
            for item in recent_questions
            if item["expectedAnswer"]
        ],
    }


def generate_recall_question_from_conversation(
    conversation_history: List[str],
    previous_questions: Optional[List[str]] = None,
    used_memory_points: Optional[List[str]] = None,
    memory_candidates: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    normalized_structured_candidates = normalize_structured_memory_candidates(
        memory_candidates,
        conversation_history,
    )
    valid_conversation_history = (
        [
            candidate["sourceText"]
            for candidate in normalized_structured_candidates
        ]
        if memory_candidates is not None
        else select_recall_memory_candidates(conversation_history)
    )

    if len(valid_conversation_history) < MIN_RECALL_MEMORY_CANDIDATES:
        return {
            "status": "SKIPPED",
            "reason": "회상 질문을 만들 유효한 memoryPoint 후보가 2개 미만입니다.",
            "memoryPoint": "",
            "question": "",
        }

    conversation_text = build_conversation_text(valid_conversation_history)

    if not conversation_text:
        return {
            "status": "SKIPPED",
            "reason": "회상 질문을 만들 만큼 의미 있는 자유대화 내용이 부족합니다.",
            "memoryPoint": "",
            "question": "",
        }

    previous_questions_text = build_previous_questions_text(previous_questions)
    used_memory_points_text = build_used_memory_points_text(used_memory_points)
    structured_recall_target = select_structured_recall_target(
        normalized_structured_candidates
    )
    structured_candidates = (
        [structured_recall_target]
        if structured_recall_target is not None
        else []
    )
    structured_candidates_text = build_structured_memory_candidates_text(
        structured_candidates
    )
    structured_candidate_instructions = (
        """
14. 아래에는 memoryPoint 엔진이 이번 회상 대상으로 확정한 후보 1개만 제공됩니다.
15. 다른 대화 내용으로 대상을 바꾸지 말고 이 후보만 질문으로 표현합니다.
16. memoryPoint는 확정 후보의 sourceText를 그대로 작성합니다.
17. answerKeyword는 확정 후보의 answerValue를 그대로 작성합니다.
18. 질문은 확정 후보의 answerType에 해당하는 정보만 묻습니다.
19. eventId와 sourceRecordId는 출력하지 않습니다.
20. eventEvidence가 있으면 같은 후보의 발화들만 하나의 사건 문맥으로 사용합니다.
21. 다른 사건의 사람, 장소, 음식, 행동을 질문에 섞지 않습니다.
22. eventEvidence에 없는 행동이나 상황을 새로 만들어 질문하지 않습니다.
""".strip()
        if structured_candidates
        else ""
    )

    prompt = f"""
당신은 노인과 자연스럽게 대화를 이어가는 한국어 AI 말동무입니다.

이 앱의 목적:
- 사용자가 검사받는 느낌을 받지 않도록 일상 대화처럼 이어갑니다.
- 사용자의 자유 대화 내용에서 나중에 다시 물어볼 회상 질문을 만듭니다.
- 회상 질문은 인지 기능 점검에 활용되지만, 사용자는 일반 대화처럼 느껴야 합니다.

해야 할 일:
1. 아래 최근 자유 대화 내용에서 나중에 다시 물어볼 만한 기억 포인트를 1개 고릅니다.
2. 이미 물어본 질문과 겹치지 않는 새로운 회상 질문을 1개 만듭니다.
3. 이미 사용한 기억 포인트와 같은 내용은 피합니다.
4. 질문은 자연스럽고 짧게 작성합니다.
5. 전체 질문은 1문장 또는 짧은 2문장으로 작성합니다.
6. 정답을 질문에 직접 포함하지 않습니다.
7. 치매, 검사, 기억력 테스트, 진단, 정답, 오답 같은 표현은 사용하지 않습니다.
8. 성함, 배우자, 고향, 요일, 날짜, 생년월일, 년도, 월, 일 관련 질문은 절대 만들지 않습니다.
9. 회상 질문은 반드시 사용자가 자유롭게 말한 일상 내용에서만 만듭니다.
10. answerKeyword에는 이번 질문의 정답이 되는 핵심 단어 또는 짧은 구절 1개만 작성합니다.
11. answerKeyword는 memoryPoint와 원래 대화에 실제로 포함된 내용이어야 합니다.
12. answerKeyword는 질문 문장에 직접 넣지 않습니다.
13. memoryPoint는 사용자가 말한 문장을 가능한 한 그대로 사용하고, 원문에 없는 표현으로 바꾸지 않습니다.
{structured_candidate_instructions}

최근 자유 대화 내용:
{conversation_text}

memoryPoint 엔진이 확정한 구조화 후보:
{structured_candidates_text}

이미 물어본 회상 질문:
{previous_questions_text}

이미 사용한 기억 포인트:
{used_memory_points_text}

절대 생성하면 안 되는 질문:
- 성함이 어떻게 되시나요?
- 생년월일이 어떻게 되시나요?
- 배우자분 성함이 어떻게 되시나요?
- 고향이 어디신가요?
- 오늘은 무슨 요일인가요?
- 오늘 날짜가 어떻게 되나요?
- 지금 몇 년도인가요?
- 지금 몇 월인가요?
- 오늘이 며칠인가요?

좋은 질문 예시:
- 그러고 보니 아까 음식 이야기를 해주셨잖아요. 어떤 음식을 드셨는지 기억나세요?
- 아까 다녀오신 곳 이야기를 해주셨는데, 어디에 다녀오셨는지 기억나세요?
- 조금 전에 통화 이야기를 해주셨는데, 누구와 통화하셨는지 기억나세요?

나쁜 질문 예시:
- 아까 김치찌개 먹었다고 했죠?
- 아들이랑 통화한 거 맞나요?
- 마트에 다녀왔다고 말했는데 기억하세요?
- 기억력 확인을 위해 질문드릴게요.
- 오늘은 무슨 요일인가요?
- 생년월일이 어떻게 되시나요?

출력 형식:
memoryPoint: 대화에서 뽑은 기억 포인트
answerKeyword: 질문의 정답이 되는 핵심 단어 또는 짧은 구절 1개
question: 자연스러운 회상 질문
""".strip()

    response = _get_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "당신은 한국어 자연 회상 질문을 생성하는 AI입니다. 초기 고정 질문이나 지남력 질문은 절대 만들지 않습니다.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.45,
    )

    content = response.choices[0].message.content.strip()

    parsed_content = parse_recall_generation_content(content)
    memory_point = parsed_content["memoryPoint"]
    answer_keyword = parsed_content["answerKeyword"]
    question = parsed_content["question"]

    if not question:
        question = content

    if not is_valid_recall_question_format(question):
        return {
            "status": "SKIPPED",
            "reason": "회상 질문은 물음표로 끝나는 짧은 질문 한 개여야 합니다.",
            "memoryPoint": memory_point,
            "answerKeyword": answer_keyword,
            "question": question,
        }

    memory_evaluation = score_recall_memory_candidate(memory_point)

    if not memory_evaluation["isValid"]:
        return {
            "status": "SKIPPED",
            "reason": "회상 질문으로 저장하기에는 memoryPoint가 너무 짧거나 구체성이 부족합니다.",
            "memoryPoint": memory_point,
            "question": question,
            "memoryQualityScore": memory_evaluation["recallScore"],
            "memoryQualityReasons": memory_evaluation["reasons"],
        }

    if is_forbidden_recall_content(memory_point) or is_forbidden_recall_content(question):
        return {
            "status": "SKIPPED",
            "reason": "초기 고정 질문 또는 지남력 질문과 유사한 내용이 생성되어 저장하지 않았습니다.",
            "memoryPoint": memory_point,
            "question": question,
        }

    matching_candidates = [
        candidate
        for candidate in valid_conversation_history
        if (
            is_similar_memory_point(memory_point, candidate)
            and is_memory_point_grounded(memory_point, candidate)
        )
    ]

    if not matching_candidates:
        return {
            "status": "SKIPPED",
            "reason": "대화 내용에서 확인되지 않는 memoryPoint가 생성되어 저장하지 않았습니다.",
            "memoryPoint": memory_point,
            "question": question,
        }

    source_text = max(
        matching_candidates,
        key=lambda candidate: memory_point_match_score(memory_point, candidate),
    )

    structured_candidate = next(
        (
            candidate
            for candidate in structured_candidates
            if clean_text(candidate["sourceText"]) == clean_text(source_text)
        ),
        None,
    )

    if structured_candidates and structured_candidate is None:
        return {
            "status": "SKIPPED",
            "reason": "memoryPoint 엔진이 확정하지 않은 원문이 선택되었습니다.",
            "memoryPoint": memory_point,
            "answerKeyword": answer_keyword,
            "question": question,
        }

    if not is_valid_answer_keyword_format(answer_keyword):
        return {
            "status": "SKIPPED",
            "reason": "answerKeyword는 한 개의 짧은 단어 또는 구절이어야 합니다.",
            "memoryPoint": memory_point,
            "answerKeyword": answer_keyword,
            "question": question,
        }

    if not is_answer_keyword_grounded(
        answer_keyword,
        memory_point,
        source_text,
    ):
        return {
            "status": "SKIPPED",
            "reason": "answerKeyword가 memoryPoint 또는 원래 대화에서 확인되지 않습니다.",
            "memoryPoint": memory_point,
            "answerKeyword": answer_keyword,
            "question": question,
        }

    if structured_candidate:
        expected_answer = structured_candidate["answerValue"]
        expected_type = MemoryAnswerType(structured_candidate["answerType"])
        generated_type = infer_answer_type(question)

        if _normalize_keyword_word(answer_keyword) != _normalize_keyword_word(
            expected_answer
        ):
            return {
                "status": "SKIPPED",
                "reason": "answerKeyword가 memoryPoint 엔진이 확정한 단서와 다릅니다.",
                "memoryPoint": memory_point,
                "answerKeyword": answer_keyword,
                "question": question,
            }

        if generated_type not in {
            MemoryAnswerType.UNKNOWN,
            expected_type,
        }:
            return {
                "status": "SKIPPED",
                "reason": "회상 질문 유형이 memoryPoint 엔진의 단서 유형과 다릅니다.",
                "memoryPoint": memory_point,
                "answerKeyword": answer_keyword,
                "question": question,
            }

        evidence = structured_candidate.get("evidence")
        if isinstance(evidence, list):
            event_actions = set().union(
                *(
                    extract_memory_action_concepts(item.get("sourceText") or "")
                    for item in evidence
                )
            )
            question_actions = extract_recall_question_action_concepts(question)

            if question_actions and not question_actions <= event_actions:
                return {
                    "status": "SKIPPED",
                    "reason": (
                        "회상 질문에 memoryPoint 사건에 없는 행동이 포함되었습니다."
                    ),
                    "memoryPoint": memory_point,
                    "answerKeyword": answer_keyword,
                    "question": question,
                }

        memory_point = structured_candidate["sourceText"]
        answer_keyword = expected_answer

    if not is_answer_keyword_compatible_with_question(
        answer_keyword,
        question,
        source_text,
    ):
        return {
            "status": "SKIPPED",
            "reason": "회상 질문이 요구하는 답변 종류와 answerKeyword가 일치하지 않습니다.",
            "memoryPoint": memory_point,
            "answerKeyword": answer_keyword,
            "question": question,
        }

    if question_reveals_memory_answer(answer_keyword, question):
        return {
            "status": "SKIPPED",
            "reason": "회상 질문에 answerKeyword가 직접 포함되어 저장하지 않았습니다.",
            "memoryPoint": memory_point,
            "answerKeyword": answer_keyword,
            "question": question,
        }

    if used_memory_points and any(
        is_similar_memory_point(memory_point, used_memory_point)
        for used_memory_point in used_memory_points
    ):
        return {
            "status": "SKIPPED",
            "reason": "이미 사용한 memoryPoint와 유사하여 저장하지 않았습니다.",
            "memoryPoint": memory_point,
            "question": question,
        }

    if previous_questions:
        for previous_question in previous_questions:
            if is_similar_recall_question(question, previous_question):
                return {
                    "status": "SKIPPED",
                    "reason": "이미 생성된 질문과 유사한 질문이 생성되어 저장하지 않았습니다.",
                    "memoryPoint": memory_point,
                    "question": question,
                }

    action_concepts, _concrete_concepts = _memory_signature(source_text)
    memory_event = MemoryEvent.from_generation(
        source_text=source_text,
        memory_point=memory_point,
        answer_keyword=answer_keyword,
        question=question,
        action=",".join(sorted(action_concepts)),
        source_record_id=(
            structured_candidate["sourceRecordId"]
            if structured_candidate
            else None
        ),
        answer_type=(
            structured_candidate["answerType"]
            if structured_candidate
            else None
        ),
        quality_score=memory_evaluation["recallScore"],
    )

    return {
        "status": "CREATED",
        "reason": "",
        "memoryPoint": memory_point,
        "sourceText": source_text,
        "answerKeywords": [answer_keyword],
        "question": question,
        "memoryQualityScore": memory_evaluation["recallScore"],
        "memoryQualityReasons": memory_evaluation["reasons"],
        "memoryEvent": memory_event.to_dict(),
        **(
            {"memoryCandidate": structured_candidate}
            if structured_candidate
            else {}
        ),
    }


def save_recall_question_to_spring(
    user_id: int,
    memory_point: str,
    question_text: str,
    answer_keywords: Optional[List[str]] = None,
    memory_candidate: Optional[Dict] = None,
    base_url: str = BASE_URL,
) -> Dict:
    expected_answer = next(
        (
            str(keyword).strip()
            for keyword in answer_keywords or []
            if str(keyword).strip()
        ),
        memory_point,
    )
    body = {
        "userId": user_id,
        "questionText": question_text,
        "questionType": "RECALL",
        "category": "CONVERSATION",
        "expectedAnswer": expected_answer,
    }
    if memory_candidate:
        recall_clue = memory_candidate.get("recallClue") or {}
        body.update(
            {
                "eventId": memory_candidate.get("eventId"),
                "clueId": recall_clue.get("clueId"),
                "sourceRecordId": memory_candidate.get("sourceRecordId"),
                "answerType": memory_candidate.get("answerType"),
            }
        )

    response = requests.post(
        f"{base_url}/api/recall/questions",
        json=body,
        timeout=10,
    )
    response.raise_for_status()
    saved_question = response.json()
    question_id = saved_question.get("questionId")

    if question_id is not None and answer_keywords:
        for attempt in range(2):
            try:
                update_response = requests.put(
                    f"{base_url}/api/recall/{question_id}",
                    json={"keywords": answer_keywords},
                    timeout=10,
                )
                update_response.raise_for_status()
                saved_question["keywords"] = answer_keywords
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        "회상 질문 키워드 갱신 재시도: questionId=%s error=%s",
                        question_id,
                        e,
                    )
                    continue

                logger.warning(
                    "회상 질문은 저장했지만 키워드 갱신에 실패함: questionId=%s error=%s",
                    question_id,
                    e,
                )
                saved_question["keywordUpdateFailed"] = True

    return saved_question


def generate_and_save_recall_question(
    user_id: int,
    conversation_history: List[str],
    previous_questions: Optional[List[str]] = None,
    used_memory_points: Optional[List[str]] = None,
    memory_candidates: Optional[List[Dict]] = None,
    base_url: str = BASE_URL,
) -> Dict:
    if previous_questions is None or used_memory_points is None:
        existing = fetch_existing_recall_questions(
            user_id=user_id,
            base_url=base_url,
        )

        if previous_questions is None:
            previous_questions = existing["previousQuestions"]

        if used_memory_points is None:
            used_memory_points = existing["usedMemoryPoints"]

    result = generate_recall_question_from_conversation(
        conversation_history=conversation_history,
        previous_questions=previous_questions,
        used_memory_points=used_memory_points,
        memory_candidates=memory_candidates,
    )

    if result.get("status") != "CREATED":
        return result

    saved_question = save_recall_question_to_spring(
        user_id=user_id,
        memory_point=result["memoryPoint"],
        question_text=result["question"],
        answer_keywords=result.get("answerKeywords"),
        memory_candidate=result.get("memoryCandidate"),
        base_url=base_url,
    )

    return {
        **result,
        "savedQuestion": saved_question,
    }


if __name__ == "__main__":
    sample_conversation = [
        "오늘 아들이랑 통화했어요.",
        "점심에는 김치찌개를 먹었어요.",
        "오후에는 집 근처 마트에 다녀왔어요.",
    ]

    sample_previous_questions = [
        "조금 전에 통화 이야기를 해주셨는데, 누구와 통화하셨는지 기억나세요?"
    ]

    sample_used_memory_points = [
        "아들과 통화했다는 것"
    ]

    result = generate_recall_question_from_conversation(
        conversation_history=sample_conversation,
        previous_questions=sample_previous_questions,
        used_memory_points=sample_used_memory_points,
    )

    print("status:", result["status"])
    print("memoryPoint:", result["memoryPoint"])
    print("question:", result["question"])
