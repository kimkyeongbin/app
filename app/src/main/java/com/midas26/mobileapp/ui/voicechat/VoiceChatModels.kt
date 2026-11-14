package com.midas26.mobileapp.ui.voicechat

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.runtime.toMutableStateList
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.midas26.mobileapp.network.RetrofitClient
import com.midas26.mobileapp.util.PrefsManager
import com.midas26.mobileapp.util.WavRecorder
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody

/** 채팅 메시지 보낸 사람. */
enum class Sender { AI, USER }

/**
 * 채팅 한 줄.
 * - [isLoading]  : "AI 입력 중" 점 3개 표시
 * - [isSpeaking] : TTS 재생 중 — 말풍선에 음파 인디케이터
 */
data class ChatMessage(
    val from: Sender,
    val text: String,
    val isLoading: Boolean  = false,
    val isSpeaking: Boolean = false
)

/**
 * 음성 대화 화면 상태 머신.
 */
sealed interface VoiceChatState {
    data object Idle       : VoiceChatState
    data object Recording  : VoiceChatState
    data object Processing : VoiceChatState   // 업로드 + 서버 STT/AI 대기
    data object Playing    : VoiceChatState
}

/**
 * 음성 대화 ViewModel.
 *
 * - 화면 진입 시 대화 세션을 서버에서 생성한다.
 * - 녹음 중지 → WAV 파일 → 서버 업로드 → Processing 유지
 * - AI 응답 수신 엔드포인트는 미구현 → [dispatchAiReply]로 나중에 연결
 */
class VoiceChatViewModel(application: Application) : AndroidViewModel(application) {

    private val prefs       = PrefsManager.from(application)
    private val wavRecorder = WavRecorder(application)
    private val api         = RetrofitClient.voiceChat

    private val userId: Int  get() = prefs.getUserId()
    private var sessionId: Long    = -1L
    private var sessionEnded       = false
    private var voiceUploaded      = false
    private var pendingRecallQuestionId: Long? = null

    fun hasUploadedVoice(): Boolean = voiceUploaded

    // ── 상태 ──────────────────────────────────────────────────────────────────

    private var _state by mutableStateOf<VoiceChatState>(VoiceChatState.Idle)
    val state: VoiceChatState get() = _state

    private val _messages: SnapshotStateList<ChatMessage> = listOf<ChatMessage>().toMutableStateList()
    val messages: List<ChatMessage> get() = _messages

    private var _recordingSeconds by mutableStateOf(0)
    val recordingSeconds: Int get() = _recordingSeconds

    private var _speakTrigger by mutableStateOf(0)
    val speakTrigger: Int get() = _speakTrigger

    // ── 문장 단위 출력 ────────────────────────────────────────────────────────

    /** 현재 재생 중인 AI 답변의 문장 목록 */
    private var currentSentences: List<String> = emptyList()
    private var sentenceIndex: Int = 0
    private var _displayedText by mutableStateOf("")
    private var advanceSentenceJob: Job? = null

    /**
     * 말풍선에 표시할 텍스트.
     * - _displayedText가 설정되어 있으면 그대로 유지 (재생 중·완료 후 모두)
     * - 아직 설정 전(초기 진입)이면 마지막 AI 메시지 전체 텍스트로 폴백
     */
    val displayedText: String get() = _displayedText.ifEmpty {
        _messages.lastOrNull { it.from == Sender.AI && !it.isLoading }?.text ?: ""
    }

    /** 문장 부호([.!?~。]) 기준으로 텍스트를 문장 목록으로 분리. */
    private fun splitSentences(text: String): List<String> =
        text.split(Regex("(?<=[.!?~。])\\s*"))
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private var timerJob: Job? = null

    // ── 초기화: 세션 시작 ─────────────────────────────────────────────────────

    init {
        viewModelScope.launch {
            runCatching { api.startSession(userId) }
                .onSuccess { resp ->
                    val data = resp.body()?.data
                    sessionId = data?.sessionId ?: -1L
                    if (sessionId > 0) prefs.saveLastSessionId(sessionId)
                    dispatchAiReply(greetingText(data?.fixedQuestionsDoneToday == true, data?.openingQuestionText))
                }
                .onFailure {
                    dispatchAiReply(greetingText(fixedQuestionsDoneToday = false, openingQuestionText = null))
                }
        }
    }

    /**
     * 세션 시작 직후 AI가 먼저 건네는 인사말.
     * 오늘 고정질문을 이미 마쳤으면 바로 자유대화로 유도하고,
     * 그렇지 않으면 서버가 내려준 첫 고정질문(FIXED_QUESTIONS[0], 단일 출처는 backend/ai)으로 이어간다.
     */
    private fun greetingText(fixedQuestionsDoneToday: Boolean, openingQuestionText: String?): String {
        val greeting = "안녕하세요! 오늘도 반갑습니다."
        return when {
            fixedQuestionsDoneToday -> "$greeting 편하게 말씀해주세요."
            !openingQuestionText.isNullOrBlank() -> "$greeting $openingQuestionText"
            else -> "$greeting 편하게 말씀해주세요."
        }
    }

    // ── 녹음 ──────────────────────────────────────────────────────────────────

    fun startRecording() {
        if (_state !is VoiceChatState.Idle) return
        _state = VoiceChatState.Recording
        _recordingSeconds = 0
        wavRecorder.start()
        timerJob = viewModelScope.launch {
            while (true) {
                delay(1000)
                _recordingSeconds += 1
            }
        }
    }

    fun stopRecording() {
        if (_state !is VoiceChatState.Recording) return
        timerJob?.cancel()
        timerJob = null

        val file = wavRecorder.stop()
        _state = VoiceChatState.Processing

        viewModelScope.launch {
            var recordId = -1L

            // 업로드 — 성공하면 recordId 확보
            if (file != null && sessionId > 0) {
                runCatching {
                    val body = file.asRequestBody("audio/wav".toMediaType())
                    val part = MultipartBody.Part.createFormData("file", file.name, body)
                    val recallQuestionId = pendingRecallQuestionId
                    val answerRole = if (recallQuestionId != null) "RECALL" else "INITIAL"

                    api.uploadVoice(
                        userId = userId,
                        sessionId = sessionId,
                        file = part,
                        recallQuestionId = recallQuestionId,
                        answerRole = answerRole
                    )
                }.onSuccess { resp ->
                    resp.body()?.data?.audioRecordId?.let {
                        recordId = it
                        voiceUploaded = true
                    }
                }
            }

            if (recordId > 0) {
                pollForReply(recordId)
            } else {
                dispatchAiReply("음성 전송에 실패했어요. 다시 시도해주세요.")
            }
        }
    }

    /** Query immediately; enforce a deadline even when an individual request stalls. */
    private suspend fun pollForReply(recordId: Long) {
        val replyText = awaitReply { api.pollReply(recordId).body()?.data?.replyText }
        if (!replyText.isNullOrBlank()) {
            withTimeoutOrNull(3_000) { refreshPendingRecallQuestion(recordId) }
            dispatchAiReply(replyText)
            return
        }
        dispatchAiReply("답변을 받아오지 못했어요. 잠시 후 다시 시도해주세요.")
    }

    private suspend fun refreshPendingRecallQuestion(recordId: Long) {
        repeat(6) { attempt ->
            val records = runCatching { api.getSessionRecords(sessionId) }
                .getOrNull()
                ?.body()
                .orEmpty()

            val current = records.firstOrNull { it.recordId == recordId }
            val role = current?.answerRole?.uppercase()
            val questionId = current?.recallQuestionId

            if (role == "INITIAL" && questionId != null) {
                pendingRecallQuestionId = questionId
                return
            }

            if (role == "RECALL") {
                pendingRecallQuestionId = null
                return
            }

            if (attempt < 5) delay(500)
        }
    }

    // ── AI 응답 수신 ──────────────────────────────────────────────────────────

    fun dispatchAiReply(text: String) {
        // 문장 분리 후 첫 문장부터 시작
        currentSentences = splitSentences(text).ifEmpty { listOf(text) }
        sentenceIndex = 0
        _displayedText = currentSentences[0]
        _messages.add(ChatMessage(Sender.AI, text, isSpeaking = true))
        _state = VoiceChatState.Playing
        _speakTrigger++
    }

    /**
     * TTS 한 문장 완료 시 호출.
     * 600ms 쉰 뒤 다음 문장으로 진행하고, 마지막 문장이면 재생 종료.
     */
    fun advanceSentence() {
        advanceSentenceJob?.cancel()
        if (_state !is VoiceChatState.Playing) return
        sentenceIndex++
        if (sentenceIndex < currentSentences.size) {
            advanceSentenceJob = viewModelScope.launch {
                delay(600L) // 문장 사이 잠시 쉬기
                if (_state !is VoiceChatState.Playing) return@launch
                _displayedText = currentSentences[sentenceIndex]
                _speakTrigger++
            }
        } else {
            finishPlaying()
        }
    }

    // ── TTS 제어 ──────────────────────────────────────────────────────────────

    fun finishPlaying() {
        advanceSentenceJob?.cancel()
        val last = _messages.lastOrNull()
        if (last != null && last.isSpeaking) {
            _messages[_messages.lastIndex] = last.copy(isSpeaking = false)
        }
        if (_state is VoiceChatState.Playing) {
            _state = VoiceChatState.Idle
        }
    }

    fun replayLastAi() {
        advanceSentenceJob?.cancel()
        val lastAi = _messages.lastOrNull { it.from == Sender.AI && !it.isLoading } ?: return
        // currentSentences가 비어 있으면 (초기 메시지 등) 그 자리에서 분리
        if (currentSentences.isEmpty()) {
            currentSentences = splitSentences(lastAi.text).ifEmpty { listOf(lastAi.text) }
        }
        sentenceIndex = 0
        _displayedText = currentSentences[0]
        val idx = _messages.lastIndexOf(lastAi)
        if (idx >= 0) _messages[idx] = lastAi.copy(isSpeaking = true)
        _state = VoiceChatState.Playing
        _speakTrigger++
    }

    // ── 세션 종료 ─────────────────────────────────────────────────────────────

    /**
     * 화면 이탈 시 호출 — 세션 종료 및 AI 분석 트리거.
     * 중복 호출되더라도 최초 1회만 실행된다.
     */
    fun endSession() {
        if (sessionId <= 0 || sessionEnded) return
        sessionEnded = true
        val endingSessionId = sessionId
        finishVoiceSession(viewModelScope, onFailure = {
            android.util.Log.w("VoiceSession", "completion failed session=$endingSessionId type=${it.javaClass.simpleName}")
        }) {
            val response = api.completeSession(endingSessionId)
            check(response.isSuccessful) { "Completion HTTP ${response.code()}" }
            android.util.Log.i("VoiceSession", "completion accepted session=$endingSessionId")
        }
    }

    fun speakMessage(index: Int) {
        val msg = _messages.getOrNull(index) ?: return
        if (msg.from != Sender.AI || msg.isLoading) return
        _messages.replaceAll { it.copy(isSpeaking = false) }
        _messages[index] = msg.copy(isSpeaking = true)
        _state = VoiceChatState.Playing
        _speakTrigger++
    }

    override fun onCleared() {
        timerJob?.cancel()
        advanceSentenceJob?.cancel()
        wavRecorder.stop()
        super.onCleared()
    }
}
