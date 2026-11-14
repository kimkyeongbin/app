package com.midas26.mobileapp.ui.voicechat

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

internal fun finishVoiceSession(
    scope: CoroutineScope,
    timeoutMs: Long = 15_000,
    onFailure: (Exception) -> Unit,
    complete: suspend () -> Unit
) = scope.launch(start = CoroutineStart.UNDISPATCHED) {
    // Navigation clears the ViewModel immediately. Finish this one request only,
    // with a deadline; do not replay a potentially committed completion POST.
    withContext(NonCancellable) {
        try {
            withTimeout(timeoutMs) { complete() }
        } catch (failure: Exception) {
            onFailure(failure)
        }
    }
}
