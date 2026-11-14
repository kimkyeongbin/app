package com.midas26.mobileapp.ui.voicechat

import java.io.IOException
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull

internal suspend fun awaitReply(
    totalTimeoutMs: Long = 30_000,
    requestTimeoutMs: Long = 5_000,
    intervalMs: Long = 1_000,
    fetch: suspend () -> String?
): String? = withTimeoutOrNull(totalTimeoutMs) {
    var reply: String? = null
    while (reply.isNullOrBlank()) {
        reply = try {
            withTimeoutOrNull(requestTimeoutMs) { fetch() }
        } catch (_: IOException) {
            null
        }
        if (reply.isNullOrBlank()) delay(intervalMs)
    }
    reply
}
