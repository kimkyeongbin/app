package com.midas26.mobileapp.network

import android.util.Log
import java.io.IOException
import java.util.concurrent.atomic.AtomicLong
import okhttp3.Call
import okhttp3.EventListener
import okhttp3.Response

internal class VoiceNetworkTiming : EventListener() {
    private val id = ids.incrementAndGet()
    private var started = 0L

    private fun record(stage: String, detail: String = "") {
        Log.d("VoiceNetworkTiming", "id=$id stage=$stage elapsed_ms=${(System.nanoTime() - started) / 1_000_000} $detail")
    }

    override fun callStart(call: Call) {
        started = System.nanoTime()
        record("start", "method=${call.request().method} path=${call.request().url.encodedPath}")
    }
    override fun requestHeadersEnd(call: Call, request: okhttp3.Request) = record("request_headers_sent")
    override fun requestBodyEnd(call: Call, byteCount: Long) = record("request_body_sent", "bytes=$byteCount")
    override fun responseHeadersEnd(call: Call, response: Response) = record("response_headers", "status=${response.code}")
    override fun responseBodyEnd(call: Call, byteCount: Long) = record("response_body", "bytes=$byteCount")
    override fun callEnd(call: Call) = record("complete")
    override fun callFailed(call: Call, ioe: IOException) = record("failed", "type=${ioe.javaClass.simpleName}")

    companion object { private val ids = AtomicLong() }
}
