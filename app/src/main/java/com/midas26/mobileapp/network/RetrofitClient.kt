package com.midas26.mobileapp.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitClient {

    private const val BASE_URL = "http://10.0.2.2:8080/"

    private var tokenProvider: (() -> String?)? = null
    private var onUnauthorized: (() -> Unit)? = null

    fun init(tokenProvider: () -> String?, onUnauthorized: (() -> Unit)? = null) {
        this.tokenProvider = tokenProvider
        this.onUnauthorized = onUnauthorized
    }

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val authInterceptor = okhttp3.Interceptor { chain ->
        val token = tokenProvider?.invoke()
        val request = if (token != null) {
            chain.request().newBuilder()
                .header("Authorization", "Bearer $token")
                .build()
        } else {
            chain.request()
        }
        val response = chain.proceed(request)
        if (response.code == 401) {
            onUnauthorized?.invoke()
        }
        response
    }

    private val client = OkHttpClient.Builder()
        .addInterceptor(authInterceptor)
        .addInterceptor(loggingInterceptor)
        .build()

    private val retrofit: Retrofit by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    internal fun voiceClient(base: OkHttpClient): OkHttpClient = base.newBuilder()
        .eventListenerFactory { VoiceNetworkTiming() }
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .callTimeout(90, TimeUnit.SECONDS)
        // Uploads create records; do not replay them after an uncertain connection failure.
        .retryOnConnectionFailure(false)
        .build()

    val instance: AuthApiService by lazy {
        retrofit.create(AuthApiService::class.java)
    }

    val voiceChat: VoiceChatApiService by lazy {
        retrofit.newBuilder()
            .client(voiceClient(client))
            .build()
            .create(VoiceChatApiService::class.java)
    }

    val location: LocationApiService by lazy {
        retrofit.create(LocationApiService::class.java)
    }

    val analysis: AnalysisApiService by lazy {
        retrofit.create(AnalysisApiService::class.java)
    }
}
