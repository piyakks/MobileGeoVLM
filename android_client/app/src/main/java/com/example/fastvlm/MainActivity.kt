package com.example.fastvlm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private lateinit var serverUrl: EditText
    private lateinit var preview: ImageView
    private lateinit var prompt: EditText
    private lateinit var answer: TextView
    private lateinit var progress: ProgressBar
    private lateinit var btnSend: Button

    private var bitmap: Bitmap? = null
    private var cameraUri: Uri? = null

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { setImage(it) }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) cameraUri?.let { setImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        serverUrl = findViewById(R.id.serverUrl)
        preview = findViewById(R.id.preview)
        prompt = findViewById(R.id.prompt)
        answer = findViewById(R.id.answer)
        progress = findViewById(R.id.progress)
        btnSend = findViewById(R.id.btnSend)

        val prefs = getSharedPreferences("settings", Context.MODE_PRIVATE)
        serverUrl.setText(prefs.getString("server_url", "http://192.168.0.10:8000"))

        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            val dir = File(cacheDir, "camera").apply { mkdirs() }
            val file = File(dir, "capture.jpg")
            cameraUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            takePicture.launch(cameraUri)
        }
        btnSend.setOnClickListener {
            prefs.edit().putString("server_url", serverUrl.text.toString().trim()).apply()
            send()
        }
    }

    private fun setImage(uri: Uri) {
        bitmap = loadBitmap(uri)?.let { downscale(it, 1024) }
        preview.setImageBitmap(bitmap)
        answer.text = ""
    }

    private fun loadBitmap(uri: Uri): Bitmap? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    } catch (e: Exception) {
        answer.text = "Failed to load image: ${e.message}"
        null
    }

    // Shrink large photos before upload; the server resizes to the model input anyway.
    private fun downscale(src: Bitmap, maxSide: Int): Bitmap {
        val scale = maxSide.toFloat() / maxOf(src.width, src.height)
        if (scale >= 1f) return src
        return Bitmap.createScaledBitmap(src, (src.width * scale).toInt(), (src.height * scale).toInt(), true)
    }

    private fun send() {
        val bmp = bitmap ?: run { answer.text = "Pick an image first."; return }
        val base = serverUrl.text.toString().trim().trimEnd('/')
        val question = prompt.text.toString().ifBlank { "Describe the image." }

        setLoading(true)
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                try {
                    val jpeg = ByteArrayOutputStream().use {
                        bmp.compress(Bitmap.CompressFormat.JPEG, 90, it)
                        it.toByteArray()
                    }
                    val body = MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("image", "image.jpg", jpeg.toRequestBody("image/jpeg".toMediaType()))
                        .addFormDataPart("prompt", question)
                        .build()
                    val request = Request.Builder().url("$base/predict").post(body).build()
                    client.newCall(request).execute().use { resp ->
                        val text = resp.body?.string().orEmpty()
                        if (!resp.isSuccessful) {
                            "Error ${resp.code}: $text"
                        } else {
                            val json = JSONObject(text)
                            "${json.getString("answer")}\n\n(${json.optInt("latency_ms")} ms)"
                        }
                    }
                } catch (e: Exception) {
                    "Request failed: ${e.message}"
                }
            }
            answer.text = result
            setLoading(false)
        }
    }

    private fun setLoading(loading: Boolean) {
        progress.visibility = if (loading) View.VISIBLE else View.GONE
        btnSend.isEnabled = !loading
    }
}
