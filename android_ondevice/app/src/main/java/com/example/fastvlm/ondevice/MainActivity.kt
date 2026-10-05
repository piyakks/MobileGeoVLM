package com.example.fastvlm.ondevice

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    companion object {
        const val INTERVAL_MS = 5000L       // run the model at most once every 5 seconds
        const val SCENE_THRESHOLD = 12      // mean abs gray diff (0-255) needed to count as a new scene
        const val CIRCLE_PROMPT = "What is in the red circle? Answer in one sentence."
    }

    /**
     * What to ask. GeoChat tasks use the tags the LoRA was trained on; their answers carry boxes that are
     * drawn automatically (like geochat_demo.py).
     */
    enum class Task(val label: String, val hint: String, val maxTokens: Int, val draws: Boolean) {
        FREE("자유 질문", "질문을 입력하세요", 64, false),
        CIRCLE("빨간 동그라미", "동그라미를 그리면 그 안의 물체를 설명합니다", 64, true),
        GROUNDING("Grounding (전체 탐지)", "이미지를 설명하고 언급한 물체마다 박스를 그립니다", 256, false),
        REFER("Refer (물체 찾기)", "찾을 물체를 입력하세요 (예: white airplane)", 48, false),
        IDENTIFY("Identify (영역 식별)", "물체 주위에 동그라미/박스를 그리세요", 48, true),
    }

    private lateinit var previewView: PreviewView
    private lateinit var stillImage: ImageView
    private lateinit var answer: TextView
    private lateinit var status: TextView
    private lateinit var prompt: EditText
    private lateinit var btnLive: Button
    private lateinit var btnGallery: Button
    private lateinit var btnAsk: Button
    private lateinit var btnClear: Button
    private lateinit var modelSpinner: Spinner
    private lateinit var taskSpinner: Spinner
    private lateinit var drawOverlay: CircleOverlayView
    private lateinit var boxOverlay: BoxOverlayView

    private var task = Task.FREE
    private var stillBitmap: Bitmap? = null   // gallery image currently shown (null in live mode)

    private var vlm: FastVLM? = null
    private var liveJob: Job? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    // Analyzer hands over one frame when requested
    @Volatile private var frameRequest: CompletableDeferred<Bitmap>? = null

    private var lastSignature: IntArray? = null
    private var lastPrompt: String? = null
    private var lastStats = ""
    private lateinit var metrics: TextView
    private var lastLatency = ""

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startLive() else status.text = "카메라 권한이 필요합니다."
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri ?: return@registerForActivityResult
        stopLive()
        val bmp = try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { d, _, _ ->
                    d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (e: Exception) {
            status.text = "이미지 로드 실패: ${e.message}"
            return@registerForActivityResult
        }
        stillBitmap = bmp
        previewView.visibility = View.GONE
        stillImage.visibility = View.VISIBLE
        stillImage.setImageBitmap(bmp)
        drawOverlay.clear()
        boxOverlay.clear()
        answer.text = ""
        btnAsk.visibility = View.VISIBLE
        if (task.draws) status.text = task.hint else askStill()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        stillImage = findViewById(R.id.stillImage)
        answer = findViewById(R.id.answer)
        status = findViewById(R.id.status)
        metrics = findViewById(R.id.metrics)
        prompt = findViewById(R.id.prompt)
        btnLive = findViewById(R.id.btnLive)
        btnGallery = findViewById(R.id.btnGallery)
        btnAsk = findViewById(R.id.btnAsk)
        btnClear = findViewById(R.id.btnClear)
        modelSpinner = findViewById(R.id.modelSpinner)
        taskSpinner = findViewById(R.id.taskSpinner)
        drawOverlay = findViewById(R.id.overlay)
        boxOverlay = findViewById(R.id.boxOverlay)

        btnLive.setOnClickListener {
            if (liveJob != null) stopLive()
            else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startLive()
            else requestCamera.launch(Manifest.permission.CAMERA)
        }
        btnGallery.setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        btnAsk.setOnClickListener { askStill() }
        btnClear.setOnClickListener { drawOverlay.clear(); boxOverlay.clear() }

        setupTaskSpinner()
        setupModelSpinner()
        startMemoryMonitor()
    }

    /** Shows this app's RAM use (PSS, includes ONNX Runtime's native memory), refreshed every second. */
    private fun startMemoryMonitor() {
        lifecycleScope.launch {
            while (isActive) {
                val ramGb = withContext(Dispatchers.Default) {
                    Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss / 1024f / 1024f
                }
                metrics.text = "지연 ${lastLatency.ifEmpty { "-" }} · RAM %.2f GB".format(ramGb)
                delay(1000)
            }
        }
    }

    // ------------------------------------------------------------------ task

    private fun setupTaskSpinner() {
        taskSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, Task.entries.map { it.label })
        taskSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) = setTask(Task.entries[position])
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setTask(t: Task) {
        task = t
        drawOverlay.drawingEnabled = t.draws
        drawOverlay.visibility = if (t.draws) View.VISIBLE else View.GONE
        drawOverlay.clear()
        boxOverlay.clear()
        btnClear.visibility = if (t.draws) View.VISIBLE else View.GONE
        // Free text is the question; Refer takes the object name; the other tasks have a fixed prompt
        prompt.isEnabled = t == Task.FREE || t == Task.REFER
        prompt.hint = t.hint
        prompt.setText(when (t) {
            Task.FREE -> "Describe the image in one sentence."
            Task.REFER -> ""
            Task.CIRCLE -> CIRCLE_PROMPT
            Task.GROUNDING -> "[grounding] describe this image in detail"
            Task.IDENTIFY -> "[identify] what is this {box}"
        })
        lastSignature = null
        status.text = t.hint
    }

    /** Where the image is shown inside the overlays: the whole view for live frames, the fitCenter rect for stills. */
    private fun imageRect(): RectF? {
        val bmp = stillBitmap ?: return null
        val vw = stillImage.width.toFloat()
        val vh = stillImage.height.toFloat()
        val scale = minOf(vw / bmp.width, vh / bmp.height)
        val w = bmp.width * scale
        val h = bmp.height * scale
        return RectF((vw - w) / 2, (vh - h) / 2, (vw + w) / 2, (vh + h) / 2)
    }

    /** Builds the model prompt and input image for the current task, or returns null with a status message. */
    private fun buildRequest(frame: Bitmap): Pair<String, Bitmap>? {
        val rect = imageRect()
        return when (task) {
            Task.FREE -> prompt.text.toString().ifBlank { "Describe the image in one sentence." } to frame
            Task.CIRCLE -> {
                if (!drawOverlay.hasCircle) { status.text = task.hint; return null }
                CIRCLE_PROMPT to drawOverlay.renderOnto(frame, rect)
            }
            Task.GROUNDING -> "[grounding] describe this image in detail" to frame
            Task.REFER -> {
                val obj = prompt.text.toString().trim()
                if (obj.isEmpty()) { status.text = task.hint; return null }
                "[refer] where is <p>$obj</p> ?" to frame
            }
            Task.IDENTIFY -> {
                // GeoChat gets the region as box coordinates on the clean image, not as a drawn circle
                val b = drawOverlay.boundingBox(rect) ?: run { status.text = task.hint; return null }
                "[identify] what is this ${GeoChatFormat.boxString(b[0], b[1], b[2], b[3])}" to frame
            }
        }
    }

    // ------------------------------------------------------------------ models

    /** Each subfolder of files/models/ holding a complete export is one selectable model. */
    private fun modelsRoot() = File(getExternalFilesDir(null), "models")

    private fun availableModels(): List<File> =
        modelsRoot().listFiles()?.filter { it.isDirectory && File(it, "fastvlm_meta.json").exists() }
            ?.sortedBy { it.name } ?: emptyList()

    private fun setupModelSpinner() {
        val models = availableModels()
        if (models.isEmpty()) {
            status.text = "모델이 없습니다. adb push <모델폴더>/. ${modelsRoot().absolutePath}/<이름>/"
            return
        }
        modelSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models.map { it.name })
        val saved = getPreferences(MODE_PRIVATE).getString("model", null)
        modelSpinner.setSelection(models.indexOfFirst { it.name == saved }.coerceAtLeast(0))
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val dir = models[position]
                getPreferences(MODE_PRIVATE).edit().putString("model", dir.name).apply()
                loadModel(dir)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun loadModel(dir: File) {
        stopLive()
        setButtonsEnabled(false)
        val previous = vlm
        vlm = null
        val missing = FastVLM.REQUIRED.filterNot { File(dir, it).exists() }
        if (missing.isNotEmpty()) {
            previous?.close()
            status.text = "${dir.name}: 파일이 없습니다: ${missing.joinToString()}"
            return
        }
        status.text = "${dir.name} 로딩 중..."
        answer.text = ""
        lastStats = ""
        lifecycleScope.launch {
            val t0 = SystemClock.elapsedRealtime()
            val loaded = withContext(Dispatchers.Default) {
                previous?.close()  // free the old model's memory before loading the next one
                try { FastVLM(dir) } catch (e: Throwable) { e }
            }
            if (loaded is FastVLM) {
                vlm = loaded
                status.text = "${dir.name} 준비 완료 (${SystemClock.elapsedRealtime() - t0} ms)"
                setButtonsEnabled(true)
            } else {
                status.text = "${dir.name} 로드 실패: ${(loaded as Throwable).message}"
            }
        }
    }

    private fun setButtonsEnabled(on: Boolean) {
        btnLive.isEnabled = on
        btnGallery.isEnabled = on
        btnAsk.isEnabled = on
    }

    // ------------------------------------------------------------------ still image

    private fun askStill() {
        val bmp = stillBitmap ?: return
        val (p, input) = buildRequest(bmp) ?: return
        lifecycleScope.launch { runModel(input, p) }
    }

    // ------------------------------------------------------------------ live camera

    private fun startLive() {
        stillBitmap = null
        stillImage.visibility = View.GONE
        previewView.visibility = View.VISIBLE
        btnAsk.visibility = View.GONE
        drawOverlay.clear()
        boxOverlay.clear()
        lastSignature = null

        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(1920, 1440), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build()
            val preview = Preview.Builder().setResolutionSelector(resolution).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { proxy -> deliverFrame(proxy) }
            // Share the preview's viewport so analysis frames are cropped to exactly what is on screen;
            // drawings and boxes in view coordinates then map onto the frame by plain scaling.
            val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
            previewView.viewPort?.let { group.setViewPort(it) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, group.build())
        }, ContextCompat.getMainExecutor(this))

        btnLive.text = "정지"
        liveJob = lifecycleScope.launch {
            var lastDrawVersion = -1
            while (isActive) {
                val t0 = SystemClock.elapsedRealtime()
                val raw = grabFrame()
                val request = raw?.let { buildRequest(it) }
                if (raw != null && request != null) {
                    val (p, input) = request
                    val sig = signature(raw)
                    val drawVersion = drawOverlay.version
                    if (sceneChanged(sig, p) || (task.draws && drawVersion != lastDrawVersion)) {
                        if (runModel(input, p)) { lastSignature = sig; lastPrompt = p; lastDrawVersion = drawVersion }
                    } else {
                        status.text = "장면 변화 없음 — 이전 답변 유지 · $lastStats"
                    }
                }
                delay(maxOf(0L, INTERVAL_MS - (SystemClock.elapsedRealtime() - t0)))
            }
        }
    }

    private fun stopLive() {
        liveJob?.cancel()
        liveJob = null
        frameRequest?.cancel()
        frameRequest = null
        cameraProvider?.unbindAll()
        btnLive.text = "실시간 시작"
    }

    private fun deliverFrame(proxy: ImageProxy) {
        proxy.use {
            val req = frameRequest ?: return
            val bmp = it.toBitmap()
            val crop = it.cropRect
            val rotation = it.imageInfo.rotationDegrees
            val upright = Bitmap.createBitmap(bmp, crop.left, crop.top, crop.width(), crop.height(),
                Matrix().apply { postRotate(rotation.toFloat()) }, true)
            frameRequest = null
            req.complete(upright)
        }
    }

    private suspend fun grabFrame(): Bitmap? {
        val req = CompletableDeferred<Bitmap>()
        frameRequest = req
        return withTimeoutOrNull(2000) { req.await() }
    }

    /** 32x32 grayscale thumbnail used to detect real scene changes. */
    private fun signature(bmp: Bitmap): IntArray {
        val small = Bitmap.createScaledBitmap(bmp, 32, 32, true)
        val px = IntArray(32 * 32)
        small.getPixels(px, 0, 32, 0, 0, 32, 32)
        return IntArray(px.size) { ((px[it] shr 16 and 0xFF) + (px[it] shr 8 and 0xFF) + (px[it] and 0xFF)) / 3 }
    }

    private fun sceneChanged(sig: IntArray, p: String): Boolean {
        val last = lastSignature ?: return true
        if (p != lastPrompt) return true
        var sum = 0L
        for (i in sig.indices) sum += abs(sig[i] - last[i])
        return sum / sig.size > SCENE_THRESHOLD
    }

    // ------------------------------------------------------------------ inference

    /** Shows the answer text and draws any boxes in it over the image. */
    private fun showAnswer(text: String) {
        val boxes = GeoChatFormat.mergeDuplicates(GeoChatFormat.parseBoxes(text))
        val colors = GeoChatFormat.colorsFor(boxes)
        answer.text = GeoChatFormat.styledAnswer(text, colors)
        boxOverlay.show(boxes, colors, imageRect())
    }

    /** Runs the model off the main thread and streams tokens into the answer view. Returns true on success. */
    private suspend fun runModel(input: Bitmap, p: String): Boolean {
        val model = vlm ?: return false
        status.text = "추론 중..."
        val job = liveJob
        return try {
            val (text, stats) = withContext(Dispatchers.Default) {
                model.generate(input, p, task.maxTokens,
                    shouldContinue = { job == null || job.isActive },
                    onToken = { partial -> runOnUiThread { showAnswer(partial) } })
            }
            showAnswer(text)
            lastStats = "비전 ${stats.visionMs} ms · 첫 토큰 ${stats.ttftMs} ms · 전체 ${stats.totalMs} ms · ${stats.tokens} 토큰"
            lastLatency = "%.1f초".format(stats.totalMs / 1000f)
            status.text = lastStats
            Log.i("FastVLM", "$lastStats | $p | $text")
            true
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            status.text = "추론 실패: ${e.message}"
            false
        }
    }

    override fun onDestroy() {
        stopLive()
        analysisExecutor.shutdown()
        vlm?.close()
        super.onDestroy()
    }
}
