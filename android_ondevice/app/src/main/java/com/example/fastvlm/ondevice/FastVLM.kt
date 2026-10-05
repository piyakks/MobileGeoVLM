package com.example.fastvlm.ondevice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * On-device FastVLM using three ONNX graphs exported by onnx_export/export_onnx.py.
 * Same pipeline as onnx_export/run_onnx.py.
 */
class FastVLM(private val modelDir: File, numThreads: Int = 4) : AutoCloseable {

    data class Stats(val visionMs: Long, val ttftMs: Long, val totalMs: Long, val tokens: Int)

    companion object {
        const val VISION = "vision_encoder.onnx"
        const val EMBED = "embed_tokens_int8.onnx"
        const val DECODER = "decoder_q4.onnx"
        val REQUIRED = listOf(VISION, EMBED, DECODER, "$DECODER.data", "vocab.json", "merges.txt", "fastvlm_meta.json")
    }

    private val env = OrtEnvironment.getEnvironment()
    private val vision: OrtSession
    private val embed: OrtSession
    private val decoder: OrtSession
    val tokenizer: QwenTokenizer

    private val imageSize: Int
    private val hidden: Int
    private val numLayers: Int
    private val kvHeads: Int
    private val headDim: Int
    private val eosId: Int
    private val preIds: IntArray
    private val assistantPrefix: String

    init {
        val meta = JSONObject(File(modelDir, "fastvlm_meta.json").readText())
        imageSize = meta.getInt("image_size")
        hidden = meta.getInt("hidden_size")
        numLayers = meta.getInt("num_layers")
        kvHeads = meta.getInt("num_kv_heads")
        headDim = meta.getInt("head_dim")
        eosId = meta.getInt("eos_token_id")
        assistantPrefix = meta.getString("assistant_prefix")

        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(numThreads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        // Load by path so the decoder's external .data file is found next to it
        vision = env.createSession(File(modelDir, VISION).absolutePath, opts)
        embed = env.createSession(File(modelDir, EMBED).absolutePath, opts)
        decoder = env.createSession(File(modelDir, DECODER).absolutePath, opts)

        tokenizer = QwenTokenizer(modelDir)
        preIds = tokenizer.encode(meta.getString("system_prompt") + meta.getString("user_prefix"))
    }

    /** Pad to square (black), resize to imageSize, scale to [0,1], CHW float. */
    private fun preprocess(src: Bitmap): FloatBuffer {
        val side = maxOf(src.width, src.height)
        val square = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        Canvas(square).apply {
            drawColor(Color.BLACK)
            drawBitmap(src, ((side - src.width) / 2).toFloat(), ((side - src.height) / 2).toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        val resized = Bitmap.createScaledBitmap(square, imageSize, imageSize, true)
        val pixels = IntArray(imageSize * imageSize)
        resized.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        if (resized !== square) resized.recycle()
        square.recycle()

        val plane = imageSize * imageSize
        val buf = FloatBuffer.allocate(3 * plane)
        val arr = buf.array()
        for (i in 0 until plane) {
            val p = pixels[i]
            arr[i] = ((p shr 16) and 0xFF) / 255f
            arr[plane + i] = ((p shr 8) and 0xFF) / 255f
            arr[2 * plane + i] = (p and 0xFF) / 255f
        }
        return buf
    }

    private fun embedIds(ids: IntArray): FloatArray {
        val input = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { ids[it].toLong() }), longArrayOf(1, ids.size.toLong()))
        input.use {
            embed.run(mapOf("input_ids" to it)).use { r ->
                val fb = (r[0] as OnnxTensor).floatBuffer
                return FloatArray(fb.remaining()).also { a -> fb.get(a) }
            }
        }
    }

    /**
     * Generate an answer for [image] and [prompt] with greedy decoding.
     * [onToken] receives the decoded text so far; return false from [shouldContinue] to stop early.
     */
    fun generate(
        image: Bitmap,
        prompt: String,
        maxNewTokens: Int = 64,
        shouldContinue: () -> Boolean = { true },
        onToken: (String) -> Unit = {},
    ): Pair<String, Stats> {
        val t0 = SystemClock.elapsedRealtime()

        // Vision encoder -> (1, N, hidden)
        val imageFeatures: FloatArray
        OnnxTensor.createTensor(env, preprocess(image), longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())).use { px ->
            vision.run(mapOf("pixel_values" to px)).use { r ->
                val fb = (r[0] as OnnxTensor).floatBuffer
                imageFeatures = FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
        val visionMs = SystemClock.elapsedRealtime() - t0

        // <system><user> [image] \n<prompt><|im_end|>\n<assistant>
        val pre = embedIds(preIds)
        val post = embedIds(tokenizer.encode("\n" + prompt + assistantPrefix))
        var inputsEmbeds = FloatArray(pre.size + imageFeatures.size + post.size).also {
            System.arraycopy(pre, 0, it, 0, pre.size)
            System.arraycopy(imageFeatures, 0, it, pre.size, imageFeatures.size)
            System.arraycopy(post, 0, it, pre.size + imageFeatures.size, post.size)
        }

        var seq = inputsEmbeds.size / hidden
        var total = seq
        var startPos = 0
        val outIds = ArrayList<Int>()
        var ttftMs = 0L

        // Empty KV cache for prefill
        val emptyShape = longArrayOf(1, kvHeads.toLong(), 0, headDim.toLong())
        var past: Map<String, OnnxTensor> = buildMap {
            for (i in 0 until numLayers) for (kv in listOf("key", "value"))
                put("past_key_values.$i.$kv", OnnxTensor.createTensor(env, FloatBuffer.allocate(0), emptyShape))
        }
        var prevResult: OrtSession.Result? = null

        try {
            for (step in 0 until maxNewTokens) {
                if (!shouldContinue()) break
                val feeds = HashMap<String, OnnxTensor>(past)
                val embedsT = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputsEmbeds), longArrayOf(1, seq.toLong(), hidden.toLong()))
                val maskT = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(total) { 1L }), longArrayOf(1, total.toLong()))
                val posT = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seq) { (startPos + it).toLong() }), longArrayOf(1, seq.toLong()))
                feeds["inputs_embeds"] = embedsT
                feeds["attention_mask"] = maskT
                feeds["position_ids"] = posT

                val result = decoder.run(feeds)
                embedsT.close(); maskT.close(); posT.close()
                // Inputs from the previous step are no longer needed
                if (prevResult == null) past.values.forEach { it.close() } else prevResult.close()
                prevResult = result

                if (step == 0) ttftMs = SystemClock.elapsedRealtime() - t0

                val logits = (result[0] as OnnxTensor).floatBuffer
                var best = 0
                var bestVal = Float.NEGATIVE_INFINITY
                for (i in 0 until logits.remaining()) {
                    val v = logits.get(i)
                    if (v > bestVal) { bestVal = v; best = i }
                }
                if (best == eosId) break
                outIds.add(best)
                onToken(tokenizer.decode(outIds))

                past = buildMap {
                    var j = 1
                    for (i in 0 until numLayers) for (kv in listOf("key", "value"))
                        put("past_key_values.$i.$kv", result[j++] as OnnxTensor)
                }
                startPos = total
                inputsEmbeds = embedIds(intArrayOf(best))
                seq = 1
                total += 1
            }
        } finally {
            if (prevResult == null) past.values.forEach { it.close() } else prevResult.close()
        }

        val text = tokenizer.decode(outIds).trim()
        return text to Stats(visionMs, ttftMs, SystemClock.elapsedRealtime() - t0, outIds.size)
    }

    override fun close() {
        vision.close(); embed.close(); decoder.close()
    }
}
